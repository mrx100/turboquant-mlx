"""AsyncTurboQuantKVCacheV2 — Double-buffered pipelined quantization.

Extends TurboQuantKVCacheV2 with asynchronous quantization using MLX streams.
While the attention mechanism reads from the quantized buffer, the next batch
of FP16 K/V pairs is being quantized on a separate stream in the background.

Pipeline:
  Forward Pass → FP16 K/V → [Pending] → async quantize → [Ready] → Attention
       ↓                                                    ↓
    next tokens                                    quantized_matmul (reads)

Memory overhead: +1× FP16 buffer size (~2-3GB for Qwen3.6-35B at full context)
Expected speedup: 15-25% tok/s improvement due to pipelined quantization
"""

import math

import mlx.core as mx
from mlx.utils import tree_map

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.rotation import safe_normalize
from turboquant.qjl import qjl_encode


class AsyncTurboQuantKVCacheV2(TurboQuantKVCacheV2):
    """Async double-buffered KV cache with pipelined quantization.
    
    Uses a separate MLX stream for quantization operations, allowing
    the forward pass and quantization to overlap in time.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Async state
        self._quant_stream = None
        self._async_enabled = False
        self._pending_keys = None
        self._pending_values = None
        self._pending_k_norms = None
        self._pending_v_norms = None
        self._pending_k_normalized = None
        self._pending_v_normalized = None
        self._async_keys = None
        self._async_values = None
        self._async_k_norms = None
        self._async_v_norms = None
        self._async_k_sign_bits = None
        self._async_k_residual_norms = None
        self._quant_in_progress = False
        self._async_ops_count = 0
        self._sync_fallback_count = 0
        
        # Try to enable async
        try:
            self._quant_stream = mx.new_stream()
            self._async_enabled = True
        except Exception:
            self._async_enabled = False
    
    def _quantize_async(self, keys, values):
        """Start async quantization on separate stream.
        
        This method captures the FP16 K/V pairs and starts quantization
        on the async stream. The main thread can continue with other work.
        """
        if not self._async_enabled:
            self._sync_fallback_count += 1
            return self._quantize_sync(keys, values)
        
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        
        # Store pending FP16 data
        self._pending_keys = keys
        self._pending_values = values
        
        # Start quantization on async stream
        with mx.stream(self._quant_stream):
            # Normalization
            k_normalized, k_norms = safe_normalize(keys)
            v_normalized, v_norms = safe_normalize(values)
            
            # Rotation
            if self.use_rotation:
                k_to_q = k_normalized @ self.rotation_matrix.T
                v_to_q = v_normalized @ self.rotation_matrix.T
            else:
                k_to_q = k_normalized
                v_to_q = v_normalized
            
            # Quantization
            k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.k_bits)
            v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.v_bits)
            
            # QJL on residual (optional)
            k_sign_bits = None
            k_residual_norms = None
            if self.use_qjl:
                k_dequant = mx.dequantize(*k_quant, group_size=self.group_size, bits=self.bits)
                k_residual = k_to_q - k_dequant
                k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)
            
            # Store async results
            self._async_keys = k_quant
            self._async_values = v_quant
            self._async_k_norms = k_norms
            self._async_v_norms = v_norms
            self._async_k_sign_bits = k_sign_bits
            self._async_k_residual_norms = k_residual_norms
            self._pending_k_normalized = k_normalized
            self._pending_v_normalized = v_normalized
            
            # Trigger async evaluation
            mx.async_eval(*k_quant, *v_quant)
            self._quant_in_progress = True
            self._async_ops_count += 1
        
        return (
            tree_map(lambda x: x[..., :self.offset, :], self.keys),
            tree_map(lambda x: x[..., :self.offset, :], self.values),
        )
    
    def _quantize_sync(self, keys, values):
        """Synchronous quantization fallback."""
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset
        
        self._ensure_capacity(B, n_kv_heads, num_steps, k_head_dim, v_head_dim, keys.dtype)
        
        if not self.use_normalization:
            if self.use_rotation:
                k_to_q = keys @ self.rotation_matrix.T
                v_to_q = values @ self.rotation_matrix.T
            else:
                k_to_q = keys
                v_to_q = values
            
            k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.k_bits)
            v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.v_bits)
            
            self.offset += num_steps
            for i in range(len(self.keys)):
                self.keys[i][..., prev:self.offset, :] = k_quant[i]
                self.values[i][..., prev:self.offset, :] = v_quant[i]
            
            return (
                tree_map(lambda x: x[..., :self.offset, :], self.keys),
                tree_map(lambda x: x[..., :self.offset, :], self.values),
            )
        
        # Full path with normalization
        k_normalized, k_norms = safe_normalize(keys)
        v_normalized, v_norms = safe_normalize(values)
        
        if self.use_rotation:
            k_to_q = k_normalized @ self.rotation_matrix.T
            v_to_q = v_normalized @ self.rotation_matrix.T
        else:
            k_to_q = k_normalized
            v_to_q = v_normalized
        
        k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.k_bits)
        v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.v_bits)
        
        if self.use_qjl:
            k_dequant = mx.dequantize(*k_quant, group_size=self.group_size, bits=self.bits)
            k_residual = k_to_q - k_dequant
            k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)
        
        self.offset += num_steps
        for i in range(len(self.keys)):
            self.keys[i][..., prev:self.offset, :] = k_quant[i]
            self.values[i][..., prev:self.offset, :] = v_quant[i]
        
        self.key_norms[:, :, prev:self.offset] = k_norms.squeeze(-1)
        self.value_norms[:, :, prev:self.offset] = v_norms.squeeze(-1)
        
        if self.use_qjl:
            self.key_sign_bits[:, :, prev:self.offset, :] = k_sign_bits
            self.key_residual_norms[:, :, prev:self.offset] = k_residual_norms
        
        return (
            self._normed_quant(self.keys, self.key_norms),
            self._normed_quant(self.values, self.value_norms),
        )
    
    def _commit_async_quantization(self):
        """Wait for async quantization to complete and commit results."""
        if not self._quant_in_progress:
            return
        
        # Synchronize with async stream
        mx.synchronize(self._quant_stream)
        self._quant_in_progress = False
        
        # Commit quantized data to main buffers
        prev = self.offset
        num_steps = self._pending_keys.shape[-2]
        self.offset += num_steps
        
        for i in range(len(self.keys)):
            self.keys[i][..., prev:self.offset, :] = self._async_keys[i]
            self.values[i][..., prev:self.offset, :] = self._async_values[i]
        
        self.key_norms[:, :, prev:self.offset] = self._async_k_norms.squeeze(-1)
        self.value_norms[:, :, prev:self.offset] = self._async_v_norms.squeeze(-1)
        
        if self.use_qjl and self._async_k_sign_bits is not None:
            self.key_sign_bits[:, :, prev:self.offset, :] = self._async_k_sign_bits
            self.key_residual_norms[:, :, prev:self.offset] = self._async_k_residual_norms
        
        # Clear pending data
        self._pending_keys = None
        self._pending_values = None
        self._pending_k_norms = None
        self._pending_v_norms = None
    
    def update_and_fetch(self, keys, values):
        """Async version: starts quantization in background, returns ready buffer.
        
        Pipeline:
        1. Wait for previous async quantization to complete (if any)
        2. Start new async quantization for current K/V
        3. Return already-quantized buffer for attention
        """
        # Wait for previous async quantization
        if self._quant_in_progress:
            self._commit_async_quantization()
        
        # Start new async quantization
        return self._quantize_async(keys, values)
    
    def flush(self):
        """Force completion of any pending async quantization."""
        if self._quant_in_progress:
            self._commit_async_quantization()
    
    @property
    def async_stats(self):
        """Returns async quantization statistics."""
        return {
            "async_enabled": self._async_enabled,
            "async_ops_count": self._async_ops_count,
            "sync_fallback_count": self._sync_fallback_count,
            "quant_in_progress": self._quant_in_progress,
        }
