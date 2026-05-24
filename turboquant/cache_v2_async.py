"""AsyncTurboQuantKVCacheV2 — Optimized KV cache with selective async quantization.

Extends TurboQuantKVCacheV2 with intelligent async/sync switching:
- Single-token decoding (T=1): Uses sync quantization (lower latency)
- Batched prefill (T>threshold): Uses async quantization with mx.async_eval
- Automatic threshold detection based on batch size

Pipeline (batched mode):
  Forward Pass → FP16 K/V → [Pending] → async quantize → [Ready] → Attention
       ↓                                                    ↓
    next tokens                                    quantized_matmul (reads)

Memory overhead: +1× FP16 buffer size (~2-3GB for Qwen3.6-35B at full context)
Expected speedup: 10-25% during prefill phase for long prompts
"""

import math

import mlx.core as mx
from mlx.utils import tree_map

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.rotation import safe_normalize
from turboquant.qjl import qjl_encode


class AsyncTurboQuantKVCacheV2(TurboQuantKVCacheV2):
    """Async double-buffered KV cache with pipelined quantization.
    
    Uses mx.async_eval for non-blocking quantization, allowing
    the forward pass and quantization to overlap in time.
    
    Automatically switches to sync mode for single-token decoding to
    avoid overhead.
    """
    
    # Async threshold: minimum tokens per call to benefit from async
    ASYNC_THRESHOLD = 16
    
    def __init__(self, *args, async_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Async state
        self._async_enabled = True  # Always enabled, uses mx.async_eval
        self._async_keys = None
        self._async_values = None
        self._async_k_norms = None
        self._async_v_norms = None
        self._async_k_sign_bits = None
        self._async_k_residual_norms = None
        self._pending_keys = None
        self._pending_values = None
        self._pending_k_normalized = None
        self._pending_v_normalized = None
        self._pending_k_norms = None
        self._pending_v_norms = None
        self._pending_k_sign_bits = None
        self._pending_k_residual_norms = None
        self._pending_num_steps = 0
        self._quant_in_progress = False
        self._async_ops_count = 0
        self._sync_fallback_count = 0
        self._async_threshold = async_threshold or self.ASYNC_THRESHOLD
    
    def _should_use_async(self, num_steps):
        """Determine if async quantization is beneficial for this batch size."""
        return self._async_enabled and num_steps >= self._async_threshold
    
    def _quantize_async(self, keys, values):
        """Start async quantization using mx.async_eval.
        
        This method captures the FP16 K/V pairs and starts quantization
        asynchronously. The main thread can continue with other work.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        
        # First call: must quantize synchronously to initialize buffers
        if self.keys is None:
            self._sync_fallback_count += 1
            return self._quantize_sync(keys, values)
        
        # Check if async is beneficial for this batch size
        if not self._should_use_async(num_steps):
            self._sync_fallback_count += 1
            return self._quantize_sync(keys, values)
        
        # Wait for previous async quantization to complete
        if self._quant_in_progress:
            if self._pending_keys is not None:
                mx.eval(self._pending_keys)
            self._commit_async_quantization()
        
        # Store pending FP16 data
        self._pending_keys = keys
        self._pending_values = values
        self._pending_num_steps = num_steps
        
        # Start quantization asynchronously
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
        
        # Schedule async evaluation
        to_eval = list(k_quant) + list(v_quant) + [k_normalized, v_normalized, k_norms, v_norms]
        if k_sign_bits is not None:
            to_eval.extend([k_sign_bits, k_residual_norms])
        mx.async_eval(*to_eval)
        
        # Store async results
        self._async_keys = k_quant
        self._async_values = v_quant
        self._async_k_norms = k_norms
        self._async_v_norms = v_norms
        self._async_k_sign_bits = k_sign_bits
        self._async_k_residual_norms = k_residual_norms
        self._pending_k_normalized = k_normalized
        self._pending_v_normalized = v_normalized
        
        self._quant_in_progress = True
        self._async_ops_count += 1
        
        # Return already-quantized buffer (from previous iteration)
        return (
            tree_map(lambda x: x[..., :self.offset, :], self.keys) if self.keys is not None else None,
            tree_map(lambda x: x[..., :self.offset, :], self.values) if self.values is not None else None,
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
        
        # Safety: ensure async data exists before eval
        if self._async_keys is None or self._async_values is None:
            self._quant_in_progress = False
            return
        
        # Evaluate pending data to ensure async ops are complete
        mx.eval(self._async_keys[0], self._async_values[0])
        self._quant_in_progress = False
        
        # Commit quantized data to main buffers
        prev = self.offset
        num_steps = self._pending_num_steps
        
        self._ensure_capacity(
            self._pending_keys.shape[0],
            self._pending_keys.shape[1],
            num_steps,
            self._pending_keys.shape[-1],
            self._pending_values.shape[-1],
            self._pending_keys.dtype,
        )
        
        self.offset += num_steps
        
        for i in range(len(self.keys)):
            self.keys[i][..., prev:self.offset, :] = self._async_keys[i]
            self.values[i][..., prev:self.offset, :] = self._async_values[i]
        
        # Fix: Handle case where async norms might be None (e.g., speculative decoding re-runs)
        if self._async_k_norms is not None:
            self.key_norms[:, :, prev:self.offset] = self._async_k_norms.squeeze(-1)
        if self._async_v_norms is not None:
            self.value_norms[:, :, prev:self.offset] = self._async_v_norms.squeeze(-1)
        
        if self.use_qjl and self._async_k_sign_bits is not None:
            self.key_sign_bits[:, :, prev:self.offset, :] = self._async_k_sign_bits
            self.key_residual_norms[:, :, prev:self.offset] = self._async_k_residual_norms
        
        # Clear pending data
        self._pending_keys = None
        self._pending_values = None
        self._pending_k_norms = None
        self._pending_v_norms = None
        self._pending_num_steps = 0
    
    def update_and_fetch(self, keys, values):
        """Smart update: uses async for batched prefill, sync for single-token decoding.
        
        Pipeline:
        1. Check batch size against async threshold
        2. If batched (T >= threshold): async quantization with double-buffering
        3. If single-token (T < threshold): sync quantization (lower latency)
        4. Return quantized buffer for attention
        """
        num_steps = keys.shape[-2]
        
        # Use async only for batched operations
        if self._should_use_async(num_steps):
            return self._quantize_async(keys, values)
        else:
            self._sync_fallback_count += 1
            return self._quantize_sync(keys, values)
    
    def flush(self):
        """Force completion of any pending async quantization."""
        if self._quant_in_progress and self._async_keys is not None:
            mx.eval(self._async_keys[0], self._async_values[0])
            self._commit_async_quantization()
    
    @property
    def async_stats(self):
        """Returns async quantization statistics."""
        return {
            "async_enabled": self._async_enabled,
            "async_ops_count": self._async_ops_count,
            "sync_fallback_count": self._sync_fallback_count,
            "quant_in_progress": self._quant_in_progress,
            "async_threshold": self._async_threshold,
        }
