"""TurboQuantKVCache V2 — PolarQuant-Rotation + MLX-native Quantisierung.

Nutzt PolarQuant's Rotation für gleichmäßige Verteilung, dann MLX's
eingebaute affine Quantisierung (mx.quantize) + optimierten quantized_matmul
Metal Kernel für maximale Hardware-Nähe.

Pre-Allokation mit step=256 wie MLX's QuantizedKVCache für minimalen
Allokations-Overhead.
"""

import mlx.core as mx
from mlx.utils import tree_map

from turboquant.rotation import generate_rotation_matrix, generate_jl_matrix
from turboquant.kernels import pack_sign_bits, qjl_encode


class TurboQuantKVCacheV2:
    """TurboQuant V2 — PolarQuant Rotation + MLX native quantized_matmul.

    Speichert Keys/Values als MLX-quantisierte Tensoren im rotierten Raum.
    Scoring nutzt mx.quantized_matmul für maximale Performance.
    Pre-Allokation mit step=256 für O(T/256) statt O(T) Reallokationen.
    """

    is_turboquant_v2 = True
    step = 256

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        group_size: int = 64,
        use_qjl: bool = False,
        use_rotation: bool = True,
        use_normalization: bool = True,
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.use_qjl = use_qjl
        self.use_rotation = use_rotation
        self.use_normalization = use_normalization
        self.offset = 0
        self._el_per_int = 8 * mx.uint32.size // bits

        if use_rotation:
            self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)
            mx.eval(self.rotation_matrix)
        else:
            self.rotation_matrix = None

        if use_qjl:
            self.jl_matrix = generate_jl_matrix(head_dim, seed=seed + 95)
            mx.eval(self.jl_matrix)
            self.combined_rot_jl = mx.concatenate(
                [self.rotation_matrix, self.jl_matrix @ self.rotation_matrix], axis=0
            )
            mx.eval(self.combined_rot_jl)

        self.keys = None
        self.values = None
        self.key_norms = None
        self.value_norms = None
        self.key_sign_bits = None
        self.key_residual_norms = None

    def _ensure_capacity(self, B, n_kv_heads, num_steps, k_head_dim, v_head_dim, dtype):
        """Pre-alloziert oder expandiert Buffer nach MLX-Muster (step=256)."""
        prev = self.offset
        if self.keys is not None and (prev + num_steps) <= self.keys[0].shape[-2]:
            return

        new_steps = (self.step + num_steps - 1) // self.step * self.step
        shape = (B, n_kv_heads, new_steps)

        def init_quant(dim):
            return (
                mx.zeros((*shape, dim // self._el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            )

        def expand_quant(x):
            new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
            return mx.concatenate([x, new_x], axis=-2)

        if self.keys is not None:
            if prev % self.step != 0:
                self.keys, self.values = tree_map(
                    lambda x: x[..., :prev, :], (self.keys, self.values)
                )
            self.keys, self.values = tree_map(
                expand_quant, (self.keys, self.values)
            )
            if self.use_normalization and self.key_norms is not None:
                k_exp = mx.zeros((B, n_kv_heads, new_steps), dtype=dtype)
                v_exp = mx.zeros((B, n_kv_heads, new_steps), dtype=dtype)
                kn = self.key_norms if prev % self.step == 0 else self.key_norms[:, :, :prev]
                vn = self.value_norms if prev % self.step == 0 else self.value_norms[:, :, :prev]
                self.key_norms = mx.concatenate([kn, k_exp], axis=-1)
                self.value_norms = mx.concatenate([vn, v_exp], axis=-1)
        else:
            self.keys = init_quant(k_head_dim)
            self.values = init_quant(v_head_dim)
            if self.use_normalization:
                self.key_norms = mx.zeros((B, n_kv_heads, new_steps), dtype=dtype)
                self.value_norms = mx.zeros((B, n_kv_heads, new_steps), dtype=dtype)

    def _normed_quant(self, quant_tuple, norms):
        """Backt Norms in quantisierte Scales/Biases ein."""
        data, scales, biases = quant_tuple
        T = self.offset
        n = norms[:, :, :T, None]
        return (data[:, :, :T, :], scales[:, :, :T, :] * n, biases[:, :, :T, :] * n)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantisiert neue KV-Paare und schreibt in pre-allozierten Buffer."""
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        self._ensure_capacity(B, n_kv_heads, num_steps, k_head_dim, v_head_dim, keys.dtype)

        # --- Lean Path: Ohne Normalisierung ---
        if not self.use_normalization:
            if self.use_rotation:
                k_to_q = keys @ self.rotation_matrix.T
                v_to_q = values @ self.rotation_matrix.T
            else:
                k_to_q = keys
                v_to_q = values

            k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.bits)
            v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.bits)

            self.offset += num_steps
            for i in range(len(self.keys)):
                self.keys[i][..., prev:self.offset, :] = k_quant[i]
                self.values[i][..., prev:self.offset, :] = v_quant[i]

            return (
                tree_map(lambda x: x[..., :self.offset, :], self.keys),
                tree_map(lambda x: x[..., :self.offset, :], self.values),
            )

        # --- Full Path: Normalisierung + optional Rotation ---
        k_norms = mx.linalg.norm(keys, axis=-1, keepdims=True)
        v_norms = mx.linalg.norm(values, axis=-1, keepdims=True)
        safe_k_norms = mx.where(k_norms < 1e-8, mx.ones_like(k_norms), k_norms)
        safe_v_norms = mx.where(v_norms < 1e-8, mx.ones_like(v_norms), v_norms)

        k_normalized = keys / safe_k_norms
        v_normalized = values / safe_v_norms

        if self.use_rotation:
            k_to_q = k_normalized @ self.rotation_matrix.T
            v_to_q = v_normalized @ self.rotation_matrix.T
        else:
            k_to_q = k_normalized
            v_to_q = v_normalized

        k_quant = mx.quantize(k_to_q, group_size=self.group_size, bits=self.bits)
        v_quant = mx.quantize(v_to_q, group_size=self.group_size, bits=self.bits)

        # QJL auf Residual (optional)
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
            if self.key_sign_bits is None:
                self.key_sign_bits = k_sign_bits
                self.key_residual_norms = k_residual_norms
            else:
                self.key_sign_bits = mx.concatenate(
                    [self.key_sign_bits, k_sign_bits], axis=2
                )
                self.key_residual_norms = mx.concatenate(
                    [self.key_residual_norms, k_residual_norms], axis=2
                )

        return (
            self._normed_quant(self.keys, self.key_norms),
            self._normed_quant(self.values, self.value_norms),
        )

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        from mlx_lm.models.base import create_causal_mask
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return create_causal_mask(N, offset=self.offset - N, window_size=window_size)
        return "causal"

    @property
    def state(self):
        if self.keys is None:
            return []
        parts = list(tree_map(lambda x: x[..., :self.offset, :], self.keys))
        parts += list(tree_map(lambda x: x[..., :self.offset, :], self.values))
        if self.use_normalization and self.key_norms is not None:
            parts += [self.key_norms[:, :, :self.offset], self.value_norms[:, :, :self.offset]]
        if self.use_qjl and self.key_sign_bits is not None:
            parts += [self.key_sign_bits, self.key_residual_norms]
        return parts

    @state.setter
    def state(self, v):
        pass

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        T = self.offset
        total = 0
        for x in self.keys:
            per_step = x.nbytes / max(x.shape[-2], 1)
            total += int(per_step * T)
        for x in self.values:
            per_step = x.nbytes / max(x.shape[-2], 1)
            total += int(per_step * T)
        if self.use_normalization and self.key_norms is not None:
            total += T * self.key_norms[:, :, :1].nbytes
            total += T * self.value_norms[:, :, :1].nbytes
        if self.use_qjl and self.key_sign_bits is not None:
            total += self.key_sign_bits.nbytes + self.key_residual_norms.nbytes
        return total

    @property
    def nbytes_equivalent_fp16(self):
        if self.keys is None:
            return 0
        B, n_kv_heads = self.keys[0].shape[:2]
        T = self.offset
        D = self.head_dim
        return B * n_kv_heads * T * D * 2 * 2
