"""TurboQuantKVCache — Drop-in Replacement für mlx-lm's KV Cache.

Speichert Keys und Values quantisiert (PolarQuant + QJL für Keys).
Indices sind 2-bit gepackt in uint32 (16 pro Wort).

Memory Layout pro Token (head_dim=128, mse_bits=2):
  - Key packed indices: 128/16 * 4 = 32 bytes (uint32, 2-bit packed)
  - Key sign bits:      128/32 * 4 = 16 bytes (uint32)
  - Key norm:           4 bytes (float32)
  - Key residual norm:  4 bytes (float32)
  - Value packed indices: 32 bytes (uint32, 2-bit packed)
  - Value norm:         4 bytes (float32)
  Total: 92 bytes vs 512 bytes (float16) = 5.6x Kompression
"""

import mlx.core as mx

from turboquant.codebook import get_codebook
from turboquant.rotation import generate_rotation_matrix, generate_jl_matrix
from turboquant.kernels import (
    polarquant_encode,
    polarquant_decode,
    qjl_encode,
    pack_2bit_indices,
    unpack_2bit_indices,
    pack_3bit_indices,
    unpack_3bit_indices,
)


class TurboQuantKVCache:
    """TurboQuant KV Cache mit PolarQuant + QJL Kompression.

    Kompatibel mit mlx-lm's Cache-Interface (update_and_fetch, offset, etc.).
    is_turboquant Flag signalisiert dem gepatchten SDPA die Custom Attention.
    Indices sind 2-bit gepackt in uint32 für maximale Kompression.
    """

    is_turboquant = True

    def __init__(self, head_dim: int = 128, mse_bits: int = 2, use_qjl: bool = True, seed: int = 42):
        self.head_dim = head_dim
        self.mse_bits = mse_bits
        self.use_qjl = use_qjl
        self.offset = 0

        # Codebook (skaliert für normalisierte Vektoren)
        self.centroids, self.boundaries = get_codebook(mse_bits, head_dim)

        # Rotation + JL Matrizen (einmal berechnen)
        self.rotation_matrix = generate_rotation_matrix(head_dim, seed=seed)
        self.jl_matrix = generate_jl_matrix(head_dim, seed=seed + 95)

        # Precompute: Kombinierte Matrix für Rotation + JL-Sketch in einem Matmul
        # q_rot = q @ Pi^T, q_sketch = q_rot @ S^T = q @ Pi^T @ S^T = q @ (S @ Pi)^T
        self.combined_rot_jl = mx.concatenate(
            [self.rotation_matrix, self.jl_matrix @ self.rotation_matrix], axis=0
        )  # (2D, D) — Pi oben, S@Pi unten
        mx.eval(self.combined_rot_jl)

        # Quantisierter Storage (packed) — wird beim ersten update_and_fetch initialisiert
        self.key_packed = None      # uint32, 2-bit packed indices
        self.key_norms = None       # float32, L2-Normen
        self.key_sign_bits = None   # uint32, QJL sign bits
        self.key_residual_norms = None  # float32, gamma
        self.value_packed = None    # uint32, 2-bit packed indices
        self.value_norms = None     # float32, L2-Normen

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantisiert neue KV-Paare, speichert gepackt, gibt dequantisiert zurück.

        Args:
            keys: (B, n_kv_heads, T_new, D) — neue Keys (nach RoPE)
            values: (B, n_kv_heads, T_new, D) — neue Values

        Returns:
            (all_keys, all_values): Dequantisierte Tensoren (für Standard-SDPA Fallback)
        """
        # --- Keys quantisieren ---
        k_indices, k_norms, k_rotated = polarquant_encode(
            keys, self.rotation_matrix, self.boundaries
        )

        # Residual für QJL (optional)
        if self.use_qjl:
            k_reconstructed_rotated = self.centroids[k_indices.astype(mx.uint32)]
            k_residual = k_rotated - k_reconstructed_rotated
            k_sign_bits, k_residual_norms = qjl_encode(k_residual, self.jl_matrix)
        else:
            k_sign_bits = None
            k_residual_norms = None

        # Pack indices
        if self.mse_bits == 2:
            k_packed = pack_2bit_indices(k_indices)
        else:
            k_packed = pack_3bit_indices(k_indices)

        # --- Values quantisieren (nur MSE, kein QJL) ---
        v_indices, v_norms, _ = polarquant_encode(
            values, self.rotation_matrix, self.boundaries
        )
        if self.mse_bits == 2:
            v_packed = pack_2bit_indices(v_indices)
        else:
            v_packed = pack_3bit_indices(v_indices)

        # --- Ans Cache anhängen ---
        if self.key_packed is None:
            self.key_packed = k_packed
            self.key_norms = k_norms
            self.key_sign_bits = k_sign_bits
            self.key_residual_norms = k_residual_norms
            self.value_packed = v_packed
            self.value_norms = v_norms
        else:
            self.key_packed = mx.concatenate([self.key_packed, k_packed], axis=2)
            self.key_norms = mx.concatenate([self.key_norms, k_norms], axis=2)
            if self.use_qjl:
                self.key_sign_bits = mx.concatenate(
                    [self.key_sign_bits, k_sign_bits], axis=2
                )
                self.key_residual_norms = mx.concatenate(
                    [self.key_residual_norms, k_residual_norms], axis=2
                )
            self.value_packed = mx.concatenate(
                [self.value_packed, v_packed], axis=2
            )
            self.value_norms = mx.concatenate([self.value_norms, v_norms], axis=2)

        self.offset += keys.shape[2]

        # mx.eval nötig: Metal-Barrier zwischen Concat und Custom Kernel
        # garantiert nur Threadgroup-Level Ordering, nicht Memory-Write
        # Completion. Ohne eval liest der Fused Kernel stale GPU-Buffer.
        # → V2 (mx.quantized_matmul) hat dieses Problem nicht.
        state = [self.key_packed, self.key_norms, self.value_packed, self.value_norms]
        if self.use_qjl and self.key_sign_bits is not None:
            state.extend([self.key_sign_bits, self.key_residual_norms])
        mx.eval(*state)

        return keys, values

    def get_key_indices(self) -> mx.array:
        """Entpackt Key-Indices. Returns (..., D) uint32."""
        if self.mse_bits == 2:
            return unpack_2bit_indices(self.key_packed, self.head_dim)
        return unpack_3bit_indices(self.key_packed, self.head_dim)

    def get_value_indices(self) -> mx.array:
        """Entpackt Value-Indices. Returns (..., D) uint32."""
        if self.mse_bits == 2:
            return unpack_2bit_indices(self.value_packed, self.head_dim)
        return unpack_3bit_indices(self.value_packed, self.head_dim)

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        """Erstellt Attention-Mask kompatibel mit mlx-lm."""
        from mlx_lm.models.base import create_causal_mask

        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return create_causal_mask(N, offset=self.offset - N, window_size=window_size)
        return "causal"

    @property
    def state(self):
        """Cache-State für mx.eval() Kompatibilität mit mlx-lm."""
        if self.key_packed is None:
            return []
        parts = [self.key_packed, self.key_norms, self.value_packed, self.value_norms]
        if self.use_qjl and self.key_sign_bits is not None:
            parts.extend([self.key_sign_bits, self.key_residual_norms])
        return parts

    @state.setter
    def state(self, v):
        if not v:
            return
        self.key_packed = v[0]
        self.key_norms = v[1]
        self.value_packed = v[2]
        self.value_norms = v[3]
        if self.use_qjl and len(v) > 4:
            self.key_sign_bits = v[4]
            self.key_residual_norms = v[5]

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return False

    def empty(self):
        return self.key_packed is None

    @property
    def nbytes(self):
        if self.key_packed is None:
            return 0
        total = self.key_packed.nbytes + self.key_norms.nbytes
        total += self.value_packed.nbytes + self.value_norms.nbytes
        if self.use_qjl and self.key_sign_bits is not None:
            total += self.key_sign_bits.nbytes + self.key_residual_norms.nbytes
        return total

    @property
    def nbytes_equivalent_fp16(self):
        """Äquivalenter Speicher wenn alles in float16 wäre."""
        if self.key_packed is None:
            return 0
        B, n_kv_heads, T, _ = self.key_packed.shape
        D = self.head_dim
        return B * n_kv_heads * T * D * 2 * 2  # 2 bytes fp16 * 2 (keys + values)
