"""TurboQuant Optimierte Attention — MLX-nativ, voll vektorisiert.

Optimierungen vs. Basisversion:
  1. Kombiniertes Rotation+JL-Sketch in einem Matmul (1 Dispatch statt 2)
  2. Value-Assoziativität: (weights @ centroids) @ Pi statt weights @ (centroids @ Pi)
     → spart O(T_kv × D²) pro Token, reduziert auf O(T_q × D²)
  3. Kein Python-Loop, volle MLX-Graph-Optimierung
"""

import math

import mlx.core as mx

from turboquant.kernels import unpack_2bit_indices

# Bit-Indizes für Sign-Bit Entpackung (einmal alloziert)
_BITS_32 = mx.arange(32, dtype=mx.uint32)


def _unpack_sign_bits(sign_bits: mx.array) -> mx.array:
    """Entpackt uint32 Sign-Bits zu ±1.0 float32."""
    expanded = (sign_bits[..., None] >> _BITS_32) & 1
    flat_D = sign_bits.shape[-1] * 32
    result = expanded.reshape(*sign_bits.shape[:-1], flat_D)
    return 2.0 * result.astype(mx.float32) - 1.0


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """Optimierte TurboQuant Attention.

    Args:
        queries: (B, n_q_heads, T_q, D)
        cache: TurboQuantKVCache
        scale: Typisch 1/sqrt(D)
        mask: "causal", bool-array, oder None

    Returns:
        output: (B, n_q_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = cache.key_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    # --- 1. Rotation (+ JL-Sketch wenn QJL aktiv) ---
    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T  # (B, n_q_heads, T_q, 2D)
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    else:
        q_rot = q_scaled @ cache.rotation_matrix.T

    # --- 2. MSE-Score ---
    key_indices = cache.get_key_indices()[:, :, :T_kv, :]
    key_centroids = cache.centroids[key_indices]

    q_rot_grouped = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
    key_centroids_expanded = key_centroids[:, :, None, :, :]

    scores = q_rot_grouped @ key_centroids_expanded.transpose(0, 1, 2, 4, 3)
    scores = scores * cache.key_norms[:, :, :T_kv][:, :, None, None, :]

    # --- 3. QJL-Score (optional) ---
    if cache.use_qjl:
        q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_signs_float = _unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
        k_signs_expanded = k_signs_float[:, :, None, :, :]

        qjl_scores = q_sketch_grouped @ k_signs_expanded.transpose(0, 1, 2, 4, 3)
        qjl_scale = math.sqrt(math.pi / 2.0) / D
        qjl_scores = qjl_scores * qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    # --- 5. Mask ---
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            q_indices = mx.arange(T_kv - T_q, T_kv)
            k_indices = mx.arange(T_kv)
            causal_mask = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

    # --- 6. Softmax ---
    weights = mx.softmax(scores, axis=-1, precise=True)

    # --- 7. Value Output via Matrix-Assoziativität ---
    # Alt: output = weights @ (centroid_values @ Pi * norms)  →  O(T_kv × D²) Decode
    # Neu: output = ((weights * norms) @ centroid_values) @ Pi  →  O(T_q × D²) Rotate
    value_indices = cache.get_value_indices()[:, :, :T_kv, :]
    value_centroids = cache.centroids[value_indices]  # (B, n_kv_heads, T_kv, D)

    # Norms in Weights einrechnen: (B, kv, reps, T_q, T_kv) * (B, kv, 1, 1, T_kv)
    value_norms = cache.value_norms[:, :, :T_kv]
    weighted_norms = weights * value_norms[:, :, None, None, :]

    # Gewichtete Centroid-Summe: (B, kv, reps, T_q, T_kv) @ (B, kv, 1, T_kv, D)
    value_centroids_expanded = value_centroids[:, :, None, :, :]
    weighted_centroids = weighted_norms @ value_centroids_expanded  # (B, kv, reps, T_q, D)

    # Einmal inverse Rotation: (B, kv, reps, T_q, D) @ (D, D)
    output = weighted_centroids @ cache.rotation_matrix  # Pi^T @ c = c @ Pi
    output = output.reshape(B, n_q_heads, T_q, D)

    return output
