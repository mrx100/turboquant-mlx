"""TurboQuant V2 Attention — Nutzt mx.quantized_matmul für Hardware-Nähe.

Pipeline:
  1. q_rot = q * scale @ Pi^T                    (MLX matmul)
  2. scores = mx.quantized_matmul(q_rot, keys)    (MLX optimierter Metal Kernel!)
  3. scores *= key_norms                           (Element-wise)
  4. [optional QJL correction]
  5. weights = softmax(scores)
  6. output_rot = mx.quantized_matmul(weights, values)  (MLX Metal Kernel!)
  7. output = output_rot @ Pi * value_norms        (Inverse Rotation)
"""

import math

import mlx.core as mx
from mlx.utils import tree_map

_BITS_32 = mx.arange(32, dtype=mx.uint32)


def _unpack_sign_bits(sign_bits: mx.array) -> mx.array:
    expanded = (sign_bits[..., None] >> _BITS_32) & 1
    flat_D = sign_bits.shape[-1] * 32
    result = expanded.reshape(*sign_bits.shape[:-1], flat_D)
    return 2.0 * result.astype(mx.float32) - 1.0


def turboquant_v2_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    cache,
    scale: float,
    mask=None,
) -> mx.array:
    """TurboQuant V2 Attention mit mx.quantized_matmul.

    Args:
        queries: (B, n_q_heads, T_q, D)
        q_keys: (quantized_data, scales, biases) — quantisierte rotierte Keys
        q_values: (quantized_data, scales, biases) — quantisierte rotierte Values
        cache: TurboQuantKVCacheV2
        scale: 1/sqrt(D)
        mask: "causal", bool-array, oder None

    Returns:
        output: (B, n_q_heads, T_q, D)
    """
    B, n_q_heads, T_q, D = queries.shape
    n_kv_heads = q_keys[0].shape[1]
    n_repeats = n_q_heads // n_kv_heads
    T_kv = cache.offset

    # --- 1. Query rotieren (optional) ---
    q_scaled = queries * scale
    if cache.use_qjl:
        q_combined = q_scaled @ cache.combined_rot_jl.T
        q_rot = q_combined[..., :D]
        q_sketch = q_combined[..., D:]
    elif cache.use_rotation:
        q_rot = q_scaled @ cache.rotation_matrix.T
    else:
        q_rot = q_scaled

    # --- 2. Scores via mx.quantized_matmul ---
    # GQA: reshape queries für Gruppenstruktur
    if n_repeats > 1:
        q_rot_grouped = q_rot.reshape(B, n_kv_heads, n_repeats, T_q, D)
        q_keys_expanded = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
    else:
        q_rot_grouped = q_rot[:, :, None, :, :]
        q_keys_expanded = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)

    # Norms sind bereits in Scales/Biases eingebacken (cache._normed_quant)
    scores = mx.quantized_matmul(
        q_rot_grouped, *q_keys_expanded,
        transpose=True, group_size=cache.group_size, bits=cache.bits,
    )

    # --- 3. QJL-Score (optional) ---
    if cache.use_qjl and cache.key_sign_bits is not None:
        q_sketch_grouped = q_sketch.reshape(B, n_kv_heads, n_repeats, T_q, D)
        k_signs = _unpack_sign_bits(cache.key_sign_bits[:, :, :T_kv, :])
        k_signs_exp = k_signs[:, :, None, :, :]

        qjl_scores = q_sketch_grouped @ k_signs_exp.transpose(0, 1, 2, 4, 3)
        qjl_scale = math.sqrt(math.pi / 2.0) / D
        qjl_scores = qjl_scores * qjl_scale * cache.key_residual_norms[:, :, :T_kv][:, :, None, None, :]
        scores = scores + qjl_scores

    # --- 4. Mask ---
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            causal_mask = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal_mask, scores, mx.finfo(scores.dtype).min)
        elif isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

    # --- 5. Softmax ---
    weights = mx.softmax(scores, axis=-1, precise=True)

    # --- 6. Value Output via quantized_matmul + inverse Rotation ---
    # weights @ dequant(values) im rotierten Raum
    if n_repeats > 1:
        q_values_expanded = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)
    else:
        q_values_expanded = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    # Norms sind bereits in Value-Scales eingebacken
    output_rot = mx.quantized_matmul(
        weights, *q_values_expanded,
        transpose=False, group_size=cache.group_size, bits=cache.bits,
    )

    # Inverse Rotation (nur wenn Rotation aktiv)
    if cache.use_rotation:
        output = output_rot @ cache.rotation_matrix
    else:
        output = output_rot
    output = output.reshape(B, n_q_heads, T_q, D)

    return output
