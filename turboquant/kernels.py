"""Custom Metal Kernels für TurboQuant.

Strategie: Matmul-Operationen (Rotation, Centroid-Lookup) via mx.matmul/mx.take.
Nur die Bit-Operationen (Quantisierung, Sign-Packing, QJL-Scoring) als Metal Kernels.
"""

import math

import mlx.core as mx
import mlx.nn as nn


# --- Kernel 1: Scalar Quantize (Binary Search auf Boundaries) ---

_QUANTIZE_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint num_boundaries = boundaries_shape[0];

    float val = rotated[elem];

    // Binary search: zähle wie viele Boundaries kleiner als val
    uint idx = 0;
    for (uint b = 0; b < num_boundaries; b++) {
        idx += (val > boundaries[b]) ? 1 : 0;
    }
    indices[elem] = static_cast<uint8_t>(idx);
"""

_quantize_kernel = mx.fast.metal_kernel(
    name="turboquant_quantize",
    input_names=["rotated", "boundaries"],
    output_names=["indices"],
    source=_QUANTIZE_SOURCE,
)


def quantize_to_indices(rotated: mx.array, boundaries: mx.array) -> mx.array:
    """Quantisiert rotierte Werte zu Centroid-Indices via Binary Search.

    Args:
        rotated: Beliebiger Shape, float32 — die rotierten (und normierten) Koordinaten
        boundaries: (num_levels - 1,) float32 — sortierte Entscheidungsgrenzen

    Returns:
        indices: Gleicher Shape wie rotated, uint8
    """
    flat = rotated.reshape(-1)
    outputs = _quantize_kernel(
        inputs=[flat, boundaries],
        grid=(flat.size, 1, 1),
        threadgroup=(min(256, flat.size), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.uint8],
    )
    return outputs[0].reshape(rotated.shape)


# --- Kernel 2: Pack Sign Bits in uint32 ---

_PACK_SIGNS_SOURCE = """
    // Ein Thread pro uint32-Wort (packt 32 Sign-Bits)
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 32;
    uint total = values_shape[0];

    uint32_t packed = 0;
    for (uint bit = 0; bit < 32; bit++) {
        uint idx = base + bit;
        if (idx < total) {
            packed |= ((values[idx] > 0.0f) ? 1u : 0u) << bit;
        }
    }
    sign_bits[word_idx] = packed;
"""

_pack_signs_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_signs",
    input_names=["values"],
    output_names=["sign_bits"],
    source=_PACK_SIGNS_SOURCE,
)


def pack_sign_bits(values: mx.array) -> mx.array:
    """Packt die Vorzeichen von values in uint32-Wörter (32 bits pro Wort).

    Args:
        values: (..., D) float32 — die Projektionswerte

    Returns:
        sign_bits: (..., D // 32) uint32 — gepackte Vorzeichen-Bits
    """
    original_shape = values.shape
    D = original_shape[-1]
    if D % 32 != 0:
        raise ValueError(f"Letzte Dimension muss durch 32 teilbar sein, ist {D}")

    flat = values.reshape(-1)
    num_words = flat.size // 32
    outputs = _pack_signs_kernel(
        inputs=[flat],
        grid=(num_words, 1, 1),
        threadgroup=(min(256, num_words), 1, 1),
        output_shapes=[(num_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (D // 32,)
    return outputs[0].reshape(out_shape)


# --- Kernel 3: QJL Score — Dot-Product mit gepackten Sign-Bits ---

_QJL_SCORE_HEADER = """
inline float popcount_xnor(uint32_t a, uint32_t b) {
    // XNOR: gleiche Bits = 1, verschiedene = 0
    // popcount(XNOR) = Anzahl übereinstimmender Bits
    // Score = 2 * matching - total = 2 * popcount(~(a^b)) - 32
    uint32_t xnor_val = ~(a ^ b);
    return static_cast<float>(2 * static_cast<int>(popcount(xnor_val)) - 32);
}
"""

_QJL_SCORE_SOURCE = """
    // Grid: (num_queries * num_keys, 1, 1)
    // Jeder Thread berechnet den QJL-Score für ein (query, key) Paar
    uint pair_idx = thread_position_in_grid.x;
    uint num_keys = key_signs_shape[0];
    uint num_words = key_signs_shape[1];  // D / 32

    uint q_idx = pair_idx / num_keys;
    uint k_idx = pair_idx % num_keys;

    float score = 0.0f;
    for (uint w = 0; w < num_words; w++) {
        // Query-Sketch ist auch als Sign-Bits gepackt
        uint32_t q_word = query_signs[q_idx * num_words + w];
        uint32_t k_word = key_signs[k_idx * num_words + w];
        score += popcount_xnor(q_word, k_word);
    }
    scores[pair_idx] = score;
"""

_qjl_score_kernel = mx.fast.metal_kernel(
    name="turboquant_qjl_score",
    input_names=["query_signs", "key_signs"],
    output_names=["scores"],
    source=_QJL_SCORE_SOURCE,
    header=_QJL_SCORE_HEADER,
)


def qjl_score(query_signs: mx.array, key_signs: mx.array) -> mx.array:
    """Berechnet QJL Inner-Product-Scores zwischen Queries und Keys.

    Nutzt XNOR + Popcount für effizientes 1-Bit Dot-Product.

    Args:
        query_signs: (T_q, D // 32) uint32 — gepackte Query-Sketch-Signs
        key_signs: (T_kv, D // 32) uint32 — gepackte Key-Residual-Signs

    Returns:
        scores: (T_q, T_kv) float32 — die QJL-Score-Anteile
    """
    T_q = query_signs.shape[0]
    T_kv = key_signs.shape[0]
    total_pairs = T_q * T_kv

    if total_pairs == 0:
        return mx.zeros((T_q, T_kv))

    outputs = _qjl_score_kernel(
        inputs=[query_signs, key_signs],
        grid=(total_pairs, 1, 1),
        threadgroup=(min(256, total_pairs), 1, 1),
        output_shapes=[(total_pairs,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(T_q, T_kv)


# --- Kernel 4: Pack 2-bit Indices in uint32 (16 Indices pro Wort) ---

_PACK_2BIT_SOURCE = """
    // Ein Thread pro uint32-Wort (packt 16 2-bit Indices)
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 16;
    uint total = indices_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 16; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(indices[idx]) & 0x3u) << (i * 2);
        }
    }
    packed_indices[word_idx] = packed;
"""

_pack_2bit_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_2bit",
    input_names=["indices"],
    output_names=["packed_indices"],
    source=_PACK_2BIT_SOURCE,
)


def pack_2bit_indices(indices: mx.array) -> mx.array:
    """Packt 2-bit Indices (0-3) in uint32-Wörter (16 Indices pro Wort).

    Args:
        indices: (..., D) uint8 mit Werten 0-3

    Returns:
        packed: (..., D // 16) uint32
    """
    original_shape = indices.shape
    D = original_shape[-1]
    if D % 16 != 0:
        raise ValueError(f"Letzte Dimension muss durch 16 teilbar sein, ist {D}")

    flat = indices.reshape(-1)
    num_words = flat.size // 16
    outputs = _pack_2bit_kernel(
        inputs=[flat],
        grid=(num_words, 1, 1),
        threadgroup=(min(256, num_words), 1, 1),
        output_shapes=[(num_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (D // 16,)
    return outputs[0].reshape(out_shape)


# Bit-Masken für 2-bit Entpackung (einmal berechnen)
_SHIFTS_2BIT = mx.array([i * 2 for i in range(16)], dtype=mx.uint32)


def unpack_2bit_indices(packed: mx.array, D: int) -> mx.array:
    """Entpackt uint32 zu 2-bit Indices (0-3).

    Args:
        packed: (..., D // 16) uint32
        D: Original-Dimension (z.B. 128)

    Returns:
        indices: (..., D) uint32 mit Werten 0-3
    """
    # Jedes uint32 zu 16 2-bit Werten expandieren
    expanded = (packed[..., None] >> _SHIFTS_2BIT) & 0x3  # (..., D//16, 16)
    return expanded.reshape(*packed.shape[:-1], D)


# --- Kernel 5: Pack 3-bit Indices in uint32 (10 Indices pro Wort, 2 bits unused) ---

_PACK_3BIT_SOURCE = """
    uint word_idx = thread_position_in_grid.x;
    uint base = word_idx * 10;
    uint total = indices_shape[0];

    uint32_t packed = 0;
    for (uint i = 0; i < 10; i++) {
        uint idx = base + i;
        if (idx < total) {
            packed |= (static_cast<uint32_t>(indices[idx]) & 0x7u) << (i * 3);
        }
    }
    packed_indices[word_idx] = packed;
"""

_pack_3bit_kernel = mx.fast.metal_kernel(
    name="turboquant_pack_3bit",
    input_names=["indices"],
    output_names=["packed_indices"],
    source=_PACK_3BIT_SOURCE,
)


def pack_3bit_indices(indices: mx.array) -> mx.array:
    """Packt 3-bit Indices (0-7) in uint32-Wörter (10 Indices pro Wort).

    Args:
        indices: (..., D) uint8 mit Werten 0-7

    Returns:
        packed: (..., ceil(D / 10)) uint32
    """
    original_shape = indices.shape
    D = original_shape[-1]
    # Padding auf Vielfaches von 10
    num_words = (D + 9) // 10

    flat = indices.reshape(-1)
    total_words = flat.size // D * num_words
    # Pad to multiple of 10
    if D % 10 != 0:
        pad_size = num_words * 10 - D
        batch_size = flat.size // D
        flat_padded = flat.reshape(-1, D)
        flat_padded = mx.pad(flat_padded, [(0, 0), (0, pad_size)])
        flat = flat_padded.reshape(-1)

    total_elements = flat.size
    total_words = total_elements // 10
    outputs = _pack_3bit_kernel(
        inputs=[flat],
        grid=(total_words, 1, 1),
        threadgroup=(min(256, total_words), 1, 1),
        output_shapes=[(total_words,)],
        output_dtypes=[mx.uint32],
    )
    out_shape = original_shape[:-1] + (num_words,)
    return outputs[0].reshape(out_shape)


_SHIFTS_3BIT = mx.array([i * 3 for i in range(10)], dtype=mx.uint32)


def unpack_3bit_indices(packed: mx.array, D: int) -> mx.array:
    """Entpackt uint32 zu 3-bit Indices (0-7).

    Args:
        packed: (..., ceil(D/10)) uint32
        D: Original-Dimension

    Returns:
        indices: (..., D) uint32 mit Werten 0-7
    """
    expanded = (packed[..., None] >> _SHIFTS_3BIT) & 0x7  # (..., words, 10)
    total = packed.shape[-1] * 10
    result = expanded.reshape(*packed.shape[:-1], total)
    # Trim to actual D
    return result[..., :D]


# --- High-Level Encode/Decode Funktionen (nutzen Kernels + MLX Ops) ---

def polarquant_encode(
    vectors: mx.array,
    rotation_matrix: mx.array,
    boundaries: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """PolarQuant Encoding: Normalisieren → Rotieren → Scalar Quantize.

    Args:
        vectors: (..., D) float32 — Input-Vektoren
        rotation_matrix: (D, D) float32 — orthogonale Matrix Pi
        boundaries: (num_levels - 1,) float32 — Quantisierungsgrenzen (unskaliert)

    Returns:
        indices: (..., D) uint8 — Centroid-Indices
        norms: (...,) float32 — L2-Normen der Inputs
        rotated_normalized: (..., D) float32 — rotierte normalisierte Vektoren (für Residual)
    """
    # L2-Norm pro Vektor
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    # Guard: Null-Vektoren abfangen
    safe_norms = mx.where(norms < 1e-8, mx.ones_like(norms), norms)
    normalized = vectors / safe_norms

    # Rotieren: x_rot = x_norm @ Pi^T
    rotated = normalized @ rotation_matrix.T

    # Scalar Quantize via Metal Kernel
    indices = quantize_to_indices(rotated, boundaries)

    return indices, norms.squeeze(-1), rotated


def polarquant_decode(
    indices: mx.array,
    rotation_matrix: mx.array,
    centroids: mx.array,
    norms: mx.array,
) -> mx.array:
    """PolarQuant Decoding: Centroid-Lookup → Inverse Rotation → Skalierung.

    Args:
        indices: (..., D) uint8 — Centroid-Indices
        rotation_matrix: (D, D) float32 — orthogonale Matrix Pi
        centroids: (num_levels,) float32 — Centroid-Werte (unskaliert)
        norms: (...,) float32 — L2-Normen

    Returns:
        reconstructed: (..., D) float32 — rekonstruierte Vektoren
    """
    # Centroid lookup
    centroid_values = centroids[indices.astype(mx.uint32)]

    # Inverse Rotation: x = c @ Pi (= Pi^T @ c da Pi orthogonal → Pi^(-1) = Pi^T)
    reconstructed = centroid_values @ rotation_matrix

    # Skalierung mit Original-Norm
    return reconstructed * norms[..., None]


def qjl_encode(
    residual: mx.array,
    jl_matrix: mx.array,
) -> tuple[mx.array, mx.array]:
    """QJL Encoding: Projizieren → Sign-Bits packen.

    Args:
        residual: (..., D) float32 — Residual-Vektoren (x_norm - dequant_mse(x_norm))
        jl_matrix: (D, D) float32 — JL Projektionsmatrix S

    Returns:
        sign_bits: (..., D // 32) uint32 — gepackte Sign-Bits
        residual_norms: (...,) float32 — ||residual||_2 (gamma)
    """
    # Residual-Norm (gamma im Paper)
    residual_norms = mx.linalg.norm(residual, axis=-1)

    # Projektion: p = residual @ S^T
    projected = residual @ jl_matrix.T

    # Sign-Bits packen via Metal Kernel
    sign_bits = pack_sign_bits(projected)

    return sign_bits, residual_norms


# --- Fused Kernel 1: TurboQuant Score (MSE + QJL in einem Kernel) ---

_FUSED_SCORE_SOURCE = """
    // Grid: (T_kv, n_repeats, 1) — ein Thread pro (key, repeat) Paar
    uint k = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;

    uint T_kv = key_norms_shape[0];
    uint D_DIV_16 = key_packed_shape[1];
    uint D_DIV_32 = key_sign_bits_shape[1];
    uint D = D_DIV_16 * 16;
    uint n_repeats = q_rot_shape[0];  // q_rot is (n_repeats, D)

    if (k >= T_kv) return;

    // q_rot und q_sketch liegen als (n_repeats, D) vor
    uint q_base = r * D;

    // 1. MSE Score: Unpack 2-bit indices + Centroid lookup + Dot product
    float mse_score = 0.0f;
    for (uint w = 0; w < D_DIV_16; w++) {
        uint32_t packed = key_packed[k * D_DIV_16 + w];
        for (uint i = 0; i < 16; i++) {
            uint idx = (packed >> (i * 2)) & 0x3u;
            float c = centroids[idx];
            mse_score += q_rot[q_base + w * 16 + i] * c;
        }
    }
    mse_score *= key_norms[k];

    // 2. QJL Score: Unpack sign bits + Dot product mit q_sketch
    float qjl_score = 0.0f;
    for (uint w = 0; w < D_DIV_32; w++) {
        uint32_t signs = key_sign_bits[k * D_DIV_32 + w];
        for (uint bit = 0; bit < 32; bit++) {
            float sign_val = ((signs >> bit) & 1u) ? 1.0f : -1.0f;
            qjl_score += q_sketch[q_base + w * 32 + bit] * sign_val;
        }
    }
    qjl_score *= qjl_scale[0] * key_residual_norms[k];

    scores[k * n_repeats + r] = mse_score + qjl_score;
"""

_fused_score_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_scores",
    input_names=["q_rot", "q_sketch", "key_packed", "centroids",
                 "key_norms", "key_sign_bits", "key_residual_norms", "qjl_scale"],
    output_names=["scores"],
    source=_FUSED_SCORE_SOURCE,
)


def fused_tq_scores(
    q_rot: mx.array,
    q_sketch: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    key_sign_bits: mx.array,
    key_residual_norms: mx.array,
    qjl_scale: mx.array,
) -> mx.array:
    """Fused TurboQuant Score: MSE + QJL in einem Kernel.

    Args:
        q_rot: (n_repeats, D) float32 — rotierte Queries für einen KV-Head
        q_sketch: (n_repeats, D) float32 — JL-Sketches
        key_packed: (T_kv, D//16) uint32 — 2-bit gepackte Keys
        centroids: (4,) float32
        key_norms: (T_kv,) float32
        key_sign_bits: (T_kv, D//32) uint32
        key_residual_norms: (T_kv,) float32
        qjl_scale: (1,) float32 — sqrt(pi/2)/D

    Returns:
        scores: (T_kv, n_repeats) float32
    """
    T_kv = key_packed.shape[0]
    n_repeats = q_rot.shape[0]

    if T_kv == 0:
        return mx.zeros((T_kv, n_repeats))

    outputs = _fused_score_kernel(
        inputs=[q_rot, q_sketch, key_packed, centroids,
                key_norms, key_sign_bits, key_residual_norms, qjl_scale],
        grid=(T_kv, n_repeats, 1),
        threadgroup=(min(256, T_kv), 1, 1),
        output_shapes=[(T_kv * n_repeats,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(T_kv, n_repeats)


# --- Fused Kernel 2: TurboQuant Value Output ---

_FUSED_VALUE_SOURCE = """
    // Grid: (D, n_repeats, 1) — ein Thread pro (output_dim, repeat) Paar
    uint d = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;

    uint T_kv = value_norms_shape[0];
    uint D_DIV_16 = value_packed_shape[1];
    uint D = D_DIV_16 * 16;
    uint n_repeats = weights_shape[1];

    if (d >= D) return;

    float result = 0.0f;

    for (uint k = 0; k < T_kv; k++) {
        float w = weights[k * n_repeats + r];
        if (w < 1e-8f) continue;

        // Decode value[k][d]: sum_j(centroid[idx_j] * Pi[j][d])
        float v_d = 0.0f;
        for (uint ww = 0; ww < D_DIV_16; ww++) {
            uint32_t packed = value_packed[k * D_DIV_16 + ww];
            for (uint i = 0; i < 16; i++) {
                uint idx = (packed >> (i * 2)) & 0x3u;
                uint j = ww * 16 + i;
                v_d += centroids[idx] * rotation_matrix[j * D + d];
            }
        }
        v_d *= value_norms[k];
        result += w * v_d;
    }

    output[r * D + d] = result;
"""

_fused_value_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_value",
    input_names=["weights", "value_packed", "centroids",
                 "value_norms", "rotation_matrix"],
    output_names=["output"],
    source=_FUSED_VALUE_SOURCE,
)


def fused_tq_value_output(
    weights: mx.array,
    value_packed: mx.array,
    centroids: mx.array,
    value_norms: mx.array,
    rotation_matrix: mx.array,
    D: int,
    n_repeats: int,
) -> mx.array:
    """Fused TurboQuant Value Output: Decode + Gewichtete Summe in einem Kernel.

    Args:
        weights: (T_kv, n_repeats) float32 — Softmax-Gewichte
        value_packed: (T_kv, D//16) uint32 — 2-bit gepackte Values
        centroids: (4,) float32
        value_norms: (T_kv,) float32
        rotation_matrix: (D, D) float32 — Pi

    Returns:
        output: (n_repeats, D) float32
    """
    outputs = _fused_value_kernel(
        inputs=[weights, value_packed, centroids, value_norms, rotation_matrix],
        grid=(D, n_repeats, 1),
        threadgroup=(min(128, D), 1, 1),
        output_shapes=[(n_repeats * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_repeats, D)


# ===========================================================================
# FUSED FULL ATTENTION KERNEL — Alles in einem Metal Dispatch
# ===========================================================================
# Rotation + Quantized Scoring + Online Softmax + Value Accumulation + Inverse Rotation
# Pattern: 1 Simdgroup (32 Threads) pro Query-Head, sequentiell über Keys.

_FUSED_ATTN_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

_FUSED_ATTN_SOURCE = """
    // Grid: (n_q_heads, 1, 1), Threadgroup: (32, 1, 1)
    // Ein Simdgroup (32 Threads) pro Query-Head
    // Thread lid handles dims [lid*4, lid*4+1, lid*4+2, lid*4+3]

    uint head = threadgroup_position_in_grid.x;
    uint lid = thread_index_in_simdgroup;

    // Dimensionen aus Input-Shapes ableiten
    uint T_kv = key_norms_shape[1];        // key_norms: (n_kv_heads, T_kv)
    uint n_kv_heads = key_norms_shape[0];
    uint n_q_heads = queries_shape[0];      // queries: (n_q_heads, D)
    uint WORDS = key_packed_shape[2];       // key_packed: (n_kv_heads, T_kv, D/16)
    uint D = WORDS * 16;
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    // Offsets
    uint q_off = head * D;
    uint kv_off_norms = kv_head * T_kv;
    uint kv_off_packed = kv_head * T_kv * WORDS;

    // ===== PHASE 1: Rotate Query =====
    // q_rot[d] = scale * sum_j(q[j] * Pi[d][j])
    // Load ALL query values into shared memory, then each thread
    // independently computes its 4 output dimensions

    threadgroup float shared_q[128];
    for (uint i = 0; i < 4; i++)
        shared_q[lid * 4 + i] = queries[q_off + lid * 4 + i] * scale[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes q_rot for dims [lid*4 .. lid*4+3]
    float q_rot[4];
    for (uint di = 0; di < 4; di++) {
        uint d = lid * 4 + di;
        float sum = 0.0f;
        for (uint j = 0; j < D; j++)
            sum += shared_q[j] * rotation_matrix[d * D + j];
        q_rot[di] = sum;
    }

    // ===== PHASE 2: Scoring + Online Softmax + Value Accumulation =====
    float max_score = -HUGE_VALF;
    float sum_exp = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Akkumulator in rotiertem Raum

    uint word_idx = lid / 4;      // Welches uint32 Wort (0..7)
    uint bit_base = (lid % 4) * 8; // Bit-Offset im Wort (0, 8, 16, 24)

    for (uint k = 0; k < T_kv; k++) {
        // -- Unpack key centroids for this thread's 4 dims --
        uint32_t k_packed = key_packed[kv_off_packed + k * WORDS + word_idx];
        float kc[4];
        for (uint i = 0; i < 4; i++) {
            uint idx = (k_packed >> (bit_base + i * 2)) & 0x3u;
            kc[i] = centroids[idx];
        }

        // -- MSE Score: dot(q_rot, key_centroids) via simd_sum --
        float partial = 0.0f;
        for (uint i = 0; i < 4; i++)
            partial += q_rot[i] * kc[i];
        float score = simd_sum(partial) * key_norms[kv_off_norms + k];

        // -- Causal mask: skip future keys --
        // offset = T_kv - 1 (current query position for T_q=1)
        if (k > T_kv - 1 + causal_offset[0])
            score = -HUGE_VALF;

        // -- Online Softmax --
        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // -- Accumulate value centroids (in rotated space) --
        uint32_t v_packed = value_packed[kv_off_packed + k * WORDS + word_idx];
        for (uint i = 0; i < 4; i++) {
            uint v_idx = (v_packed >> (bit_base + i * 2)) & 0x3u;
            float vc = centroids[v_idx] * value_norms[kv_off_norms + k];
            acc[i] = acc[i] * factor + exp_score * vc;
        }
    }

    // Normalize by sum_exp
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint i = 0; i < 4; i++)
        acc[i] *= inv_sum;

    // ===== PHASE 3: Inverse Rotation =====
    // output[d] = sum_j(acc[j] * Pi[j][d])
    // acc is distributed: thread lid holds acc for dims [lid*4..lid*4+3]
    // Store acc to threadgroup memory so all threads can read all 128 values
    threadgroup float shared_acc[128];
    for (uint i = 0; i < 4; i++)
        shared_acc[lid * 4 + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes 4 output dimensions
    for (uint di = 0; di < 4; di++) {
        uint d = lid * 4 + di;
        float sum = 0.0f;
        for (uint j = 0; j < D; j++)
            sum += shared_acc[j] * rotation_matrix[j * D + d];
        output[head * D + d] = sum;
    }
"""

_fused_attn_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_attention",
    input_names=[
        "queries", "rotation_matrix", "key_packed", "centroids",
        "key_norms", "value_packed", "value_norms", "scale", "causal_offset",
    ],
    output_names=["output"],
    source=_FUSED_ATTN_SOURCE,
    header=_FUSED_ATTN_HEADER,
)


def fused_tq_attention(
    queries: mx.array,
    rotation_matrix: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    value_packed: mx.array,
    value_norms: mx.array,
    scale: float,
    n_q_heads: int,
    D: int,
    causal_offset: int = 0,
) -> mx.array:
    """Vollständig fused TurboQuant Attention in EINEM Metal Dispatch.

    Rotation + Quantized Scoring + Online Softmax + Value Accumulation + Inverse Rotation.

    Args:
        queries: (n_q_heads, D) float32 — ALLE Query-Heads (flattened batch)
        rotation_matrix: (D, D) float32 — Pi
        key_packed: (n_kv_heads, T_kv, D//16) uint32 — 2-bit packed
        centroids: (4,) float32
        key_norms: (n_kv_heads, T_kv) float32
        value_packed: (n_kv_heads, T_kv, D//16) uint32
        value_norms: (n_kv_heads, T_kv) float32
        scale: Attention scale (1/sqrt(D))
        n_q_heads: Total query heads
        D: Head dimension
        causal_offset: 0 for generation (T_q=1)

    Returns:
        output: (n_q_heads, D) float32
    """
    scale_arr = mx.array([scale], dtype=mx.float32)
    offset_arr = mx.array([causal_offset], dtype=mx.int32)

    outputs = _fused_attn_kernel(
        inputs=[
            queries, rotation_matrix, key_packed, centroids,
            key_norms, value_packed, value_norms, scale_arr, offset_arr,
        ],
        grid=(n_q_heads * 32, 1, 1),  # Total threads = n_heads × 32
        threadgroup=(32, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)


# ===========================================================================
# HIGH-OCCUPANCY FUSED KERNEL — 32 Simdgroups, Rotation außerhalb
# ===========================================================================
# Arbeitet im rotierten Raum: Query wird VOR dem Kernel rotiert (MLX GEMM),
# Output wird NACH dem Kernel zurückrotiert (MLX GEMM).
# 32 Simdgroups verarbeiten Keys PARALLEL mit Cross-Simdgroup Online Softmax.
# Pattern: Wie MLX's sdpa_vector.h — 1024 Threads pro Head.

_FUSED_ATTN_NOROT_HEADER = """
#include <metal_simdgroup>
using namespace metal;
"""

_FUSED_ATTN_NOROT_SOURCE = """
    // Grid: total threads = n_q_heads * 1024
    // Threadgroup: (1024, 1, 1) = 32 Simdgroups × 32 Lanes
    // Jede Threadgroup verarbeitet einen Query-Head.
    // Simdgroups verarbeiten Keys mit Stride 32 (parallel).

    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;   // 0..31 (Simdgroup-Index)
    uint lane_id = tid & 31;   // 0..31 (Lane innerhalb Simdgroup)

    // Dimensionen
    uint T_kv = key_norms_shape[1];
    uint n_kv_heads = key_norms_shape[0];
    uint n_q_heads = q_rot_shape[0];
    uint WORDS = key_packed_shape[2];   // D/16 für 2-bit
    uint D = WORDS * 16;
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;

    // Query laden (bereits skaliert und rotiert)
    float q[4];
    uint q_off = head * D;
    for (uint i = 0; i < 4; i++)
        q[i] = q_rot[q_off + lane_id * 4 + i];

    // KV-Offsets
    uint kv_off_norms = kv_head * T_kv;
    uint kv_off_packed = kv_head * T_kv * WORDS;

    // Dim-Mapping: Lane → Welches uint32-Wort und Bit-Offset
    uint word_idx = lane_id >> 2;        // 0..7
    uint bit_base = (lane_id & 3) << 3;  // 0, 8, 16, 24

    // ===== Scoring + Online Softmax + Value Accumulation (pro Simdgroup) =====
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Simdgroups verarbeiten Keys mit Stride (32 Simdgroups parallel)
    for (uint k = simd_id; k < T_kv; k += 32) {
        // Unpack Key-Centroids für 4 Dims dieses Threads
        uint32_t k_packed = key_packed[kv_off_packed + k * WORDS + word_idx];
        float partial = 0.0f;
        for (uint i = 0; i < 4; i++) {
            uint idx = (k_packed >> (bit_base + i * 2)) & 0x3u;
            partial += q[i] * centroids[idx];
        }
        float score = simd_sum(partial) * key_norms[kv_off_norms + k];

        // Online Softmax
        float new_max = max(local_max, score);
        float factor = metal::fast::exp(local_max - new_max);
        float exp_s = metal::fast::exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        // Value Accumulation (im rotierten Raum)
        uint32_t v_packed = value_packed[kv_off_packed + k * WORDS + word_idx];
        for (uint i = 0; i < 4; i++) {
            uint v_idx = (v_packed >> (bit_base + i * 2)) & 0x3u;
            float vc = centroids[v_idx] * value_norms[kv_off_norms + k];
            local_acc[i] = local_acc[i] * factor + exp_s * vc;
        }
    }

    // ===== Cross-Simdgroup Reduktion via Threadgroup Memory =====
    // 32 Simdgroups × (max + sum + 128 acc dims)
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup float tg_acc[32 * 128];

    // Lane 0 schreibt max/sum (identisch nach simd_sum)
    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    // Alle Lanes schreiben ihre 4 Acc-Dims
    for (uint i = 0; i < 4; i++)
        tg_acc[simd_id * 128 + lane_id * 4 + i] = local_acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Jede Lane berechnet 4 Output-Dimensionen durch Kombination aller Simdgroups
    float global_max = -1e10f;
    for (uint s = 0; s < 32; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < 32; s++) {
        float factor = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * factor;
        for (uint i = 0; i < 4; i++)
            result[i] += tg_acc[s * 128 + lane_id * 4 + i] * factor;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < 4; i++)
        output_rot[head * D + lane_id * 4 + i] = result[i] * inv;
"""

_fused_attn_norot_kernel = mx.fast.metal_kernel(
    name="turboquant_fused_attention_norot",
    input_names=[
        "q_rot", "key_packed", "centroids",
        "key_norms", "value_packed", "value_norms",
    ],
    output_names=["output_rot"],
    source=_FUSED_ATTN_NOROT_SOURCE,
    header=_FUSED_ATTN_NOROT_HEADER,
)


def fused_tq_attention_norot(
    q_rot: mx.array,
    key_packed: mx.array,
    centroids: mx.array,
    key_norms: mx.array,
    value_packed: mx.array,
    value_norms: mx.array,
    n_q_heads: int,
    n_kv_heads: int,
    D: int,
) -> mx.array:
    """High-Occupancy Fused Attention ohne Rotation (32 Simdgroups).

    Query muss VOR dem Call rotiert sein, Output ist im rotierten Raum.

    Args:
        q_rot: (n_q_heads, D) float32 — skalierte, rotierte Queries
        key_packed: (n_kv_heads, T_kv, D//16) uint32 — 2-bit packed
        centroids: (4,) float32
        key_norms: (n_kv_heads, T_kv) float32
        value_packed: (n_kv_heads, T_kv, D//16) uint32
        value_norms: (n_kv_heads, T_kv) float32

    Returns:
        output_rot: (n_q_heads, D) float32 — im rotierten Raum
    """
    outputs = _fused_attn_norot_kernel(
        inputs=[q_rot, key_packed, centroids, key_norms, value_packed, value_norms],
        grid=(n_q_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_q_heads * D,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, D)
