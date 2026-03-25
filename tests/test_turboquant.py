"""Isolierte Unit-Tests für TurboQuant Kernkomponenten.

Testet numerische Korrektheit OHNE LLM-Modell.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant.codebook import get_codebook, get_codebook_unscaled
from turboquant.rotation import generate_rotation_matrix, generate_jl_matrix
from turboquant.kernels import (
    polarquant_encode,
    polarquant_decode,
    pack_2bit_indices,
    unpack_2bit_indices,
    pack_3bit_indices,
    unpack_3bit_indices,
    qjl_encode,
)


# --- Codebook Tests ---

class TestCodebook:
    def test_2bit_centroid_symmetry(self):
        """Lloyd-Max Centroids für N(0,1) sind symmetrisch um 0."""
        centroids, _ = get_codebook_unscaled(2)
        c = centroids.tolist()
        assert len(c) == 4
        assert abs(c[0] + c[3]) < 1e-6  # c[0] = -c[3]
        assert abs(c[1] + c[2]) < 1e-6  # c[1] = -c[2]

    def test_2bit_centroid_values(self):
        """2-bit Centroids matchen Paper-Werte: ±0.453, ±1.51."""
        centroids, _ = get_codebook_unscaled(2)
        c = sorted(centroids.tolist())
        assert abs(c[0] - (-1.51)) < 0.01
        assert abs(c[1] - (-0.453)) < 0.01
        assert abs(c[2] - 0.453) < 0.01
        assert abs(c[3] - 1.51) < 0.01

    def test_scaling_by_head_dim(self):
        """Centroids werden mit 1/sqrt(head_dim) skaliert."""
        c_unscaled, _ = get_codebook_unscaled(2)
        c_scaled, _ = get_codebook(2, head_dim=128)
        ratio = (c_unscaled / c_scaled).tolist()
        expected = math.sqrt(128)
        for r in ratio:
            assert abs(r - expected) < 0.01

    def test_3bit_has_8_centroids(self):
        centroids, boundaries = get_codebook(3, head_dim=128)
        assert centroids.shape == (8,)
        assert boundaries.shape == (7,)

    def test_boundaries_between_centroids(self):
        """Boundaries liegen zwischen aufeinanderfolgenden Centroids."""
        centroids, boundaries = get_codebook_unscaled(2)
        c = sorted(centroids.tolist())
        b = sorted(boundaries.tolist())
        for i, bi in enumerate(b):
            assert c[i] < bi < c[i + 1]


# --- Rotation Tests ---

class TestRotation:
    def test_orthogonality(self):
        """Rotationsmatrix ist orthogonal: Pi @ Pi^T = I."""
        Pi = generate_rotation_matrix(128, seed=42)
        identity = Pi @ Pi.T
        mx.eval(identity)
        diff = mx.abs(identity - mx.eye(128)).max().item()
        assert diff < 1e-5, f"Orthogonalität verletzt: max diff {diff}"

    def test_determinant_unit(self):
        """|det(Pi)| = 1 (orthogonale Matrix)."""
        Pi = generate_rotation_matrix(128, seed=42)
        Pi_np = np.array(Pi)
        det = np.linalg.det(Pi_np)
        assert abs(abs(det) - 1.0) < 1e-3, f"|det| = {abs(det)}, erwartet 1"

    def test_reproducibility(self):
        """Gleicher Seed → gleiche Matrix."""
        Pi1 = generate_rotation_matrix(128, seed=99)
        Pi2 = generate_rotation_matrix(128, seed=99)
        diff = mx.abs(Pi1 - Pi2).max().item()
        assert diff == 0.0

    def test_different_seeds_different_matrices(self):
        Pi1 = generate_rotation_matrix(128, seed=42)
        Pi2 = generate_rotation_matrix(128, seed=43)
        diff = mx.abs(Pi1 - Pi2).max().item()
        assert diff > 0.1

    def test_rotation_preserves_norm(self):
        """||Pi @ x|| = ||x|| für orthogonale Pi."""
        Pi = generate_rotation_matrix(128, seed=42)
        x = mx.random.normal((128,))
        mx.eval(x)
        norm_before = mx.linalg.norm(x).item()
        norm_after = mx.linalg.norm(Pi @ x).item()
        assert abs(norm_before - norm_after) < 1e-4


# --- PolarQuant Encode/Decode Tests ---

class TestPolarQuant:
    def test_encode_decode_roundtrip(self):
        """Encode → Decode rekonstruiert Vektor mit begrenztem Fehler."""
        Pi = generate_rotation_matrix(128, seed=42)
        _, boundaries = get_codebook(2, head_dim=128)
        centroids, _ = get_codebook(2, head_dim=128)

        x = mx.random.normal((1, 1, 1, 128))  # (B, H, T, D)
        mx.eval(x)

        indices, norms, rotated = polarquant_encode(x, Pi, boundaries)
        reconstructed = polarquant_decode(indices, Pi, centroids, norms)

        # Relativer Fehler
        orig_norm = mx.linalg.norm(x).item()
        error = mx.linalg.norm(x - reconstructed).item()
        relative_error = error / orig_norm
        assert relative_error < 0.5, f"Relativer Fehler zu hoch: {relative_error}"

    def test_norms_are_positive(self):
        Pi = generate_rotation_matrix(128, seed=42)
        _, boundaries = get_codebook(2, head_dim=128)
        x = mx.random.normal((1, 1, 4, 128))
        mx.eval(x)
        _, norms, _ = polarquant_encode(x, Pi, boundaries)
        assert (norms > 0).all().item()

    def test_indices_in_range(self):
        """2-bit Indices sind in [0, 3]."""
        Pi = generate_rotation_matrix(128, seed=42)
        _, boundaries = get_codebook(2, head_dim=128)
        x = mx.random.normal((1, 1, 4, 128))
        mx.eval(x)
        indices, _, _ = polarquant_encode(x, Pi, boundaries)
        assert (indices >= 0).all().item()
        assert (indices <= 3).all().item()


# --- Pack/Unpack Tests ---

class TestPacking:
    def test_2bit_roundtrip(self):
        """Pack → Unpack 2-bit gibt Originalindizes zurück."""
        indices = mx.random.randint(0, 4, (1, 1, 4, 128)).astype(mx.uint32)
        mx.eval(indices)
        packed = pack_2bit_indices(indices)
        unpacked = unpack_2bit_indices(packed, 128)
        diff = mx.abs(indices.astype(mx.int32) - unpacked.astype(mx.int32)).max().item()
        assert diff == 0, f"2-bit pack/unpack mismatch: max diff {diff}"

    def test_3bit_roundtrip(self):
        """Pack → Unpack 3-bit gibt Originalindizes zurück."""
        indices = mx.random.randint(0, 8, (1, 1, 4, 128)).astype(mx.uint32)
        mx.eval(indices)
        packed = pack_3bit_indices(indices)
        unpacked = unpack_3bit_indices(packed, 128)
        diff = mx.abs(indices.astype(mx.int32) - unpacked.astype(mx.int32)).max().item()
        assert diff == 0, f"3-bit pack/unpack mismatch: max diff {diff}"

    def test_2bit_packed_shape(self):
        """2-bit packing: 128 Elemente → 128/16 = 8 uint32."""
        indices = mx.zeros((1, 1, 1, 128), dtype=mx.uint32)
        packed = pack_2bit_indices(indices)
        assert packed.shape[-1] == 8  # 128 / 16 = 8

    def test_3bit_packed_shape(self):
        """3-bit packing: 128 Elemente → ceil(128/10) = 13 uint32 (10 Werte pro uint32)."""
        indices = mx.zeros((1, 1, 1, 128), dtype=mx.uint32)
        packed = pack_3bit_indices(indices)
        # 10 Werte à 3 bit = 30 bit pro uint32 → ceil(128/10) = 13
        assert packed.shape[-1] == 13


# --- QJL Tests ---

class TestQJL:
    def test_sign_bits_shape(self):
        """QJL produziert D/32 uint32 sign bits pro Vektor."""
        S = generate_jl_matrix(128, seed=137)
        residual = mx.random.normal((1, 1, 4, 128))
        mx.eval(residual)
        sign_bits, res_norms = qjl_encode(residual, S)
        assert sign_bits.shape == (1, 1, 4, 4)  # 128/32 = 4 uint32
        assert res_norms.shape == (1, 1, 4)

    def test_residual_norms_positive(self):
        S = generate_jl_matrix(128, seed=137)
        residual = mx.random.normal((1, 1, 4, 128))
        mx.eval(residual)
        _, res_norms = qjl_encode(residual, S)
        assert (res_norms >= 0).all().item()


# --- V2 Norm-Baking Tests ---

class TestNormBaking:
    def test_normed_quant_mathematical_equivalence(self):
        """norm * dequant(data, scale, bias) = dequant(data, norm*scale, norm*bias)."""
        from turboquant.cache_v2 import TurboQuantKVCacheV2

        cache = TurboQuantKVCacheV2(head_dim=128, bits=3, group_size=64, seed=42)

        # Simuliere einen Token
        keys = mx.random.normal((1, 8, 1, 128))
        values = mx.random.normal((1, 8, 1, 128))
        mx.eval(keys, values)

        # Manuell: raw quantize + separate norm multiply
        k_norms = mx.linalg.norm(keys, axis=-1, keepdims=True)
        safe_norms = mx.where(k_norms < 1e-8, mx.ones_like(k_norms), k_norms)
        k_rotated = (keys / safe_norms) @ cache.rotation_matrix.T
        k_quant = mx.quantize(k_rotated, group_size=64, bits=3)

        # Weg A: Separate norm multiply
        k_dequant = mx.dequantize(*k_quant, group_size=64, bits=3)
        result_a = k_dequant * k_norms

        # Weg B: Norm in scales/biases gebacken
        data, scales, biases = k_quant
        norms_sq = k_norms.squeeze(-1)[:, :, :, None]  # (B, H, T, 1)
        normed_quant = (data, scales * norms_sq, biases * norms_sq)
        result_b = mx.dequantize(*normed_quant, group_size=64, bits=3)

        mx.eval(result_a, result_b)
        diff = mx.abs(result_a - result_b).max().item()
        assert diff < 1e-5, f"Norm-Baking nicht äquivalent: max diff {diff}"

    def test_v2_sdpa_produces_finite_output(self):
        """V2 SDPA mit norm-baking produziert endliche Werte."""
        from turboquant.cache_v2 import TurboQuantKVCacheV2
        from turboquant.attention_v2 import turboquant_v2_sdpa

        cache = TurboQuantKVCacheV2(head_dim=128, bits=3, group_size=64, seed=42)
        keys = mx.random.normal((1, 8, 4, 128))
        values = mx.random.normal((1, 8, 4, 128))
        mx.eval(keys, values)

        q_keys, q_values = cache.update_and_fetch(keys, values)

        queries = mx.random.normal((1, 32, 1, 128))
        mx.eval(queries)
        scale = 1.0 / math.sqrt(128)

        output = turboquant_v2_sdpa(queries, q_keys, q_values, cache, scale, mask=None)
        mx.eval(output)

        assert output.shape == (1, 32, 1, 128)
        assert mx.isfinite(output).all().item(), "Output enthält NaN/Inf!"


# --- V2 Full Attention vs Reference ---

class TestV2Attention:
    def test_v2_attention_close_to_fp16(self):
        """V2 3-bit Attention-Output ist nahe am fp16 Reference."""
        from turboquant.cache_v2 import TurboQuantKVCacheV2
        from turboquant.attention_v2 import turboquant_v2_sdpa

        D = 128
        T_kv = 16
        n_q_heads = 32
        n_kv_heads = 8

        cache = TurboQuantKVCacheV2(head_dim=D, bits=3, group_size=64, seed=42)

        # Synthetische KV
        keys = mx.random.normal((1, n_kv_heads, T_kv, D))
        values = mx.random.normal((1, n_kv_heads, T_kv, D))
        mx.eval(keys, values)

        q_keys, q_values = cache.update_and_fetch(keys, values)

        queries = mx.random.normal((1, n_q_heads, 1, D))
        mx.eval(queries)
        scale = 1.0 / math.sqrt(D)

        # V2 Output
        v2_output = turboquant_v2_sdpa(queries, q_keys, q_values, cache, scale, mask=None)
        mx.eval(v2_output)

        # fp16 Reference: standard attention
        n_repeats = n_q_heads // n_kv_heads
        q_scaled = queries * scale
        keys_expanded = mx.repeat(keys, n_repeats, axis=1)
        values_expanded = mx.repeat(values, n_repeats, axis=1)

        scores_ref = (q_scaled @ keys_expanded.transpose(0, 1, 3, 2))
        weights_ref = mx.softmax(scores_ref, axis=-1, precise=True)
        ref_output = weights_ref @ values_expanded
        mx.eval(ref_output)

        # Cosine Similarity — sollte hoch sein
        v2_flat = v2_output.reshape(-1)
        ref_flat = ref_output.reshape(-1)
        cos_sim = (mx.sum(v2_flat * ref_flat) / (mx.linalg.norm(v2_flat) * mx.linalg.norm(ref_flat))).item()

        assert cos_sim > 0.85, f"Cosine Similarity zu niedrig: {cos_sim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
