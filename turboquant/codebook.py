"""Lloyd-Max optimale Scalar Quantizer für die Gaussian N(0,1) Verteilung.

Berechnet optimale Centroids und Boundaries für b=1,2,3 Bits.
Die Centroids minimieren den MSE unter der Normalverteilung.
"""

import math

import mlx.core as mx
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def _lloyd_max_gaussian(num_levels: int, max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max Iteration für N(0,1).

    Returns (centroids, boundaries) wobei boundaries die Entscheidungsgrenzen sind.
    """
    # Initiale Boundaries: gleichmäßig verteilt über [-3, 3]
    boundaries = np.linspace(-3.0, 3.0, num_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    centroids = np.zeros(num_levels)

    for _ in range(max_iter):
        # Schritt 1: Centroids als bedingten Erwartungswert berechnen
        # c_i = E[X | b_{i-1} < X <= b_i]
        old_centroids = centroids.copy()
        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]

            numerator, _ = quad(lambda x: x * norm.pdf(x), lo, hi)
            denominator, _ = quad(norm.pdf, lo, hi)

            if denominator < 1e-15:
                centroids[i] = (lo + hi) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else old_centroids[i]
                continue
            centroids[i] = numerator / denominator

        # Schritt 2: Boundaries als Mittelpunkt zwischen Centroids
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        if np.max(np.abs(centroids - old_centroids)) < tol:
            break

    # Innere Boundaries zurückgeben (ohne -inf/+inf)
    inner_boundaries = boundaries[1:-1]
    return centroids, inner_boundaries


# Precomputed Codebooks für b=1,2,3
_CODEBOOKS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ensure_codebooks():
    if _CODEBOOKS:
        return
    for bits in (1, 2, 3):
        num_levels = 2**bits
        centroids, boundaries = _lloyd_max_gaussian(num_levels)
        _CODEBOOKS[bits] = (centroids, boundaries)


def get_codebook(bits: int, head_dim: int) -> tuple[mx.array, mx.array]:
    """Gibt (centroids, boundaries) zurück, skaliert mit 1/sqrt(head_dim).

    Args:
        bits: Anzahl Bits pro Koordinate (1, 2 oder 3)
        head_dim: Dimension des Attention-Head (z.B. 128)

    Returns:
        centroids: mx.array shape (2^bits,) — die optimalen Centroid-Werte
        boundaries: mx.array shape (2^bits - 1,) — die Entscheidungsgrenzen
    """
    if bits not in (1, 2, 3):
        raise ValueError(f"Unterstützte Bits: 1, 2, 3. Erhalten: {bits}")

    _ensure_codebooks()
    centroids_np, boundaries_np = _CODEBOOKS[bits]

    scale = 1.0 / math.sqrt(head_dim)
    centroids = mx.array(centroids_np * scale, dtype=mx.float32)
    boundaries = mx.array(boundaries_np * scale, dtype=mx.float32)
    return centroids, boundaries


def get_codebook_unscaled(bits: int) -> tuple[mx.array, mx.array]:
    """Gibt (centroids, boundaries) ohne Skalierung zurück.

    Nützlich wenn die Skalierung separat erfolgt (z.B. nach Normalisierung).
    """
    if bits not in (1, 2, 3):
        raise ValueError(f"Unterstützte Bits: 1, 2, 3. Erhalten: {bits}")

    _ensure_codebooks()
    centroids_np, boundaries_np = _CODEBOOKS[bits]
    return mx.array(centroids_np, dtype=mx.float32), mx.array(boundaries_np, dtype=mx.float32)
