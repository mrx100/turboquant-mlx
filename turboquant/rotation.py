"""Random Rotation Matrix (PolarQuant) und JL Matrix (QJL).

Die Rotation transformiert beliebige Vektoren in eine gleichmäßige
Verteilung auf der Einheitskugel, was die Scalar Quantization optimiert.
Die JL-Matrix wird für die 1-Bit Residual-Kompression verwendet.
"""

import mlx.core as mx


def generate_rotation_matrix(head_dim: int, seed: int = 42) -> mx.array:
    """Erzeugt eine orthogonale Rotationsmatrix Pi via QR-Zerlegung.

    Args:
        head_dim: Dimension (z.B. 128)
        seed: Seed für Reproduzierbarkeit

    Returns:
        Pi: mx.array shape (head_dim, head_dim), orthogonal, float32
    """
    mx.random.seed(seed)
    G = mx.random.normal((head_dim, head_dim))
    mx.eval(G)

    # QR nur auf CPU möglich
    Q, R = mx.linalg.qr(G, stream=mx.cpu)
    mx.eval(Q)

    # Vorzeichen-Korrektur: sicherstellen dass det(Q) = +1
    # (QR kann negative Diagonale in R haben → Q reflektiert statt rotiert)
    diag_sign = mx.sign(mx.diag(R))
    Q = Q * diag_sign[None, :]
    mx.eval(Q)

    return Q


def generate_jl_matrix(head_dim: int, seed: int = 137) -> mx.array:
    """Erzeugt die Johnson-Lindenstrauss Projektionsmatrix S.

    S hat i.i.d. N(0,1) Einträge. Keine Orthogonalisierung nötig —
    die JL-Eigenschaft gilt für beliebige Gauss-Matrizen.

    Args:
        head_dim: Dimension (z.B. 128)
        seed: Seed (anderer Default als Rotation, um Unabhängigkeit zu sichern)

    Returns:
        S: mx.array shape (head_dim, head_dim), float32
    """
    mx.random.seed(seed)
    S = mx.random.normal((head_dim, head_dim))
    mx.eval(S)
    return S
