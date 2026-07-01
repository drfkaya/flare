"""
flare.rff — Orthogonal Random Fourier Features (Yu et al., 2016).

Vanilla RFF'den farkı: ω vektörleri blok-bloq orthogonalize
edilerek spektral uzayda eşit dağılım sağlanır.

m ≤ d:  Tüm m vektör birbirine dik (tam orthogonal)
m > d:  ⌈m/d⌉ blok, her blokta d vektör birbirine dik
"""

import numpy as np


class RFF:
    """Orthogonal Random Fourier Features.

    Parameters
    ----------
    m     : int   – number of random features
    d     : int   – input dimensionality
    sigma : float – kernel bandwidth
    seed  : int   – RNG seed for reproducibility
    """

    __slots__ = ("omega", "b", "c", "m")

    def __init__(self, m, d, sigma, seed=None):
        self.m = m
        rng = np.random.default_rng(seed)

        n_blocks = int(np.ceil(m / d))
        blocks = []
        for _ in range(n_blocks):
            G = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(G)
            blocks.append(Q)
        omega = np.vstack(blocks)[:m]
        self.omega = omega / sigma

        self.b = rng.uniform(0, 2 * np.pi, m)
        self.c = np.sqrt(2.0 / m)

    def transform(self, X):
        """Map X (n, d) → Φ (n, m)."""
        return self.c * np.cos(X @ self.omega.T + self.b)