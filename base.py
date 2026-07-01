"""
flare.base — Shared constants and utility functions.

Exports
-------
EPS                 : numerical floor constant
sigmoid(z)          : clipped logistic function
stabilize_fisher()  : symmetrise + positive-definite floor
"""

import numpy as np
from scipy.special import expit

# ── Numerical floor ─────────────────────────────────────────────
EPS = 1e-10

np.seterr(over="ignore", under="ignore", divide="ignore")


# ── Sigmoid ─────────────────────────────────────────────────────
def sigmoid(z):
    """Numerically stable sigmoid clipped to (EPS, 1 − EPS)."""
    return np.clip(expit(z), EPS, 1.0 - EPS)


# ── Fisher stabilisation ────────────────────────────────────────
def stabilize_fisher(fisher, dim, floor=EPS):
    """Ensure symmetric positive-definiteness of a Fisher matrix."""
    fisher = 0.5 * (fisher + fisher.T)
    fisher += floor * np.eye(dim)
    return fisher