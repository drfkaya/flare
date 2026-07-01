"""
flare — Kernel Approximation with Laplace Explainability
=========================================================

Package structure
─────────────────
base.py  – shared constants & utility functions
rff.py   – Orthogonal Random Fourier Features
logit.py – FLARE model (binary & multi-class classification)

v2.6: Conformal prediction removed.
"""

from .rff import RFF
from .logit import FLARE
from .base import EPS

__all__ = ["RFF", "FLARE", "EPS"]
__version__ = "2.6.0"