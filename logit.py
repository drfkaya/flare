"""
flare.logit — FLARE: Fourier-Laplace Analytical Risk Estimation.

Binary & multi-class classification combining:
  - RFF-based kernel approximation
  - β_equiv: equivalent LR coefficients via Jacobian projection
  - Two-component analytical uncertainty
  - Sandwich (HC2) robust covariance
  - Effective df + t-distribution finite-sample correction
  - Epistemic / aleatoric decomposition
  - RFF-Attribution with Simpson quadrature (v2.7)
"""

import numpy as np
from scipy.linalg import solve
from scipy.stats import t as t_dist

from .base import EPS, sigmoid, stabilize_fisher
from .rff import RFF


class FLARE:
    """Fourier-Laplace Analytical Risk Estimation.

    Parameters
    ----------
    m    : int   – number of random Fourier features
    l2   : float – L2 regularisation coefficient (lambda)
    seed : int   – global RNG seed
    """

    def __init__(self, m=256, l2=1e-3, seed=42):
        self.m = m
        self.l2 = l2
        self.seed = seed

    # ──────────────────────────────────────────────────────────────
    # SIGMOID  (backward compat — helpers.py calls self._sigmoid)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _sigmoid(z):
        return sigmoid(z)

    # ──────────────────────────────────────────────────────────────
    # DESIGN MATRIX  Phi_tilde = [phi(X), 1]
    # ──────────────────────────────────────────────────────────────
    def _design(self, X):
        """Augmented feature matrix with bias column."""
        Phi = self.rff_.transform(X)
        return np.hstack([Phi, np.ones((Phi.shape[0], 1))])

    # ──────────────────────────────────────────────────────────────
    # FISHER SCORING + ARMIJO LINE SEARCH
    # ──────────────────────────────────────────────────────────────
    def _scoring(self, Phi, y, max_iter=50, tol=1e-6):
        """Newton-Raphson (Fisher scoring) with Armijo backtracking."""
        n, m = Phi.shape
        beta = np.zeros(m)

        l2_diag = np.full(m, self.l2)
        l2_diag[-1] = 0.0

        def loglik(b):
            p = sigmoid(Phi @ b)
            return (
                np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
                - 0.5 * np.sum(l2_diag * b * b)
            )

        ll_old = loglik(beta)

        for _ in range(max_iter):
            eta = Phi @ beta
            p = sigmoid(eta)
            w = p * (1.0 - p)

            grad = Phi.T @ (y - p) - l2_diag * beta

            fisher = (Phi.T * w) @ Phi + np.diag(l2_diag)
            fisher = stabilize_fisher(fisher, m)

            try:
                delta = solve(fisher, grad, assume_a="pos")
            except np.linalg.LinAlgError:
                break

            step = 1.0
            improved = False
            directional = grad @ delta
            for _ in range(20):
                b_new = beta + step * delta
                ll_new = loglik(b_new)
                if ll_new >= ll_old + 1e-4 * step * directional:
                    beta = b_new
                    ll_old = ll_new
                    improved = True
                    break
                step *= 0.5

            if not improved or np.linalg.norm(delta) < tol:
                break

        eta = Phi @ beta
        p = sigmoid(eta)
        w = p * (1.0 - p)
        fisher = (Phi.T * w) @ Phi + np.diag(l2_diag)
        fisher = stabilize_fisher(fisher, m)

        return beta, fisher, loglik(beta)

    # ──────────────────────────────────────────────────────────────
    # NEGATIVE LOG MARGINAL LIKELIHOOD
    # ──────────────────────────────────────────────────────────────
    def _neg_log_marginal(self, Xs, ys, sigma, fixed_seed, strict=True):
        """Negative log marginal likelihood S(sigma)."""
        d = Xs.shape[1]
        mi = 50 if strict else 15
        tl = 1e-6 if strict else 1e-3

        rff = RFF(self.m, d, sigma, fixed_seed)
        Phi = np.hstack(
            [rff.transform(Xs), np.ones((Xs.shape[0], 1))]
        )
        _, fisher, ll = self._scoring(Phi, ys, mi, tl)

        s, ld = np.linalg.slogdet(fisher)
        if s <= 0 or not np.isfinite(ld):
            return np.inf

        dim_total = self.m + 1
        return -ll + 0.5 * ld - 0.5 * dim_total * np.log(2.0 * np.pi)

    # ──────────────────────────────────────────────────────────────
    # SIGMA SELECTION
    # ──────────────────────────────────────────────────────────────
    def _select_sigma(self, X, y):
        """Data-adaptive kernel bandwidth via marginal likelihood."""
        from scipy.optimize import minimize_scalar

        n, d = X.shape
        rng = np.random.default_rng(self.seed)

        n_min = max(100, 2 * d)
        n_sub = min(n, max(n_min, 2 * self.m))
        idx = rng.choice(n, n_sub, replace=False)
        Xs, ys = X[idx], y[idx]

        dists = np.linalg.norm(
            Xs[:, None, :] - Xs[None, :, :], axis=2
        )
        np.fill_diagonal(dists, np.nan)
        sigma_med = np.nanmedian(dists)
        if not np.isfinite(sigma_med) or sigma_med <= 0:
            sigma_med = np.std(Xs) * np.sqrt(d)

        sigma_lo = sigma_med * 0.01
        sigma_hi = sigma_med * 10.0

        grid = sigma_med * np.logspace(-2, 2, 9)
        fixed_seed = self.seed + 1234

        cands = []
        for sigma in grid:
            sc = self._neg_log_marginal(
                Xs, ys, sigma, fixed_seed, strict=False
            )
            if np.isfinite(sc):
                cands.append((sc, sigma))
        if not cands:
            return sigma_med, (sigma_med * 0.5) ** 2

        cands.sort()
        best = (np.inf, cands[0][1])
        for _, sigma in cands[:2]:
            sc = self._neg_log_marginal(
                Xs, ys, sigma, fixed_seed, strict=True
            )
            if sc < best[0]:
                best = (sc, sigma)
        sigma_grid = best[1]

        def objective(log_sigma):
            sigma = np.exp(log_sigma)
            sigma = np.clip(sigma, sigma_lo, sigma_hi)
            val = self._neg_log_marginal(
                Xs, ys, sigma, fixed_seed, strict=True
            )
            return val if np.isfinite(val) else 1e20

        log_bounds = (
            np.log(max(sigma_grid * 0.1, sigma_lo)),
            np.log(min(sigma_grid * 10.0, sigma_hi)),
        )

        opt = minimize_scalar(
            objective,
            bounds=log_bounds,
            method="bounded",
            options={"xatol": 1e-5},
        )

        if opt.success and np.isfinite(opt.fun):
            sigma_star = np.exp(opt.x)
        else:
            sigma_star = sigma_grid

        sigma_star = np.clip(sigma_star, sigma_lo, sigma_hi)

        s_star_fast = self._neg_log_marginal(
            Xs, ys, sigma_star, fixed_seed, strict=False
        )

        delta_fracs = [0.02, 0.05, 0.10]
        s_dd_positive = []

        for frac in delta_fracs:
            delta = sigma_star * frac

            sigma_plus = np.clip(
                sigma_star + delta, sigma_lo, sigma_hi
            )
            sigma_minus = np.clip(
                sigma_star - delta, sigma_lo, sigma_hi
            )

            if (
                abs(sigma_plus - (sigma_star + delta)) > 1e-10
                or abs(sigma_minus - (sigma_star - delta)) > 1e-10
            ):
                continue

            s_plus = self._neg_log_marginal(
                Xs, ys, sigma_plus, fixed_seed, strict=False
            )
            s_minus = self._neg_log_marginal(
                Xs, ys, sigma_minus, fixed_seed, strict=False
            )

            if np.isfinite(s_plus) and np.isfinite(s_minus):
                s_dd = (
                    (s_plus + s_minus - 2.0 * s_star_fast)
                    / (delta ** 2)
                )
                if s_dd > EPS:
                    s_dd_positive.append(s_dd)

        if len(s_dd_positive) >= 2:
            s_dd_sub = np.median(s_dd_positive)
        elif len(s_dd_positive) == 1:
            s_dd_sub = s_dd_positive[0]
        else:
            sigma_var = (sigma_med * 0.5) ** 2
            return sigma_star, sigma_var

        s_dd_full = s_dd_sub * (n / n_sub)
        sigma_var = 1.0 / s_dd_full

        return sigma_star, sigma_var

    # ──────────────────────────────────────────────────────────────
    # SIGMA SENSITIVITY
    # ──────────────────────────────────────────────────────────────
    def _sigma_sensitivity(self, X):
        """dη/dσ at X using analytical chain rule."""
        rff = self.rff_
        beta_rff = self.coef_[:-1]

        theta = X @ rff.omega.T + rff.b
        wx = X @ rff.omega.T

        dphi_dsigma = rff.c * np.sin(theta) * (wx / self.sigma_)
        deta_dsigma = dphi_dsigma @ beta_rff

        return deta_dsigma

    # ──────────────────────────────────────────────────────────────
    # JACOBIAN + β_equiv
    # ──────────────────────────────────────────────────────────────
    def _compute_jacobian(self, X_train):
        """Jacobian projection at sample mean and β_equiv."""
        rff = self.rff_
        x_bar = X_train.mean(axis=0)
        beta_rff = self.coef_[:-1]

        theta = rff.omega @ x_bar + rff.b
        self.J_ = -rff.c * (
            np.sin(theta)[:, None] * rff.omega
        )

        self.beta_equiv_ = self.J_.T @ beta_rff

        phi_xbar = self._design(x_bar.reshape(1, -1))[0]
        self.beta_equiv_bias_ = float(
            phi_xbar @ self.coef_ - x_bar @ self.beta_equiv_
        )

        if self.cov_ is not None:
            Sigma_rff = self.cov_[:-1, :-1]
            self.beta_equiv_cov_ = self.J_.T @ Sigma_rff @ self.J_
        else:
            self.beta_equiv_cov_ = None

    # ──────────────────────────────────────────────────────────────
    # GRADIENT AT ARBITRARY POINT  (Simpson + attribution)
    # ──────────────────────────────────────────────────────────────
    def _gradient_at_point(self, x):
        """∂η/∂x at arbitrary x ∈ R^d.

        Returns
        -------
        grad : (d,) array — ∂η/∂x_1, ..., ∂η/∂x_d
        """
        rff = self.rff_
        beta_rff = self.coef_[:-1]
        c = rff.c

        theta = rff.omega @ x + rff.b                    # (m,)
        grad_eta = -c * (np.sin(theta) * beta_rff) @ rff.omega  # (d,)
        return grad_eta

    # ──────────────────────────────────────────────────────────────
    # ETA AT ARBITRARY POINT
    # ──────────────────────────────────────────────────────────────
    def _eta_at_point(self, x):
        """η = Φ̃(x)'β at arbitrary x."""
        phi = self._design(x.reshape(1, -1))[0]
        return float(phi @ self.coef_)

    # ──────────────────────────────────────────────────────────────
    # RFF-ATTRIBUTION: Simpson / Trapezoidal
    # ──────────────────────────────────────────────────────────────
    def compute_attribution(self, x, x_ref=None, method='simpson'):
        """Path-dependent gradient decomposition with L/NL split.

        Parameters
        ----------
        x      : (d,) — patient covariate vector
        x_ref  : (d,) — reference point (default: training mean)
        method : 'simpson' or 'trapezoidal'

        Returns
        -------
        dict with keys:
            L            : (d,) linear components
            NL           : (d,) nonlinear components
            delta_eta    : float, total logit shift
            gap          : float, approximation error (%)
            method       : str, which quadrature was used
            grad_ref     : (d,) gradient at reference
            grad_x       : (d,) gradient at patient
            grad_mid     : (d,) gradient at midpoint (Simpson only)
        """
        if x_ref is None:
            x_ref = self._x_bar_

        dx = x - x_ref                                     # (d,)
        eta_x = self._eta_at_point(x)
        eta_ref = self._eta_at_point(x_ref)
        delta_eta = eta_x - eta_ref

        grad_ref = self._gradient_at_point(x_ref)          # (d,)
        grad_x = self._gradient_at_point(x)                # (d,)

        # Linear component (same for both methods)
        L = dx * grad_ref

        if method == 'simpson':
            x_mid = 0.5 * (x + x_ref)
            grad_mid = self._gradient_at_point(x_mid)      # (d,)
            total = dx * (grad_ref + 4.0 * grad_mid + grad_x) / 6.0
        else:
            grad_mid = None
            total = dx * 0.5 * (grad_ref + grad_x)

        NL = total - L

        gap = abs(np.sum(total) - delta_eta) / max(abs(delta_eta), EPS) * 100.0

        result = {
            'L': L,
            'NL': NL,
            'delta_eta': delta_eta,
            'gap': gap,
            'method': method,
            'grad_ref': grad_ref,
            'grad_x': grad_x,
        }
        if grad_mid is not None:
            result['grad_mid'] = grad_mid

        return result

    # ──────────────────────────────────────────────────────────────
    # ATTRIBUTION COMPARISON: Simpson vs Trapezoidal
    # ──────────────────────────────────────────────────────────────
    def compare_attribution_methods(self, X, n_sample=None, seed=None):
        """Compare Simpson vs Trapezoidal gap on a sample.

        Parameters
        ----------
        X        : (n, d) test matrix
        n_sample : int or None (default: min(200, n))
        seed     : int or None

        Returns
        -------
        dict with median gaps, per-patient gaps, improvement
        """
        n = X.shape[0]
        if n_sample is None:
            n_sample = min(200, n)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, n_sample, replace=False)

        gaps_trap = np.zeros(n_sample)
        gaps_simp = np.zeros(n_sample)

        for i, ix in enumerate(idx):
            res_t = self.compute_attribution(
                X[ix], method='trapezoidal')
            res_s = self.compute_attribution(
                X[ix], method='simpson')
            gaps_trap[i] = res_t['gap']
            gaps_simp[i] = res_s['gap']

        med_trap = float(np.median(gaps_trap))
        med_simp = float(np.median(gaps_simp))

        return {
            'median_gap_trapezoidal': med_trap,
            'median_gap_simpson': med_simp,
            'improvement': med_trap - med_simp,
            'gaps_trapezoidal': gaps_trap,
            'gaps_simpson': gaps_simp,
        }

    # ──────────────────────────────────────────────────────────────
    # BINARY PROBA HELPER
    # ──────────────────────────────────────────────────────────────
    def _predict_proba_binary(self, X):
        """P(y=0) and P(y=1) columns."""
        Phi = self._design(X)
        p = sigmoid(Phi @ self.coef_)
        return np.column_stack([1.0 - p, p])

    # =================================================================
    # FIT
    # =================================================================
    def fit(self, X, y, sigma=None):
        """Fit FLARE model.

        Parameters
        ----------
        X     : (n, d) feature matrix
        y     : (n,) binary or multi-class labels
        sigma : float or None (auto via Laplace evidence)

        Returns
        -------
        self (fitted model)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least 2 classes.")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ == 2:
            return self._fit_binary(X, y, sigma)
        return self._fit_multiclass(X, y, sigma)

    # ── Binary ───────────────────────────────────────────────────
    def _fit_binary(self, X, y, sigma):
        """Fit binary classifier with full uncertainty pipeline."""
        # 1. sigma
        if sigma is not None:
            self.sigma_ = sigma
            self.sigma_var_ = 0.0
        else:
            self.sigma_, self.sigma_var_ = self._select_sigma(X, y)

        # 2. RFF + Fisher scoring
        self.rff_ = RFF(self.m, X.shape[1], self.sigma_, self.seed)
        Phi = self._design(X)
        self.coef_, fisher, self.loglik_ = self._scoring(Phi, y)

        # 3. Sandwich (HC2)
        n, m = Phi.shape
        eta = Phi @ self.coef_
        p = sigmoid(eta)
        w = p * (1.0 - p)
        residuals = y - p

        sandwich_floor = max(self.l2, EPS)

        F_raw = (Phi.T * w) @ Phi
        F_raw = stabilize_fisher(F_raw, m, floor=sandwich_floor)

        use_lambda_free = True
        try:
            cond = np.linalg.cond(F_raw)
            if cond > 1e12:
                use_lambda_free = False
        except np.linalg.LinAlgError:
            use_lambda_free = False

        if use_lambda_free:
            F_inv = solve(F_raw, np.eye(m), assume_a="pos")
        else:
            F_inv = solve(fisher, np.eye(m), assume_a="pos")

        Phi_Finv = Phi @ F_inv
        h = w * np.sum(Phi_Finv * Phi, axis=1)
        h = np.clip(h, 0.0, 0.999)

        adj_weights = residuals ** 2 / (1.0 - h)
        G = (Phi.T * adj_weights) @ Phi
        G = stabilize_fisher(G, m, floor=sandwich_floor)

        cov_raw = F_inv @ G @ F_inv
        self.cov_ = 0.5 * (cov_raw + cov_raw.T)

        # 4. Effective df
        self.df_eff_ = np.trace(F_inv @ F_raw)
        self.se_ = np.sqrt(np.maximum(np.diag(self.cov_), EPS))

        # 5. Jacobian → β_equiv
        self._compute_jacobian(X)

        # 6. Store training mean for attribution reference
        self._x_bar_ = X.mean(axis=0)

        self.n_train_ = len(y)
        self.n_features_ = X.shape[1]
        return self

    # ── Multi-class (One-vs-Rest) ────────────────────────────────
    def _fit_multiclass(self, X, y, sigma):
        """One-vs-Rest multi-class."""
        if sigma is None:
            maj = self.classes_[
                np.bincount(y.astype(int)).argmax()
            ]
            sigma_result = self._select_sigma(
                X, (y == maj).astype(float)
            )
            sigma = (
                sigma_result[0]
                if isinstance(sigma_result, tuple)
                else sigma_result
            )
            self.sigma_var_ = (
                sigma_result[1]
                if isinstance(sigma_result, tuple)
                else 0.0
            )
        else:
            self.sigma_var_ = 0.0
        self.sigma_ = sigma

        self.models_ = {}
        for cls in self.classes_:
            m = FLARE(self.m, self.l2, self.seed)
            m.fit(X, (y == cls).astype(float), sigma=sigma)
            self.models_[cls] = m

        self.rff_ = self.models_[self.classes_[0]].rff_
        self.cov_ = None
        self.J_ = None
        self.beta_equiv_ = None
        self.beta_equiv_bias_ = None
        self.beta_equiv_cov_ = None
        self._x_bar_ = X.mean(axis=0)
        return self

    # =================================================================
    # PREDICTIONS
    # =================================================================
    def predict_eta(self, X, return_se=False):
        """Logit prediction η = Φ̃(x)'β."""
        if self.n_classes_ > 2:
            raise NotImplementedError("predict_eta is binary-only.")

        Phi = self._design(X)
        eta = Phi @ self.coef_

        if not return_se:
            return eta

        var_laplace = np.sum((Phi @ self.cov_) * Phi, axis=1)

        if hasattr(self, "sigma_var_") and self.sigma_var_ > 0:
            deta_dsigma = self._sigma_sensitivity(X)
            var_sigma = (deta_dsigma ** 2) * self.sigma_var_
        else:
            var_sigma = 0.0

        var_total = var_laplace + var_sigma
        se = np.sqrt(np.maximum(var_total, EPS))
        return eta, se

    def predict_proba(self, X, return_se=False):
        """Class probabilities with optional Delta Method SE."""
        if self.n_classes_ == 2:
            if return_se:
                eta, se_eta = self.predict_eta(X, True)
                p = sigmoid(eta)
                se_p = p * (1.0 - p) * se_eta
                return (
                    np.column_stack([1.0 - p, p]),
                    np.column_stack([se_p, se_p]),
                )
            return self._predict_proba_binary(X)

        etas = np.column_stack(
            [m.predict_eta(X) for m in self.models_.values()]
        )
        etas -= etas.max(axis=1, keepdims=True)
        exp_eta = np.exp(etas)
        probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)

        if not return_se:
            return probs

        ses = np.column_stack(
            [m.predict_eta(X, True)[1] for m in self.models_.values()]
        )
        return probs, probs * (1.0 - probs) * ses

    def predict(self, X, threshold=0.5):
        """Predicted class labels."""
        probs = self.predict_proba(X)
        if self.n_classes_ == 2:
            return (probs[:, 1] >= threshold).astype(int)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_ci(self, X, z=None):
        """Prediction CI with full uncertainty decomposition."""
        if self.n_classes_ > 2:
            raise NotImplementedError("CI is binary-only.")

        Phi = self._design(X)
        eta = Phi @ self.coef_

        var_laplace = np.sum((Phi @ self.cov_) * Phi, axis=1)

        if hasattr(self, "sigma_var_") and self.sigma_var_ > 0:
            deta_dsigma = self._sigma_sensitivity(X)
            var_sigma = (deta_dsigma ** 2) * self.sigma_var_
        else:
            var_sigma = np.zeros_like(var_laplace)

        var_epistemic = var_laplace + var_sigma
        se_eta = np.sqrt(np.maximum(var_epistemic, EPS))

        p = sigmoid(eta)
        se_p = p * (1.0 - p) * se_eta

        if z is None:
            df = max(self.n_train_ - self.df_eff_, 1)
            z = t_dist.ppf(0.975, df=df)

        ci_lo = sigmoid(eta - z * se_eta)
        ci_hi = sigmoid(eta + z * se_eta)

        var_aleatoric = p * (1.0 - p)

        return {
            "p_hat": p,
            "se": se_p,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
            "eta": eta,
            "se_eta": se_eta,
            "var_laplace": var_laplace,
            "var_sigma": var_sigma,
            "var_aleatoric": var_aleatoric,
            "df_eff": self.df_eff_,
        }

    # ──────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────
    def score(self, X, y):
        """ROC-AUC score."""
        from sklearn.metrics import roc_auc_score

        probs = self.predict_proba(X)
        if self.n_classes_ == 2:
            return roc_auc_score(y, probs[:, 1])
        return roc_auc_score(
            y, probs, multi_class="ovr", average="macro"
        )

    def _compute_hessian_patient(self, x_k):
        """Patient-level Hessian: ∂²η/∂x_i∂x_j at x=x^(k)."""
        rff = self.rff_
        beta_rff = self.coef_[:-1]
        c = rff.c

        theta = rff.omega @ x_k + rff.b
        cos_theta_beta = np.cos(theta) * beta_rff

        W = rff.omega
        H = -c * (W.T * cos_theta_beta) @ W

        return H