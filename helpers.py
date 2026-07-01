""""
helpers.py — FLARE Helper Functions (v3.1)
==========================================
Model-external analysis utilities.
FLARE class is not modified by any function here.

Categories:
  A. DeLong Test
  B. LR Coefficients
  C. β_equiv — Equivalent Linear Coefficients
  D. Local Attribution — RFF-Attribution (Midpoint IG)
  E. Interaction Tests — ∂²η Wald, Bonferroni, by-reference
  F. Risk Groups — β_equiv by group
  G. LR vs FLARE — Comparison table
  H. Residuals — Pearson, Deviance
  I. Evaluation — AUC, Calibration, AIC/BIC, Summary
  J. Uncertainty — SE discrimination, patient risk profile
  K. Validation — Coverage, CV, posterior predictive
  L. True Beta — Synthetic data recovery
  M. Data Pipeline — Multi-objective param search, CV evaluation
  N. Visualisation — Plot data
  O. Internal Helpers
  P. Competitor Models — LR, RF, XGBoost, NGBoost
  Q. Feature Importance — Full comparison tables
  R. Interaction Analysis Pipeline
  S. Patient-Level Hessian Analysis

Changes from v3:
  - Removed duplicate _sig_stars; unified as _sig_label
  - _flare_shap_values refactored to use _analytical_grad helper
  - posterior_predictive_ci: fixed seed=0 edge case
  - interaction_tests: clarified sig star logic
  - Updated docstring to match actual section layout
  - Formatting consistency pass
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, accuracy_score,
                              precision_score, recall_score,
                              matthews_corrcoef, brier_score_loss)
#from core import FLARE
from flare import FLARE
EPS = 1e-10


# ═══════════════════════════════════════════════════════════════
# O. INTERNAL HELPERS (defined first — used everywhere)
# ═══════════════════════════════════════════════════════════════

def _sig_label(p):
    """Significance star label."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    return ''


def _analytical_grad(rff, beta_rff, X):
    """∂η/∂x_j at X. Returns (n, d).

    grad = -c · (sin(θ) ⊙ β_rff) · ω
    """
    theta = X @ rff.omega.T + rff.b
    return (-rff.c
            * (np.sin(theta) * beta_rff[None, :])
            @ rff.omega)


def _compute_jacobian_at_points(model, X):
    """J[i,k,j] = ∂φ_k/∂x_j at each X[i].

    Shape: (n, m, d).
    """
    rff = model.rff_
    theta = X @ rff.omega.T + rff.b
    sin_theta = np.sin(theta)
    return -rff.c * (sin_theta[:, :, None]
                     * rff.omega[None, :, :])


# ═══════════════════════════════════════════════════════════════
# A. DELONG TEST
# ═══════════════════════════════════════════════════════════════

def delong_test(y_true, prob_1, prob_2):
    """DeLong test for comparing two AUC values.

    Returns
    -------
    dict: auc_1, auc_2, diff, se, z, p
    """
    def _placement(y_true, y_score):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        m, n = len(pos), len(neg)
        V10 = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p)
                        for p in pos])
        V01 = np.array([np.mean(pos > v) + 0.5 * np.mean(pos == v)
                        for v in neg])
        return np.mean(V10), V10, V01, m, n

    auc1, V10_1, V01_1, m, n = _placement(y_true, prob_1)
    auc2, V10_2, V01_2, _, _ = _placement(y_true, prob_2)

    var1 = np.var(V10_1, ddof=1) / m + np.var(V01_1, ddof=1) / n
    var2 = np.var(V10_2, ddof=1) / m + np.var(V01_2, ddof=1) / n
    cov_val = (np.cov(V10_1, V10_2)[0, 1] / m
               + np.cov(V01_1, V01_2)[0, 1] / n)

    diff = auc1 - auc2
    se = np.sqrt(max(var1 + var2 - 2 * cov_val, 1e-12))
    z = diff / se
    p = 2 * norm.sf(abs(z))
    return {'auc_1': auc1, 'auc_2': auc2, 'diff': diff,
            'se': se, 'z': z, 'p': p}


# ═══════════════════════════════════════════════════════════════
# B. LR COEFFICIENTS
# ═══════════════════════════════════════════════════════════════

def lr_coefficients_full(lr, X_train, y_train, feature_names,
                         z=1.96):
    """LR coefficients: β, SE, Wald Z, p, HR, CI.

    Returns
    -------
    dict: beta, se, z, p, hr, ci_lo, ci_up
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    beta = lr.coef_[0]
    intercept = lr.intercept_[0]

    eta = X @ beta + intercept
    p_hat = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    w = p_hat * (1.0 - p_hat)

    X_aug = np.hstack([X, np.ones((n, 1))])
    fisher = (X_aug.T * w) @ X_aug
    l2_reg = (1.0 / max(getattr(lr, 'C', 1e10), 1e-10)
              * np.eye(d + 1))
    l2_reg[-1, -1] = 0.0
    fisher += l2_reg

    try:
        fisher = 0.5 * (fisher + fisher.T)
        cov = np.linalg.inv(fisher)
        cov = 0.5 * (cov + cov.T)
        se = np.sqrt(np.maximum(np.diag(cov), EPS))
    except Exception:
        se = np.full(d + 1, np.nan)

    se_beta = se[:d]
    z_scores = beta / np.maximum(se_beta, EPS)
    p_values = 2 * norm.sf(np.abs(z_scores))
    ci_lo = beta - z * se_beta
    ci_up = beta + z * se_beta
    hr = np.exp(beta)

    print(f"\n  {'=' * 80}")
    print(f"  LOGISTIC REGRESSION COEFFICIENTS")
    print(f"  {'=' * 80}")
    print(f"  {'Var':<12} {'β':>10} {'SE':>8} {'Z':>8} "
          f"{'p':>12} {'HR':>8} {'95% CI':>22} {'Sig':>5}")
    print(f"  {'-' * 80}")
    for j in range(d):
        sig = _sig_label(p_values[j])
        print(f"  {feature_names[j]:<12} {beta[j]:>10.4f} "
              f"{se_beta[j]:>8.4f} {z_scores[j]:>8.3f} "
              f"{p_values[j]:>12.4e} {hr[j]:>8.4f} "
              f"[{ci_lo[j]:>9.4f}, {ci_up[j]:>8.4f}] {sig:>5}")
    print(f"  {'Intercept':<12} {intercept:>10.4f}")
    return {'beta': beta, 'se': se_beta, 'z': z_scores,
            'p': p_values, 'hr': hr, 'ci_lo': ci_lo,
            'ci_up': ci_up, 'intercept': intercept}


# ═══════════════════════════════════════════════════════════════
# C. β_equiv — EQUIVALENT LINEAR COEFFICIENTS
# ═══════════════════════════════════════════════════════════════

def beta_equiv_summary(model, feature_names, z=1.96):
    """β_equiv table: equivalent LR coefficients at x̄.

    Returns
    -------
    dict: feature, beta, se, z, p, hr, ci_lo, ci_up, bias
    """
    beta = model.beta_equiv_
    bias = model.beta_equiv_bias_
    cov = getattr(model, 'beta_equiv_cov_', None)
    d = len(beta)

    se = (np.sqrt(np.maximum(np.diag(cov), EPS))
          if cov is not None else np.full(d, np.nan))

    z_sc = beta / np.maximum(se, EPS)
    pv = 2 * norm.sf(np.abs(z_sc))
    lo = beta - z * se
    hi = beta + z * se

    print(f"\n  {'=' * 80}")
    print(f"  β_equiv — Equivalent LR Coefficients (at x̄)")
    print(f"  {'=' * 80}")
    print(f"  Intercept (α) = {bias:.4f}")
    print(f"\n  {'Var':<12} {'β_eq':>10} {'SE':>8} {'Z':>8} "
          f"{'p':>12} {'HR':>8} {'95% CI':>22} {'Sig':>5}")
    print(f"  {'-' * 80}")
    for j in range(d):
        sig = _sig_label(pv[j])
        print(f"  {feature_names[j]:<12} {beta[j]:>10.4f} "
              f"{se[j]:>8.4f} {z_sc[j]:>8.3f} "
              f"{pv[j]:>12.4e} {np.exp(beta[j]):>8.4f} "
              f"[{lo[j]:>9.4f}, {hi[j]:>8.4f}] {sig:>5}")

    return {'feature': feature_names, 'beta': beta, 'se': se,
            'z': z_sc, 'p': pv, 'hr': np.exp(beta),
            'ci_lo': lo, 'ci_up': hi, 'bias': bias}


# ═══════════════════════════════════════════════════════════════
# D. LOCAL ATTRIBUTION — RFF-Attribution (Midpoint IG)
# ═══════════════════════════════════════════════════════════════

def rff_attribution_patient(model, X_test, feature_names, idx):
    """Single-patient RFF-Attribution (Simpson quadrature)."""
    x = X_test[idx]

    # YENİ: FLARE'nin kendi Simpson metodu
    attr = model.compute_attribution(x, method='simpson')

    L        = attr['L']
    NL       = attr['NL']
    total    = L + NL
    eta_x    = model._eta_at_point(x)
    eta_bar  = model._eta_at_point(model._x_bar_)
    delta_eta = attr['delta_eta']
    gap      = abs(float(np.sum(total)) - delta_eta)
    gap_pct  = gap / max(abs(delta_eta), EPS) * 100

    print(f"\n  RFF-Attribution (Simpson) — Patient #{idx}")
    print(f"  η(x) = {eta_x:.4f}, η(x̄) = {eta_bar:.4f}, "
          f"Δη = {delta_eta:.4f}")
    print(f"  {'Var':<12} {'Attr(L)':>10} {'Attr(NL)':>10} "
          f"{'Total':>10} {'Dir':>8}")
    print(f"  {'-' * 55}")
    for j, name in enumerate(feature_names):
        direction = ("↑" if float(total[j]) > 0.01
                     else "↓" if float(total[j]) < -0.01
                     else "≈")
        print(f"  {name:<12} {float(L[j]):>10.4f} "
              f"{float(NL[j]):>10.4f} "
              f"{float(total[j]):>10.4f} {direction:>8}")
    print(f"  {'-' * 55}")
    print(f"  {'Sum':<12} {'':>10} {'':>10} "
          f"{float(np.sum(total)):>10.4f}")
    print(f"  {'Δη':<12} {'':>10} {'':>10} "
          f"{delta_eta:>10.4f}")
    print(f"  {'Gap':<12} {'':>10} {'':>10} "
          f"{gap:>10.4f}  ({gap_pct:.1f}%)")

    quality = ("✓ Excellent" if gap_pct < 5
               else "✓ Good" if gap_pct < 15
               else "~ Fair" if gap_pct < 30
               else "✗ Weak (expected: nonlinear model)")
    print(f"  {'Quality':<12} {'':>10} {'':>10} {'':>10}  "
          f"{quality}")

    return {'attr_linear': L, 'attr_nonlinear': NL,
            'total': total, 'eta_x': eta_x, 'eta_bar': eta_bar,
            'delta_eta': delta_eta, 'gap': gap,
            'gap_pct': gap_pct}
def print_coverage_report(coverage: dict) -> None:
    """
    binary_posterior_coverage() sonucunu okunabilir tablo olarak yazdırır.

    Parameters
    ----------
    coverage : dict
        binary_posterior_coverage() dönüş değeri.
        Beklenen anahtarlar:
          coverage, target, quality, n_bins, total_n, bins
        Her bins elemanı:
          bin, n, observed, predicted, ci_lo, ci_hi,
          se_model, se_sampling, se_total, covered
    """
    print(f"\n  {'─' * 75}")
    print(f"  Posterior Coverage")
    print(f"  {'─' * 75}")
    print(f"  Coverage    : {coverage['coverage']:.1%} "
          f"(target: {coverage['target']:.1%}) "
          f"— {coverage['quality']}")
    print(f"  Bins        : {coverage['n_bins']}")
    print(f"  Total       : {coverage['total_n']}")

    # ── Tablo başlığı ──
    hdr = (f"  {'Bin':<16} {'n':>5} {'Obs':>9} "
           f"{'Pred':>10} {'CI_lo':>8} {'CI_hi':>8} "
           f"{'SE_m':>9} {'SE_s':>9} {'SE_t':>9} "
           f"{'OK':>4}")
    sep = f"  {'-' * 96}"

    print(f"\n{hdr}")
    print(sep)

    # ── Satırlar ──
    for b in coverage['bins']:
        mark = '✓' if b['covered'] else '✗'
        print(f"  {b['bin']:<16} {b['n']:>5} "
              f"{b['observed']:>9.4f} "
              f"{b['predicted']:>10.4f} "
              f"{b['ci_lo']:>8.4f} {b['ci_hi']:>8.4f} "
              f"{b['se_model']:>9.4f} "
              f"{b['se_sampling']:>9.4f} "
              f"{b['se_total']:>9.4f} {mark:>4}")

    # ── Özet satırı ──
    n_ok = sum(1 for b in coverage['bins'] if b['covered'])
    n_total = len(coverage['bins'])
    print(sep)
    print(f"  {'Toplam':<16} {'':>5} {'':>9} "
          f"{'':>10} {'':>8} {'':>8} "
          f"{'':>9} {'':>9} {'':>9} "
          f"{n_ok}/{n_total:>3}")
def rff_attribution_bulk(model, X, x_ref=None):
    """Bulk RFF-Attribution for all patients.

    Uses midpoint gradient integration (same as patient version).
    NO rescaling — preserves true approximation gap.

    Returns
    -------
    dict: attribution (n,d), attribution_2 (n,d),
          total_per_feature (n,d), decomp_sum (n,),
          eta_diff (n,), gap (n,), gap_pct (n,)
    """
    n, d = X.shape
    if x_ref is None:
        x_ref = X.mean(axis=0).reshape(1, -1)
    x_ref = np.atleast_2d(x_ref)

    rff = model.rff_
    beta_rff = model.coef_[:-1]

    eta_X = model.predict_eta(X)
    eta_ref = model.predict_eta(x_ref)
    delta_eta = eta_X - eta_ref

    grad_X = _analytical_grad(rff, beta_rff, X)
    grad_ref = _analytical_grad(rff, beta_rff, x_ref)

    dx = X - x_ref

    attr_linear = dx * grad_ref
    attr_nonlinear = dx * 0.5 * (grad_X - grad_ref)
    total_per_feature = dx * 0.5 * (grad_ref + grad_X)

    decomp_sum = total_per_feature.sum(axis=1)
    gap = np.abs(decomp_sum - delta_eta)

    # NaN-safe gap percentage
    significant = np.abs(delta_eta) > 0.1
    gap_pct_raw = gap / np.maximum(np.abs(delta_eta), EPS) * 100
    gap_pct = np.where(significant, gap_pct_raw, np.nan)

    return {
        'attribution': attr_linear,
        'attribution_2': attr_nonlinear,
        'total_per_feature': total_per_feature,
        'decomp_sum': decomp_sum,
        'eta_diff': delta_eta,
        'gap': gap,
        'gap_pct': gap_pct,
    }


# ═══════════════════════════════════════════════════════════════
# D2. FLARE SHAP HELPER
# ═══════════════════════════════════════════════════════════════

def _flare_shap_values(model, X, x_ref=None):
    """FLARE SHAP-like values using RFF Jacobian.

    For linear model on features Φ, exact SHAP is:
      φ_j = (x_j - x̄_j) * ∂η/∂x_j  (at midpoint)

    Returns
    -------
    shap_vals : (n, d)
    eta_ref   : float
    """
    rff = model.rff_
    beta_rff = model.coef_[:-1]

    if x_ref is None:
        x_ref = X.mean(axis=0, keepdims=True)

    grad = _analytical_grad(rff, beta_rff, X)
    grad_ref = _analytical_grad(rff, beta_rff, x_ref)

    dx = X - x_ref
    shap_vals = dx * 0.5 * (grad + grad_ref)

    eta_ref = float(model.predict_eta(x_ref))

    return shap_vals, eta_ref


# ═══════════════════════════════════════════════════════════════
# E. INTERACTION TESTS
# ═══════════════════════════════════════════════════════════════

def interaction_test(model, X_ref, j, k, feature_names,
                     eps=1e-5, z=1.96):
    """Single pair ∂²η/∂x_j∂x_k Wald test (Delta method).

    Returns
    -------
    dict: pair, estimate, se, z, p, ci_lo, ci_up
    """
    X_ref = np.atleast_2d(X_ref)

    Xpp = X_ref.copy(); Xpp[0, j] += eps; Xpp[0, k] += eps
    Xpm = X_ref.copy(); Xpm[0, j] += eps; Xpm[0, k] -= eps
    Xmp = X_ref.copy(); Xmp[0, j] -= eps; Xmp[0, k] += eps
    Xmm = X_ref.copy(); Xmm[0, j] -= eps; Xmm[0, k] -= eps

    cp = float(
        (model.predict_eta(Xpp) - model.predict_eta(Xpm)
         - model.predict_eta(Xmp) + model.predict_eta(Xmm))
        / (4 * eps ** 2))

    rff = model.rff_
    j_jk = ((rff.transform(Xpp) - rff.transform(Xpm)
             - rff.transform(Xmp) + rff.transform(Xmm))
            / (4 * eps ** 2))
    m_rff = j_jk.shape[1]
    j_full = np.zeros((1, m_rff + 1))
    j_full[:, :m_rff] = j_jk

    if model.cov_ is not None:
        var_cp = float(j_full @ model.cov_ @ j_full.T)
    else:
        var_cp = 0.0

    se_cp = np.sqrt(max(var_cp, EPS))
    z_cp = cp / se_cp if se_cp > EPS else 0.0
    p_cp = 2 * norm.sf(abs(z_cp))

    name = f"{feature_names[j]}×{feature_names[k]}"
    return {'pair': name, 'estimate': cp, 'se': se_cp,
            'z': z_cp, 'p': p_cp,
            'ci_lo': cp - z * se_cp, 'ci_up': cp + z * se_cp,
            'j': j, 'k': k}


def interaction_tests(model, X_ref, feature_names,
                      holm=True, eps=1e-5, z=1.96,
                      report_threshold=0.2):
    """All pairs ∂²η Wald test + Holm-Bonferroni.

    Parameters
    ----------
    holm : bool (default True)
        True  → Holm-Bonferroni (step-down)
        False → Klasik Bonferroni
    report_threshold : float (default 0.2)
        Only pairs with p < report_threshold are printed.

    Returns
    -------
    list of dicts (all pairs, sorted by p-value)
    """
    X_ref = np.atleast_2d(X_ref)
    d = min(X_ref.shape[1], len(feature_names))

    pairs_list = [(j, k) for j in range(d)
                  for k in range(j + 1, d)]
    results = []
    for j, k in pairs_list:
        r = interaction_test(model, X_ref, j, k,
                             feature_names, eps, z)
        results.append(r)

    n_tests = len(results)

    if holm:
        sorted_idx = sorted(range(len(results)),
                            key=lambda i: results[i]['p'])
        for rank, idx in enumerate(sorted_idx):
            results[idx]['holm_threshold'] = (
                0.05 / (n_tests - rank))
            results[idx]['holm_reject'] = (
                results[idx]['p'] < 0.05 / (n_tests - rank))
            if rank > 0:
                prev_idx = sorted_idx[rank - 1]
                if not results[prev_idx]['holm_reject']:
                    results[idx]['holm_reject'] = False
        alpha_adj_label = "Holm-Bonferroni"
    else:
        alpha_adj = 0.05 / n_tests
        for r in results:
            r['holm_threshold'] = alpha_adj
            r['holm_reject'] = r['p'] < alpha_adj
        alpha_adj_label = "Bonferroni"

    results.sort(key=lambda x: x['p'])

    reported = [r for r in results
                if r['p'] < report_threshold]

    print(f"\n  {'=' * 80}")
    print(f"  INTERACTION TESTS ({n_tests} pairs, "
          f"{alpha_adj_label})")
    print(f"  {'=' * 80}")

    if not reported:
        print(f"  No pairs with p < {report_threshold}")
    else:
        print(f"  {'Pair':<24} {'∂²η':>10} {'SE':>8} "
              f"{'Z':>8} {'p':>12} {'Sig':>5}")
        print(f"  {'-' * 70}")
        for r in reported:
            if r['holm_reject'] and r['p'] < 0.001:
                sig = '***'
            elif r['holm_reject'] and r['p'] < 0.01:
                sig = '**'
            elif r['holm_reject']:
                sig = '*'
            else:
                sig = ''
            print(f"  {r['pair']:<24} "
                  f"{r['estimate']:>10.4f} "
                  f"{r['se']:>8.4f} {r['z']:>8.3f} "
                  f"{r['p']:>12.4e} {sig:>5}")

    return results


def interaction_by_reference(model, X_train, feature_names,
                             pairs=None, eps=1e-5, z=1.96):
    """∂²η at different reference points.

    Low (−1σ), Mean (x̄), High (+1σ)
    """
    d = X_train.shape[1]
    x_bar = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    ref_points = {
        'Low (−1σ)': x_bar - x_std,
        'Mean (x̄)':  x_bar.copy(),
        'High (+1σ)': x_bar + x_std,
    }

    if pairs is None:
        pairs = [(j, k) for j in range(d)
                 for k in range(j + 1, d)]

    for j, k in pairs:
        name_j = feature_names[j]
        name_k = feature_names[k]

        print(f"\n  {'=' * 75}")
        print(f"  INTERACTION: {name_j} × {name_k} "
              f"at Different References")
        print(f"  {'=' * 75}")
        print(f"  {'Reference':<14} {name_j:>10} {name_k:>10} "
              f"{'∂²η̂':>10} {'SE':>8} {'Z':>8} "
              f"{'p':>12} {'Sig':>5}")
        print(f"  {'-' * 78}")

        for ref_name, x_ref_val in ref_points.items():
            r = interaction_test(model, x_ref_val.reshape(1, -1),
                                 j, k, feature_names, eps, z)
            sig = _sig_label(r['p'])

            print(f"  {ref_name:<14} "
                  f"{float(x_ref_val[j]):>10.1f} "
                  f"{float(x_ref_val[k]):>10.1f} "
                  f"{r['estimate']:>10.3f} {r['se']:>8.3f} "
                  f"{r['z']:>8.2f} {r['p']:>12.3e} {sig:>5}")


# ═══════════════════════════════════════════════════════════════
# F. RISK GROUPS — β_equiv by Group
# ═══════════════════════════════════════════════════════════════

def beta_equiv_by_group(model, X_test, y_test, feature_names,
                        n_groups=3):
    """Risk-group β_equiv + Wald Z/p.

    Each group's own x̄_group is used as reference point
    for Jacobian projection.
    """
    probs = model.predict_proba(X_test)[:, 1]
    edges = np.quantile(
        probs, np.linspace(0, 1, n_groups + 1)[1:-1])
    group_idx = np.digitize(probs, edges)

    groups = sorted(np.unique(group_idx))
    labels = ['Low Risk', 'Mid Risk', 'High Risk']
    d = len(feature_names)
    rff = model.rff_
    beta_rff = model.coef_[:-1]

    for g, label in zip(groups, labels[:len(groups)]):
        mask = group_idx == g
        n_g = int(mask.sum())
        if n_g < 10:
            continue

        Xg = X_test[mask]
        pg = probs[mask]

        x_g_bar = Xg.mean(axis=0)
        theta = rff.omega @ x_g_bar + rff.b
        J_g = -rff.c * (np.sin(theta)[:, None] * rff.omega)

        beta_eq_g = J_g.T @ beta_rff
        eta_g = float(model.predict_eta(
            x_g_bar.reshape(1, -1)))
        bias_g = eta_g - float(x_g_bar @ beta_eq_g)

        if model.cov_ is not None:
            cov_g = (J_g.T @ model.cov_[:-1, :-1] @ J_g)
            se_g = np.sqrt(np.maximum(np.diag(cov_g), EPS))
        else:
            se_g = np.full(d, np.nan)

        se_g = np.maximum(se_g, EPS)
        z_g = beta_eq_g / se_g
        p_g = 2 * norm.sf(np.abs(z_g))
        hr_g = np.exp(beta_eq_g)

        print(f"\n  {'=' * 85}")
        print(f"  {label} (n={n_g}, p̄={pg.mean():.3f}, "
              f"α={bias_g:.4f})")
        print(f"  {'=' * 85}")
        print(f"  {'Var':<12} {'β_eq':>10} {'SE':>8} {'Z':>8} "
              f"{'p':>12} {'HR':>8} {'95% CI':>22} {'Sig':>5}")
        print(f"  {'-' * 85}")

        for j in range(d):
            sig = _sig_label(p_g[j])
            ci_lo = beta_eq_g[j] - 1.96 * se_g[j]
            ci_up = beta_eq_g[j] + 1.96 * se_g[j]
            print(f"  {feature_names[j]:<12} "
                  f"{beta_eq_g[j]:>10.4f} {se_g[j]:>8.4f} "
                  f"{z_g[j]:>8.3f} {p_g[j]:>12.4e} "
                  f"{hr_g[j]:>8.4f} "
                  f"[{ci_lo:>9.4f}, {ci_up:>8.4f}] {sig:>5}")


# ═══════════════════════════════════════════════════════════════
# G. LR vs FLARE COMPARISON
# ═══════════════════════════════════════════════════════════════

def lr_vs_flare_table(lr_res, beq_res, feature_names):
    """LR β and FLARE β_equiv side-by-side comparison."""
    print(f"\n  {'=' * 72}")
    print(f"  LR vs FLARE — Coefficient Comparison")
    print(f"  {'=' * 72}")
    print(f"\n  {'Var':<10} "
          f"{'β_LR':>8} {'HR_LR':>7} {'p_LR':>10} "
          f"{'β_eq':>8} {'HR_eq':>7} {'p_eq':>10}")
    print(f"  {'-' * 72}")
    d = len(feature_names)
    for j in range(d):
        def _ps(pval):
            return f"{pval:.2e}" if pval > 0.0001 else "<1e-4"
        print(f"  {feature_names[j]:<10} "
              f"{lr_res['beta'][j]:>8.4f} "
              f"{lr_res['hr'][j]:>7.3f} "
              f"{_ps(lr_res['p'][j]):>10} "
              f"{beq_res['beta'][j]:>8.4f} "
              f"{beq_res['hr'][j]:>7.3f} "
              f"{_ps(beq_res['p'][j]):>10}")


# ═══════════════════════════════════════════════════════════════
# H. RESIDUALS
# ═══════════════════════════════════════════════════════════════

def residuals_pearson(model, X, y):
    """Pearson residuals: r = (y - p) / √(p(1-p))."""
    probs = model.predict_proba(X)
    p = np.clip(probs[:, 1], EPS, 1.0 - EPS)
    return (y - p) / np.sqrt(p * (1.0 - p))


def residuals_deviance(model, X, y):
    """Deviance residuals."""
    probs = model.predict_proba(X)
    p = np.clip(probs[:, 1], EPS, 1.0 - EPS)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return np.sign(y - p) * np.sqrt(np.maximum(-2.0 * ll, 0.0))


def residuals_summary(model, X, y, threshold_pearson=2.0,
                      threshold_deviance=2.5):
    """Residual analysis with outlier detection.

    Returns
    -------
    dict: pearson, deviance, outlier indices, stats
    """
    r_p = residuals_pearson(model, X, y)
    r_d = residuals_deviance(model, X, y)

    out_p = np.where(np.abs(r_p) > threshold_pearson)[0]
    out_d = np.where(np.abs(r_d) > threshold_deviance)[0]

    print(f"\n  {'=' * 60}")
    print(f"  RESIDUAL ANALYSIS")
    print(f"  {'=' * 60}")
    print(f"  Pearson  — mean: {r_p.mean():>8.4f}, "
          f"std: {r_p.std():>8.4f}, "
          f"max|: {np.abs(r_p).max():>8.4f}")
    print(f"  Deviance — mean: {r_d.mean():>8.4f}, "
          f"std: {r_d.std():>8.4f}, "
          f"max|: {np.abs(r_d).max():>8.4f}")
    print(f"  Outliers (Pearson  |r|>{threshold_pearson}): "
          f"{len(out_p)} ({100 * len(out_p) / len(y):.1f}%)")
    print(f"  Outliers (Deviance |d|>{threshold_deviance}): "
          f"{len(out_d)} ({100 * len(out_d) / len(y):.1f}%)")

    return {
        'pearson': r_p, 'deviance': r_d,
        'outliers_pearson': out_p, 'outliers_deviance': out_d,
        'stats': {
            'pearson_mean': float(r_p.mean()),
            'pearson_std': float(r_p.std()),
            'deviance_mean': float(r_d.mean()),
            'deviance_std': float(r_d.std()),
        },
    }


# ═══════════════════════════════════════════════════════════════
# I. EVALUATION
# ═══════════════════════════════════════════════════════════════

def compute_metrics(y_true, probs, preds):
    """ROC-AUC, PR-AUC, F1, ACC, MCC, Brier."""
    return {
        'AUC': roc_auc_score(y_true, probs),
        'PR-AUC': average_precision_score(y_true, probs),
        'F1': f1_score(y_true, preds, zero_division=0),
        'ACC': accuracy_score(y_true, preds),
        'MCC': matthews_corrcoef(y_true, preds),
        'Brier': brier_score_loss(
            y_true, np.clip(probs, EPS, 1 - EPS)),
    }


def auc_metrics(model, X, y):
    """Model-based AUC metrics."""
    probs = model.predict_proba(X)
    preds = model.predict(X)
    if model.n_classes_ == 2:
        return compute_metrics(y, probs[:, 1], preds)
    return {
        'roc_auc': roc_auc_score(
            y, probs, multi_class='ovr', average='macro'),
        'pr_auc': average_precision_score(
            y, probs, average='macro'),
        'f1': f1_score(y, preds, average='macro'),
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(
            y, preds, average='macro', zero_division=0),
        'recall': recall_score(
            y, preds, average='macro', zero_division=0),
    }


def calibration_metrics(model, X, y, n_bins=10):
    probs = model.predict_proba(X)
    p1 = np.clip(probs[:, 1], EPS, 1.0 - EPS)
    brier = float(np.mean((y - p1) ** 2))

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins_edges = np.unique(np.quantile(p1, quantiles))
    n_actual_bins = len(bins_edges) - 1

    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_actual_bins):
        lo, hi = bins_edges[i], bins_edges[i + 1]
        if i == n_actual_bins - 1:
            mask = (p1 >= lo) & (p1 <= hi)
        else:
            mask = (p1 >= lo) & (p1 < hi)

        if mask.sum() == 0:
            continue

        mp = float(p1[mask].mean())
        mt = float(y[mask].mean())
        n_bin = int(mask.sum())
        gap = abs(mp - mt)
        ece += n_bin * gap
        mce = max(mce, gap)

        bin_data.append({
            'bin_lo': lo, 'bin_hi': hi,
            'n': n_bin, 'mean_pred': mp, 'mean_true': mt,
        })

    ece /= len(y)
    return {'brier': brier, 'ece': float(ece),
            'mce': float(mce), 'bins': bin_data}


def aic_bic(model, X, y):
    """AIC, BIC with effective degrees of freedom.

    Uses df_eff_ (if available) instead of raw param count.
    Regularized models have fewer effective parameters than
    nominal — df_eff_ = tr(H⁻¹ ΦᵀWΦ) accounts for shrinkage.

    Hodges & Sargent (2001), Wood (2017) GAM Ch.6.

    Returns
    -------
    dict: aic, bic, loglik, k, k_raw, n
    """
    if hasattr(model, 'loglik_') and model.loglik_ is not None:
        ll = model.loglik_
    else:
        probs = model.predict_proba(X)
        p = np.clip(probs[:, 1], EPS, 1.0 - EPS)
        ll = float(np.sum(y * np.log(p)
                          + (1 - y) * np.log(1 - p)))

    k_raw = len(model.coef_)
    k = getattr(model, 'df_eff_', float(k_raw))
    n = len(y)
    return {'aic': -2 * ll + 2 * k,
            'bic': -2 * ll + k * np.log(n),
            'loglik': ll, 'k': k, 'k_raw': k_raw, 'n': n}


def print_model_summary(model, X, y):
    """Full model summary: AIC/BIC, AUC, Calibration."""
    n = len(y)
    info = aic_bic(model, X, y)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    metrics = compute_metrics(y, probs, preds)
    cal = calibration_metrics(model, X, y)

    print(f"\n  {'=' * 60}")
    print(f"  FLARE MODEL SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Samples        = {n}")
    print(f"  Features (d)   = {model.n_features_}")
    print(f"  Classes        = {model.n_classes_}")
    print(f"  RFF features(m)= {model.m}")
    print(f"  σ (bandwidth)  = {model.sigma_:.6f}")
    print(f"  λ (L2)         = {model.l2}")
    print(f"  df_eff         = {info['k']:.1f} "
          f"(raw={info['k_raw']})")
    print(f"  Log-likelihood = {info['loglik']:.4f}")
    print(f"  AIC            = {info['aic']:.4f}")
    print(f"  BIC            = {info['bic']:.4f}")
    print(f"  {'─' * 55}")
    print(f"  ROC-AUC        = {metrics['AUC']:.4f}")
    print(f"  PR-AUC         = {metrics['PR-AUC']:.4f}")
    print(f"  F1             = {metrics['F1']:.4f}")
    print(f"  Accuracy       = {metrics['ACC']:.4f}")
    print(f"  MCC            = {metrics['MCC']:.4f}")
    print(f"  {'─' * 55}")
    print(f"  Brier Score    = {cal['brier']:.4f}")
    print(f"  ECE            = {cal['ece']:.4f}")
    print(f"  MCE            = {cal['mce']:.4f}")
    print(f"  {'=' * 60}")
    return {'loglik': info['loglik'], **metrics, **cal}


# ═══════════════════════════════════════════════════════════════
# J. UNCERTAINTY + PATIENT RISK PROFILE
# ═══════════════════════════════════════════════════════════════

def uncertainty_analysis(model, X_test, y_test):
    """SE discrimination: do misclassified instances
    have higher uncertainty?

    Returns
    -------
    dict: ratio, t_stat, t_p, u_stat, u_p
    """
    probs, se = model.predict_proba(X_test, return_se=True)
    se_p1 = se[:, 1]
    y_pred = model.predict(X_test)

    correct = y_pred == y_test
    se_c = se_p1[correct]
    se_i = se_p1[~correct]
    ratio = se_i.mean() / max(se_c.mean(), EPS)

    t_stat, t_p = stats.ttest_ind(se_i, se_c, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(
        se_i, se_c, alternative='greater')

    print(f"\n  {'=' * 60}")
    print(f"  UNCERTAINTY DISCRIMINATION")
    print(f"  {'=' * 60}")
    print(f"  Correct:   n={correct.sum():>4}, "
          f"SE={se_c.mean():.4f} ± {se_c.std():.4f}")
    print(f"  Incorrect: n={(~correct).sum():>4}, "
          f"SE={se_i.mean():.4f} ± {se_i.std():.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Welch t: t={t_stat:.3f}, p={t_p:.6f}")
    print(f"  Mann-Whitney: U={u_stat:.0f}, p={u_p:.6f}")

    return {'ratio': ratio, 't_stat': t_stat, 't_p': t_p,
            'u_stat': u_stat, 'u_p': u_p}


def patient_risk_profile(model, X, feature_names, idx, z=1.96):
    """Single-patient full risk profile."""
    x = X[idx:idx + 1]
    ci_result = model.predict_ci(x, z=z)

    p1 = float(ci_result['p_hat'][0])
    se1 = float(ci_result['se'][0])
    ci_lo = float(ci_result['ci_lo'][0])
    ci_hi = float(ci_result['ci_hi'][0])
    ci_width = float(ci_result['ci_width'][0])
    eta = float(ci_result['eta'][0])

    rff = model.rff_
    beta_rff = model.coef_[:-1]
    grad = _analytical_grad(rff, beta_rff, x)[0]

    J = _compute_jacobian_at_points(model, x)[0]
    Sigma_rff = model.cov_[:-1, :-1]
    var_f = np.einsum('ik,ij,jk->k', J, Sigma_rff, J)
    se_features = np.sqrt(np.maximum(var_f, EPS))

    order = np.argsort(-np.abs(grad))

    if ci_width < 0.20:
        conf_label = "High confidence"
    elif ci_width < 0.40:
        conf_label = "Moderate confidence"
    else:
        conf_label = "Low confidence"

    return {
        'idx': int(idx),
        'p_hat': p1,
        'se': se1,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'ci_width': ci_width,
        'confidence': conf_label,
        'prediction': int(p1 >= 0.5),
        'eta': eta,
        'gradient': np.asarray(grad).ravel(),
        'se_features': np.asarray(se_features).ravel(),
        'ranked_features': np.asarray(order).ravel(),
        'feature_names': feature_names,
    }


def print_patient_risk_report(profile):
    """Formatted report for patient_risk_profile output."""
    p = profile
    fn = p['feature_names']
    grad = p['gradient']
    se_f = p['se_features']
    order = p['ranked_features']
    d = len(fn)

    print(f"\n  {'=' * 75}")
    print(f"  PATIENT #{p['idx']} — Risk Profile")
    print(f"  {'=' * 75}")
    print(f"  p̂     = {p['p_hat']:.4f}")
    print(f"  SE    = {p['se']:.4f}")
    print(f"  95% CI = [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]  "
          f"(width={p['ci_width']:.4f})")
    print(f"  Pred  = {p['prediction']}   ({p['confidence']})")
    print(f"  η     = {p['eta']:.4f}")

    print(f"\n  {'Var':<12} {'∂η/∂x':>10} {'SE':>8} "
          f"{'Z':>8} {'p':>12} {'Sig':>5} {'Dir':>8}")
    print(f"  {'-' * 68}")
    for rank_j in range(d):
        j = int(order[rank_j])
        g_j = float(grad[j])
        s_j = float(se_f[j])
        z_sc = g_j / max(s_j, EPS)
        p_val = 2.0 * norm.sf(abs(z_sc))
        sig = _sig_label(p_val)

        direction = ("↑" if g_j > 0.01
                     else "↓" if g_j < -0.01
                     else "≈")
        print(f"  {fn[j]:<12} {g_j:>10.4f} {s_j:>8.4f} "
              f"{z_sc:>8.3f} {p_val:>12.2e} {sig:>5} "
              f"{direction:>8}")
def binary_posterior_coverage(model, X, y, z=1.96,
                               n_bins=20, method='equal_frequency'):
    probs = model.predict_proba(X)[:, 1]

    # Per-patient epistemic SE (Eq. 23 → Eq. 24)
    eta, se_eta = model.predict_eta(X, return_se=True)
    patient_se_p = probs * (1 - probs) * se_eta  # Delta method

    if method == 'equal_frequency':
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(probs, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    bins = []
    for i in range(len(edges) - 1):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == len(edges) - 2:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])

        if mask.sum() == 0:
            continue

        bin_probs = probs[mask]
        bin_y = y[mask]
        n_bin = int(mask.sum())

        obs = float(bin_y.mean())
        pred = float(bin_probs.mean())

        # Epistemic: RMS of per-patient SEs within bin
        se_model = float(np.sqrt(np.mean(patient_se_p[mask] ** 2)))

        # Aleatoric: Bernoulli sampling noise
        se_sampling = float(np.sqrt(pred * (1 - pred) / n_bin))

        # Total (Eq. 28)
        se_total = float(np.sqrt(se_model ** 2 + se_sampling ** 2))

        ci_lo = max(0.0, pred - z * se_total)
        ci_hi = min(1.0, pred + z * se_total)

        covered = (ci_lo <= obs) and (obs <= ci_hi)

        bins.append({
            'bin': f'[{edges[i]:.2f},{edges[i+1]:.2f})',
            'n': n_bin,
            'observed': obs,
            'predicted': pred,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'se_model': se_model,
            'se_sampling': se_sampling,
            'se_total': se_total,
            'covered': covered,
        })
    total_covered = sum(1 for b in bins if b['covered'])
    total_bins = len(bins)
    coverage = total_covered / total_bins if total_bins > 0 else 0

    return {
        'bins': bins,
        'coverage': coverage,
        'target': 0.95,
        'n_bins': total_bins,
        'total_n': len(y),
        'quality': ('✓ Well-calibrated' if coverage >= 0.80
                    else '⚠ Under-calibrated'),
        'method': method,
    }

def compute_patient_hessians_with_ci(model, X, feature_names,
                                      selected, top_n=10):
    """Patient-level Hessian with SE, Z, p-values.

    ∂²η/∂xᵢ∂xⱼ = βᵀ g_ij(x)  (linear in β)
    g_ij[k] = -c · cos(θₖ) · ωₖᵢ · ωₖⱼ,  k=1..m
    g_ij[m+1] = 0  (bias)

    Var(∂²η) = g_ijᵀ Σ_β g_ij   (sandwich HC2)

    Returns
    -------
    pair_results : dict  {idx: [pair_dicts]}
        Her hasta için üst-üçgende sıralanmış (pair, hess, se, z, p).
    full_matrices : dict {idx: ndarray (d, d)}
        Her hasta için tam simetrik Hessian matrisi.
        global_vs_patient_interactions / unique_patient_interactions
        tarafından kullanılır.
    """
    d = X.shape[1]
    m = model.m
    beta = model.coef_.ravel()
    Sigma = model.cov_
    W = model.rff_.omega
    c = model.rff_.c
    b = model.rff_.b
    beta_rff = beta[:-1]

    Sigma_rff = Sigma[:m, :m]

    pair_results = {}
    full_matrices = {}

    for idx in selected:
        xi = X[idx]

        theta = W @ xi + b
        cos_theta = np.cos(theta)
        g_base = -c * cos_theta

        H = np.zeros((d, d))
        pairs = []

        for i in range(d):
            for j in range(i + 1, d):        # ← DÜZELTME: i+1 (diyagonal hariç)
                g_rff = g_base * W[:, i] * W[:, j]

                hess_val = float(beta_rff @ g_rff)

                var_val = float(g_rff @ Sigma_rff @ g_rff)
                se_val = np.sqrt(max(var_val, 1e-30))

                z_val = (hess_val / se_val
                         if se_val > 1e-15 else 0.0)
                p_val = float(2 * norm.sf(abs(z_val)))

                H[i, j] = hess_val
                H[j, i] = hess_val

                pairs.append({
                    'pair': (f'{feature_names[i]}'
                             f' × {feature_names[j]}'),
                    'hess': hess_val,
                    'se': se_val,
                    'z': z_val,
                    'p': p_val,
                })

        pairs.sort(key=lambda x: -abs(x['hess']))
        pair_results[idx] = pairs[:top_n]
        full_matrices[idx] = H

    return pair_results, full_matrices





def print_patient_hessians_with_ci(results, probs):
    """Print patient-level Hessian with SE, Z, p."""
    for idx, pairs in results.items():
        print(f"\n  {'─' * 100}")
        print(f"  Patient #{idx} (p̂={probs[idx]:.4f})")
        print(f"  {'─' * 100}")
        print(f"  {'Rank':<5} {'Pair':<45} "
              f"{'∂²η':>10} {'SE':>9} "
              f"{'Z':>8} {'p':>12} {'Sig':>5}")
        print(f"  {'-' * 100}")

        for rank, p in enumerate(pairs, 1):
            sig = _sig_label(p['p'])
            print(f"  {rank:<5} {p['pair']:<45} "
                  f"{p['hess']:>10.6f} {p['se']:>9.6f} "
                  f"{p['z']:>8.3f} {p['p']:>12.2e} "
                  f"{sig:>5}")


def patient_posterior_plausibility(model, X, y, z=1.96):
    """Patient-level plausibility classification.

    CI in logit space → always [0,1].
    σ(η ± z × SE_η) (asymmetric, safe).

    4 categories:
      1. Confident Correct
      2. Uncertain Correct
      3. Uncertain Wrong
      4. Confident Wrong
    """
    eta, se_eta = model.predict_eta(X, return_se=True)
    lo = model._sigmoid(eta - z * se_eta)
    hi = model._sigmoid(eta + z * se_eta)

    probs = model.predict_proba(X)
    p1 = probs[:, 1]

    n = len(y)
    pred = (p1 >= 0.5).astype(int)
    correct = (pred == y)

    certain = (lo > 0.5) | (hi < 0.5)

    category = np.full(n, 2, dtype=int)
    category[correct & certain] = 1
    category[~correct & ~certain] = 3
    category[~correct & certain] = 4

    labels = {
        1: 'Confident Correct',
        2: 'Uncertain Correct',
        3: 'Uncertain Wrong',
        4: 'Confident Wrong',
    }
    counts = {}
    rates = {}
    for cat in [1, 2, 3, 4]:
        counts[cat] = int((category == cat).sum())
        rates[cat] = counts[cat] / n

    if certain.sum() > 0:
        confident_accuracy = float(correct[certain].mean())
    else:
        confident_accuracy = np.nan

    return {
        'category': category, 'labels': labels,
        'counts': counts, 'rates': rates,
        'n': n, 'z': z,
        'confident_accuracy': confident_accuracy,
        'n_confident': int(certain.sum()),
        'n_uncertain': int((~certain).sum()),
    }


# ═══════════════════════════════════════════════════════════════
# K. VALIDATION
# ═══════════════════════════════════════════════════════════════

def posterior_predictive_ci(model, X_test, B=1000, z=1.96,
                            seed=None):
    """Sampling-based CI validation.

    Draws B samples from N(β̂, Σ_β) and compares
    sampling CI with analytical CI.

    Returns
    -------
    dict: analytical (lo, hi), sampling (lo, hi),
          max_diff_lo, max_diff_hi
    """
    rng = np.random.default_rng(
        seed if seed is not None else model.seed)
    beta_samples = rng.multivariate_normal(
        model.coef_, model.cov_, size=B)
    Phi = model._design(X_test)
    eta_samples = Phi @ beta_samples.T
    p_samples = model._sigmoid(eta_samples)

    ci_lo_samp = np.percentile(p_samples, 2.5, axis=1)
    ci_hi_samp = np.percentile(p_samples, 97.5, axis=1)

    eta, se_eta = model.predict_eta(X_test, return_se=True)
    ci_lo_ana = model._sigmoid(eta - z * se_eta)
    ci_hi_ana = model._sigmoid(eta + z * se_eta)

    return {
        'analytical': (ci_lo_ana, ci_hi_ana),
        'sampling': (ci_lo_samp, ci_hi_samp),
        'max_diff_lo': float(
            np.max(np.abs(ci_lo_ana - ci_lo_samp))),
        'max_diff_hi': float(
            np.max(np.abs(ci_hi_ana - ci_hi_samp))),
    }


def m_sensitivity_analysis(X, y, m_values, l2=1e-3, seed=42):
    """RFF dimension sensitivity.

    Returns
    -------
    list of dicts: m, AUC, mean_SE, SE_ratio
    """
    results = []
    for m in m_values:
        model = FLARE(m=m, l2=l2, seed=seed)
        model.fit(X, y)
        probs, se = model.predict_proba(X, return_se=True)
        p1 = probs[:, 1]

        auc = roc_auc_score(y, p1)
        mean_se = float(se[:, 1].mean())

        preds = (p1 >= 0.5).astype(int)
        correct = preds == y
        if (~correct).sum() > 0 and correct.sum() > 0:
            ratio = (se[~correct, 1].mean()
                     / max(se[correct, 1].mean(), EPS))
        else:
            ratio = np.nan

        results.append({
            'm': m, 'AUC': auc,
            'mean_SE': mean_se, 'SE_ratio': ratio,
        })
    return results


def cv_evaluate_kfold(X, y, model_class, params, cv=5, seed=42):
    """K-fold CV: AUC, F1, Accuracy.

    Returns
    -------
    dict: mean/std per metric, folds
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, cv)

    all_metrics = []
    for i in range(cv):
        te = folds[i]
        tr = np.concatenate(
            [folds[j] for j in range(cv) if j != i])

        m = model_class(**params)
        m.fit(X[tr], y[tr], conformal=False)

        probs = m.predict_proba(X[te])
        preds = m.predict(X[te])

        fm = {
            'roc_auc': roc_auc_score(y[te], probs[:, 1]),
            'f1': f1_score(y[te], preds, zero_division=0),
            'accuracy': accuracy_score(y[te], preds),
        }
        all_metrics.append(fm)

    result = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        result[f'{key}_mean'] = float(np.mean(vals))
        result[f'{key}_std'] = float(np.std(vals))
    result['folds'] = all_metrics
    return result


# ═══════════════════════════════════════════════════════════════
# L. TRUE BETA (SYNTHETIC)
# ═══════════════════════════════════════════════════════════════

def true_beta_comparison(meta, model, lr, feature_names):
    """β_recovery on synthetic data.

    Compares FLARE β_eq and LR β against true β.
    Reports MSE, bias, and per-variable error.
    """
    if 'true_weights' not in meta:
        return

    w_true = np.array(meta['true_weights'])
    d = len(w_true)

    print(f"\n  {'=' * 80}")
    print(f"  TRUE β COMPARISON (Synthetic Data)")
    print(f"  {'=' * 80}")
    print(f"  {'Var':<10} {'β_true':>8} {'β_LR':>8} "
          f"{'β_eq':>8} {'Err_LR':>8} {'Err_eq':>8} "
          f"{'★':>5}")
    print(f"  {'-' * 60}")

    lr_coef = lr.coef_[0]
    flare_eq = model.beta_equiv_

    for j in range(d):
        err_lr = abs(lr_coef[j] - w_true[j])
        err_eq = abs(flare_eq[j] - w_true[j])
        better = "FLARE" if err_eq < err_lr else "LR"
        print(f"  {feature_names[j]:<10} {w_true[j]:>8.4f} "
              f"{lr_coef[j]:>8.4f} {flare_eq[j]:>8.4f} "
              f"{err_lr:>8.4f} {err_eq:>8.4f}  ★{better}")

    mse_lr = np.mean((lr_coef - w_true) ** 2)
    mse_eq = np.mean((flare_eq - w_true) ** 2)
    print(f"  {'-' * 60}")
    print(f"  {'MSE':<10} {'':>8} {mse_lr:>8.4f} "
          f"{mse_eq:>8.4f}")

    true_bias = meta.get('true_bias', 0)
    lr_bias = lr.intercept_[0]
    flare_bias = model.beta_equiv_bias_
    err_lr_bias = abs(lr_bias - true_bias)
    err_eq_bias = abs(flare_bias - true_bias)
    better_bias = ("FLARE" if err_eq_bias < err_lr_bias
                   else "LR")

    print(f"  {'-' * 60}")
    print(f"  {'Bias':<10} {true_bias:>8.4f} "
          f"{lr_bias:>8.4f} {flare_bias:>8.4f} "
          f"{err_lr_bias:>8.4f} {err_eq_bias:>8.4f}  "
          f"★{better_bias}")


# ═══════════════════════════════════════════════════════════════
# M. DATA PIPELINE — Multi-Objective Param Search
# ═══════════════════════════════════════════════════════════════
def find_best_params(X, y, m_cand=None, l2_cand=None,
                     sigma_cand=None, cv=3, seed=42):
    """Multi-objective grid search.

    Primary: Coverage + Nonlinearity + SE Ratio
    Secondary: Gap (β_equiv reliability)
    Tertiary: CC↑, CW↓

    Filters:
        - σ_ratio > MAX_SIGMA_RATIO → eliminated (model too linear)
        - CW > 0.15 → eliminated
        - Coverage < 0.50 → eliminated

    OBJ:
        OBJ = w_cov   × |coverage - 0.95|
            + w_nl    × (1 - nonlinearity)
            + w_ser   × (1 - se_ratio_norm)
            + w_gap   × gap
            + w_cc    × (1 - CC)
            + w_cw    × CW
        Lower OBJ = better.

    SE Ratio:
        mean(SE | incorrect) / mean(SE | correct).
        Higher = model knows when it's wrong.
        Normalised to [0, 1] via sigmoid-like clip.

    Nonlinearity:
        mean |Attr(NL)| / mean |Attr(L)| from RFF-Attribution.
        0 = perfectly linear (FLARE = LR), 1 = fully nonlinear.
    """
    n, d = X.shape

    # ── Sabitler ──
    MAX_SIGMA_RATIO = 10.0
    MIN_COVERAGE = 0.50
    MIN_SE_RATIO_SAMPLES = 5

    # Ağırlıklar (toplam = 1.0)
    W_COV  = 0.25
    W_NL   = 0.20
    W_SER  = 0.20
    W_GAP  = 0.15
    W_CC   = 0.12
    W_CW   = 0.08

    # ── m Adayları ──
    if m_cand is None:
        m_base = max(16, int(np.sqrt(n)))
        multipliers = [1.5, 2.0, 3.0]
        m_list = sorted(set(
            max(16, int(m_base * mult))
            for mult in multipliers))
    elif isinstance(m_cand, int):
        m_list = [m_cand]
    else:
        m_list = sorted(list(m_cand))

    # ── Lambda Adayları ──
    if l2_cand is None:
        l2_list = [1e-3, 1e-2, 1e-1]
    elif isinstance(l2_cand, float):
        l2_list = [l2_cand]
    else:
        l2_list = sorted(list(l2_cand))

    total_fits = len(m_list) * len(l2_list) * cv
    print(f"\n  {'─' * 70}")
    print(f"  Hyperparameter Search (Multi-Objective v3)")
    print(f"  Data: n={n}, d={d}")
    print(f"  m: {m_list}")
    print(f"  λ: {l2_list}")
    print(f"  Total fits: {total_fits}")
    print(f"  Filters: σ_ratio<{MAX_SIGMA_RATIO}, "
          f"coverage≥{MIN_COVERAGE}, CW≤0.15")
    print(f"  Weights: Cov={W_COV}, NL={W_NL}, "
          f"SEr={W_SER}, Gap={W_GAP}, "
          f"CC={W_CC}, CW={W_CW}")
    print(f"  {'─' * 70}")

    # ── Referans: LR β ──
    from sklearn.linear_model import LogisticRegression
    lr_ref = LogisticRegression(C=1.0, max_iter=2000,
                                solver='lbfgs')
    lr_ref.fit(X, y)
    lr_beta = lr_ref.coef_[0]

    # ── Veri ölçeği (σ_ratio için) ──
    data_scale = float(np.median(np.std(X, axis=0)))

    skf = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=seed)

    results_log = []

    for m in m_list:
        for l2 in l2_list:
            fold_aucs = []
            fold_gaps = []
            fold_cc = []
            fold_cw = []
            fold_pstd = []
            fold_coverages = []
            fold_sigma_ratios = []
            fold_nl_ratios = []
            fold_se_ratios = []

            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                try:
                    model = FLARE(m=m, l2=l2, seed=seed)
                    model.fit(X_tr, y_tr, sigma=sigma_cand)

                    probs_va = model.predict_proba(X_va)[:, 1]
                    auc = roc_auc_score(y_va, probs_va)
                    fold_aucs.append(auc)

                    # ── σ_ratio ──
                    sigma_ratio = (model.sigma_
                                   / max(data_scale, EPS))
                    fold_sigma_ratios.append(sigma_ratio)

                    # ── Gap (helpers'dan) ──
                    gap_result = compute_patient_gaps(
                        model, X_va,
                        x_ref=X_tr.mean(axis=0),
                        threshold=0.1)
                    fold_gaps.append(gap_result['median_gap'])

                    # ── Coverage (helpers'dan) ──
                    cov_result = binary_posterior_coverage(
                        model, X_va, y_va,
                        method='equal_frequency')
                    fold_coverages.append(cov_result['coverage'])

                    # ── Plausibility (CC, CW) ──
                    plaus = patient_posterior_plausibility(
                        model, X_va, y_va)
                    fold_cc.append(plaus['rates'][1])
                    fold_cw.append(plaus['rates'][4])

                    # ── SE Ratio ──
                    ci = model.predict_ci(X_va, z=1.96)
                    se_p = ci['se']
                    pred = (ci['p_hat'] >= 0.5).astype(int)
                    correct = (pred == y_va)

                    if (correct.sum() >= MIN_SE_RATIO_SAMPLES
                            and (~correct).sum()
                            >= MIN_SE_RATIO_SAMPLES):
                        se_ratio = float(
                            se_p[~correct].mean()
                            / max(se_p[correct].mean(), EPS))
                    else:
                        se_ratio = np.nan
                    fold_se_ratios.append(se_ratio)

                    # ── pStd ──
                    fold_pstd.append(float(np.std(probs_va)))

                    # ── Nonlinearity ──
                    bulk = rff_attribution_bulk(
                        model, X_va,
                        x_ref=X_tr.mean(axis=0).reshape(1, -1))
                    abs_L = np.abs(bulk['attribution']).mean()
                    abs_NL = np.abs(bulk['attribution_2']).mean()
                    nl_ratio = (abs_NL / max(abs_L, EPS)
                                if abs_L > EPS else 0.0)
                    fold_nl_ratios.append(nl_ratio)

                except Exception:
                    fold_aucs.append(0.5)
                    fold_gaps.append(100.0)
                    fold_cc.append(0.0)
                    fold_cw.append(1.0)
                    fold_pstd.append(0.0)
                    fold_coverages.append(0.0)
                    fold_sigma_ratios.append(999.0)
                    fold_nl_ratios.append(0.0)
                    fold_se_ratios.append(np.nan)

            # ── Fold ortalamaları ──
            mean_sigma_ratio = float(np.mean(fold_sigma_ratios))
            mean_coverage = float(np.mean(fold_coverages))
            mean_nl = float(np.mean(fold_nl_ratios))
            mean_gap = float(np.mean(fold_gaps))
            mean_auc = float(np.mean(fold_aucs))
            mean_cc = float(np.mean(fold_cc))
            mean_cw = float(np.mean(fold_cw))
            mean_pstd = float(np.mean(fold_pstd))

            # SE ratio: NaN-safe
            se_valid = [v for v in fold_se_ratios
                        if v is not None and not np.isnan(v)]
            mean_se_ratio = (float(np.mean(se_valid))
                             if se_valid else 1.0)

            results_log.append({
                'm': m,
                'l2': l2,
                'auc': mean_auc,
                'auc_std': float(np.std(fold_aucs)),
                'gap': mean_gap,
                'gap_std': float(np.std(fold_gaps)),
                'coverage': mean_coverage,
                'coverage_std': float(np.std(fold_coverages)),
                'cc': mean_cc,
                'cw': mean_cw,
                'pstd': mean_pstd,
                'sigma': mean_sigma_ratio,
                'nl_ratio': mean_nl,
                'nl_ratio_std': float(np.std(fold_nl_ratios)),
                'se_ratio': mean_se_ratio,
            })

    # ══════════════════════════════════════════════════════
    #  FİLTRELEME
    # ══════════════════════════════════════════════════════
    filtered = []
    rejected = {'sigma': 0, 'coverage': 0, 'cw': 0}

    for r in results_log:
        if r['sigma'] > MAX_SIGMA_RATIO:
            rejected['sigma'] += 1
            continue
        if r['coverage'] < MIN_COVERAGE:
            rejected['coverage'] += 1
            continue
        if r['cw'] > 0.15:
            rejected['cw'] += 1
            continue
        filtered.append(r)

    if not filtered:
        print(f"\n  ⚠ Tüm adaylar elendi. Filtreler gevşetiliyor.")
        print(f"    Elendi: σ_ratio>{MAX_SIGMA_RATIO}: "
              f"{rejected['sigma']}, "
              f"cov<{MIN_COVERAGE}: {rejected['coverage']}, "
              f"CW>0.15: {rejected['cw']}")
        filtered = results_log

    # ══════════════════════════════════════════════════════
    #  OBJEKTİF HESAPLAMA
    # ══════════════════════════════════════════════════════

    # SE ratio normalizasyonu: [1, ∞) → [0, 1]
    # SE_ratio=1.0 → 0.0 (ayırt edemiyor)
    # SE_ratio=2.0 → 0.67
    # SE_ratio=3.0 → 0.80
    # Asimptotik: se_norm = 1 - 1/se_ratio
    for r in filtered:
        sr = max(r['se_ratio'], 1.0)
        se_norm = 1.0 - 1.0 / sr

        r['se_ratio_norm'] = se_norm

        r['obj'] = (
            W_COV * abs(r['coverage'] - 0.95)
            + W_NL * (1.0 - min(r['nl_ratio'], 1.0))
            + W_SER * (1.0 - se_norm)
            + W_GAP * r['gap'] / 100.0
            + W_CC * (1.0 - r['cc'])
            + W_CW * r['cw']
        )

    filtered.sort(key=lambda x: x['obj'])
    best = filtered[0]

    # ══════════════════════════════════════════════════════
    #  YAZDIR
    # ══════════════════════════════════════════════════════
    print(f"\n  Filtrelenen: σ_ratio>{MAX_SIGMA_RATIO}: "
          f"{rejected['sigma']}, "
          f"cov<{MIN_COVERAGE}: {rejected['coverage']}, "
          f"CW>0.15: {rejected['cw']}")
    print(f"  Kalan: {len(filtered)}/{len(results_log)}")

    print(f"\n  {'Rank':<5} {'m':<5} {'λ':<8} "
          f"{'AUC':>6} {'Cov':>7} {'NL':>6} "
          f"{'SEr':>6} {'CC':>6} {'CW':>6} "
          f"{'Gap':>8} {'σ/r':>6} {'OBJ':>8}")
    print(f"  {'-' * 82}")

    for rank, r in enumerate(filtered, 1):
        star = " ★" if r == best else ""
        print(f"  {rank:<5} {r['m']:<5} {r['l2']:<8.0e} "
              f"{r['auc']:>6.4f} "
              f"{r['coverage']:>6.1%} "
              f"{r['nl_ratio']:>6.3f} "
              f"{r['se_ratio']:>6.2f} "
              f"{r['cc']:>5.1%} {r['cw']:>5.1%} "
              f"{r['gap']:>8.2f} "
              f"{r['sigma']:>6.1f} "
              f"{r['obj']:>8.4f}{star}")

    print(f"\n  ★ Best: m={best['m']}, λ={best['l2']:.0e}")
    print(f"    AUC={best['auc']:.4f}, "
          f"Coverage={best['coverage']:.1%}, "
          f"NL={best['nl_ratio']:.3f}")
    print(f"    SE_ratio={best['se_ratio']:.2f}, "
          f"CC={best['cc']:.1%}, "
          f"CW={best['cw']:.1%}")
    print(f"    Gap={best['gap']:.2f}, "
          f"σ_ratio={best['sigma']:.1f}")

    return {
        'm': best['m'],
        'l2': best['l2'],
        'cv_auc': best['auc'],
        'cv_gap': best['gap'],
        'cv_cc': best['cc'],
        'cv_cw': best['cw'],
        'cv_coverage': best['coverage'],
        'cv_nl_ratio': best['nl_ratio'],
        'cv_sigma_ratio': best['sigma'],
        'cv_se_ratio': best['se_ratio'],
        'search_log': results_log,
        'filtered_log': filtered,
        'rejected': rejected,
        'weights': {
            'cov': W_COV, 'nl': W_NL, 'ser': W_SER,
            'gap': W_GAP, 'cc': W_CC, 'cw': W_CW,
        },
        'filters': {
            'max_sigma_ratio': MAX_SIGMA_RATIO,
            'min_coverage': MIN_COVERAGE,
        },
    }


def cv_evaluation(X, y, m, l2, sigma, cv_outer, seed=42):
    """StratifiedKFold CV: AUC."""
    skf = StratifiedKFold(n_splits=cv_outer, shuffle=True,
                          random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        try:
            k = FLARE(m=m, l2=l2, seed=seed)
            k.fit(X[tr], y[tr], sigma=sigma, conformal=False)
            probs_te = k.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], probs_te))
        except Exception:
            aucs.append(0.5)
    return {'mean': np.mean(aucs), 'std': np.std(aucs),
            'folds': aucs}


# ═══════════════════════════════════════════════════════════════
# N. VISUALISATION DATA
# ═══════════════════════════════════════════════════════════════

def calibration_plot_data(model, X, y, n_bins=10):
    """Calibration curve data."""
    cal = calibration_metrics(model, X, y, n_bins)
    bins = cal['bins']
    return {
        'mean_predicted': [b['mean_pred'] for b in bins
                           if b['n'] > 0],
        'fraction_positive': [b['mean_true'] for b in bins
                              if b['n'] > 0],
        'counts': [b['n'] for b in bins if b['n'] > 0],
        'brier': cal['brier'],
        'ece': cal['ece'],
        'mce': cal['mce'],
    }


def residuals_plot_data(model, X, y):
    """Fitted vs residuals data."""
    probs = model.predict_proba(X)[:, 1]
    return {
        'fitted': probs,
        'pearson': residuals_pearson(model, X, y),
        'deviance': residuals_deviance(model, X, y),
    }


def roc_curve_data(model, X, y):
    """ROC curve data."""
    from sklearn.metrics import roc_curve, auc as calc_auc
    p1 = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, p1)
    return {'fpr': fpr, 'tpr': tpr,
            'thresholds': thresholds,
            'auc': float(calc_auc(fpr, tpr))}


def pr_curve_data(model, X, y):
    """Precision-Recall curve data."""
    from sklearn.metrics import (precision_recall_curve,
                                 average_precision_score)
    p1 = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, p1)
    return {'precision': precision, 'recall': recall,
            'thresholds': thresholds,
            'ap': float(average_precision_score(y, p1))}


# ═══════════════════════════════════════════════════════════════
# P. COMPETITOR MODELS — LR, RF, XGBoost, NGBoost
# ═══════════════════════════════════════════════════════════════

# ── P1. Bireysel Eğitim Fonksiyonları ────────────────────────

def fit_competitor_lr(X_train, y_train, C=1.0):
    """Logistic Regression (sklearn)."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C=C, max_iter=2000, solver='lbfgs')
    model.fit(X_train, y_train)
    return model


def fit_competitor_rf(X_train, y_train, n_estimators=500,
                      max_depth=None, seed=42):
    """Random Forest (sklearn)."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def fit_competitor_xgb(X_train, y_train, n_estimators=500,
                       seed=42):
    """XGBoost."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=6,
        learning_rate=0.1, eval_metric='logloss',
        random_state=seed, verbosity=0)
    model.fit(X_train, y_train)
    return model


def fit_competitor_ngb(X_train, y_train, n_estimators=500,
                       seed=42):
    """NGBoost — Natural Gradient Boosting (Bernoulli)."""
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    y_train_int = y_train.astype(int)
    model = NGBClassifier(
        Dist=Bernoulli, n_estimators=n_estimators,
        random_state=seed, verbose=False)
    model.fit(X_train, y_train_int)
    return model


# ── P2. Belirsizlik Yardımcıları ─────────────────────────────

def get_rf_uncertainty(rf, X):
    """RF belirsizliği: ağaç varyansı.

    Returns
    -------
    mean_p : (n,) ortalama olasılık
    std_p  : (n,) ağaç standart sapması
    """
    tree_preds = np.array([
        tree.predict_proba(X)[:, 1]
        for tree in rf.estimators_])
    return np.mean(tree_preds, axis=0), np.std(tree_preds, axis=0)


def get_ngb_uncertainty(ngb, X):
    """NGBoost belirsizliği: Bernoulli entropisi.

    Returns
    -------
    probs   : (n,) P(y=1)
    entropy : (n,) Shannon entropisi
    """
    probs = ngb.predict_proba(X)
    if probs.ndim == 2:
        probs = probs[:, 1]
    probs = np.asarray(probs, dtype=np.float64).ravel()

    p_clip = np.clip(probs, EPS, 1.0 - EPS)
    entropy = (-p_clip * np.log(p_clip)
               - (1.0 - p_clip) * np.log(1.0 - p_clip))
    return probs, entropy


# ── P3. Hepsi Bir Arada Eğitim ───────────────────────────────

def fit_all_competitors(X_train, y_train, X_test, y_test,
                        seed=42, verbose=True):
    """LR, RF, XGBoost, NGBoost — aynı split üzerinde eğit.

    Returns
    -------
    dict: {name: {model, probs, preds, metrics, [se|entropy]}}
    """
    results = {}

    # ── LR ──
    if verbose:
        print("  [1/4] LR...", end=" ")
    lr = fit_competitor_lr(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    preds = lr.predict(X_test)
    results['LR'] = {
        'model': lr, 'probs': probs, 'preds': preds,
        'metrics': compute_metrics(y_test, probs, preds)}
    if verbose:
        print(f"AUC={results['LR']['metrics']['AUC']:.4f}")

    # ── RF ──
    if verbose:
        print("  [2/4] RF (500 trees)...", end=" ")
    rf = fit_competitor_rf(X_train, y_train, seed=seed)
    rf_probs, rf_se = get_rf_uncertainty(rf, X_test)
    rf_preds = (rf_probs >= 0.5).astype(int)
    results['RF'] = {
        'model': rf, 'probs': rf_probs, 'se': rf_se,
        'preds': rf_preds,
        'metrics': compute_metrics(y_test, rf_probs, rf_preds)}
    if verbose:
        print(f"AUC={results['RF']['metrics']['AUC']:.4f}")

    # ── XGBoost ──
    if verbose:
        print("  [3/4] XGBoost...", end=" ")
    xgb = fit_competitor_xgb(X_train, y_train, seed=seed)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    results['XGBoost'] = {
        'model': xgb, 'probs': xgb_probs, 'preds': xgb_preds,
        'metrics': compute_metrics(
            y_test, xgb_probs, xgb_preds)}
    if verbose:
        print(f"AUC={results['XGBoost']['metrics']['AUC']:.4f}")

    # ── NGBoost ──
    if verbose:
        print("  [4/4] NGBoost...", end=" ")
    try:
        ngb = fit_competitor_ngb(X_train, y_train, seed=seed)
        ngb_probs, ngb_entropy = get_ngb_uncertainty(ngb, X_test)
        ngb_preds = (ngb_probs >= 0.5).astype(int)
        results['NGBoost'] = {
            'model': ngb, 'probs': ngb_probs,
            'entropy': ngb_entropy, 'preds': ngb_preds,
            'metrics': compute_metrics(
                y_test, ngb_probs, ngb_preds)}
        if verbose:
            print(
                f"AUC={results['NGBoost']['metrics']['AUC']:.4f}")
    except ImportError:
        if verbose:
            print("ATLANDI (pip install ngboost)")
    except Exception as e:
        if verbose:
            print(f"BAŞARISIZ: {e}")

    if verbose:
        print(f"\n  ✓ {len(results)}/4 rakip eğitildi")
    return results


# ── P4. Karşılaştırma Tabloları ──────────────────────────────

def benchmark_table(flare_model, competitors, X_test, y_test,
                    feature_names=None):
    """Tam benchmark: metrikler + DeLong testi (FLARE vs her rakip).

    Returns
    -------
    dict: {name: {metrics, probs, delong}}
    """
    flare_probs = flare_model.predict_proba(X_test)[:, 1]
    flare_preds = flare_model.predict(X_test)
    flare_m = compute_metrics(y_test, flare_probs, flare_preds)

    print(f"\n  {'=' * 100}")
    print(f"  BENCHMARK — FLARE vs Competitors")
    print(f"  {'=' * 100}")
    print(f"  n_test={len(y_test)}, "
          f"pos_rate={y_test.mean():.1%}, "
          f"m={flare_model.m}, σ={flare_model.sigma_:.4f}")
    print(f"\n  {'Model':<12} {'AUC':>8} {'PR-AUC':>8} "
          f"{'F1':>8} {'ACC':>8} {'MCC':>8} {'Brier':>8}  "
          f"{'ΔAUC':>8} {'p(DeLong)':>12} {'Sig':>5}")
    print(f"  {'-' * 100}")

    print(f"  {'▶ FLARE':<12} {flare_m['AUC']:>8.4f} "
          f"{flare_m['PR-AUC']:>8.4f} "
          f"{flare_m['F1']:>8.4f} "
          f"{flare_m['ACC']:>8.4f} "
          f"{flare_m['MCC']:>8.4f} "
          f"{flare_m['Brier']:>8.4f}  "
          f"{'—':>8} {'—':>12} {'':>5}")

    all_results = {
        'FLARE': {'metrics': flare_m, 'probs': flare_probs}}

    for name, res in competitors.items():
        m = res['metrics']
        dl = delong_test(y_test, flare_probs, res['probs'])

        sig = ("***" if dl['p'] < 0.001
               else "**" if dl['p'] < 0.01
               else "*" if dl['p'] < 0.05
               else "." if dl['p'] < 0.1 else "")

        print(f"  {name:<12} {m['AUC']:>8.4f} "
              f"{m['PR-AUC']:>8.4f} "
              f"{m['F1']:>8.4f} "
              f"{m['ACC']:>8.4f} "
              f"{m['MCC']:>8.4f} "
              f"{m['Brier']:>8.4f}  "
              f"{dl['diff']:>+8.4f} "
              f"{dl['p']:>12.4e} {sig:>5}")

        all_results[name] = {**res, 'delong': dl}

    print(f"  {'-' * 100}")
    aucs = {n: r['metrics']['AUC']
            for n, r in all_results.items()}
    best_model = max(aucs, key=aucs.get)
    print(f"  En iyi AUC : {best_model} ({aucs[best_model]:.4f})")

    flare_wins = sum(
        1 for n, r in all_results.items()
        if n != 'FLARE'
        and r.get('delong', {}).get('diff', 0) > 0)
    total = sum(1 for n in all_results if n != 'FLARE')
    print(f"  FLARE DeLong kazanma: {flare_wins}/{total}")
    print(f"  Pozitif ΔAUC = FLARE daha iyi. "
          f"*** p<0.001, ** p<0.01, * p<0.05")

    return all_results


def uncertainty_comparison(flare_model, X_test, y_test,
                           competitors=None):
    """Belirsizlik discrimination karşılaştırması.

    Returns
    -------
    dict: {name: {ratio, t_stat, t_p}}
    """
    flare_probs, flare_se = flare_model.predict_proba(
        X_test, return_se=True)
    se_p1 = flare_se[:, 1]
    flare_preds = (flare_probs[:, 1] >= 0.5).astype(int)
    flare_correct = (flare_preds == y_test)
    n_corr = int(flare_correct.sum())
    n_incorr = int((~flare_correct).sum())

    print(f"\n  {'=' * 80}")
    print(f"  BELİRSİZLİK DISCRIMINATION — Karşılaştırma")
    print(f"  {'=' * 80}")
    print(f"  n_test={len(y_test)}, "
          f"doğru={n_corr}, yanlış={n_incorr}")
    print(f"\n  {'Model':<12} {'Metrik':<8} "
          f"{'Doğru':>20} {'Yanlış':>20} "
          f"{'Ratio':>8} {'p':>10}")
    print(f"  {'-' * 80}")

    results = {}

    def _row(name, metric_name, values, correct_mask):
        vc = values[correct_mask]
        vi = values[~correct_mask]
        if len(vc) == 0 or len(vi) == 0:
            print(f"  {name:<12} {metric_name:<8} "
                  f"{'YETERSİZ VERİ':>50}")
            return
        ratio = vi.mean() / max(vc.mean(), EPS)
        _, t_p = stats.ttest_ind(vi, vc, equal_var=False)

        print(f"  {name:<12} {metric_name:<8} "
              f"{vc.mean():>8.4f}±{vc.std():.4f}   "
              f"{vi.mean():>8.4f}±{vi.std():.4f}   "
              f"{ratio:>8.2f} {t_p:>10.4e}")

        results[name] = {
            'ratio': float(ratio), 't_p': float(t_p),
            'se_correct_mean': float(vc.mean()),
            'se_incorrect_mean': float(vi.mean()),
        }

    _row('FLARE', 'SE', se_p1, flare_correct)

    if competitors and 'RF' in competitors:
        rf_se = competitors['RF']['se']
        rf_preds = competitors['RF']['preds']
        rf_correct = (rf_preds == y_test)
        _row('RF', 'σ_ağaç', rf_se, rf_correct)

    if competitors and 'NGBoost' in competitors:
        ent = competitors['NGBoost']['entropy']
        ngb_preds = competitors['NGBoost']['preds']
        ngb_correct = (ngb_preds == y_test)
        _row('NGBoost', 'Entropy', ent, ngb_correct)

    print(f"  {'-' * 80}")
    print(f"  Ratio > 1.0 → model yanlışken daha belirsiz")
    print(f"  RF:    500 ağacın P(y=1) tahimlerinin std'si")
    print(f"  NGBoost: Shannon entropy H(p) = -p·ln(p)-(1-p)·ln(1-p)")
    return results


def calibration_comparison(flare_model, competitors, X_test,
                           y_test, n_bins=10):
    """Kalibrasyon karşılaştırması: Brier, ECE, MCE.

    Returns
    -------
    dict: {name: {brier, ece, mce}}
    """
    flare_cal = calibration_metrics(
        flare_model, X_test, y_test, n_bins)

    print(f"\n  {'=' * 65}")
    print(f"  KALİBRASYON KARŞILAŞTIRMASI")
    print(f"  {'=' * 65}")
    print(f"  {'Model':<12} {'Brier':>10} {'ECE':>10} "
          f"{'MCE':>10} {'Durum':>15}")
    print(f"  {'-' * 65}")

    results = {}

    def _cal_row(name, cal_dict):
        status = ("✓ İyi" if cal_dict['ece'] < 0.05
                  else "~ Orta" if cal_dict['ece'] < 0.10
                  else "✗ Zayıf")
        print(f"  {name:<12} {cal_dict['brier']:>10.4f} "
              f"{cal_dict['ece']:>10.4f} "
              f"{cal_dict['mce']:>10.4f} {status:>15}")
        results[name] = cal_dict

    _cal_row('FLARE', flare_cal)

    for name, res in competitors.items():
        probs = res['probs']
        p_clip = np.clip(probs, EPS, 1.0 - EPS)
        brier = brier_score_loss(y_test, p_clip)

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p_clip, quantiles))
        n_bins_actual = len(edges) - 1
        ece, mce = 0.0, 0.0

        for i in range(n_bins_actual):
            lo, hi = edges[i], edges[i + 1]
            mask = ((p_clip >= lo) & (p_clip < hi)
                    if i < n_bins_actual - 1
                    else (p_clip >= lo) & (p_clip <= hi))
            if mask.sum() == 0:
                continue
            gap = abs(float(p_clip[mask].mean())
                      - float(y_test[mask].mean()))
            ece += int(mask.sum()) * gap
            mce = max(mce, gap)
        ece /= len(y_test)

        _cal_row(name,
                 {'brier': brier, 'ece': float(ece),
                  'mce': float(mce)})

    print(f"  {'-' * 65}")
    eces = {n: r['ece'] for n, r in results.items()}
    best_cal = min(eces, key=eces.get)
    print(f"  En iyi kalibre: {best_cal} "
          f"(ECE={eces[best_cal]:.4f})")
    return results


def feature_importance_comparison(flare_model, competitors,
                                  feature_names):
    """Öznitelik önem karşılaştırması.

    FLARE   → |β_equiv| (Cox-like yorumlama)
    RF      → Mean Decrease Impurity (MDI)
    XGBoost → Gain-based importance

    Returns
    -------
    dict: {name: importance_array}
    """
    d = len(feature_names)

    beq = np.abs(flare_model.beta_equiv_)
    beq_norm = beq / max(beq.sum(), EPS)
    results = {'FLARE': beq_norm}

    print(f"\n  {'=' * 85}")
    print(f"  ÖZNİTELİK ÖNEMİ KARŞILAŞTIRMASI")
    print(f"  {'=' * 85}")

    has_rf = 'RF' in competitors
    has_xgb = 'XGBoost' in competitors

    if has_rf:
        rf_imp = competitors['RF']['model'].feature_importances_
        rf_norm = rf_imp / max(rf_imp.sum(), EPS)
        results['RF'] = rf_norm

    if has_xgb:
        xgb_imp = (competitors['XGBoost']['model']
                   .feature_importances_)
        xgb_norm = xgb_imp / max(xgb_imp.sum(), EPS)
        results['XGBoost'] = xgb_norm

    header = f"  {'Sıra':<5} {'Öznitelik':<14} {'|β_eq|':>8}"
    if has_rf:
        header += f" {'RF_MDI':>8}"
    if has_xgb:
        header += f" {'XGB_gain':>10}"
    print(header)
    print(f"  {'-' * 85}")

    order = np.argsort(-beq_norm)
    for rank_j in range(d):
        j = int(order[rank_j])
        row = (f"  {rank_j+1:<5} "
               f"{feature_names[j]:<14} "
               f"{beq_norm[j]:>8.4f}")
        if has_rf:
            row += f" {results['RF'][j]:>8.4f}"
        if has_xgb:
            row += f" {results['XGBoost'][j]:>10.4f}"
        print(row)

    from scipy.stats import spearmanr
    print()
    if has_rf:
        rho, p = spearmanr(beq_norm, results['RF'])
        print(f"  Spearman(FLARE, RF):       "
              f"ρ={rho:.3f}, p={p:.4f}")
    if has_xgb:
        rho, p = spearmanr(beq_norm, results['XGBoost'])
        print(f"  Spearman(FLARE, XGBoost):  "
              f"ρ={rho:.3f}, p={p:.4f}")
    if has_rf and has_xgb:
        rho, p = spearmanr(results['RF'], results['XGBoost'])
        print(f"  Spearman(RF, XGBoost):     "
              f"ρ={rho:.3f}, p={p:.4f}")

    return results


# ── P5. Tam Pipeline ─────────────────────────────────────────

def run_full_benchmark(dataset_name, flare_params=None,
                       test_size=0.2, seed=42, verbose=True):
    """Tek fonksiyonla tam benchmark.

    1. Veri yükle
    2. Train/test split
    3. FLARE eğit
    4. 4 rakibi eğit
    5. Benchmark tablosu (AUC + DeLong)
    6. Belirsizlik karşılaştırması
    7. Kalibrasyon karşılaştırması
    8. Öznitelik önem karşılaştırması
    9. β_equiv tablosu
    10. Model özeti

    Parameters
    ----------
    dataset_name : str — datasets.py'deki ad
    flare_params : dict — FLARE hiperparametreleri (m, l2)
    test_size    : float — test oranı
    seed         : int — RNG seed
    verbose      : bool — çıktı

    Returns
    -------
    dict: flare, competitors, benchmark, uncertainty,
          calibration, importance, X_test, y_test,
          feature_names, meta
    """
    from sklearn.model_selection import train_test_split
    # NOTE: This imports from a local datasets.py module,
    #       NOT the HuggingFace 'datasets' package.
    from datasets import load_dataset

    X, y, meta = load_dataset(
        dataset_name, normalize=True, seed=seed)
    feature_names = meta.get(
        'feature_names',
        [f'x{i+1}' for i in range(X.shape[1])])

    if verbose:
        print(f"\n  {'═' * 70}")
        print(f"  FULL BENCHMARK — {meta['name']}")
        print(f"  {'═' * 70}")
        print(f"  n={meta['n_final']}, d={meta['d_final']}, "
              f"pos={meta['pos_rate']:.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=seed)

    if verbose:
        print(f"  Train: {len(y_tr)}, Test: {len(y_te)}")

    params = flare_params or {}
    flare = FLARE(seed=seed, **params)
    flare.fit(X_tr, y_tr)

    if verbose:
        flare_p = flare.predict_proba(X_te)[:, 1]
        flare_m = compute_metrics(
            y_te, flare_p, flare.predict(X_te))
        print(f"  FLARE: AUC={flare_m['AUC']:.4f}, "
              f"σ={flare.sigma_:.4f}, m={flare.m}")
        print(f"  {'─' * 70}")

    competitors = fit_all_competitors(
        X_tr, y_tr, X_te, y_te,
        seed=seed, verbose=verbose)

    bench = benchmark_table(
        flare, competitors, X_te, y_te, feature_names)
    unc = uncertainty_comparison(
        flare, X_te, y_te, competitors)
    cal = calibration_comparison(
        flare, competitors, X_te, y_te)
    imp = feature_importance_comparison(
        flare, competitors, feature_names)

    beta_equiv_summary(flare, feature_names)
    print_model_summary(flare, X_te, y_te)

    return {
        'flare': flare,
        'competitors': competitors,
        'benchmark': bench,
        'uncertainty': unc,
        'calibration': cal,
        'importance': imp,
        'X_train': X_tr, 'y_train': y_tr,
        'X_test': X_te, 'y_test': y_te,
        'feature_names': feature_names,
        'meta': meta,
    }


# ═══════════════════════════════════════════════════════════════
# Q. FEATURE IMPORTANCE — FULL COMPARISON TABLES
# ═══════════════════════════════════════════════════════════════

def raw_coefficients_comparison(flare_model, lr_model,
                                feature_names):
    """Tablo A: β_eq vs β_LR raw coefficients.

    Prints |Δ| and Δ% per feature.

    Returns
    -------
    dict: feature, beta_eq, beta_lr, delta, delta_pct
    """
    d = len(feature_names)
    beta_eq = flare_model.beta_equiv_
    beta_lr = lr_model.coef_[0]

    delta = np.abs(beta_eq - beta_lr)
    base = np.maximum(np.abs(beta_lr), EPS)
    delta_pct = (delta / base) * 100

    print(f"\n  {'=' * 65}")
    print(f"  RAW COEFFICIENTS (FLARE β_eq vs LR β)")
    print(f"  {'=' * 65}")
    print(f"  {'Var':<22} {'β_eq':>10} {'β_LR':>10} "
          f"{'|Δ|':>10} {'Δ%':>8}")
    print(f"  {'-' * 62}")

    for j in range(d):
        print(f"  {feature_names[j]:<22} "
              f"{beta_eq[j]:>10.4f} {beta_lr[j]:>10.4f} "
              f"{delta[j]:>10.4f} {delta_pct[j]:>7.1f}%")

    return {
        'feature': feature_names,
        'beta_eq': beta_eq, 'beta_lr': beta_lr,
        'delta': delta, 'delta_pct': delta_pct,
    }


def feature_importance_ranking(flare_model, lr_model=None,
                                rf_model=None, xgb_model=None,
                                feature_names=None):
    """Tablo B: Normalized importance ranking + Spearman.

    FLARE → |β_eq|, LR → |β|, RF → MDI, XGB → gain.
    All normalized to proportion (sum=1).

    Returns
    -------
    dict: importances {name: array}, order, spearman
    """
    from scipy.stats import spearmanr

    d = len(feature_names)
    models = {}

    beq_abs = np.abs(flare_model.beta_equiv_)
    models['FLARE'] = beq_abs / max(beq_abs.sum(), EPS)

    if lr_model is not None:
        lr_abs = np.abs(lr_model.coef_[0])
        models['LR'] = lr_abs / max(lr_abs.sum(), EPS)

    if rf_model is not None:
        rf_imp = rf_model.feature_importances_
        models['RF'] = rf_imp / max(rf_imp.sum(), EPS)

    if xgb_model is not None:
        xgb_imp = xgb_model.feature_importances_
        models['XGBoost'] = xgb_imp / max(xgb_imp.sum(), EPS)

    names = list(models.keys())

    # ── Print table ──
    print(f"\n  {'=' * 85}")
    print(f"  NORMALIZED IMPORTANCE RANKING")
    print(f"  (Proportion of total sum — not raw coefficients)")
    print(f"  {'=' * 85}")

    header = f"  {'Rank':<5} {'Feature':<22}"
    for name in names:
        header += f" {name + '(%)':>10}"
    print(header)
    print(f"  {'-' * (27 + 10 * len(names))}")

    order = np.argsort(-models['FLARE'])
    for rank_j in range(d):
        j = int(order[rank_j])
        row = f"  {rank_j + 1:<5} {feature_names[j]:<22}"
        for name in names:
            row += f" {models[name][j] * 100:>9.2f}%"
        print(row)

    # ── Spearman correlations ──
    spearman_results = {}
    print(f"\n  Spearman Rank Korelasyonları:")
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            rho, p = spearmanr(models[n1], models[n2])
            spearman_results[(n1, n2)] = (rho, p)
            label = f"Spearman({n1}, {n2})"
            print(f"    {label:<35} ρ={rho:.3f}, p={p:.4f}")

    return {
        'importances': models, 'order': order,
        'spearman': spearman_results,
    }


# ═══════════════════════════════════════════════════════════════
# R. INTERACTION ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_interaction_analysis(model, X_train, feature_names,
                              report_threshold=0.2,
                              top_n_ref=3):
    """Full interaction pipeline: global Wald + by-reference.

    1. ∂²η Wald tests for all pairs at x̄
    2. Holm-Bonferroni correction
    3. For top significant pairs: by-reference analysis
       (−1σ, x̄, +1σ)

    Returns
    -------
    list of dicts: all interaction test results
    """
    x_bar = X_train.mean(axis=0)

    interactions = interaction_tests(
        model, x_bar, feature_names,
        holm=True, report_threshold=report_threshold)

    if interactions:
        reported = [r for r in interactions
                    if r['p'] < report_threshold]
        if reported:
            top_pairs = [(r['j'], r['k'])
                         for r in reported[:top_n_ref]]
        else:
            top_pairs = [(0, 1)]

        for j, k in top_pairs:
            interaction_by_reference(
                model, X_train, feature_names,
                pairs=[(j, k)])

    return interactions


# ═══════════════════════════════════════════════════════════════
# S. PATIENT-LEVEL HESSIAN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def select_extreme_patients(probs, n=3):
    """Select patients at low, median, high risk.

    Parameters
    ----------
    probs : array — predicted P(y=1)
    n     : int — up to 3

    Returns
    -------
    list of int: deduplicated indices, sorted by prob
    """
    idx_low = int(np.argmin(probs))
    idx_high = int(np.argmax(probs))
    median_p = np.median(probs)
    idx_mid = int(np.argmin(np.abs(probs - median_p)))

    candidates = [idx_low, idx_mid, idx_high][:n]
    seen = dict.fromkeys(candidates)

    sorted_indices = sorted(seen.keys(), key=lambda i: probs[i])

    return sorted_indices


def print_patient_selection(selected, probs):
    """Print selected patient summary."""
    risk_labels = ['Low Risk', 'Mid Risk', 'High Risk']
    print(f"\n  Selected patients — risk profiles:")
    for i, idx in enumerate(selected):
        label = risk_labels[i] if i < len(risk_labels) else f"#{i}"
        print(f"    {label:<10} (idx={idx}): p̂={probs[idx]:.4f}")


def compute_patient_hessians(model, X_test, feature_names,
                              idx_list, top_n=10):
    """Per-patient Hessian ∂²η pairs.

    Parameters
    ----------
    model         : FLARE (must have _compute_hessian_patient)
    X_test        : array
    feature_names : list
    idx_list      : list of int
    top_n         : int — top pairs to print per patient

    Returns
    -------
    dict: {idx: ndarray (d, d)}
    """
    d = len(feature_names)
    all_probs = model.predict_proba(X_test)[:, 1]
    hessians = {}

    print(f"\n  {'═' * 75}")
    print(f"  PATIENT-LEVEL HESSIAN INTERACTIONS (∂²η)")
    print(f"  {'═' * 75}")

    for idx in idx_list:
        x_k = X_test[idx]
        p_k = float(all_probs[idx])
        H_k = model._compute_hessian_patient(x_k)
        hessians[idx] = H_k

        # Extract + sort upper triangle
        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs.append({
                    'i': i, 'j': j,
                    'name': f"{feature_names[i]} × "
                            f"{feature_names[j]}",
                    'val': float(H_k[i, j]),
                })
        pairs.sort(key=lambda x: abs(x['val']), reverse=True)

        print(f"\n  ── Patient #{idx} (p̂={p_k:.4f}) ──")
        print(f"  {'Rank':<5} {'Pair':<42} "
              f"{'∂²η':>12} {'Dir':>5}")
        print(f"  {'-' * 66}")

        for rank, p in enumerate(pairs[:top_n], 1):
            v = p['val']
            d_str = ("  +↑" if v > 0
                     else "  -↓" if v < 0
                     else "   ≈")
            print(f"  {rank:<5} {p['name']:<42} "
                  f"{v:>12.6f} {d_str}")

    return hessians


def global_vs_patient_interactions(model, X_train,
                                    feature_names, idx_list,
                                    hessians, top_n=5):
    """Global Hessian (x̄) vs per-patient comparison.

    Parameters
    ----------
    model, X_train, feature_names : standard
    idx_list  : list of int
    hessians  : dict {idx: ndarray} from compute_patient_hessians
    top_n     : int

    Returns
    -------
    tuple: (results_list, H_global)
    """
    d = len(feature_names)
    x_bar = X_train.mean(axis=0)
    H_global = model._compute_hessian_patient(x_bar)

    # Top global pairs by |∂²η|
    global_pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            global_pairs.append(
                (i, j, abs(float(H_global[i, j]))))
    global_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = global_pairs[:top_n]

    print(f"\n  {'═' * 75}")
    print(f"  GLOBAL vs PATIENT-LEVEL INTERACTION COMPARISON")
    print(f"  {'═' * 75}")

    header = f"  {'Pair':<42} {'Global':>12}"
    for idx in idx_list:
        header += f" {'#' + str(idx):>12}"
    header += f" {'ΔMax':>12}"
    print(f"\n{header}")
    sep = 42 + 12 + 12 * len(idx_list) + 12
    print(f"  {'-' * sep}")

    results = []
    for i, j, _ in top_pairs:
        g_val = float(H_global[i, j])
        pair_name = f"{feature_names[i]} × {feature_names[j]}"
        line = f"  {pair_name:<42} {g_val:>12.6f}"

        vals = []
        for idx in idx_list:
            h_val = float(hessians[idx][i, j])
            line += f" {h_val:>12.6f}"
            vals.append(h_val)

        delta = max(vals) - min(vals)
        line += f" {delta:>12.6f}"
        print(line)

        results.append({
            'i': i, 'j': j, 'pair': pair_name,
            'global': g_val,
            'patients': {idx: float(hessians[idx][i, j])
                         for idx in idx_list},
            'delta': delta,
        })

    return results, H_global


def unique_patient_interactions(model, X_train, X_test,
                                 feature_names, idx_list,
                                 hessians, H_global=None,
                                 threshold=2.0, top_n=5):
    """Interactions weak globally but strong locally.

    Finds pairs where |local| / |global| > threshold.

    Parameters
    ----------
    model, X_train, X_test, feature_names : standard
    idx_list   : list of int
    hessians   : dict {idx: ndarray}
    H_global   : ndarray or None (computed if None)
    threshold  : float (default 2.0)
    top_n      : int

    Returns
    -------
    dict: {idx: list of unique interaction dicts}
    """
    d = len(feature_names)
    all_probs = model.predict_proba(X_test)[:, 1]

    if H_global is None:
        x_bar = X_train.mean(axis=0)
        H_global = model._compute_hessian_patient(x_bar)

    print(f"\n  {'═' * 75}")
    print(f"  UNIQUE PATIENT INTERACTIONS")
    print(f"  (globally weak, locally >{threshold}x stronger)")
    print(f"  {'═' * 75}")

    all_results = {}

    for idx in idx_list:
        p_k = float(all_probs[idx])
        H_k = hessians[idx]

        unique = []
        for i in range(d):
            for j in range(i + 1, d):
                local_val = abs(float(H_k[i, j]))
                global_val = abs(float(H_global[i, j]))

                ratio = (local_val / global_val
                         if global_val > EPS
                         else (local_val / EPS
                               if local_val > EPS else 0))

                if ratio > threshold:
                    unique.append({
                        'pair': f"{feature_names[i]} × "
                                f"{feature_names[j]}",
                        'local': float(H_k[i, j]),
                        'global': float(H_global[i, j]),
                        'ratio': ratio,
                    })

        unique.sort(key=lambda x: x['ratio'], reverse=True)

        print(f"\n  ── Patient #{idx} (p̂={p_k:.4f}) ──")
        if unique:
            print(f"  {'Pair':<42} {'Patient':>12} "
                  f"{'Global':>12} {'Ratio':>8}")
            print(f"  {'-' * 76}")
            for u in unique[:top_n]:
                print(f"  {u['pair']:<42} "
                      f"{u['local']:>12.6f} "
                      f"{u['global']:>12.6f} "
                      f"{u['ratio']:>8.1f}x")
        else:
            print(f"  (no interactions >{threshold}x "
                  f"stronger locally)")

        all_results[idx] = unique

    return all_results
def analytical_grad(model, X):
    """∂η/∂x at X. Public wrapper for _analytical_grad.

    Parameters
    ----------
    model : FLARE
    X     : (n, d)

    Returns
    -------
    grad : (n, d)
    """
    return _analytical_grad(model.rff_, model.coef_[:-1], X)
def compute_patient_gaps(model, X_test, x_ref=None,
                          threshold=0.1):
    """Simpson gap: |Δη - Simpson_integral| / |Δη|.

    Aligned with paper Eq. 46, 48–50.

    Simpson's rule (Eq. 46):
      Δη_j ≈ (x_j - x̄_j) · (1/6)[∂η/∂x_j|_x̄
                                   + 4·∂η/∂x_j|_x_mid
                                   + ∂η/∂x_j|_x]

    Gap metric (Eq. 50):
      Gap = |Σ_j(L_j + NL_j) - Δη| / |Δη| × 100

    Parameters
    ----------
    model     : FLARE
    X_test    : (n, d)
    x_ref     : (1, d) or None → X_test.mean
    threshold : float, min |Δη| to include (default 0.1)

    Returns
    -------
    dict:
        gap           : (n,)  absolute gap
        gap_pct       : (n,)  percentage gap (NaN if below threshold)
        delta_eta     : (n,)  η(x) - η(x_ref)
        median_gap    : float median gap percentage
        n_significant : int   samples above threshold
    """
    if x_ref is None:
        x_ref = X_test.mean(axis=0).reshape(1, -1)
    x_ref = np.atleast_2d(x_ref)

    rff = model.rff_
    beta_rff = model.coef_[:-1]

    # Üç noktada gradyan (Eq. 46)
    grad_ref  = _analytical_grad(rff, beta_rff, x_ref)    # ∂η/∂x|_x̄
    grad_test = _analytical_grad(rff, beta_rff, X_test)   # ∂η/∂x|_x
    x_mid     = 0.5 * (X_test + x_ref)
    grad_mid  = _analytical_grad(rff, beta_rff, x_mid)    # ∂η/∂x|_x_mid

    dx = X_test - x_ref

    # Simpson toplamı (Eq. 46): L + NL per feature
    simpson_sum = (
        dx * (1.0 / 6.0)
        * (grad_ref + 4.0 * grad_mid + grad_test)
    ).sum(axis=1)

    # Gerçek Δη
    eta_test = model.predict_eta(X_test)
    eta_ref  = model.predict_eta(x_ref)
    delta_eta = eta_test - eta_ref

    # Gap (Eq. 50)
    gap = np.abs(simpson_sum - delta_eta)

    significant = np.abs(delta_eta) > threshold
    gap_pct = np.full(len(X_test), np.nan)
    if significant.any():
        gap_pct[significant] = (
            gap[significant]
            / np.abs(delta_eta[significant]) * 100)

    median_gap = (float(np.nanmedian(gap_pct))
                  if significant.any() else 0.0)

    return {
        'gap': gap,
        'gap_pct': gap_pct,
        'delta_eta': delta_eta,
        'median_gap': median_gap,
        'n_significant': int(significant.sum()),
    }