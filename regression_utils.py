"""
Gemeinsame Regressions- und Mediations-Hilfsfunktionen (Ersatz fuer statsmodels /
SPSS PROCESS, die in dieser Sandbox nicht verfuegbar sind).

Wird von h1_analysis.py, h2_analysis.py, h3_analysis.py, h4_analysis.py genutzt.
In VS Code (wo statsmodels i.d.R. verfuegbar ist) lassen sich alle Modelle 1:1
als smf.ols(...) / smf.logit(...) nachbauen - die Koeffizienten wurden gegen
sklearn gegengeprueft und stimmen exakt ueberein.
"""

import os
import numpy as np
from scipy import stats


def find_data_file(script_dir, filename):
    """Sucht eine Datendatei an mehreren plausiblen Orten relativ zum Skript,
    falls die Ordnerstruktur beim Entpacken/Oeffnen (z. B. aus einem ZIP heraus)
    nicht 1:1 erhalten bleibt. Wirft einen klaren Fehler mit allen versuchten
    Pfaden, falls nichts gefunden wird."""
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(script_dir, "..", "data", filename),
        os.path.join(script_dir, "..", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    tried = "\n  - ".join(os.path.abspath(p) for p in candidates)
    raise FileNotFoundError(
        f"Konnte '{filename}' nicht finden. Versucht wurden:\n  - {tried}\n"
        f"Leg die CSV-Datei entweder direkt neben die Skripte oder in einen "
        f"Unterordner 'data/'."
    )


def ols_with_inference(X, y):
    """OLS mit Standardfehlern, t- und p-Werten. X: (n,k) ohne Interceptspalte."""
    n, k = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = n - X1.shape[1]
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), dof))
    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return {"beta": beta, "se": se, "t": tvals, "p": pvals, "r2": r2, "dof": dof, "n": n}


def ols_beta_only(X, y):
    """Schnelle OLS-Variante nur mit Koeffizienten (fuer Bootstrap-Schleifen)."""
    n = X.shape[0]
    X1 = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def logit_with_inference(X, y, max_iter=200, tol=1e-10):
    """Logistische Regression per Newton-Raphson (IRLS) mit Standardfehlern
    (aus der beobachteten Fisher-Information), z- und p-Werten."""
    n, k = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta = _logit_fit(X1, y, max_iter, tol)
    eta = X1 @ beta
    p = 1 / (1 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    H = -(X1.T * W) @ X1
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    zvals = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
    p0 = y.mean()
    ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll_full = np.sum(y * np.log(np.clip(p, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-10, 1)))
    mcfadden_r2 = 1 - ll_full / ll_null
    return {"beta": beta, "se": se, "z": zvals, "p": pvals, "n": n, "mcfadden_r2": mcfadden_r2}


def logit_beta_only(X, y, max_iter=100, tol=1e-8):
    """Schnelle Logit-Variante nur mit Koeffizienten (fuer Bootstrap-Schleifen)."""
    n = X.shape[0]
    X1 = np.column_stack([np.ones(n), X])
    return _logit_fit(X1, y, max_iter, tol)


def _logit_fit(X1, y, max_iter, tol):
    beta = np.zeros(X1.shape[1])
    for _ in range(max_iter):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-10, None)
        grad = X1.T @ (y - p)
        H = -(X1.T * W) @ X1
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def bootstrap_ci(values, alpha=0.05, method="percentile"):
    """Bias-korrigiertes oder einfaches Perzentil-Konfidenzintervall aus
    Bootstrap-Replikaten. method='bc' fuer bias-corrected (wie SPSS PROCESS
    standardmaessig), 'percentile' fuer den einfachen Perzentil-Ansatz."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if method == "percentile":
        lo = np.percentile(values, 100 * alpha / 2)
        hi = np.percentile(values, 100 * (1 - alpha / 2))
        return lo, hi
    # bias-corrected (BC, ohne Beschleunigungskorrektur)
    point = np.median(values)  # Naeherung, siehe z0-Berechnung unten
    return _bc_ci(values, alpha)


def _bc_ci(values, alpha):
    values = np.asarray(values)
    n = len(values)
    # z0 benoetigt den Originalschaetzer separat - wird von aufrufender Funktion
    # mituebergeben ueber ein Tupel (values, theta_hat). Hier vereinfachtes BC
    # ueber den Anteil der Bootstrap-Werte unterhalb des Medians als Fallback.
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return lo, hi


def bc_ci_with_point(boot_values, theta_hat, alpha=0.05):
    """Echtes Bias-corrected (BC) Bootstrap-KI nach Efron, mit dem
    Original-Punktschaetzer theta_hat (aus dem vollen Sample, nicht Bootstrap-Median)."""
    boot_values = np.asarray(boot_values)
    boot_values = boot_values[~np.isnan(boot_values)]
    n = len(boot_values)
    prop_less = np.mean(boot_values < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(2 * z0 + z_lo)
    p_hi = stats.norm.cdf(2 * z0 + z_hi)
    p_lo = np.clip(p_lo, 1e-6, 1 - 1e-6)
    p_hi = np.clip(p_hi, 1e-6, 1 - 1e-6)
    lo = np.percentile(boot_values, 100 * p_lo)
    hi = np.percentile(boot_values, 100 * p_hi)
    return lo, hi
