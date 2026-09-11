import csv
import numpy as np
import os
from regression_utils import find_data_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from scipy import stats

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")
BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")


def to_float(s):
    return float(s.strip().replace(",", "."))


# ---------------------------------------------------------------------------
# Manuelle Regressionsfunktionen 
# ---------------------------------------------------------------------------

def ols_with_inference(X, y):
    """Gewoehnliche kleinste Quadrate mit Standardfehlern, t- und p-Werten.
    X: (n, k) Designmatrix OHNE Interceptspalte (wird automatisch ergaenzt).
    """
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


def logit_with_inference(X, y, max_iter=200, tol=1e-10):
    """Logistische Regression per Newton-Raphson (IRLS) mit Standardfehlern
    (aus der beobachteten Fisher-Information), z- und p-Werten.
    X: (n, k) Designmatrix OHNE Interceptspalte.
    """
    n, k = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta = np.zeros(X1.shape[1])
    for _ in range(max_iter):
        eta = X1 @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        W = np.clip(W, 1e-10, None)
        grad = X1.T @ (y - p)
        H = -(X1.T * W) @ X1
        step = np.linalg.solve(H, grad)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X1 @ beta
    p = 1 / (1 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-10, None)
    H = -(X1.T * W) @ X1
    cov = np.linalg.inv(-H)
    se = np.sqrt(np.diag(cov))
    zvals = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
    # Log-Likelihood fuer Nullmodell (nur Intercept) zum Vergleich (McFadden R2)
    p0 = y.mean()
    ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll_full = np.sum(y * np.log(np.clip(p, 1e-10, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-10, 1)))
    mcfadden_r2 = 1 - ll_full / ll_null
    return {"beta": beta, "se": se, "z": zvals, "p": pvals, "n": n, "mcfadden_r2": mcfadden_r2}


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


# ---------------------------------------------------------------------------
# PREISEXPERIMENT
# ---------------------------------------------------------------------------

print("=" * 70)
print("PREISEXPERIMENT: H1 (logistische Regression, quadratischer Term)")
print("=" * 70)

with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]
print(f"Finale Analysestichprobe gesamt: N = {len(final)} (Soll: 246)")

# Kontrollbedingung (849) separat als Vergleichsmassstab, nicht Teil der Kurve
kontrolle = [r for r in final if r["Treatment"] == "849"]
annahme_quote_kontrolle = np.mean([int(r["Codierung"]) for r in kontrolle])
print(f"Kontrollbedingung 849 EUR: N={len(kontrolle)}, Annahmequote={annahme_quote_kontrolle:.3f}")

# Kurve ueber die 5 Treatmentbedingungen mit Preissenkung (vgl. Abschnitt 3.1)
treatment_rows = [r for r in final if r["Treatment"] != "849"]
print(f"Treatmentbedingungen (ohne Kontrolle): N = {len(treatment_rows)}")

treat = np.array([int(r["Treatment"]) for r in treatment_rows], dtype=float)
y = np.array([int(r["Codierung"]) for r in treatment_rows], dtype=float)

# Skalierung auf Tausend-Euro-Einheiten fuer numerische Stabilitaet
treat_k = treat / 1000.0
treat_k_c = treat_k - treat_k.mean()  # zentriert, reduziert Kollinearitaet
X = np.column_stack([treat_k_c, treat_k_c ** 2])

res = logit_with_inference(X, y)
labels = ["Intercept", "Treatment (in 1000 EUR, zentriert)", "Treatment^2"]
print(f"\n{'Parameter':<38}{'Koeff.':>10}{'SE':>10}{'z':>8}{'p':>10}")
for lbl, b, se, z, p in zip(labels, res["beta"], res["se"], res["z"], res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{z:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"McFadden R^2 = {res['mcfadden_r2']:.4f}, N = {res['n']}")

# Deskriptive Annahmequoten je Bedingung als Ergaenzung
print("\nDeskriptiv - Annahmequote je Bedingung (finale Stichprobe):")
for t in sorted(set(int(r["Treatment"]) for r in final)):
    grp = [int(r["Codierung"]) for r in final if int(r["Treatment"]) == t]
    print(f"  {t:>6} EUR: N={len(grp):>3}, Annahmequote={np.mean(grp):.3f}")


# ---------------------------------------------------------------------------
# BEWERTUNGSEXPERIMENT
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("BEWERTUNGSEXPERIMENT: H1 (OLS-Regression, quadratischer Term)")
print("=" * 70)

with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows2 = list(csv.DictReader(f, delimiter=";"))

final2 = [r for r in rows2
          if r["Manipulationscheck_Rolle"].strip() == "Ja"
          and r["Manipulationscheck_Forderung"].strip() == "Ja"]
print(f"Finale Analysestichprobe: N = {len(final2)} (Soll: 295)")

ford = np.array([int(r["Forderung_des_Studierenden"]) for r in final2], dtype=float)
punkte = np.array([int(r["Final_vergebene_Punktzahl"]) for r in final2], dtype=float)

ford_c = ford - ford.mean()
X2 = np.column_stack([ford_c, ford_c ** 2])

res2 = ols_with_inference(X2, punkte)
labels2 = ["Intercept", "Forderung (Punkte, zentriert)", "Forderung^2"]
print(f"\n{'Parameter':<38}{'Koeff.':>10}{'SE':>10}{'t':>8}{'p':>10}")
for lbl, b, se, t, p in zip(labels2, res2["beta"], res2["se"], res2["t"], res2["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res2['r2']:.4f}, df_resid = {res2['dof']}, N = {res2['n']}")

print("\nDeskriptiv - Mittelwert Punktzahl je Bedingung (finale Stichprobe):")
for f_ in sorted(set(int(r["Forderung_des_Studierenden"]) for r in final2)):
    grp = [int(r["Final_vergebene_Punktzahl"]) for r in final2 if int(r["Forderung_des_Studierenden"]) == f_]
    print(f"  {f_:>2} Punkte gefordert: N={len(grp):>3}, Mittelwert Punktzahl={np.mean(grp):.3f}, SD={np.std(grp, ddof=1):.3f}")
