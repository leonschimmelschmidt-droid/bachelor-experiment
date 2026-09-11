import csv
import time
import numpy as np
from scipy.optimize import linprog
from scipy import stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file, ols_with_inference, sig_stars

BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
N_BOOT = 1000
TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]


def to_float(s):
    return float(s.strip().replace(",", "."))


def quantreg_slope(x, y, tau):
    """Quantilsregression y = alpha + beta*x, exakte Loesung ueber LP.
    Gibt (alpha, beta) zurueck."""
    n = len(y)
    c = np.concatenate([[0, 0], np.full(n, tau), np.full(n, 1 - tau)])
    A_eq = np.zeros((n, 2 + 2 * n))
    A_eq[:, 0] = 1.0
    A_eq[:, 1] = x
    A_eq[np.arange(n), 2 + np.arange(n)] = 1.0
    A_eq[np.arange(n), 2 + n + np.arange(n)] = -1.0
    res = linprog(c, A_eq=A_eq, b_eq=y, bounds=[(None, None), (None, None)] + [(0, None)] * (2 * n),
                   method="highs")
    if not res.success:
        raise RuntimeError(f"LP nicht geloest (tau={tau}): {res.message}")
    return res.x[0], res.x[1]


with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Manipulationscheck_Rolle"].strip() == "Ja"
         and r["Manipulationscheck_Forderung"].strip() == "Ja"]
print(f"Finale Analysestichprobe: N = {len(final)} (Soll: 295)")

X = np.array([to_float(r["Forderung_des_Studierenden"]) for r in final])
X_c = X - X.mean()
Y = np.array([to_float(r["Final_vergebene_Punktzahl"]) for r in final])
n = len(Y)

res_mean = ols_with_inference(X_c.reshape(-1, 1), Y)
print(f"\nZum Vergleich, lineare OLS auf den Mittelwert (ohne quadratischen Term): "
      f"b = {res_mean['beta'][1]:.4f}, p = {res_mean['p'][1]:.4f}")

point_betas = {tau: quantreg_slope(X_c, Y, tau)[1] for tau in TAUS}

t0 = time.time()
np.random.seed(42)
boot = {tau: np.zeros(N_BOOT) for tau in TAUS}
for i in range(N_BOOT):
    idx = np.random.randint(0, n, size=n)
    Xb, Yb = X_c[idx], Y[idx]
    for tau in TAUS:
        try:
            _, b = quantreg_slope(Xb, Yb, tau)
        except RuntimeError:
            b = np.nan
        boot[tau][i] = b
    if (i + 1) % 200 == 0:
        print(f"  Bootstrap {i+1}/{N_BOOT} ({time.time()-t0:.0f}s)")

print("\n" + "=" * 78)
print("Quantilsregression: Punktzahl ~ Forderung (zentriert), je Quantil")
print("=" * 78)
for tau in TAUS:
    beta = point_betas[tau]
    b = boot[tau]
    se = np.nanstd(b, ddof=1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    z = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    print(f"tau = {tau:.2f}:  b = {beta:>7.4f}  SE = {se:.4f}  "
          f"95%-KI [{lo:>7.4f}, {hi:>7.4f}]  p = {p:.4f} {sig_stars(p)}")

diff_boot = boot[0.90] - boot[0.10]
point_diff = point_betas[0.90] - point_betas[0.10]
lo_d, hi_d = np.nanpercentile(diff_boot, [2.5, 97.5])
z_d = point_diff / np.nanstd(diff_boot, ddof=1)
p_d = 2 * (1 - stats.norm.cdf(np.abs(z_d)))
print(f"\nDivergenz b(0,90) - b(0,10) = {point_diff:.4f}, 95%-KI [{lo_d:.4f}, {hi_d:.4f}], "
      f"p = {p_d:.4f} {sig_stars(p_d)}")
