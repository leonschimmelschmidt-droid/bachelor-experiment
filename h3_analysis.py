import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import (
    find_data_file,
    ols_with_inference, ols_beta_only, sig_stars, bc_ci_with_point,
)

np.random.seed(42)

BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
N_BOOT = 5000


def to_float(s):
    return float(s.strip().replace(",", "."))


with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Manipulationscheck_Rolle"].strip() == "Ja"
         and r["Manipulationscheck_Forderung"].strip() == "Ja"]
print(f"Finale Analysestichprobe: N = {len(final)} (Soll: 295)")

X = np.array([to_float(r["Forderung_des_Studierenden"]) for r in final])
X_c = X - X.mean()
M = np.array([to_float(r["Boomerang_Variable"]) for r in final])
Y = np.array([to_float(r["Final_vergebene_Punktzahl"]) for r in final])

# ---------------------------------------------------------------------------
# Pfad a: M ~ X
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PFAD a (X -> M): OLS, X = Forderung (Punkte, zentriert)")
print("=" * 70)
res_a = ols_with_inference(X_c.reshape(-1, 1), M)
a_lbl = ["Intercept", "Forderung"]
for lbl, b, se, t, p in zip(a_lbl, res_a["beta"], res_a["se"], res_a["t"], res_a["p"]):
    print(f"{lbl:<20}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
    print(f"R^2 = {res_a['r2']:.4f}, df_resid = {res_a['dof']}, N = {res_a['n']}")
a_point = res_a["beta"][1]

# ---------------------------------------------------------------------------
# Pfad b + c': Y ~ X + M
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PFAD b + DIREKTER EFFEKT c' (X, M -> Y): OLS")
print("=" * 70)
X_full = np.column_stack([X_c, M])
res_full = ols_with_inference(X_full, Y)
full_lbl = ["Intercept", "X: Forderung (c')", "M: Boomerang_Variable (Reaktanz, b)"]
for lbl, b, se, t, p in zip(full_lbl, res_full["beta"], res_full["se"], res_full["t"], res_full["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res_full['r2']:.4f}, df_resid = {res_full['dof']}, N = {res_full['n']}")
b_point = res_full["beta"][2]
c_prime = res_full["beta"][1]

# ---------------------------------------------------------------------------
# Totaler Effekt c: Y ~ X
# ---------------------------------------------------------------------------
res_total = ols_with_inference(X_c.reshape(-1, 1), Y)
c_total = res_total["beta"][1]
print(f"\nTotaler Effekt c (X -> Y, ohne Mediator): {c_total:.4f} "
      f"(p={res_total['p'][1]:.4f} {sig_stars(res_total['p'][1])}), "
      f"R^2 = {res_total['r2']:.4f}")

# ---------------------------------------------------------------------------
# Indirekter Effekt a*b + Bootstrap-KI
# ---------------------------------------------------------------------------
indirect_point = a_point * b_point
n = len(Y)
boot_indirect = np.zeros(N_BOOT)
for i in range(N_BOOT):
    idx = np.random.randint(0, n, size=n)
    Xb, Mb, Yb = X_c[idx], M[idx], Y[idx]
    ab = ols_beta_only(Xb.reshape(-1, 1), Mb)[1]
    beta_full = ols_beta_only(np.column_stack([Xb, Mb]), Yb)
    bb = beta_full[2]
    boot_indirect[i] = ab * bb

lo, hi = bc_ci_with_point(boot_indirect, indirect_point)
sig = "JA" if (lo > 0 or hi < 0) else "Nein"

print("\n" + "=" * 90)
print(f"{'Pfad':<12}{'a':>10}{'b':>10}{'Indirekt (a*b)':>18}   95%-BC-Bootstrap-KI      Sig.?")
print("=" * 90)
print(f"{'X->M->Y':<12}{a_point:>10.4f}{b_point:>10.4f}{indirect_point:>18.4f}   "
      f"[{lo:>8.4f}, {hi:>8.4f}]   {sig}")

print(f"\nDirekter Effekt c' (Forderung -> Punkte, kontrolliert fuer Reaktanz): "
      f"{c_prime:.4f} (p={res_full['p'][1]:.4f} {sig_stars(res_full['p'][1])})")
print(f"Totaler Effekt c (Forderung -> Punkte, ohne Mediator): "
      f"{c_total:.4f} (p={res_total['p'][1]:.4f} {sig_stars(res_total['p'][1])})")

print("\nHinweis zur Interpretation von H3:")
print("H3 sagt vorher, dass hoehere Reaktanz zu NIEDRIGERER Punktzahl fuehrt,")
print("also einen NEGATIVEN Pfad b. Das Vorzeichen von b oben entscheidet,")
print("ob die Richtung ueberhaupt zur Hypothese passt - unabhaengig davon,")
print("ob der indirekte Effekt signifikant ist.")
