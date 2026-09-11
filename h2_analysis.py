import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import (
    find_data_file,
    ols_with_inference, ols_beta_only,
    logit_with_inference, logit_beta_only,
    sig_stars, bc_ci_with_point,
)

np.random.seed(42)  # Reproduzierbarkeit der Bootstrap-Ergebnisse

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")

N_BOOT = 5000


def to_float(s):
    return float(s.strip().replace(",", "."))


with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final_mit_kontrolle = [r for r in rows
         if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]
print(f"Finale Analysestichprobe (inkl. Kontrollbedingung): N = {len(final_mit_kontrolle)} (Soll: 246)")

final = [r for r in final_mit_kontrolle if r["Treatment"] != "849"]
print(f"Finale Analysestichprobe (nur Treatmentbedingungen, wie H1): N = {len(final)} (Soll: 195)")

X = np.array([to_float(r["Treatment"]) for r in final]) / 1000.0  # in 1000 EUR
X_c = X - X.mean()
M1 = np.array([to_float(r["Wahrnehmungskontrast"]) for r in final])
M2 = np.array([to_float(r["Konzession"]) for r in final])
M3 = np.array([to_float(r["Boomerang_Variable"]) for r in final])
Y = np.array([to_float(r["Codierung"]) for r in final])

mediator_names = ["M1: Wahrnehmungskontrast (Kontrasteffekt)",
                   "M2: Konzession (Aufwaertsphase)",
                   "M3: Boomerang_Variable (Abwaertsphase)"]
mediators = [M1, M2, M3]

# ---------------------------------------------------------------------------
# Pfad a: M_i ~ X  (einfache OLS je Mediator)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PFAD a (X -> M_i): OLS, X = Treatment in 1000 EUR (zentriert)")
print("=" * 78)
a_results = []
for name, M in zip(mediator_names, mediators):
    res = ols_with_inference(X_c.reshape(-1, 1), M)
    a_results.append(res)
    b, se, t, p = res["beta"][1], res["se"][1], res["t"][1], res["p"][1]
    print(f"{name:<45} a={b:>8.4f}  SE={se:>7.4f}  t={t:>6.2f}  p={p:>7.4f} {sig_stars(p)}")

# ---------------------------------------------------------------------------
# Pfad b + c': Y ~ X + M1 + M2 + M3  (voller Logit)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PFAD b + DIREKTER EFFEKT c' (X, M1, M2, M3 -> Y): Logit")
print("=" * 78)
X_full = np.column_stack([X_c, M1, M2, M3])
full_res = logit_with_inference(X_full, Y)
labels = ["Intercept", "X: Treatment (c')", "M1: Wahrnehmungskontrast",
          "M2: Konzession", "M3: Boomerang_Variable"]
for lbl, b, se, z, p in zip(labels, full_res["beta"], full_res["se"], full_res["z"], full_res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{z:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"McFadden R^2 = {full_res['mcfadden_r2']:.4f}, N = {full_res['n']}")

# ---------------------------------------------------------------------------
# Totaler Effekt c: Y ~ X (ohne Mediatoren)
# ---------------------------------------------------------------------------
total_res = logit_with_inference(X_c.reshape(-1, 1), Y)
c_total = total_res["beta"][1]
print(f"\nTotaler Effekt c (X -> Y, ohne Mediatoren): {c_total:.4f} "
      f"(p={total_res['p'][1]:.4f} {sig_stars(total_res['p'][1])})")

# ---------------------------------------------------------------------------
# Indirekte Effekte a_i * b_i + Bootstrap-KI (BC, 5000 Replikationen)
# ---------------------------------------------------------------------------
a_point = np.array([res["beta"][1] for res in a_results])
b_point = full_res["beta"][2:5]  # M1, M2, M3 im vollen Modell
indirect_point = a_point * b_point

n = len(Y)
boot_indirect = np.zeros((N_BOOT, 3))
for i in range(N_BOOT):
    idx = np.random.randint(0, n, size=n)
    Xb, Mb1, Mb2, Mb3, Yb = X_c[idx], M1[idx], M2[idx], M3[idx], Y[idx]
    # Falls im Bootstrap-Sample Y konstant wird (nur 0en oder nur 1en),
    # ist Logit nicht definiert -> dieses Replikat ueberspringen (NaN).
    if Yb.min() == Yb.max():
        boot_indirect[i, :] = np.nan
        continue
    a1 = ols_beta_only(Xb.reshape(-1, 1), Mb1)[1]
    a2 = ols_beta_only(Xb.reshape(-1, 1), Mb2)[1]
    a3 = ols_beta_only(Xb.reshape(-1, 1), Mb3)[1]
    try:
        beta_full = logit_beta_only(np.column_stack([Xb, Mb1, Mb2, Mb3]), Yb)
    except np.linalg.LinAlgError:
        boot_indirect[i, :] = np.nan
        continue
    b1, b2, b3 = beta_full[2], beta_full[3], beta_full[4]
    boot_indirect[i, 0] = a1 * b1
    boot_indirect[i, 1] = a2 * b2
    boot_indirect[i, 2] = a3 * b3

n_valid = np.sum(~np.isnan(boot_indirect[:, 0]))
print(f"\nBootstrap: {n_valid}/{N_BOOT} gueltige Replikationen (Rest wegen "
      f"entarteter Y-Verteilung im Resample verworfen)")

print("\n" + "=" * 100)
print(f"{'Mediator':<40}{'a-Pfad':>9}{'b-Pfad':>9}{'Indirekt (a*b)':>16}   95%-BC-Bootstrap-KI      Sig.?")
print("=" * 100)
for i, name in enumerate(mediator_names):
    lo, hi = bc_ci_with_point(boot_indirect[:, i], indirect_point[i])
    sig = "JA" if (lo > 0 or hi < 0) else "Nein"
    print(f"{name:<40}{a_point[i]:>9.4f}{b_point[i]:>9.4f}{indirect_point[i]:>16.4f}   "
          f"[{lo:>8.4f}, {hi:>8.4f}]   {sig}")

print(f"\nDirekter Effekt c' (X -> Y, kontrolliert fuer M1-M3): "
      f"{full_res['beta'][1]:.4f} (p={full_res['p'][1]:.4f} {sig_stars(full_res['p'][1])})")
print(f"Totaler Effekt c (X -> Y, ohne Mediatoren): "
      f"{c_total:.4f} (p={total_res['p'][1]:.4f} {sig_stars(total_res['p'][1])})")

# ---------------------------------------------------------------------------
# Zusatz: Korrelationen zwischen den drei Mediatoren 
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Korrelationsmatrix der drei Mediatoren (zur Einordnung)")
print("=" * 70)
mat = np.corrcoef([M1, M2, M3])
names_short = ["Kontrasteffekt", "Konzession", "Boomerang"]
print(f"{'':<16}" + "".join(f"{n:>14}" for n in names_short))
for i, n in enumerate(names_short):
    print(f"{n:<16}" + "".join(f"{mat[i, j]:>14.3f}" for j in range(3)))
