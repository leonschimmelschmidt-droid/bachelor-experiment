"""
Ergaenzende Mediationsanalyse H2, allein mit Item 2a (Angemessenheit_invertiert)
statt der kombinierten Boomerang-Variable (angekuendigt in 3.2.3, bisher nicht
berichtet):

"Die Mediationsanalyse wird ergaenzend allein mit Item 2a berichtet."

Hintergrund: Item 2d ist als doppellaeufiges Item (Double-Barreled) kritisiert,
weil es emotionales Erleben und die spaetere Ablehnung selbst vermengt (vgl.
3.2.3). Die Boomerang-Variable (M3 im Hauptmodell) besteht aus Item 2a
(invertiert) UND Item 2d und koennte daher den beschriebenen Messfehler
teilweise erben. Dieses Zusatzmodell ersetzt M3 durch Item 2a allein
(Angemessenheit_invertiert), um zu pruefen, ob der Reaktanzpfad auch ohne
Item 2d bestehen bleibt.

Modell identisch zu h2_analysis.py, nur M3 ausgetauscht:
  M1 = Wahrnehmungskontrast (unveraendert)
  M2 = Konzession (unveraendert)
  M3'= Angemessenheit_invertiert (Item 2a allein, statt Boomerang_Variable)
"""

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

np.random.seed(42)

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
final = [r for r in final_mit_kontrolle if r["Treatment"] != "849"]
print(f"Finale Analysestichprobe (nur Treatmentbedingungen, wie H2): N = {len(final)} (Soll: 195)")

X = np.array([to_float(r["Treatment"]) for r in final]) / 1000.0
X_c = X - X.mean()
M1 = np.array([to_float(r["Wahrnehmungskontrast"]) for r in final])
M2 = np.array([to_float(r["Konzession"]) for r in final])
M3 = np.array([to_float(r["Angemessenheit_invertiert"]) for r in final])  # Item 2a allein
Y = np.array([to_float(r["Codierung"]) for r in final])

mediator_names = ["M1: Wahrnehmungskontrast",
                   "M2: Konzession",
                   "M3': Angemessenheit_invertiert (Item 2a allein)"]
mediators = [M1, M2, M3]

print("\n" + "=" * 78)
print("PFAD a (X -> M_i): OLS, X = Treatment in 1000 EUR (zentriert)")
print("=" * 78)
a_results = []
for name, M in zip(mediator_names, mediators):
    res = ols_with_inference(X_c.reshape(-1, 1), M)
    a_results.append(res)
    b, se, t, p = res["beta"][1], res["se"][1], res["t"][1], res["p"][1]
    print(f"{name:<48} a={b:>8.4f}  SE={se:>7.4f}  t={t:>6.2f}  p={p:>7.4f} {sig_stars(p)}")

print("\n" + "=" * 78)
print("PFAD b + DIREKTER EFFEKT c' (X, M1, M2, M3' -> Y): Logit")
print("=" * 78)
X_full = np.column_stack([X_c, M1, M2, M3])
full_res = logit_with_inference(X_full, Y)
labels = ["Intercept", "X: Treatment (c')", "M1: Wahrnehmungskontrast",
          "M2: Konzession", "M3': Angemessenheit_invertiert"]
for lbl, b, se, z, p in zip(labels, full_res["beta"], full_res["se"], full_res["z"], full_res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{z:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"McFadden R^2 = {full_res['mcfadden_r2']:.4f}, N = {full_res['n']}")

total_res = logit_with_inference(X_c.reshape(-1, 1), Y)
c_total = total_res["beta"][1]
print(f"\nTotaler Effekt c (X -> Y, ohne Mediatoren, unveraendert ggue. H2): "
      f"{c_total:.4f} (p={total_res['p'][1]:.4f} {sig_stars(total_res['p'][1])})")

a_point = np.array([res["beta"][1] for res in a_results])
b_point = full_res["beta"][2:5]
indirect_point = a_point * b_point

n = len(Y)
boot_indirect = np.zeros((N_BOOT, 3))
for i in range(N_BOOT):
    idx = np.random.randint(0, n, size=n)
    Xb, Mb1, Mb2, Mb3, Yb = X_c[idx], M1[idx], M2[idx], M3[idx], Y[idx]
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
print(f"\nBootstrap: {n_valid}/{N_BOOT} gueltige Replikationen")

print("\n" + "=" * 100)
print(f"{'Mediator':<48}{'a-Pfad':>9}{'b-Pfad':>9}{'Indirekt (a*b)':>16}   95%-BC-Bootstrap-KI      Sig.?")
print("=" * 100)
for i, name in enumerate(mediator_names):
    lo, hi = bc_ci_with_point(boot_indirect[:, i], indirect_point[i])
    sig = "JA" if (lo > 0 or hi < 0) else "Nein"
    print(f"{name:<48}{a_point[i]:>9.4f}{b_point[i]:>9.4f}{indirect_point[i]:>16.4f}   "
          f"[{lo:>8.4f}, {hi:>8.4f}]   {sig}")
