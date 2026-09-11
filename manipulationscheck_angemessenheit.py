"""
Manipulationscheck: Wahrgenommene Angemessenheit (Item 2a) 

Verwendet dieselbe Treatmentstichprobe wie H1/H2 (ohne Kontrollbedingung,
N = 195), da nur dort eine ueberhoehte Erstforderung ueberhaupt vorliegt, deren
Angemessenheit variieren kann.
"""

import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file, ols_with_inference, sig_stars

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")


def to_float(s):
    return float(s.strip().replace(",", "."))


with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]
treat = [r for r in final if r["Treatment"] != "849"]
print(f"Treatmentstichprobe (wie H1/H2): N = {len(treat)} (Soll: 195)")

X = np.array([to_float(r["Treatment"]) for r in treat]) / 1000.0
X_c = X - X.mean()
angemessenheit = np.array([to_float(r["Angemessenheit"]) for r in treat])

print("\n" + "=" * 70)
print("Manipulationscheck: Angemessenheit (Item 2a) ~ Treatment (OLS)")
print("=" * 70)
res = ols_with_inference(X_c.reshape(-1, 1), angemessenheit)
labels = ["Intercept", "Treatment (in 1000 EUR, zentriert)"]
for lbl, b, se, t, p in zip(labels, res["beta"], res["se"], res["t"], res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res['r2']:.4f}, df_resid = {res['dof']}, N = {res['n']}")

print("\nMittelwerte der Angemessenheit je Bedingung:")
for t_ in sorted(set(int(r["Treatment"]) for r in treat)):
    vals = np.array([to_float(r["Angemessenheit"]) for r in treat if int(r["Treatment"]) == t_])
    print(f"  {t_:>6} EUR: M = {vals.mean():.3f} (n = {len(vals)})")
