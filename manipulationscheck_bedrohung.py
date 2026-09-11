"""
Manipulationscheck: Wahrgenommene Bedrohung (Items 1-3) ~ Forderung
Bewertungsexperiment.

Bedrohungswahrnehmung_Index = Mittelwert der drei Items (bereits in der
CSV vorberechnet).
"""

import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file, ols_with_inference, sig_stars

BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")


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
threat_index = np.array([to_float(r["Bedrohungswahrnehmung_Index"]) for r in final])

print("\n" + "=" * 70)
print("Manipulationscheck: Bedrohungswahrnehmung_Index ~ Forderung (OLS)")
print("=" * 70)
res = ols_with_inference(X_c.reshape(-1, 1), threat_index)
labels = ["Intercept", "Forderung (zentriert)"]
for lbl, b, se, t, p in zip(labels, res["beta"], res["se"], res["t"], res["p"]):
    print(f"{lbl:<28}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res['r2']:.4f}, df_resid = {res['dof']}, N = {res['n']}")

# Mittelwerte je Bedingung, zur Illustration
print("\nMittelwerte des Bedrohungswahrnehmung_Index je Bedingung:")
for cond in sorted(set(X)):
    vals = threat_index[X == cond]
    print(f"  {cond:.0f} Punkte: M = {vals.mean():.3f} (n = {len(vals)})")

# Einzelitems zur Vollstaendigkeit
print("\nEinzelitems (Mittelwerte je Bedingung):")
for item_col in ["Wahrgenommener_Druck", "Eingeschraenkte_Bewertungsfreiheit", "Wahrnehmung_als_manipulativ"]:
    vals_all = np.array([to_float(r[item_col]) for r in final])
    means = [vals_all[X == cond].mean() for cond in sorted(set(X))]
    print(f"  {item_col:<38}" + "  ".join(f"{m:.2f}" for m in means))
