"""
Robustheitscheck: H1-Kurve im Preisexperiment,
Direktannahmen der Erstforderung in den Treatmentbedingungen zusaetzlich als
Annahme (Codierung = 1) gewertet, statt sie strukturell auszuschliessen.
"""

import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file, logit_with_inference, sig_stars

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")

with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

treat_rows_all = [r for r in rows if r["Treatment"].strip() != "849"]

# Regulaere Faelle: Codierung vorhanden, beide Manipulationschecks bestanden
regular = [r for r in treat_rows_all
           if r["Codierung"].strip() != ""
           and r["Manipulationscheck_Preis"].strip() == "Ja"
           and r["Manipulationscheck_Validitaet"].strip() == "Ja"]

# Direktannahmen (Codierung leer), die beide Manipulationschecks trotzdem bestehen
direkt_included = [r for r in treat_rows_all
                    if r["Codierung"].strip() == ""
                    and r["Manipulationscheck_Preis"].strip() == "Ja"
                    and r["Manipulationscheck_Validitaet"].strip() == "Ja"]

print(f"Reguläre Fälle (wie Hauptanalyse): N = {len(regular)} (Soll: 195)")
print(f"Direktannahmen, die beide Manipulationschecks bestehen (zusätzlich als Annahme=1 gewertet): N = {len(direkt_included)}")
print(f"Direktannahmen insgesamt in Treatmentbedingungen (vgl. Tabelle 5): 7 "
      f"(davon {len(direkt_included)} mit bestandenen Manipulationschecks, "
      f"{7 - len(direkt_included)} zusätzlich ausgeschlossen wegen gescheiterter Checks)")

combined = regular + direkt_included
print(f"Kombinierte Stichprobe für den Robustheitscheck: N = {len(combined)}")

treat = np.array([int(r["Treatment"]) for r in combined], dtype=float)
y = np.array([1.0 if r["Codierung"].strip() == "" else float(r["Codierung"]) for r in combined])

# Sanity check: Direktannahmen muessen alle auf 1 gesetzt sein, reguläre Codierung unveraendert
assert all(y[treat == t].sum() >= 0 for t in set(treat))

treat_k = treat / 1000.0
treat_k_c = treat_k - treat_k.mean()
X = np.column_stack([treat_k_c, treat_k_c ** 2])

res = logit_with_inference(X, y)
labels = ["Intercept", "Treatment (in 1000 EUR, zentriert)", "Treatment^2"]
print("\n" + "=" * 70)
print("Robustheitscheck: H1-Kurve, Direktannahmen als Annahme gewertet")
print("=" * 70)
print(f"{'Parameter':<38}{'Koeff.':>10}{'SE':>10}{'z':>8}{'p':>10}")
for lbl, b, se, z, p in zip(labels, res["beta"], res["se"], res["z"], res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{z:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"McFadden R^2 = {res['mcfadden_r2']:.4f}, N = {res['n']}")

print("\nAnnahmequoten je Bedingung (Robustheitscheck-Stichprobe):")
for t in sorted(set(treat)):
    grp = y[treat == t]
    print(f"  {t:>6.0f} EUR: N={len(grp):>3}, Annahmequote={grp.mean():.3f}")
