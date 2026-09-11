"""
H4-Test: Bewertungsexperiment - Wahrgenommene Fairness -> Reaktanz
Plus die in Abschnitt 3.3.3 versprochene Trennschaerfe-Pruefung (Item 5 vs. Item 6).

H4: "Im Bewertungsexperiment gilt: je unfairer die Erstforderung wahrgenommen
     wird, desto staerker ist die dadurch ausgeloeste Reaktanz."

Modell (einfache OLS-Regression):
  X = Wahrgenommene_Fairness   (Item 6, hoeher = fairer)
  Y = Boomerang_Variable       (Reaktanz-Komposit aus Item 4 + invertiertem Item 5)

Erwartung laut H4: NEGATIVER Zusammenhang (mehr Fairness -> weniger Reaktanz).

Trennschaerfe-Pruefung Item 5 (Angemessenheit_invertiert) vs. Item 6
(Wahrgenommene_Fairness): beide erfassen konzeptuell verwandte, aber nicht
identische Konstrukte (Angemessenheit der Erstforderung vs. Fairness des
Verfahrens/der Forderung). Korrelation sollte moderat sein - hoch genug fuer
inhaltliche Naehe, aber klar unter dem ueblichen Schwellenwert fuer
Redundanz (r > .85, vgl. Diskriminanzvaliditaet).
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

fairness = np.array([to_float(r["Wahrgenommene_Fairness"]) for r in final])
reaktanz = np.array([to_float(r["Boomerang_Variable"]) for r in final])
item5 = np.array([to_float(r["Angemessenheit_invertiert"]) for r in final])

fairness_c = fairness - fairness.mean()

print("\n" + "=" * 70)
print("H4: Fairness -> Reaktanz (OLS)")
print("=" * 70)
res = ols_with_inference(fairness_c.reshape(-1, 1), reaktanz)
labels = ["Intercept", "Wahrgenommene_Fairness (zentriert)"]
for lbl, b, se, t, p in zip(labels, res["beta"], res["se"], res["t"], res["p"]):
    print(f"{lbl:<38}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res['r2']:.4f}, df_resid = {res['dof']}, N = {res['n']}")

direction = "negativ (erwartungskonform)" if res["beta"][1] < 0 else "positiv (NICHT erwartungskonform)"
print(f"\n-> Richtung des Effekts: {direction}")

print("\n" + "=" * 70)
print("Trennschaerfe-Pruefung: Item 5 (Angemessenheit_invertiert) vs. Item 6 (Fairness)")
print("=" * 70)
r = np.corrcoef(item5, fairness)[0, 1]
print(f"Korrelation Item 5 <-> Item 6 (Fairness): r = {r:.3f}")
if abs(r) > 0.85:
    verdict = "hoch, moegliche Redundanz - Items messen evtl. dasselbe Konstrukt"
elif abs(r) > 0.5:
    verdict = "moderat bis hoch - inhaltlich verwandt, aber empirisch unterscheidbar"
else:
    verdict = "eher schwach - Items erfassen weitgehend getrennte Konstrukte"
print(f"-> Einordnung: {verdict}")

# Zusatz: haelt Fairness als Praediktor stand, wenn Item 5 (Angemessenheit_invertiert)
# separat kontrolliert wird? Zeigt, ob Fairness einen EIGENEN Beitrag zur Reaktanz
# leistet, der nicht nur die (invertierte) Angemessenheitswahrnehmung widerspiegelt.
# Reaktanz besteht per Definition aus Veraergerung + Angemessenheit_invertiert, daher
# wird hier NICHT Boomerang_Variable, sondern nur Veraergerung als Y verwendet, um
# Teil-Ganzes-Konfundierung zu vermeiden.
veraergerung = np.array([to_float(r_["Veraergerung"]) for r_ in final])
print("\n" + "=" * 70)
print("Ergaenzung: Fairness + Item 5 -> Veraergerung (Item 4, um Teil-Ganzes-")
print("Konfundierung mit der Reaktanz-Variable selbst zu vermeiden)")
print("=" * 70)
item5_c = item5 - item5.mean()
X2 = np.column_stack([fairness_c, item5_c])
res2 = ols_with_inference(X2, veraergerung)
labels2 = ["Intercept", "Fairness (zentriert)", "Item 5: Angemessenheit_invertiert (zentriert)"]
for lbl, b, se, t, p in zip(labels2, res2["beta"], res2["se"], res2["t"], res2["p"]):
    print(f"{lbl:<48}{b:>10.4f}{se:>10.4f}{t:>8.2f}{p:>10.4f} {sig_stars(p)}")
print(f"R^2 = {res2['r2']:.4f}, N = {res2['n']}")
