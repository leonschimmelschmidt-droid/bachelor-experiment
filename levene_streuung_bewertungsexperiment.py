"""
Streuungsanalyse (Levene-Test), Bewertungsexperiment: Ergaenzung zu H1 (4.2).

Hintergrund: Der quadratische Term in der H1-Kurve (Tabelle 10) ist nicht
signifikant, obwohl H1 und H3 zwei gegenlaeufige, theoretisch begruendete
Mechanismen postulieren (Ankereffekt nach oben, vgl. Abschnitt 2.1;
Reaktanz nach unten, vgl. Abschnitt 2.2/H3). Falls beide Mechanismen
gleichzeitig auf unterschiedliche Teile der Stichprobe wirken, statt die
gesamte Stichprobe einheitlich zu verschieben, wuerde sich das nicht im
Mittelwert zeigen (die Effekte heben sich auf), sondern in der Streuung
(die Effekte addieren sich).

Test: Levene-Test auf Gleichheit der Varianzen, medianzentriert
(Brown-Forsythe-Variante, robust gegenueber Abweichungen von der
Normalverteilung), sowohl ueber alle vier Bedingungen als auch im
direkten Vergleich der niedrigsten (9 Punkte) und hoechsten (15 Punkte)
Bedingung.
"""

import csv
import numpy as np
from scipy import stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file

BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")


def to_float(s):
    return float(s.strip().replace(",", "."))


with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Manipulationscheck_Rolle"].strip() == "Ja"
         and r["Manipulationscheck_Forderung"].strip() == "Ja"]
print(f"Finale Analysestichprobe: N = {len(final)} (Soll: 295)")

groups = {}
for cond in [9, 11, 13, 15]:
    vals = np.array([to_float(r["Final_vergebene_Punktzahl"]) for r in final
                      if int(to_float(r["Forderung_des_Studierenden"])) == cond])
    groups[cond] = vals
    print(f"  {cond} Punkte: N={len(vals)}, M={vals.mean():.3f}, SD={vals.std(ddof=1):.3f}, "
          f"Range=[{vals.min():.0f}, {vals.max():.0f}]")

print("\n" + "=" * 70)
print("Levene-Test auf Gleichheit der Varianzen (medianzentriert)")
print("=" * 70)
W4, p4 = stats.levene(groups[9], groups[11], groups[13], groups[15], center="median")
print(f"Alle vier Bedingungen: W = {W4:.4f}, p = {p4:.6f}")

W2, p2 = stats.levene(groups[9], groups[15], center="median")
print(f"9 vs. 15 Punkte:       W = {W2:.4f}, p = {p2:.6f}")

print("\n" + "=" * 70)
print("Verteilung an den Raendern, 9 vs. 15 Punkte")
print("=" * 70)
for cond in [9, 15]:
    g = groups[cond]
    ge12 = np.sum(g >= 12)
    le6 = np.sum(g <= 6)
    print(f"{cond} Punkte (N={len(g)}): >=12 Punkte: {ge12} ({100*ge12/len(g):.1f}%), "
          f"<=6 Punkte: {le6} ({100*le6/len(g):.1f}%), Spannweite: [{g.min():.0f}, {g.max():.0f}]")
