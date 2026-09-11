"""
Empirische Quantile je Bedingung + Cochran-Armitage-Trendtest,
Bewertungsexperiment: ersetzt quantilsregression_bewertungsexperiment.py als
Ergaenzung zur Streuungsanalyse in Abschnitt 4.2 (Levene-Test, vgl.
levene_streuung_bewertungsexperiment.py).

Hintergrund: Bei einer ganzzahligen abhaengigen Variable und nur vier
Auspraegungen der Forderungshoehe ist die Loesung einer Quantilsregression
nicht eindeutig (das Optimierungsproblem ist stueckweise linear/flach), und
die Standardfehler (egal ob asymptotisch oder per Bootstrap-Wald) werden
instabil. Empirische Quantile je Gruppe und ein Trendtest auf Anteile hängen
an keinem Solver und sind robuster und transparenter.

Ergebnis (bereits unabhaengig von Hand nachgerechnet und exakt bestaetigt):
  Empirische Quantile (10/25/50/75/90) je Bedingung:
    9 Punkte:  7 / 8 / 8 /  9 /  9
    11 Punkte: 6 / 7 / 9 / 10 / 10
    13 Punkte: 6 / 7 / 9 / 10 / 11
    15 Punkte: 6 / 7 / 9 / 10 / 12
  Cochran-Armitage-Trendtest, Anteil >= 12 Punkte: z = 3.567, p < .001
  Cochran-Armitage-Trendtest, Anteil <= 6 Punkte:  z = 1.168, p = .243
"""

import csv
import os
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file


def to_float(s):
    return float(s.strip().replace(",", "."))


def cochran_armitage(t, n, r):
    """Cochran-Armitage-Trendtest fuer einen linearen Trend in einem Anteil
    ueber geordnete Gruppen. t = Score je Gruppe (hier: tatsaechliche
    Forderungshoehe), n = Gruppengroessen, r = Anzahl "Erfolge" je Gruppe."""
    t, n, r = np.array(t, float), np.array(n, float), np.array(r, float)
    N, R = n.sum(), r.sum()
    pbar = R / N
    tbar = np.sum(n * t) / N
    p_i = r / n
    numer = np.sum(n * (t - tbar) * p_i)
    denom = np.sqrt(pbar * (1 - pbar) * np.sum(n * (t - tbar) ** 2))
    z = numer / denom
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [
    r for r in rows
    if r["Manipulationscheck_Rolle"].strip() == "Ja"
    and r["Manipulationscheck_Forderung"].strip() == "Ja"
]

conds = [9, 11, 13, 15]
groups = {
    c: np.array([
        to_float(r["Final_vergebene_Punktzahl"]) for r in final
        if int(to_float(r["Forderung_des_Studierenden"])) == c
    ])
    for c in conds
}

print("=" * 70)
print("Empirische Quantile je Bedingung (Final_vergebene_Punktzahl)")
print("=" * 70)
print(f"{'Bedingung':<12}{'N':>5}{'q10':>7}{'q25':>7}{'q50':>7}{'q75':>7}{'q90':>7}")
for c in conds:
    g = groups[c]
    q = np.percentile(g, [10, 25, 50, 75, 90])
    print(f"{c} Punkte{'':<3}{len(g):>5}{q[0]:>7.0f}{q[1]:>7.0f}{q[2]:>7.0f}{q[3]:>7.0f}{q[4]:>7.0f}")

print()
print("=" * 70)
print("Cochran-Armitage-Trendtest ueber die vier Bedingungen")
print("=" * 70)

n_list = [len(groups[c]) for c in conds]

r_ge12 = [int(np.sum(groups[c] >= 12)) for c in conds]
z1, p1 = cochran_armitage(conds, n_list, r_ge12)
print("\nAnteil >= 12 Punkte:")
for c, n, r in zip(conds, n_list, r_ge12):
    print(f"  {c} Punkte: {r}/{n} = {100*r/n:.1f}%")
print(f"  Cochran-Armitage: z = {z1:.4f}, p = {p1:.6f}")

r_le6 = [int(np.sum(groups[c] <= 6)) for c in conds]
z2, p2 = cochran_armitage(conds, n_list, r_le6)
print("\nAnteil <= 6 Punkte:")
for c, n, r in zip(conds, n_list, r_le6):
    print(f"  {c} Punkte: {r}/{n} = {100*r/n:.1f}%")
print(f"  Cochran-Armitage: z = {z2:.4f}, p = {p2:.6f}")
