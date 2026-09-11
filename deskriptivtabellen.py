import csv
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")
BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")


def to_float(s):
    return float(s.strip().replace(",", "."))


def fmt(m, sd):
    return f"{m:.2f} ({sd:.2f})"


# ---------------------------------------------------------------------------
# Preisexperiment
# ---------------------------------------------------------------------------
with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]
print(f"Preisexperiment, finale Analysestichprobe: N = {len(final)} (Soll: 246)")

cols_preis = ["Codierung", "Wahrnehmungskontrast", "Konzession", "Boomerang_Variable", "Angemessenheit_invertiert"]

print("\n" + "=" * 100)
print(f"{'Treatment':<12}{'N':>5}   " + "   ".join(f"{c:<28}" for c in cols_preis))
print("=" * 100)
for cond in sorted(set(int(r["Treatment"]) for r in final)):
    grp = [r for r in final if int(r["Treatment"]) == cond]
    n = len(grp)
    line = f"{cond:<12}{n:>5}   "
    for c in cols_preis:
        vals = np.array([to_float(r[c]) for r in grp])
        line += f"{fmt(vals.mean(), vals.std(ddof=1)):<28}   "
    print(line)

# Gesamt (alle Bedingungen gepoolt)
n = len(final)
line = f"{'Gesamt':<12}{n:>5}   "
for c in cols_preis:
    vals = np.array([to_float(r[c]) for r in final])
    line += f"{fmt(vals.mean(), vals.std(ddof=1)):<28}   "
print("-" * 100)
print(line)

# ---------------------------------------------------------------------------
# Bewertungsexperiment
# ---------------------------------------------------------------------------
with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows2 = list(csv.DictReader(f, delimiter=";"))

final2 = [r for r in rows2
          if r["Manipulationscheck_Rolle"].strip() == "Ja"
          and r["Manipulationscheck_Forderung"].strip() == "Ja"]
print(f"\n\nBewertungsexperiment, finale Analysestichprobe: N = {len(final2)} (Soll: 295)")

cols_bewert = ["Final_vergebene_Punktzahl", "Boomerang_Variable", "Wahrgenommene_Fairness", "Bedrohungswahrnehmung_Index"]

print("\n" + "=" * 100)
print(f"{'Forderung':<12}{'N':>5}   " + "   ".join(f"{c:<28}" for c in cols_bewert))
print("=" * 100)
for cond in sorted(set(int(to_float(r["Forderung_des_Studierenden"])) for r in final2)):
    grp = [r for r in final2 if int(to_float(r["Forderung_des_Studierenden"])) == cond]
    n = len(grp)
    line = f"{cond:<12}{n:>5}   "
    for c in cols_bewert:
        vals = np.array([to_float(r[c]) for r in grp])
        line += f"{fmt(vals.mean(), vals.std(ddof=1)):<28}   "
    print(line)

n = len(final2)
line = f"{'Gesamt':<12}{n:>5}   "
for c in cols_bewert:
    vals = np.array([to_float(r[c]) for r in final2])
    line += f"{fmt(vals.mean(), vals.std(ddof=1)):<28}   "
print("-" * 100)
print(line)
