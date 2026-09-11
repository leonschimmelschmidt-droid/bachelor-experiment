"""
Frage: Bringt eine ueberhoehte Erstforderung mit anschliessender Preissenkung
(Treatmentbedingungen, 1.500-10.000 Euro) ueberhaupt einen Zugewinn gegenueber
der direkten Nennung der Zielforderung (Kontrollbedingung, 849 Euro)?

Das ist eine eigenstaendige Frage, unabhaengig vom Kurvenverlauf (H1) und der
Mediationsanalyse (H2), die beide die Kontrollbedingung ausschliessen (vgl.
Abschnitt 3.1: die Kontrolle ist keine weitere Stufe desselben Verlaufs).

Test: 2x2-Kontingenztafel (Annahme ja/nein x Kontrolle/gepoolte Treatments),
Chi2-Test mit Yates-Korrektur (Standardtest bei 2x2-Tafeln) sowie ergaenzend
Fisher-Exact-Test (praeziser bei den hier vorliegenden, nicht sehr grossen
Zellenhaeufigkeiten).
"""

import csv
import numpy as np
from scipy import stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from regression_utils import find_data_file

PREIS_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")

with open(PREIS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Codierung"].strip() != ""
         and r["Manipulationscheck_Preis"].strip() == "Ja"
         and r["Manipulationscheck_Validitaet"].strip() == "Ja"]
print(f"Finale Analysestichprobe gesamt: N = {len(final)} (Soll: 246)")

kontrolle = [r for r in final if r["Treatment"] == "849"]
treat = [r for r in final if r["Treatment"] != "849"]

k_annahme = sum(int(r["Codierung"]) for r in kontrolle)
t_annahme = sum(int(r["Codierung"]) for r in treat)

print(f"\nKontrollbedingung (849 EUR, direkt): N = {len(kontrolle)}, "
      f"Annahmen = {k_annahme}, Quote = {k_annahme/len(kontrolle):.4f}")
print(f"Gepoolte Treatmentbedingungen (1.500-10.000 EUR): N = {len(treat)}, "
      f"Annahmen = {t_annahme}, Quote = {t_annahme/len(treat):.4f}")

table = [[k_annahme, len(kontrolle) - k_annahme],
         [t_annahme, len(treat) - t_annahme]]
print(f"\nKontingenztafel (Annahme/Ablehnung x Kontrolle/Treatment): {table}")

chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=True)
print(f"\nChi2-Test (Yates-Korrektur): chi2({dof}) = {chi2:.4f}, p = {p_chi2:.4f}")
print(f"Erwartete Haeufigkeiten (min. Zelle sollte > 5 sein): {expected.flatten()}")

oddsratio, p_fisher = stats.fisher_exact(table)
print(f"Fisher-Exact-Test: OR = {oddsratio:.4f}, p = {p_fisher:.4f}")
