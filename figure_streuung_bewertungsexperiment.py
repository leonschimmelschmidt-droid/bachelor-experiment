"""
Abbildung (neu): Verteilung der vergebenen Punktzahl je Bedingung,
Bewertungsexperiment - Begleitgrafik zur Streuungsanalyse in Abschnitt 4.2
(Levene-Test, vgl. levene_streuung_bewertungsexperiment.py).

Boxplot je Bedingung (Median, IQR, Whisker) mit ueberlagerten, leicht
gejitterten Einzelfaellen, damit sowohl die Zusammenfassung als auch die
tatsaechliche Streuung/Randbesetzung sichtbar wird. Stil konsistent mit
create_figures.py (gedeckte Farben, serifenlose Anmerkungen, 300dpi).
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regression_utils import find_data_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BEWERT_CSV = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 150,
})
MAIN_COLOR = "#2C4A6E"
FIT_COLOR = "#8C2F2F"


def to_float(s):
    return float(s.strip().replace(",", "."))


def de_num(x, decimals=2):
    s = f"{x:.{decimals}f}"
    return s.replace("-", "−").replace(".", ",")


with open(BEWERT_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f, delimiter=";"))

final = [r for r in rows
         if r["Manipulationscheck_Rolle"].strip() == "Ja"
         and r["Manipulationscheck_Forderung"].strip() == "Ja"]

conds = [9, 11, 13, 15]
groups = []
for c in conds:
    vals = np.array([to_float(r["Final_vergebene_Punktzahl"]) for r in final
                      if int(to_float(r["Forderung_des_Studierenden"])) == c])
    groups.append(vals)

fig, ax = plt.subplots(figsize=(6.3, 4.2))

bp = ax.boxplot(groups, positions=conds, widths=1.2, patch_artist=True,
                 showmeans=False, whis=1.5, zorder=3)
for box in bp["boxes"]:
    box.set(facecolor=MAIN_COLOR, alpha=0.18, edgecolor=MAIN_COLOR, lw=1.3)
for median in bp["medians"]:
    median.set(color=FIT_COLOR, lw=2)
for whisker in bp["whiskers"]:
    whisker.set(color="#555555", lw=1.1)
for cap in bp["caps"]:
    cap.set(color="#555555", lw=1.1)
for flier in bp["fliers"]:
    flier.set(marker="", markersize=0)  # Ausreisser stattdessen ueber Jitter-Punkte sichtbar

rng = np.random.default_rng(7)
for c, vals in zip(conds, groups):
    jitter = c + rng.uniform(-0.35, 0.35, size=len(vals))
    ax.scatter(jitter, vals, s=14, color=MAIN_COLOR, alpha=0.35, edgecolors="none", zorder=2)

for c, vals in zip(conds, groups):
    ax.annotate(f"SD={de_num(vals.std(ddof=1))}\nn={len(vals)}", (c, 15.6),
                ha="center", va="bottom", fontsize=8, color="#555555")

ax.set_xlabel("Forderung des Studierenden (Punkte)")
ax.set_ylabel("Final vergebene Punktzahl")
ax.set_xticks(conds)
ax.set_ylim(-0.5, 17.5)
ax.set_yticks(range(0, 16, 3))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "Abbildung_neu_streuung_bewertungsexperiment.png"), bbox_inches="tight", dpi=300)
fig.savefig(os.path.join(OUT_DIR, "Abbildung_neu_streuung_bewertungsexperiment.pdf"), bbox_inches="tight")
plt.close(fig)
print("gespeichert.")
