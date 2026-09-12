"""
Robustheitscheck zu H1: getrennte Schaetzung nach Erhebungsphase,
Bewertungsexperiment.

Hintergrund
-----------
Teil 1 schaetzt die H1-Kurve getrennt fuer beide Erhebungsphasen. Die
Render-Phase wurde ueber eine eigene Weberhebung erfasst, die den Punktwert
direkt speichert. Sie ist von der Rueckrechnung nicht betroffen und liefert
damit einen von ihr unabhaengigen Test.

Teil 2 macht transparent, warum die SoSci-Teilstichprobe diesen Test NICHT
leisten kann: Eine Umkehrung der Kodierung kehrt auch das Vorzeichen des
quadratischen Terms um, bei identischem p-Wert. Die Teilstichprobe ist deshalb
zur Absicherung der Rueckrechnung ungeeignet und wird im Paper ausdruecklich
nur als Konsistenzpruefung berichtet.

Teil 3 zeigt, welches Argument welchen Teil der Rueckrechnung traegt. Das
Vorzeichen der Korrelation mit der wahrgenommenen Angemessenheit klaert die
Richtung (auf- oder absteigende Optionsliste), die Konvergenz der Mittelwerte
klaert den Bezugswert (16). Eine beliebige absteigende Umrechnung wuerde das
Vorzeichen ebenfalls reparieren, aber die Mittelwerte auseinandertreiben.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from regression_utils import find_data_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PFAD = find_data_file(SCRIPT_DIR, "Auswertung_Bewertungsexperiment.csv")
OPTIONEN = 16  # Antwortoptionen von NV08: 15 bis 0 Punkte


def ols(y, spalten, namen):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(s, float) for s in spalten])
    y = np.asarray(y, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * s2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    return {"namen": ["Intercept"] + namen, "beta": beta, "se": se,
            "t": t, "p": p, "r2": r2, "n": n}


def quadratisch(x, y):
    xz = x - x.mean()
    erg = ols(y, [xz, xz ** 2], ["Forderung (zentriert)", "Forderung^2"])
    b1, b2 = erg["beta"][1], erg["beta"][2]
    erg["scheitel"] = x.mean() - b1 / (2 * b2) if b2 != 0 else float("nan")
    return erg


def analysestichprobe():
    df = pd.read_csv(PFAD, sep=";", decimal=",")
    return df[(df["Manipulationscheck_Rolle"] == "Ja")
              & (df["Manipulationscheck_Forderung"] == "Ja")]


def main():
    bf = analysestichprobe()

    print("=" * 78)
    print("TEIL 1  H1 getrennt nach Erhebungsphase")
    print("=" * 78)
    print(f"  {'Phase':<16} {'N':>4} {'b1':>9} {'p(b1)':>8} {'b2':>9} {'p(b2)':>8} {'R2':>7} {'Scheitel':>9}")
    for phase, bezeichnung in [("render", "Render"), ("sosci_import", "SoSci")]:
        g = bf[bf["Datenquelle"] == phase]
        x = g["Forderung_des_Studierenden"].to_numpy(float)
        y = g["Final_vergebene_Punktzahl"].to_numpy(float)
        e = quadratisch(x, y)
        print(f"  {bezeichnung:<16} {e['n']:>4} {e['beta'][1]:>9.4f} {e['p'][1]:>8.4f} "
              f"{e['beta'][2]:>9.4f} {e['p'][2]:>8.4f} {e['r2']:>7.3f} {e['scheitel']:>9.2f}")
    print("\n  Die Render-Phase ist von der Rueckrechnung nicht betroffen.")
    print("  Der quadratische Term ist dort signifikant negativ. H1 steht damit")
    print("  unabhaengig von der Rueckrechnung.")

    print()
    print("=" * 78)
    print("TEIL 2  Warum die SoSci-Teilstichprobe die Rueckrechnung nicht absichert")
    print("=" * 78)
    g = bf[bf["Datenquelle"] == "sosci_import"]
    x = g["Forderung_des_Studierenden"].to_numpy(float)
    y_korr = g["Final_vergebene_Punktzahl"].to_numpy(float)
    rohcode = OPTIONEN - y_korr
    varianten = [
        ("16 - Code (adoptiert)", y_korr),
        ("Rohcode (unkorrigiert)", rohcode),
        ("Code - 1 (Gegenannahme)", rohcode - 1),
        ("20 - Code (falscher Bezug)", 20 - rohcode),
    ]
    print(f"  {'Annahme':<28} {'b2':>9} {'p(b2)':>8} {'Mittelwert':>12}")
    for name, y in varianten:
        e = quadratisch(x, y)
        print(f"  {name:<28} {e['beta'][2]:>9.4f} {e['p'][2]:>8.4f} {y.mean():>12.2f}")
    print("\n  Eine Umkehrung der Kodierung dreht das Vorzeichen von b2 mit, bei")
    print("  identischem p-Wert. Die SoSci-Teilstichprobe ist deshalb eine")
    print("  Konsistenzpruefung, kein unabhaengiger Test.")

    print()
    print("=" * 78)
    print("TEIL 3  Welches Argument traegt welchen Teil der Rueckrechnung")
    print("=" * 78)
    render = bf[bf["Datenquelle"] == "render"]
    r_render = np.corrcoef(render["Final_vergebene_Punktzahl"].to_numpy(float),
                           render["Wahrgenommene_Angemessenheit"].to_numpy(float))[0, 1]
    ang = g["Wahrgenommene_Angemessenheit"].to_numpy(float)
    print("  Richtung der Optionsliste, Korrelation Punktzahl x Angemessenheit:")
    print(f"    Render (Referenz, nicht umgerechnet) r = {r_render:+.3f}")
    for name, y in varianten:
        print(f"    SoSci, {name:<28} r = {np.corrcoef(y, ang)[0, 1]:+.3f}")
    print("\n    Jede absteigende Umrechnung repariert das Vorzeichen. Das Argument")
    print("    klaert damit die Richtung, nicht den Bezugswert.")
    # Fuer den Mittelwertvergleich alle erhobenen Faelle, wie in Abschnitt 3.4
    alle = pd.read_csv(PFAD, sep=";", decimal=",")
    a_render = alle[alle["Datenquelle"] == "render"]["Final_vergebene_Punktzahl"].to_numpy(float)
    a_sosci = alle[alle["Datenquelle"] == "sosci_import"]["Final_vergebene_Punktzahl"].to_numpy(float)
    a_roh = OPTIONEN - a_sosci
    print("\n  Bezugswert, Mittelwerte beider Phasen (alle erhobenen Faelle):")
    print(f"    Render (Referenz)                          {a_render.mean():>8.2f}")
    for name, y in [("16 - Code (adoptiert)", a_sosci),
                    ("Rohcode (unkorrigiert)", a_roh),
                    ("Code - 1 (Gegenannahme)", a_roh - 1),
                    ("20 - Code (falscher Bezug)", 20 - a_roh)]:
        print(f"    SoSci, {name:<28} {y.mean():>8.2f}")
    print("\n    Nur mit 16 als Bezugswert fallen die Mittelwerte zusammen.")
    print("    Beide Argumente zusammen legen die Rueckrechnung eindeutig fest.")
 


if __name__ == "__main__":
    main()
