"""
Sensitivitaetsanalyse zur Teststaerke des quadratischen Terms, Preisexperiment.

Vorgehen
--------
Fuer eine Reihe angenommener wahrer Werte von beta2 wird die tatsaechliche
Designmatrix des Preisexperiments verwendet, also die beobachteten
Treatmentstufen und Zellbesetzungen. Der Intercept wird jeweils so kalibriert,
dass die mittlere Annahmequote der Stichprobe erhalten bleibt; der lineare Term
wird auf null gesetzt, sodass allein die Kruemmung variiert. Aus dem so
festgelegten Modell werden binaere Ergebnisse gezogen, das Modell wird erneut
geschaetzt, und es wird gezaehlt, wie oft der quadratische Term den einseitigen
Test gegen H0: beta2 >= 0 besteht. Der Anteil ist die Teststaerke.

Zusaetzlich wird die Kruemmung in eine interpretierbare Groesse uebersetzt: die
Spannweite der modellierten Annahmequote zwischen Scheitelpunkt und Rand des
getesteten Bereichs.

Wie in den uebrigen Skripten geht die Kontrollbedingung (849 Euro) nicht in die
Schaetzung ein.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from regression_utils import find_data_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PFAD = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")

ALPHA = 0.05          # einseitig, entsprechend H0: beta2 >= 0 in Abschnitt 4.2
ZIEL_POWER = 0.80
REPLIKATIONEN = 4000
SEED = 20260915

GITTER = [-0.010, -0.020, -0.030, -0.040, -0.050, -0.055,
          -0.060, -0.070, -0.080, -0.100, -0.120]


def logistische_regression(y, spalten, maxiter=100, toleranz=1e-10):
    """Logistische Regression per Newton-Raphson/IRLS, analog regression_utils."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(s, float) for s in spalten])
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(maxiter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        neu = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(neu - beta)) < toleranz:
            beta = neu
            break
        beta = neu
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * w) @ X)))
    return beta, se


def kalibrierter_intercept(xz, beta2, zielquote, schritte=80):
    """Intercept so waehlen, dass die mittlere Annahmewahrscheinlichkeit der
    beobachteten Annahmequote entspricht."""
    b0 = np.log(zielquote / (1.0 - zielquote))
    for _ in range(schritte):
        eta = b0 + beta2 * xz ** 2
        b0 -= (1.0 / (1.0 + np.exp(-eta))).mean() - zielquote
    return b0


def wahrscheinlichkeiten(xz, beta2, zielquote):
    b0 = kalibrierter_intercept(xz, beta2, zielquote)
    return 1.0 / (1.0 + np.exp(-(b0 + beta2 * xz ** 2)))


def teststaerke(xz, beta2, zielquote, rng, replikationen=REPLIKATIONEN):
    pr = wahrscheinlichkeiten(xz, beta2, zielquote)
    n = len(xz)
    treffer = 0
    gueltig = 0
    for _ in range(replikationen):
        y = (rng.random(n) < pr).astype(float)
        if y.sum() == 0 or y.sum() == n:
            continue
        try:
            beta, se = logistische_regression(y, [xz, xz ** 2])
        except np.linalg.LinAlgError:
            continue
        gueltig += 1
        if beta[2] < 0 and stats.norm.cdf(beta[2] / se[2]) < ALPHA:
            treffer += 1
    return treffer / gueltig if gueltig else float("nan")


def main():
    df = pd.read_csv(PFAD, sep=";", decimal=",")
    analyse = df[df["Codierung"].notna()
                 & (df["Manipulationscheck_Preis"] == "Ja")
                 & (df["Manipulationscheck_Validitaet"] == "Ja")]
    treat = analyse[analyse["Treatment"] != 849]

    x = (treat["Treatment"] / 1000.0).to_numpy(float)
    xz = x - x.mean()
    y = treat["Codierung"].to_numpy(float)
    zielquote = y.mean()

    print("Designgrundlage")
    print(f"  Faelle (ohne Kontrollbedingung)   N = {len(x)}")
    stufen = sorted(treat["Treatment"].unique())
    print("  Zellbesetzung                     "
          + ", ".join(f"{int(t)} EUR: {int((treat['Treatment'] == t).sum())}" for t in stufen))
    print(f"  Annahmequote                      {zielquote:.3f}")
    print(f"  Test                              einseitig gegen H0: beta2 >= 0, alpha = {ALPHA}")
    print(f"  Replikationen je Gitterpunkt      {REPLIKATIONEN}")

    beobachtet, se_beob = logistische_regression(y, [xz, xz ** 2])
    print(f"\n  Beobachtet im Paper (Tabelle 9)   beta2 = {beobachtet[2]:+.4f} "
          f"(SE {se_beob[2]:.4f})")

    rng = np.random.default_rng(SEED)
    print("\nTeststaerke je angenommenem wahrem quadratischem Term")
    print(f"  {'beta2':>8} {'Power':>8} {'Scheitel':>10} {'Rand':>8} {'Abfall':>10}")
    ergebnisse = []
    for b2 in GITTER:
        pw = teststaerke(xz, b2, zielquote, rng)
        pr = wahrscheinlichkeiten(xz, b2, zielquote)
        ergebnisse.append((b2, pw))
        print(f"  {b2:>8.3f} {pw:>8.2f} {pr.max()*100:>9.0f}% {pr.min()*100:>7.0f}% "
              f"{(pr.max()-pr.min())*100:>8.1f} Pp")

    # Schwelle linear zwischen den beiden umschliessenden Gitterpunkten interpolieren
    schwelle = None
    for (b_u, p_u), (b_o, p_o) in zip(ergebnisse, ergebnisse[1:]):
        if p_u < ZIEL_POWER <= p_o:
            schwelle = b_u + (ZIEL_POWER - p_u) * (b_o - b_u) / (p_o - p_u)
            break

    if schwelle is not None:
        pr = wahrscheinlichkeiten(xz, schwelle, zielquote)
        print(f"\nErgebnis")
        print(f"  Ein quadratischer Term wird erst ab etwa beta2 = {schwelle:.3f} mit")
        print(f"  {ZIEL_POWER*100:.0f} Prozent Wahrscheinlichkeit signifikant. Das entspricht einem")
        print(f"  Abfall der modellierten Annahmequote von {pr.max()*100:.0f} Prozent am Scheitelpunkt")
        print(f"  auf {pr.min()*100:.0f} Prozent am Rand des getesteten Bereichs, also rund")
        print(f"  {(pr.max()-pr.min())*100:.0f} Prozentpunkten.")
       
    else:
        print(f"\n  Die Zielpower von {ZIEL_POWER:.0%} liegt ausserhalb des gewaehlten Gitters.")


if __name__ == "__main__":
    main()
