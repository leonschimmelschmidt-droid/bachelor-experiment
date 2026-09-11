"""
H4-Robustheitspruefung: Regression der Boomerang-Variable auf die wahrgenommene
Fairness, mit und ohne Kontrolle fuer die Forderungshoehe.

Hintergrund: Wahrgenommene Fairness und Reaktanz gehen beide auf die manipulierte
Forderungshoehe zurueck. 
"""

import csv
import math
import os

import numpy as np

PFAD = "Auswertung_Bewertungsexperiment.csv"


# --------------------------------------------------------------- Hilfsfunktionen
def zu_zahl(text):
    """Deutsches Zahlenformat ('2,5') in float. Leere Felder -> None."""
    text = text.strip()
    if text == "":
        return None
    return float(text.replace(",", "."))


def betacf(a, b, x, iterationen=200, eps=3e-16):
    """Kettenbruch fuer die unvollstaendige Betafunktion (Lentz-Algorithmus)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, iterationen + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularisierte unvollstaendige Betafunktion I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    vorfaktor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return vorfaktor * betacf(a, b, x) / a
    return 1.0 - vorfaktor * betacf(b, a, 1.0 - x) / b


def p_wert(t, df):
    """Zweiseitiger p-Wert der t-Verteilung."""
    return betai(0.5 * df, 0.5, df / (df + t * t))


def p_formatiert(p):
    return "<,001" if p < 0.001 else f"{p:.3f}".replace("0.", ",")


# -------------------------------------------------------------------- Daten laden
with open(PFAD, encoding="utf-8-sig", newline="") as datei:
    zeilen = list(csv.DictReader(datei, delimiter=";"))

# Analysestichprobe nach Abschnitt 3.3.3: Rollencheck und Forderungscheck
# muessen beide bestanden sein.
faelle = [z for z in zeilen
          if z["Manipulationscheck_Rolle"].strip() == "Ja"
          and z["Manipulationscheck_Forderung"].strip() == "Ja"]

boomerang = np.array([zu_zahl(z["Boomerang_Variable"]) for z in faelle])
fairness = np.array([zu_zahl(z["Wahrgenommene_Fairness"]) for z in faelle])
forderung = np.array([zu_zahl(z["Forderung_des_Studierenden"]) for z in faelle])

print(f"Analysestichprobe: N = {len(faelle)}\n")

# Praediktoren zentrieren, wie in den uebrigen Tabellen der Arbeit
fairness_z = fairness - fairness.mean()
forderung_z = forderung - forderung.mean()


# ----------------------------------------------------------------------- OLS
def ols(y, praediktoren, beschriftungen, titel):
    """Kleinste-Quadrate-Schaetzung mit Standardfehlern, t- und p-Werten."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + list(praediktoren))
    k = X.shape[1]

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuen = y - X @ beta
    df = n - k
    sigma2 = residuen @ residuen / df
    kovarianz = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(kovarianz))
    t = beta / se

    r2 = 1.0 - (residuen @ residuen) / ((y - y.mean()) @ (y - y.mean()))

    print(titel)
    print(f"  {'Praediktor':<26}{'Koeff.':>9}{'SE':>8}{'t':>9}{'p':>10}")
    for name, b_, se_, t_ in zip(["Intercept"] + beschriftungen, beta, se, t):
        print(f"  {name:<26}{b_:9.3f}{se_:8.3f}{t_:9.2f}"
              f"{p_formatiert(p_wert(t_, df)):>10}")
    print(f"  {'Beobachtungen (N)':<26}{n:9d}")
    print(f"  {'R2':<26}{r2:9.4f}\n")
    return beta, r2


# ------------------------------------------- Modell 1: Replikation Tabelle 16
beta1, r2_1 = ols(boomerang, [fairness_z],
                  ["Fairness (zentriert)"],
                  "Modell 1 - nur Fairness (entspricht Tabelle 16)")

# ------------------------- Modell 2: mit Forderungshoehe als Kontrollvariable
beta2, r2_2 = ols(boomerang, [fairness_z, forderung_z],
                  ["Fairness (zentriert)", "Forderung (zentriert)"],
                  "Modell 2 - mit Kontrolle fuer die Forderungshoehe (Tabelle 18)")

# ------------------------------------------------------------- Zusatzkennwerte
print(f"Delta R2 (Modell 2 - Modell 1): {r2_2 - r2_1:.4f}")
print("Korrelation Fairness / Forderungshoehe: "
      f"r = {np.corrcoef(fairness, forderung)[0, 1]:.3f}")
print(f"Aenderung des Fairness-Koeffizienten: {beta1[1]:.3f} -> {beta2[1]:.3f}")
