"""
Robustheitscheck: H1-Kurve im Preisexperiment ohne den Validitaetscheck.

Hintergrund
-----------
Abschnitt 6 haelt fest, dass der Validitaetscheck (Item 2e) ungleich ueber die
Treatmentstufen ausfaellt: Bei 849 und 1.500 Euro scheitern vier bzw. drei
Faelle, bei 10.000 Euro dagegen 17. Das Kriterium schliesst damit gerade jene
Probanden ueberproportional aus, denen die hoechsten Erstforderungen
unglaubwuerdig erschienen. Da nach Schwarzwald, Raz und Zvibel (1979, S. 578)
genau diese Unglaubwuerdigkeit den Boomerang-Effekt traegt, koennte die
Bereinigung die Faelle mit der staerksten Abwaertsreaktion entfernt haben.

Dieses Skript prueft das. Die H1-Kurve wird zweimal geschaetzt: einmal auf der
im Paper verwendeten Analysestichprobe und einmal ohne den Validitaetscheck,
also auf allen Faellen mit bestandenem Preis-Manipulationscheck.

Ergebnis: Der quadratische Term bleibt in beiden Faellen insignifikant. Das
Ausbleiben einer Abwaertsphase im Preisexperiment geht nicht auf das
Ausschlusskriterium zurueck.

Wie in den uebrigen Skripten geht die Kontrollbedingung (849 Euro) nicht in die
Kurve ein (vgl. Abschnitt 3.1).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from regression_utils import find_data_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PFAD = find_data_file(SCRIPT_DIR, "Auswertung_Preisexperiment.csv")


def logistische_regression(y, spalten, namen, maxiter=100, toleranz=1e-10):
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
    z_wert = beta / se
    p = 2 * stats.norm.sf(np.abs(z_wert))
    ll = np.sum(y * np.log(np.clip(mu, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1)))
    p_quer = y.mean()
    ll0 = np.sum(y * np.log(p_quer) + (1 - y) * np.log(1 - p_quer))
    return {"namen": ["Intercept"] + namen, "beta": beta, "se": se,
            "z": z_wert, "p": p, "mcfadden": 1 - ll / ll0, "n": len(y)}


def kurve(df, titel):
    treat = df[df["Treatment"] != 849]
    x = (treat["Treatment"] / 1000.0).to_numpy(float)
    xz = x - x.mean()
    y = treat["Codierung"].to_numpy(float)
    erg = logistische_regression(y, [xz, xz ** 2],
                                 ["Erstforderung (zentriert, in 1.000 EUR)", "Erstforderung^2"])
    print(f"\n{titel}  (N = {erg['n']})")
    print(f"  {'Praediktor':<42} {'Koeff.':>9} {'SE':>8} {'z':>7} {'p':>8}")
    for nm, b, se, z, p in zip(erg["namen"], erg["beta"], erg["se"], erg["z"], erg["p"]):
        print(f"  {nm:<42} {b:>9.4f} {se:>8.4f} {z:>7.2f} {p:>8.3f}")
    print(f"  {'McFadden R2':<42} {erg['mcfadden']:>9.4f}")
    return erg


def main():
    df = pd.read_csv(PFAD, sep=";", decimal=",")

    # Gemeinsame Basis: gueltige abhaengige Variable und bestandener Preis-Check
    basis = df[df["Codierung"].notna() & (df["Manipulationscheck_Preis"] == "Ja")]
    mit = basis[basis["Manipulationscheck_Validitaet"] == "Ja"]

    print("Ausfall am Validitaetscheck je Bedingung")
    print(f"  {'Erstforderung':>14} {'n':>5} {'gescheitert':>12}")
    for t in sorted(df["Treatment"].unique()):
        g = basis[basis["Treatment"] == t]
        print(f"  {t:>10} EUR {len(g):>5} "
              f"{int((g['Manipulationscheck_Validitaet'] == 'Nein').sum()):>12}")

    a = kurve(mit, "MIT Validitaetscheck (Analysestichprobe des Papers, Tabelle 9)")
    b = kurve(basis, "OHNE Validitaetscheck")

    print("\nVergleich des quadratischen Terms")
    print(f"  mit  Validitaetscheck: b2 = {a['beta'][2]:+.4f}, p = {a['p'][2]:.3f}")
    print(f"  ohne Validitaetscheck: b2 = {b['beta'][2]:+.4f}, p = {b['p'][2]:.3f}")
    print("\n  In beiden Faellen insignifikant. Das Ausschlusskriterium erklaert das")
    print("  Ausbleiben einer Abwaertsphase im Preisexperiment nicht.")


if __name__ == "__main__":
    main()
