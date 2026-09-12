"""
Korrektur der Bewertungsvariable aus der SoSci-Phase, Bewertungsexperiment.
"""

import sys
import pandas as pd

PFAD = "data/Auswertung_Bewertungsexperiment.csv"
SPALTE = "Final_vergebene_Punktzahl"
PHASE = "sosci_import"
OPTIONEN = 16  # Anzahl der Antwortoptionen von NV08: 15 bis 0 Punkte


def main(pfad: str = PFAD) -> None:
    df = pd.read_csv(pfad, sep=";", decimal=",")
    maske = df["Datenquelle"] == PHASE

    if not maske.any():
        sys.exit("Keine Faelle der SoSci-Phase gefunden. Abbruch.")

    vorher = df.loc[maske, SPALTE].mean()
    render = df.loc[~maske, SPALTE].mean()

    # Schutz gegen doppelte Ausfuehrung: nach der Korrektur liegen die
    # Mittelwerte beider Phasen dicht beieinander.
    if abs(vorher - render) < 1.0:
        sys.exit(
            f"Die Datei scheint bereits korrigiert zu sein "
            f"(SoSci {vorher:.2f} gegenueber Render {render:.2f}). Keine Aenderung."
        )

    df.loc[maske, SPALTE] = OPTIONEN - df.loc[maske, SPALTE]
    nachher = df.loc[maske, SPALTE].mean()

    df.to_csv(pfad, sep=";", decimal=",", index=False)

    print(f"Korrigierte Faelle:            {int(maske.sum())}")
    print(f"Mittelwert SoSci vorher:       {vorher:.2f}")
    print(f"Mittelwert SoSci nachher:      {nachher:.2f}")
    print(f"Mittelwert Render (Referenz):  {render:.2f}")
    print(f"Spanne gesamt nachher:         {df[SPALTE].min():.0f} bis {df[SPALTE].max():.0f}")
    print(f"\nDatei geschrieben: {pfad}")
    


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PFAD)
