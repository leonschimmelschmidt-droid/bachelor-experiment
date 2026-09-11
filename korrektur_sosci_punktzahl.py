"""
Korrektur der Bewertungsvariable aus der SoSci-Phase, Bewertungsexperiment.

Hintergrund
-----------
Die Frage NV08 ("Welche endgueltige Punktzahl wuerden Sie dem Studierenden
geben?") ist in SoSci Survey als Auswahlfrage angelegt. SoSci speichert bei
Auswahlfragen die *Position* der gewaehlten Antwortoption, nicht den in der
Option angezeigten Wert. Die Optionsliste ist absteigend angeordnet:

    [01] = 15 Punkte  ...  [16] = 0 Punkte

Daraus folgt:  vergebene Punktzahl = 16 - Positionscode

Das urspruengliche Import-Skript hat den Positionscode unveraendert als
Rohwert uebernommen. Betroffen sind ausschliesslich die 138 Faelle mit
Datenquelle == "sosci_import". Die 250 Faelle der Render-Phase wurden ueber
eine eigene Weberhebung erfasst, die den Punktwert direkt speichert, und sind
nicht betroffen. Das Preisexperiment ist ebenfalls nicht betroffen, da dessen
Daten ausschliesslich aus der Render-Phase stammen (vgl. Abschnitt 3.4).

Wirkung
-------
Vor der Korrektur lagen die Mittelwerte beider Erhebungsphasen um mehr als
drei Punkte auseinander (6,35 gegenueber 9,75), und der Zusammenhang zwischen
Forderungshoehe und vergebener Punktzahl hatte in den beiden Phasen
entgegengesetzte Vorzeichen. Nach der Korrektur stimmen die Mittelwerte
nahezu ueberein (9,65 gegenueber 9,75) und beide Phasen zeigen denselben
Zusammenhang.

Die Analysestichprobe bleibt bei N = 295, da die Manipulationschecks
(Rollenkontrolle und erinnerte Forderung) die Bewertungsvariable nicht
heranziehen.

Dieses Skript ist einmalig auszufuehren. Es prueft vorab, ob die Datei bereits
korrigiert wurde, und bricht in diesem Fall ohne Aenderung ab.
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
    print("Anschliessend die Auswertungsskripte erneut ausfuehren.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PFAD)
