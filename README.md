# Kapitel-4-Auswertung: H1-H4 (VS Code)

Diese vier Skripte prüfen die Hypothesen H1-H4 auf Basis der finalen, bereinigten
Analysestichproben (Preisexperiment N=246, Bewertungsexperiment N=295), exakt
nach den in Abschnitt 3.2.3/3.3.3/3.4 des Papers definierten Ausschlusskriterien.

## Setup

```bash
pip install -r requirements.txt
```

Alle Modelle (OLS, logistische Regression per
Newton-Raphson/IRLS, Bootstrap-Mediation) sind in `regression_utils.py` von Hand
mit NumPy/SciPy implementiert und wurden gegen `sklearn` gegengeprüft (identische
Koeffizienten). 
## Skripte

- `h1_analysis.py` — H1: umgekehrt U-förmiger Zusammenhang (quadratische
  Regression) für beide Experimente.
- `h2_analysis.py` — H2: Mediationsanalyse Preisexperiment (paralleles
  Mehrfachmediatormodell, Aufwärtsphase via Konzession, Abwärtsphase via
  Boomerang-Variable), Bootstrap-KI (5000 Replikationen, BC).
- `h3_analysis.py` — H3: Mediationsanalyse Bewertungsexperiment (Forderung →
  Reaktanz/Boomerang-Variable → vergebene Punktzahl), Bootstrap-KI.
- `h4_analysis.py` — H4: Fairness → Reaktanz, plus die in 3.3.3 angekündigte
  Trennschärfe-Prüfung Item 5 (Angemessenheit_invertiert) vs. Item 6 (Fairness).

- `create_figures.py` — erzeugt Abbildung 6-10 (siehe unten) als PNG (300dpi)
  und Vektor-PDF im Ordner `figures/`, direkt einsetzbar in Word.

Jedes Skript einfach direkt ausführen, z. B.:

```bash
python3 h1_analysis.py
python3 create_figures.py
```

## Abbildungen

`create_figures.py` erzeugt fünf Grafiken (Zählung schließt an die bestehenden
Abbildungen 1-5 im Paper an):

- **Abbildung 6** — Preisexperiment: Annahmequote je Erstforderung mit
  angepasster quadratischer Kurve (H1). Kontrollbedingung (849 €) bewusst
  nicht massstabsgetreu, sondern durch eine gestrichelte Trennlinie separiert
  dargestellt, da sie nicht Teil der Kurve ist.
- **Abbildung 7** — Bewertungsexperiment: vergebene Punktzahl je Forderung
  mit angepasster quadratischer Kurve (H1).
- **Abbildung 8** — Preisexperiment: Forest-Plot der drei indirekten Effekte
  aus dem Mehrfachmediatormodell (H2), mit 95%-BC-Bootstrap-KI.
- **Abbildung 9** — Bewertungsexperiment: Pfaddiagramm des Mediationsmodells
  (H3) mit a-, b-, c'- und c-Koeffizienten.
- **Abbildung 10** — Bewertungsexperiment: Streudiagramm Fairness vs.
  Reaktanz mit Regressionsgerade (H4).

Farben und Stil sind bewusst gedeckt gehalten (dezentes Blau/Rot, dünne
Gitterlinien) und in Graustufen noch unterscheidbar (unterschiedliche
Markerformen/Linienstile für Kontroll- vs. Treatmentpunkte). PNG für die
Einbindung in Word, PDF als Vektor-Backup falls später Nachbearbeitung nötig
ist (z. B. Beschriftungsgröße).

## Daten

`data/Auswertung_Preisexperiment.csv` und `data/Auswertung_Bewertungsexperiment.csv`
sind die finalen, bereits mit den Ausschluss-/Manipulationscheck-Spalten
angereicherten Exporte (Snapshot vom 27.08.2026). Wenn du die CSVs später erneut
aktualisierst, einfach hier ersetzen (Spaltennamen müssen gleich bleiben) — die
Skripte filtern die Analysestichprobe selbst über die
`Manipulationscheck_*`-Spalten, du musst also nicht vorher von Hand filtern.

## Ergebnisse (Snapshot, siehe Chat für die volle Einordnung)

- **H1**: In keinem der beiden Experimente ist der quadratische Term
  signifikant → umgekehrt U-förmiger Verlauf statistisch nicht absicherbar.
- **H2**: Boomerang-Variable (Abwärtsphase) vermittelt signifikant
  (BC-KI schließt 0 nicht ein), Konzession (Aufwärtsphase) und Kontrasteffekt
  nicht signifikant → H2 nur teilweise bestätigt, wie schon vor der
  Datenbereinigung.
- **H3**: Anders als in der alten (vor-bereinigten) Präsentation ist der
  indirekte Effekt über die Reaktanz-Variable jetzt signifikant und
  erwartungskonform negativ, wird aber vom positiven direkten Effekt
  überkompensiert (inkonsistente Mediation/Suppression) — sauber im Text zu
  erklären, nicht einfach als "signifikant" verkaufen.
- **H4**: Deutlich stärkster Befund — Fairness sagt Reaktanz klar negativ
  vorher (R²≈.38, p<.001). Item 5 und Item 6 korrelieren moderat (r≈-.66),
  sind aber empirisch unterscheidbar (Trennschärfe gegeben).
