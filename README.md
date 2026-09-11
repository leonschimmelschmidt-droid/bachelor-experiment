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
- `h2_ergaenzung_item2a.py` - Ergaenzende Mediationsanalyse H2, allein mit Item 2a (Angemessenheit_invertiert) 
- `h3_analysis.py` — H3: Mediationsanalyse Bewertungsexperiment (Forderung →
  Reaktanz/Boomerang-Variable → vergebene Punktzahl), Bootstrap-KI.
- `h4_analysis.py` — H4: Fairness → Reaktanz, plus die in 3.3.3 angekündigte
  Trennschärfe-Prüfung Item 5 (Angemessenheit_invertiert) vs. Item 6 (Fairness).
- `h4_kontrolle_forderungshoehe.py` - H4-Robustheitspruefung: Regression der Boomerang-Variable auf die wahrgenommene
Fairness, mit und ohne Kontrolle fuer die Forderungshoehe.

- `deskriptivtabellen.py` - Deskriptivtabellen (N, M, SD je Bedingung) fuer beide Experimente. Verwendet jeweils dieselbe finale Analysestichprobe wie die Haupttests (H1/H2 bzw. H3/H4): Preisexperiment N = 246 (alle 6 Bedingungen inkl. Kontrolle), Bewertungsexperiment N = 295 (alle 4 Bedingungen).
- `kontrollvergleich_preisexperiment.py` - Kontroll-vs.-Treatment-Vergleich, Preisexperiment
- `levene-streuung-bewertungsexperiment.py` - Streuungsanalyse (Levene-Test), Bewertungsexperiment: Ergaenzung zu H1
- `manipulationscheck_angemessenheit.py` - Manipulationscheck: Wahrgenommene Angemessenheit (Item 2a) ~ Erstforderung,
Preisexperiment
- `manipulationscheck_bedrohung.py` - Manipulationscheck: Wahrgenommene Bedrohung (Items 1-3) ~ Forderung
Bewertungsexperiment.
- `quantile_trend_bewertungsexperiment.py`- Empirische Quantile je Bedingung + Cochran-Armitage-Trendtest,
Bewertungsexperiment: ersetzt quantilsregression_bewertungsexperiment.py als
Ergaenzung zur Streuungsanalyse in Abschnitt 4.2 (Levene-Test, vgl.
levene_streuung_bewertungsexperiment.py).
- `robustheitscheck_direktannahmen.py` - Robustheitscheck: H1-Kurve im Preisexperiment,
Direktannahmen der Erstforderung in den Treatmentbedingungen zusaetzlich als
Annahme (Codierung = 1) gewertet, statt sie strukturell auszuschliessen.


- `create_figures.py` — erzeugt Abbildung 6, 7, 9, 10 und 11 (siehe unten) als PNG (300dpi)
  und Vektor-PDF im Ordner `figures/`.
  

## Abbildungen

`create_figures.py` erzeugt fünf Grafiken (Zählung schließt an die bestehenden
Abbildungen 1-5 im Paper an):

- **Abbildung 6** — Preisexperiment: Annahmequote je Erstforderung mit
  angepasster quadratischer Kurve (H1). Kontrollbedingung (849 €) bewusst
  nicht massstabsgetreu, sondern durch eine gestrichelte Trennlinie separiert
  dargestellt, da sie nicht Teil der Kurve ist.
- **Abbildung 7** — Bewertungsexperiment: vergebene Punktzahl je Forderung
  mit angepasster quadratischer Kurve (H1).
- **Abbildung 9** — Preisexperiment: Forest-Plot der drei indirekten Effekte
  aus dem Mehrfachmediatormodell (H2), mit 95%-BC-Bootstrap-KI.
- **Abbildung 10** — Bewertungsexperiment: Pfaddiagramm des Mediationsmodells
  (H3) mit a-, b-, c'- und c-Koeffizienten.
- **Abbildung 11** — Bewertungsexperiment: Streudiagramm Fairness vs.
  Reaktanz mit Regressionsgerade (H4).

`figure_streuung_bewertungsexperiment.py` erzeugt **Abbildung 8**

## Daten

`data/Auswertung_Preisexperiment.csv` und `data/Auswertung_Bewertungsexperiment.csv`
sind die finalen Rohdaten. 


