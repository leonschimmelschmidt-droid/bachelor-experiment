# Kapitel-4-Auswertung: H1-H4 (VS Code)

Diese Skripte prüfen die Hypothesen H1-H4 auf Basis der finalen, bereinigten
Analysestichproben (Preisexperiment N=246, Bewertungsexperiment N=295), exakt
nach den in Abschnitt 3.2.3/3.3.3/3.4 des Papers definierten Ausschlusskriterien.

## Setup

Alle Modelle (OLS, logistische Regression per
Newton-Raphson/IRLS, Bootstrap-Mediation) sind in `regression_utils.py` von Hand
mit NumPy/SciPy implementiert und wurden gegen `sklearn` gegengeprüft (identische
Koeffizienten). 
## Skripte

### Hypothesentests

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

### Deskriptives, Manipulationschecks und Ergänzungen

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

### Robustheitschecks

- `robustheitscheck_direktannahmen.py` - Robustheitscheck: H1-Kurve im Preisexperiment,
Direktannahmen der Erstforderung in den Treatmentbedingungen zusaetzlich als
Annahme (Codierung = 1) gewertet, statt sie strukturell auszuschliessen.
- `robustheitscheck_ohne_validitaetscheck.py` - Robustheitscheck: H1-Kurve im
Preisexperiment ohne den Validitaetscheck (Item 2e) als Ausschlusskriterium.
Prueft die in Abschnitt 6 aufgeworfene Frage, ob das ungleich ueber die
Treatmentstufen verteilte Kriterium die Faelle mit der staerksten
Abwaertsreaktion entfernt hat. Der quadratische Term bleibt mit und ohne
Kriterium insignifikant (b2 = 0,007, p = ,735 auf 251 Faellen).
- `h1_robustheit_erhebungsphase.py` - Robustheitscheck zu H1, Bewertungsexperiment:
getrennte Schaetzung nach Erhebungsphase. Die Render-Phase ist von der
SoSci-Rueckrechnung nicht betroffen und liefert einen von ihr unabhaengigen Test
(N = 203, b2 = -0,079, p = ,001, Scheitelpunkt 13,4). Das Skript weist zusaetzlich
aus, warum die SoSci-Teilstichprobe diesen Test nicht leisten kann, und welches
Argument welchen Teil der Rueckrechnung traegt.

### Datenaufbereitung und Darstellung

- `korrektur_sosci_punktzahl.py` - Einmalige Korrektur der Bewertungsvariable aus der
SoSci-Phase (vgl. Abschnitt "Korrektur der SoSci-Punktzahlen"). Bereits auf die
im Repository liegende Datendatei angewendet; das Skript bricht bei erneuter
Ausfuehrung ohne Aenderung ab.
- `abbildungsformat.py` - Hilfsfunktionen fuer die Achsenformatierung:
`dezimalkomma(ax)` setzt ein Dezimalkomma statt eines Punkts, `scheitelpunkt(ax, x)`
markiert den Scheitelpunkt in Abbildung 7. Werden in `create_figures.py` importiert.
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
  mit angepasster quadratischer Kurve (H1). Der Scheitelpunkt der Kurve liegt
  bei einer Forderung von 13,4 Punkten und ist als senkrechte Linie markiert.
- **Abbildung 9** — Preisexperiment: Forest-Plot der drei indirekten Effekte
  aus dem Mehrfachmediatormodell (H2), mit 95%-BC-Bootstrap-KI.
- **Abbildung 10** — Bewertungsexperiment: Pfaddiagramm des Mediationsmodells
  (H3) mit a-, b-, c'- und c-Koeffizienten.
- **Abbildung 11** — Bewertungsexperiment: Streudiagramm Fairness vs.
  Reaktanz mit Regressionsgerade (H4).

`figure_streuung_bewertungsexperiment.py` erzeugt **Abbildung 8**

- **Abbildung 8** - Verteilung der vergebenen Punktzahl je Bedingung,
Bewertungsexperiment - Begleitgrafik zur Streuungsanalyse.

Alle Achsen mit Dezimalstellen verwenden das Dezimalkomma (vgl. `abbildungsformat.py`).

## Daten und Codebuch

`data/Auswertung_Preisexperiment.csv` (356 Fälle) und
`data/Auswertung_Bewertungsexperiment.csv` (388 Fälle) sind die Rohdaten: eine Zeile je
Beobachtung, keine Vorab-Aggregation. Beide Dateien sind semikolongetrennt und verwenden
das Dezimalkomma:

```python
pd.read_csv("data/Auswertung_Preisexperiment.csv", sep=";", decimal=",")
```

### Korrektur der SoSci-Punktzahlen

Die Frage NV08 des Bewertungsexperiments ("Welche endgültige Punktzahl würden Sie dem
Studierenden geben?") ist in SoSci Survey als Auswahlfrage angelegt. SoSci speichert bei
Auswahlfragen die **Position der gewählten Antwortoption** und nicht den in der Option
angezeigten Wert. Die Optionsliste ist absteigend angeordnet, von `[01] = 15 Punkte` bis
`[16] = 0 Punkte`. Daraus folgt:

```
vergebene Punktzahl = 16 − Positionscode
```

Das ursprüngliche Import-Skript hat den Positionscode unverändert als Rohwert übernommen.
Betroffen waren ausschließlich die 138 Fälle mit `Datenquelle == "sosci_import"`. Die
250 Fälle der Render-Phase stammen aus einer eigens entwickelten Weberhebung, die den
Punktwert direkt speichert, und waren nicht betroffen. Das **Preisexperiment ist nicht
betroffen**, da dessen Daten ausschließlich aus der Render-Phase stammen.

Vor der Korrektur lagen die Mittelwerte beider Erhebungsphasen um mehr als drei Punkte
auseinander (6,35 gegenüber 9,75), und der Zusammenhang zwischen Forderungshöhe und
vergebener Punktzahl hatte in beiden Phasen entgegengesetzte Vorzeichen. Nach der
Korrektur stimmen die Mittelwerte nahezu überein (9,65 gegenüber 9,75), und beide Phasen
zeigen denselben Zusammenhang.

Die im Repository liegende Datendatei ist bereits korrigiert. Die Analysestichprobe bleibt
unverändert bei 295 Fällen, da die Manipulationschecks die Bewertungsvariable nicht
heranziehen. Die Korrektur ist in `korrektur_sosci_punktzahl.py` dokumentiert und
nachvollziehbar; das Skript bricht bei erneuter Ausführung ohne Änderung ab.

Dass der Befund zu H1 nicht an dieser Rückrechnung hängt, prüft
`h1_robustheit_erhebungsphase.py`. Die Render-Phase ist von ihr nicht betroffen und zeigt
den umgekehrt U-förmigen Verlauf für sich allein.

### Wie die Analysestichproben entstehen

**Preisexperiment (356 → 246).** Sieben Fälle entfallen strukturell, weil sie bereits die
überhöhte Erstforderung angenommen und damit nie über die Zielforderung entschieden haben;
bei ihnen ist `Codierung` leer. Von den verbleibenden 349 Fällen scheitern 103 an mindestens
einem der beiden Manipulationschecks. Es bleiben 246 Fälle.

**Bewertungsexperiment (388 → 295).** 93 Fälle scheitern am Rollen- oder am Forderungscheck.
Es bleiben 295 Fälle.

Beide Filter sind in den Skripten identisch umgesetzt:

```python
# Preisexperiment
df = df[df["Codierung"].notna()
        & (df["Manipulationscheck_Preis"] == "Ja")
        & (df["Manipulationscheck_Validitaet"] == "Ja")]          # -> 246

# Bewertungsexperiment
df = df[(df["Manipulationscheck_Rolle"] == "Ja")
        & (df["Manipulationscheck_Forderung"] == "Ja")]           # -> 295
```

### Hinweis zur Spalte `Boomerang_Variable`

`Boomerang_Variable` enthält in beiden Datensätzen den **Reaktanzindex**, also den Mittelwert
aus der affektiven und der umgepolten kognitiven Komponente. Das Paper verwendet durchgängig
die Bezeichnung Reaktanzindex. Die Spalte behielt ihre ursprüngliche Bezeichnung, damit die
Auswertungsskripte unverändert und die Ergebnisse reproduzierbar bleiben. Beide Bezeichnungen
meinen dieselbe Variable.

Die Itemtexte in den folgenden Tabellen sind wörtlich aus den Fragebögen in Anhang C und D
des Papers übernommen.

### Codebuch Preisexperiment

| Spalte | Bedeutung | Werte | Ableitung und Verwendung |
|---|---|---|---|
| `Datum` | Erhebungsdatum | TT.MM.JJJJ | Direkt erhoben, nicht ausgewertet |
| `Zeitstempel` | Uhrzeit des Abschlusses | HH:MM:SS | Direkt erhoben, nicht ausgewertet |
| `Treatment` | Höhe der Erstforderung in Euro | 849, 1500, 4000, 6000, 8000, 10000 (849 = Kontrolle) | Zufällige Zuweisung. Unabhängige Variable |
| `Entscheidung_1` | Entscheidung über die Erstforderung | `Annahme`, `Ablehnung` | Direkt erhoben |
| `Entscheidung_2` | Entscheidung über die Zielforderung (849 €) | `Annahme`, `Ablehnung`, `Nicht benötigt (Direkte Annahme)` | Direkt erhoben |
| `Codierung` | **Abhängige Variable:** Annahme der Zielforderung | 1 = Annahme, 0 = Ablehnung, leer = kein Wert | Aus `Entscheidung_2`. Leer bei Direktannahme in T1–T5 (7 Fälle) |
| `Erinnerter_Preis` | Item 1: „Wie hoch war das erste Angebot, das Ihnen der Vertreter an der Tür gemacht hat?" | Offene Eingabe in Euro, leer = keine Angabe (5 Fälle) | Grundlage des Preis-Manipulationschecks |
| `Angemessenheit` | Item 2a: „Der erste Angebotspreis wirkte auf mich angemessen." | 1–5 (1 = Stimme gar nicht zu) | Grundlage von `Angemessenheit_invertiert` |
| `Wahrnehmungskontrast` | Item 2b: „Das finale Angebot von 849 € wirkte auf mich attraktiv." | 1–5 | Mediator M1. Erfasst die Attraktivität der Zielforderung, nicht den Vergleich beider Forderungen; im Paper als Limitation diskutiert |
| `Konzession` | Item 2c: „Das finale Angebot von 849 € wirkte wie ein Entgegenkommen bzw. Zugeständnis des Verkäufers." | 1–5 | Mediator M2 |
| `Reaktanz` | Item 2d: „Das erste Angebot hat in mir Verärgerung ausgelöst und dazu geführt, dass ich auch das finale Angebot abgelehnt habe." | 1–5 | Affektive Komponente. Doppelläufiges Item, im Paper als Limitation diskutiert |
| `Validitaetscheck` | Item 2e: „Das Szenario war für mich verständlich und nachvollziehbar." | 1–5 | Grundlage des Validitätschecks |
| `Angemessenheit_invertiert` | Umgepolte Angemessenheit, hohe Werte = Widerstand | 1–5 | `6 - Angemessenheit`. Ergänzende Mediationsanalyse (`h2_ergaenzung_item2a.py`) |
| `Boomerang_Variable` | **Reaktanzindex** (siehe Hinweis oben) | 1–5 in Schritten von 0,5 | `(Angemessenheit_invertiert + Reaktanz) / 2`. Mediator M3 |
| `Manipulationscheck_Preis` | Erstforderung korrekt erinnert? | `Ja`, `Nein` | `Ja`, wenn die Abweichung höchstens 10 % von `Treatment` beträgt. Fehlende Angabe zählt als `Nein` |
| `Manipulationscheck_Validitaet` | Szenario verstanden? | `Ja`, `Nein` | `Ja`, wenn `Validitaetscheck >= 3` |

### Codebuch Bewertungsexperiment

| Spalte | Bedeutung | Werte | Ableitung und Verwendung |
|---|---|---|---|
| `Fallnummer` | Laufende Fallnummer | 1–388 | Beim Zusammenführen beider Phasen vergeben |
| `Zeitpunkt` | Zeitstempel der Teilnahme | Millisekunden seit 01.01.1970 | Im Export auf fünf signifikante Stellen gerundet, nicht ausgewertet |
| `Datenquelle` | Erhebungsphase | `sosci_import` (138), `render` (250) | Plattformwechsel, im Paper als Limitation diskutiert. Maßgeblich für die Rückrechnung der Punktzahl (siehe oben) |
| `Fragebogenversion` | Fragebogenversion | `sosci_nv_2026`, `render_v2` | Direkt erhoben |
| `SoSci-Fallnummer` | Ursprüngliche Fallnummer der SoSci-Phase | Ganzzahl, leer bei `render` | Rückverfolgbarkeit |
| `Treatment_Code` | Codierte Treatmentstufe | 1 = 9, 2 = 11, 3 = 13, 4 = 15 Punkte | Zufällige Zuweisung |
| `Forderung_des_Studierenden` | Geforderte Punktzahl | 9, 11, 13, 15 (9 = Kontrolle) | Unabhängige Variable |
| `Final_vergebene_Punktzahl` | **Abhängige Variable:** vergebene Punktzahl (Frage NV08) | Skala 0–15, beobachtet 5–15 | Direkt erhoben. Für die Fälle der SoSci-Phase zurückgerechnet als `16 - Positionscode` der Antwortoption, siehe Abschnitt „Korrektur der SoSci-Punktzahlen" |
| `Antwort_Rollenkontrolle` | Erinnerte eigene Rolle | `professor`, `student`, `external`, `none` | Korrekt ist `professor` |
| `Erinnerte_Forderung` | Erinnerte geforderte Punktzahl | Offene Eingabe | Grundlage des Forderungschecks |
| `Wahrgenommener_Druck` | Item 1: „Die Forderung des Studierenden hat bei mir Druck ausgelöst." | 1–5 | Bestandteil des Bedrohungsindex |
| `Eingeschraenkte_Bewertungsfreiheit` | Item 2: „Die Forderung hat meine Freiheit bei der Bewertung eingeschränkt." | 1–5 | Bestandteil des Bedrohungsindex |
| `Wahrnehmung_als_manipulativ` | Item 3: „Die Forderung des Studierenden wirkte auf mich manipulativ." | 1–5 | Bestandteil des Bedrohungsindex |
| `Veraergerung` | Item 4: „Die Forderung des Studierenden hat mich verärgert." | 1–5 | Affektive Komponente des Reaktanzindex |
| `Wahrgenommene_Angemessenheit` | Item 5: „Die Forderung des Studierenden wirkte auf mich angemessen." | 1–5 | Kognitive Komponente. Grundlage von `Angemessenheit_invertiert` |
| `Wahrgenommene_Fairness` | Item 6: „Die Forderung des Studierenden wirkte auf mich fair." | 1–5 | Prädiktor in H4. Fließt bewusst nicht in den Reaktanzindex ein |
| `Bearbeitungsdauer_in_Sekunden` | Bearbeitungsdauer | Ganzzahl in Sekunden | Dokumentation, nicht ausgewertet |
| `Angemessenheit_invertiert` | Umgepolte Angemessenheit, hohe Werte = Widerstand | 1–5 | `6 - Wahrgenommene_Angemessenheit`. Trennschärfeprüfung in `h4_analysis.py` |
| `Boomerang_Variable` | **Reaktanzindex** (siehe Hinweis oben) | 1–5 in Schritten von 0,5 | `(Veraergerung + Angemessenheit_invertiert) / 2`. Mediator in H3, abhängige Variable in H4 |
| `Manipulationscheck_Rolle` | Rolle korrekt erinnert? | `Ja`, `Nein` | `Ja`, wenn `Antwort_Rollenkontrolle == "professor"` |
| `Manipulationscheck_Forderung` | Forderung korrekt erinnert? | `Ja`, `Nein` | `Ja`, wenn `Erinnerte_Forderung == Forderung_des_Studierenden` |
| `Bedrohungswahrnehmung_Index` | Index der wahrgenommenen Freiheitsbedrohung | 1–5 | Mittelwert der Items 1–3. Manipulationscheck in `manipulationscheck_bedrohung.py` |

### Prüfsummen

Aus den Rohdaten lassen sich die zentralen Kennzahlen des Papers direkt reproduzieren:

| Kennzahl | Preisexperiment | Bewertungsexperiment |
|---|---|---|
| Erhobene Fälle | 356 | 388 |
| Strukturell ausgeschlossen | 7 | – |
| An Checks gescheitert | 103 | 93 |
| Finale Analysestichprobe | 246 | 295 |

Für das Bewertungsexperiment lässt sich zusätzlich prüfen, ob die Datendatei die korrigierte
Fassung ist:

| Kennzahl | Erwartet |
|---|---|
| Spanne `Final_vergebene_Punktzahl` | 5 bis 15 |
| Mittelwert der SoSci-Phase | 9,65 |
| Mittelwert der Render-Phase | 9,75 |

