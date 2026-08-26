# Modellauswahl im ersten Formular

Das erste Formular dient zur Auswahl einer passenden Modellfamilie. Für die
aktuell vorgesehenen Modelle sind folgende Angaben auszuwählen. Alle Modelle
gelten derzeit fest für **ein Produkt**; der Produktumfang wird deshalb nicht
im Formular abgefragt.

## Newsvendor

- **Entscheidung:** Bestellmenge
- **Ziel:** Gewinn maximieren
- **Nachfrage:** Unsicher / saisonal
- **Planungszeitraum:** Für eine einmalige Entscheidung

Bei einem einzelnen Planungszeitraum wird keine Art der
Bestandsüberprüfung abgefragt. Nach der Analyse führt diese Kombination zum
Eingabeformular des Newsvendor-Modells.

### Unterstützte Nebenbedingungen

- Mindestbestellmenge
- Höchstbestellmenge
- Budgetgrenze
- Lagerkapazität

## Q–R-Politik

- **Entscheidung:** Bestellmenge
- **Ziel:** Kosten minimieren
- **Nachfrage:** Unsicher / saisonal
- **Planungszeitraum:** Für mehrere aufeinanderfolgende Zeiträume
- **Bestandsüberprüfung:** Kontinuierlich – Bestand laufend überwachen

Nach der Analyse führt diese Kombination zum Eingabeformular der
Q–R-Politik.

### Unterstützte Nebenbedingungen

- Mindestbestellmenge \(Q\)
- Höchstbestellmenge \(Q\)
- Mindestmeldebestand \(R\)
- Höchstmeldebestand \(R\)
- Lagerkapazität als konservative Grenze \(Q + R \leq C\)

Eine Budgetgrenze wird bei der Q–R-Politik derzeit nicht unterstützt, da das
Modell keinen Einkaufspreis pro Einheit verwendet. Die Kapazitätsgrenze
\(Q + R \leq C\) ist eine konservative Grenze für die Bestandsposition und
nicht der erwartete physische Höchstbestand.

Bestellmenge, Meldebestand und ausgegebener Sicherheitsbestand werden als
ganze Produkteinheiten empfohlen. Dabei wird \(Q\) auf die nächste Einheit
gerundet; \(R\) und der Sicherheitsbestand werden vorsichtshalber
aufgerundet. Statistische Eingaben und erwartete Kosten dürfen weiterhin
Dezimalwerte enthalten.

## (s, S)-Politik

- **Entscheidung:** Bestellmenge
- **Ziel:** Kosten minimieren
- **Nachfrage:** Unsicher / saisonal
- **Planungszeitraum:** Für mehrere aufeinanderfolgende Zeiträume
- **Bestandsüberprüfung:** Periodisch – Bestand in festen Abständen prüfen

Diese Kombination ist bereits als zukünftige `(s, S)`-Route vorgesehen. Da
das Modell noch nicht implementiert ist, wird nach der Analyse ein
entsprechender Platzhalter angezeigt.

# Demobeispiele für das zweite Formular

Das zweite Formular **Problembeschreibung** ist die natürlichsprachliche
Quelle für die gesamte Problemdefinition. Ein vollständiger Beispieltext
soll deshalb nicht nur die Modellstruktur, sondern auch alle in
`REQUIRED_FIELDS` definierten Modellparameter und gegebenenfalls optionale
Rahmenbedingungen enthalten. Die daraus erzeugte Eingabemaske dient dann zur
Kontrolle und Korrektur der erkannten Werte.

> **Hinweis zum aktuellen Implementierungsstand:** Die Werte optionaler
> Rahmenbedingungen werden bereits als vorausgefüllte Kandidaten aus dem
> Freitext übernommen. Die Extraktion der modellspezifischen
> `REQUIRED_FIELDS` in `model_specific_inputs` ist im aktuellen Workflow noch
> nicht angeschlossen. Diese Felder müssen daher momentan in **Angaben zur
> Berechnung** noch einmal eingetragen werden. Die Texte unten enthalten
> trotzdem bereits alle Werte und dokumentieren damit zugleich das erwartete
> Zielverhalten der Vorbefüllung.

Für eine vollständige Vorführung eignet sich besonders das folgende
Newsvendor-Beispiel, weil die Quelldaten und das Soll-Ergebnis kompakt und
öffentlich dokumentiert sind.

## Empfohlene End-to-End-Demo: Weihnachtsornamente

Das Beispiel basiert auf dem Fall „Theodore's Gift Shop“ aus den
[MIT-OpenCourseWare-Unterlagen zum Newsvendor-Modell](https://ocw.mit.edu/courses/15-772j-d-lab-supply-chains-fall-2014/6952be57b43aa185119c6f114908bcc5_MIT15_772JF14_Newsboy.pdf).
Die Quelle verwendet US-Dollar. Für die Demo werden dieselben Zahlen als Euro
eingegeben. Das ist keine Währungsumrechnung, ändert aber die optimale Menge
nicht, solange alle Geldbeträge einheitlich behandelt werden.

### Formular 1: Grundlegende Angaben

- **Entscheidung:** Bestellmenge
- **Ziel:** Gewinn maximieren
- **Nachfrage:** Unsicher / saisonal
- **Planungszeitraum:** Für eine einmalige Entscheidung

### Formular 2: Problembeschreibung

Folgenden Text vollständig in das Freitextfeld kopieren:

> Ein Geschenkeladen muss bereits im Juli einmalig datierte
> Weihnachtsornamente für die kommende Saison bestellen. Während der Saison
> kann nicht nachbestellt werden. Ein Ornament kostet im Einkauf 55 Euro und
> wird für 80 Euro verkauft. Nicht verkaufte Ornamente können nach Weihnachten
> für 40 Euro pro Stück verkauft werden. Für die saisonale Nachfrage werden
> vier Szenarien erwartet: 5 Stück mit 20 Prozent Wahrscheinlichkeit, 6 Stück
> mit 25 Prozent, 7 Stück mit 30 Prozent und 8 Stück mit 25 Prozent. Gesucht
> ist die einmalige Bestellmenge mit dem höchsten erwarteten Gewinn. Wegen der
> begrenzten Ausstellungsfläche dürfen höchstens 7 Ornamente bestellt werden.

Nach **Problem analysieren** sollte die Anwendung das **Newsvendor-Modell –
Single Product, Single Period** erkennen und zu **Angaben zur Berechnung**
wechseln. Die fünf Pflichtfelder sollten aus dem Text mit den unten gezeigten
Werten vorbefüllt werden; beim aktuellen Implementierungsstand sind sie noch
manuell zu übernehmen. Zusätzlich sollte eine **Höchstbestellmenge von 7**
als vorausgewählte Rahmenbedingung erkannt werden.

### Angaben zur Berechnung

| Feld | Eingabe |
| --- | ---: |
| Verkaufspreis pro Einheit | 80 |
| Einkaufskosten pro Einheit | 55 |
| Restwert pro unverkaufter Einheit | 40 |
| Mögliche Nachfragewerte | `5, 6, 7, 8` |
| Wahrscheinlichkeiten | `0.20, 0.25, 0.30, 0.25` |
| Betriebliche Rahmenbedingungen berücksichtigen? | Ja |
| Höchstbestellmenge | 7 |

Die Höchstbestellmenge ist eine realistische Demo-Erweiterung und nicht Teil
der MIT-Quelle. Sie verändert das Soll-Ergebnis nicht, da die dort bereits
optimale Bestellmenge 7 beträgt.

### Erwartetes Ergebnis

- **Empfohlene Bestellmenge:** 7 Einheiten
- **Erwarteter Gewinn:** 149,00 €
- **Solverstatus:** Optimal

Die Szenariotabelle sollte Folgendes zeigen:

| Nachfrage | Wahrscheinlichkeit | Verkauft | Restbestand |
| ---: | ---: | ---: | ---: |
| 5 | 20 % | 5 | 2 |
| 6 | 25 % | 6 | 1 |
| 7 | 30 % | 7 | 0 |
| 8 | 25 % | 7 | 0 |

Das Ergebnis stimmt mit der MIT-Musterrechnung überein: Bei sieben bestellten
Ornamenten beträgt der erwartete Erlös 534 und nach Abzug der Einkaufskosten
von \(7 \cdot 55 = 385\) der erwartete Gewinn 149.

## Alternativtext für die Q–R-Politik: T-Shirts im Campusladen

Dieses kleinere Beispiel stammt aus einer Cornell-ORIE-3120-Übung zur
Q–R-Politik. Eine
[archivierte Kopie der Übung mit Musterlösung](https://www.studocu.com/en-us/document/cornell-university/industrial-data-and-systems-analysis/orie-3120-rec-4-answers-material/12623299)
nennt genau die Kosten- und Nachfragegrößen, die unser Modell benötigt.
Cornell stellt außerdem
[offizielle Kursunterlagen zur Q–R-Bestandssteuerung](https://people.orie.cornell.edu/pfrazier/orie3120/demos/QR.pdf)
bereit.

Gegenüber dem bisherigen HP-Druckerfall hat dieses Beispiel drei Vorteile:

- Es verwendet genau die sechs in `REQUIRED_FIELDS` definierten Eingaben.
- Es benötigt weder Einkaufspreis noch Pipelinekosten oder ein vorgegebenes
  Servicelevel.
- Bestellmenge und Meldebestand liegen im zwei- bis dreistelligen Bereich und
  lassen sich in der WebUI gut lesen.

### Formular 2: Problembeschreibung

Folgenden Text vollständig in das Freitextfeld kopieren:

> Ein Campusladen überwacht den Bestand eines T-Shirt-Modells laufend und
> möchte seine jährlichen Bestell-, Lagerhaltungs- und Fehlmengenkosten
> minimieren. Sobald die Bestandsposition den Meldebestand erreicht, wird eine
> feste Menge nachbestellt. Pro Tag werden im Mittel 20 T-Shirts verkauft; die
> Tagesnachfrage ist unabhängig und normalverteilt mit einer
> Standardabweichung von 4 T-Shirts. Bei 365 Verkaufstagen beträgt die
> jährliche Nachfrage 7.300 T-Shirts. Die Lieferzeit beträgt drei Tage. Damit
> liegt die mittlere Nachfrage während der Lieferzeit bei 60 T-Shirts und ihre
> Standardabweichung bei 6,928 T-Shirts. Jede Bestellung kostet 250 Euro, die
> Lagerhaltung 50 Euro je T-Shirt und Jahr und eine Fehlmenge 10 Euro je
> T-Shirt. Der Lieferant akzeptiert Bestellmengen von mindestens 250 und
> höchstens 300 T-Shirts. Der Meldebestand soll zwischen 60 und 80 T-Shirts
> liegen. Im Lager darf die konservativ gerechnete Bestandsposition aus
> Bestellmenge und Meldebestand zusammen höchstens 400 T-Shirts betragen.

Zusammen mit den oben beschriebenen Q–R-Auswahlen in Formular 1 sollte die
Anwendung die **Q–R Policy** erkennen und zu **Angaben zur Berechnung**
wechseln. Die sechs Pflichtfelder sollten aus dem Text mit den folgenden
Werten vorbefüllt werden; beim aktuellen Implementierungsstand sind sie noch
manuell zu übernehmen.

### Angaben zur Berechnung

Die Quelle gibt eine tägliche Nachfrage mit Mittelwert 20 und
Standardabweichung 4 sowie eine Lieferzeit von drei Tagen an. Bei unabhängigen
Tagesnachfragen folgt daraus
\(\mu_L = 3 \cdot 20 = 60\) und
\(\sigma_L = \sqrt{3} \cdot 4 \approx 6{,}928\). Für die WebUI werden die
Dollarbeträge der Quelle ohne Wechselkursumrechnung als einheitliche
Euro-Demowerte behandelt.

| Feld | Eingabe |
| --- | ---: |
| Jährliche Nachfrage | 7300 |
| Fixe Bestellkosten | 250 |
| Jährliche Lagerhaltungskosten | 50 |
| Fehlmengenkosten | 10 |
| Mittlere Nachfrage während der Lieferzeit | 60 |
| Standardabweichung der Lieferzeitnachfrage | 6.928 |
| Betriebliche Rahmenbedingungen berücksichtigen? | Ja |
| Mindestbestellmenge Q | 250 |
| Höchstbestellmenge Q | 300 |
| Mindestmeldebestand R | 60 |
| Höchstmeldebestand R | 80 |
| Maximale Bestandsposition Q + R | 400 |

Die Grenzen für \(Q\), \(R\) und \(Q+R\) sind plausible
**Demo-Erweiterungen**, aber nicht Teil der Cornell-Quelle. Sie bilden alle von
der WebUI unterstützten Q–R-Nebenbedingungen ab und sind für die unten
genannten Ergebnisse nicht bindend.

### Referenzergebnis und aktuelles WebUI-Ergebnis

Die Cornell-Musterrechnung liefert vor der operativen Ganzzahlrundung ungefähr
\(Q = 274{,}012\) und \(R = 66{,}142\) T-Shirts. Nach der in diesem Repository
beschriebenen Rundungsregel entspräche das einer Bestellmenge von 274, einem
Meldebestand von 67 und einem Sicherheitsbestand von 7 T-Shirts.

Mit den oben angegebenen Werten und allen fünf Demo-Nebenbedingungen liefert
der aktuelle Stand dieses Repositories dagegen:

- **Empfohlene Bestellmenge Q:** 270 Einheiten
- **Meldebestand R:** 79 Einheiten
- **Sicherheitsbestand:** 19 Einheiten
- **Erwartete jährliche Kosten:** 14.460,99 €
- **Solverstatus:** Optimal
- **Iterationen:** 2

Die verbleibende Abweichung ist kein Unterschied des Anwendungsfalls. Die
Cornell-Musterrechnung verwendet für ihre Iteration
\(F(R) = 1 - hQ/(pD)\). Dieselbe Näherung wird auch im
[LINDO-Kapitel zum Q–R-Modell](https://www.lindo.com/downloads/LINGO_text/Chapter17.pdf)
angegeben. Die aktuelle Implementierung verwendet dagegen
\(F(R) = 1 - Q/(pD)\). Dadurch wird der Lagerkostensatz \(h\) bei der
Servicewahrscheinlichkeit nicht berücksichtigt und der Meldebestand zu hoch
berechnet. Das neue Beispiel passt daher sehr gut zur Vorführung von
**Modellauswahl, Eingabefeldern und Nebenbedingungen**, macht aber zugleich
die noch zu korrigierende Ergebnislogik sichtbar.

## Alternativtext für die (s, S)-Route

> Eine Apotheke prüft ihren Bestand einmal pro Woche. Die Nachfrage zwischen
> den Prüfzeitpunkten ist unsicher. Wenn der Bestand bei einer Prüfung zu
> niedrig ist, soll bis zu einem Zielbestand aufgefüllt werden. Gesucht ist
> eine kostengünstige Bestellpolitik für mehrere aufeinanderfolgende Wochen.

Zusammen mit den oben beschriebenen `(s, S)`-Auswahlen in Formular 1 sollte
die Anwendung die periodische Bestandsüberprüfung erkennen. Erwartet wird der
Hinweis **Problemart noch nicht unterstützt**, weil das `(s, S)`-Modell noch
nicht implementiert ist.
