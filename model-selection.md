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

## Nicht unterstützte Nebenbedingungen

Nicht unterstützte oder nicht eindeutig zuordenbare Nebenbedingungen werden
in **Angaben zur Berechnung** als Hinweis angezeigt. Sie verhindern die
Berechnung nicht:

- Unterstützte und vom Nutzer aktivierte Rahmenbedingungen werden weiterhin
  an den Solver übergeben.
- Nicht unterstützte Vorgaben werden nicht an den Solver übergeben und haben
  daher keinen Einfluss auf das Ergebnis.
- Die ignorierten Vorgaben bleiben in der Überprüfung und beim Ergebnis
  sichtbar.
- Ist eine ignorierte Vorgabe zwingend einzuhalten, sollte der Nutzer zur
  Problembeschreibung zurückkehren und das Solverergebnis nicht als
  Entscheidungsgrundlage verwenden.

Formulierungen, die lediglich die Modellstruktur beschreiben, sollen nicht als
Nebenbedingungen extrahiert werden. Beispielsweise beschreibt „Während der
Saison kann nicht nachbestellt werden“ beim Newsvendor den
`single_period`-Planungshorizont und keine zusätzliche Solver-Nebenbedingung.

# Demobeispiele für das zweite Formular

Das zweite Formular **Problembeschreibung** ist die natürlichsprachliche
Quelle für die gesamte Problemdefinition. Ein vollständiger Beispieltext
soll deshalb nicht nur die Modellstruktur, sondern auch alle in
`REQUIRED_FIELDS` definierten Modellparameter und gegebenenfalls optionale
Rahmenbedingungen enthalten. Die daraus erzeugte Eingabemaske dient dann zur
Kontrolle und Korrektur der erkannten Werte.

> **Hinweis zur Vorbefüllung:** Sowohl die Werte optionaler
> Rahmenbedingungen als auch die in `REQUIRED_FIELDS` definierten
> Modellparameter werden aus dem Freitext als unbestätigte Kandidaten
> übernommen. In **Angaben zur Berechnung** müssen alle erkannten Werte
> geprüft und mit **Eingaben prüfen** bestätigt werden. Fehlende oder nicht
> eindeutig erkannte Werte bleiben leer und können dort manuell ergänzt
> werden.

Für eine vollständige Vorführung eignet sich das folgende Newsvendor-Beispiel
mit Restwert und begrenzter Bestellmenge.

## Empfohlene End-to-End-Demo: Weihnachtsbäume

### In den Chatbot eingeben

> Wir verkaufen Weihnachtsbäume in einer einzigen Saison und bestellen sie
> einmal vor Saisonbeginn. Während der Saison können wir nicht nachbestellen.
> Die Nachfrage ist unsicher. Ein Baum wird für 25 € verkauft und kostet im
> Einkauf 10 €. Nicht verkaufte Bäume können nach der Saison noch für 3 € pro
> Baum verwertet werden. Wir möchten die Bestellmenge finden, mit der wir
> voraussichtlich den höchsten Gewinn erzielen. Zusätzlich dürfen höchstens
> 250 Weihnachtsbäume bestellt werden.

Bei Rückfragen bestätigen, dass **einmal vor der Saison bestellt** wird und
**während der Saison keine Nachbestellung möglich ist**.

### Werte für das Formular

| Feld | Eingabe |
| --- | ---: |
| Erwartete Nachfrage | 276 |
| Nachfragequantile | `0.1: 150; 0.5: 300; 0.9: 350` |
| Prognosezeitraum | 55 Tage |
| Verkaufspreis pro Baum | 25 € |
| Einkaufskosten pro Baum | 10 € |
| Restwert pro unverkauftem Baum | 3 € |
| Höchstbestellmenge | 250 Bäume |

Vor der Berechnung prüfen, ob neben Verkaufserlös und Einkaufskosten auch der
**Restwerterlös** sowie die **Höchstbestellmenge von 250** berücksichtigt werden.
Die drei Quantile vereinfachen die ursprüngliche Nachfrageverteilung. Deshalb
ist der erwartete Gewinn nicht direkt mit einem Ergebnis aus der vollständigen
Verteilung vergleichbar.

## Alternativdemo: EOQ für E-Bikes bei Bike-World

### In den Chatbot eingeben

> Die Bike-World GmbH verkauft ein E-Bike-Modell dauerhaft und bestellt es
> regelmäßig nach. Für die kommenden 365 Tage erwarten wir einen Bedarf von
> 1.200 E-Bikes, ungefähr gleichmäßig über die Zeit. Es gibt keine
> Mengenrabatte und wir planen keine Fehlbestände. Jede Bestellung verursacht
> feste Bestellkosten von 150 €. Die Lagerhaltung kostet 25 € pro E-Bike und
> Jahr. Wir möchten die Bestellmenge bestimmen, bei der die jährlichen
> Bestell- und Lagerhaltungskosten zusammen am geringsten sind.

Bei Rückfragen die **regelmäßige Nachbestellung** bestätigen. Vor der
Berechnung die Modellannahmen auf der Prüfseite einzeln bestätigen.

### Werte für das Formular

| Feld | Eingabe |
| --- | ---: |
| Erwartete Nachfrage | 1.200 E-Bikes |
| Prognosezeitraum | 365 Tage |
| Bestellkosten pro Bestellung | 150 € |
| Lagerhaltungskosten pro E-Bike und Jahr | 25 € |

### Erwartetes Ergebnis

- **Grundmodell:** Wirtschaftliche Bestellmenge (EOQ)
- **Empfohlene Bestellmenge:** 120 E-Bikes pro Bestellung
- **Jährliche Bestellkosten:** 1.500 €
- **Jährliche Lagerhaltungskosten:** 1.500 €
- **Jährliche relevante Gesamtkosten:** 3.000 €

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
> jährliche Nachfrage 7.300 T-Shirts. Die Lieferzeit beträgt drei Tage. Es liegen folgende Nachfragequantile vor: 0.1 mit 51,12; 0.5 mit 60; 0.9 mit 68,88. Jede Bestellung kostet 250 Euro, die
> Lagerhaltung 50 Euro je T-Shirt und Jahr und eine Fehlmenge 10 Euro je
> T-Shirt. Der Lieferant akzeptiert Bestellmengen von mindestens 250 und
> höchstens 300 T-Shirts. Der Meldebestand soll zwischen 60 und 80 T-Shirts
> liegen. Im Lager darf die konservativ gerechnete Bestandsposition aus
> Bestellmenge und Meldebestand zusammen höchstens 400 T-Shirts betragen.

Zusammen mit den oben beschriebenen Q–R-Auswahlen in Formular 1 sollte die
Anwendung die **Q–R Policy** erkennen und zu **Angaben zur Berechnung**
wechseln. Die sechs Pflichtfelder sollten aus dem Text mit den folgenden
Werten vorbefüllt werden. Auch die fünf Rahmenbedingungen sollten
vorausgewählt und mit ihren Zahlenwerten befüllt sein. Vor der Berechnung
müssen alle erkannten Angaben noch geprüft und bestätigt werden.

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

### Erwartetes Ergebnis

Die Cornell-Musterrechnung liefert vor der operativen Ganzzahlrundung ungefähr
\(Q = 274{,}012\) und \(R = 66{,}142\) T-Shirts. Nach der in diesem Repository
beschriebenen Rundungsregel und allen fünf nichtbindenden
Demo-Nebenbedingungen liefert die WebUI:

- **Empfohlene Bestellmenge Q:** 274 Einheiten
- **Meldebestand R:** 67 Einheiten
- **Sicherheitsbestand:** 7 Einheiten
- **Erwartete jährliche Kosten:** 14.011,35 €
- **Solverstatus:** Optimal
- **Iterationen:** 5

Der Solver verwendet wie die Cornell-Musterrechnung
\(F(R) = 1 - hQ/(pD)\). Dieselbe Näherung wird auch im
[LINDO-Kapitel zum Q–R-Modell](https://www.lindo.com/downloads/LINGO_text/Chapter17.pdf)
angegeben. Die Fixpunktiteration rechnet intern mit ungerundeten Werten und
rundet erst die operative Empfehlung. Dadurch stimmt das WebUI-Ergebnis bis
auf diese dokumentierte Ganzzahlrundung mit der Referenz überein.
