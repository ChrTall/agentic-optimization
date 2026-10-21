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

## Alternativdemo: Q,R für T-Shirts

### In den Chatbot eingeben

> Wir verkaufen ein T-Shirt-Modell dauerhaft und bestellen regelmäßig nach.
> Der Lagerbestand wird laufend überwacht. Die Lieferzeit beträgt konstant
> drei Tage. Die Nachfrage ist unsicher: Für diese drei Tage erwarten wir
> 60 verkaufte T-Shirts; die Nachfragequantile sind Q10 = 51,12, Q50 = 60
> und Q90 = 68,88. Wir möchten die Bestellmenge und den Meldebestand
> bestimmen und die jährlichen Bestell-, Lagerhaltungs- und Fehlmengenkosten
> minimieren. Jede Bestellung kostet 250 €, die Lagerhaltung 50 € je T-Shirt
> und Jahr und jede fehlende Einheit 10 €.

Bei Rückfragen die **regelmäßige Nachbestellung** und die **laufende
Bestandsprüfung** bestätigen. Die konstante Lieferzeit auf der Prüfseite
ausdrücklich bestätigen.

### Werte für das Formular

| Feld | Eingabe |
| --- | ---: |
| Erwartete Nachfrage | 60 T-Shirts |
| Nachfragequantile | `0.1: 51.12; 0.5: 60; 0.9: 68.88` |
| Prognosezeitraum | 3 Tage |
| Lieferzeit | 3 Tage |
| Bestellkosten pro Bestellung | 250 € |
| Lagerhaltungskosten pro T-Shirt und Jahr | 50 € |
| Fehlmengenkosten pro fehlender Einheit | 10 € |

Erwartet wird das Grundmodell **Kontinuierliche Bestandsüberprüfung (Q,R)**,
das sowohl eine Bestellmenge als auch einen Meldebestand empfiehlt.

### Erwartetes Ergebnis

| Ergebnisgröße | Wert |
| --- | ---: |
| Jahresbedarf (aus 60 Stück in 3 Tagen hochgerechnet) | 7.300 |
| Erwartete Nachfrage während der Lieferzeit | 60 |
| Standardabweichung der Lieferzeitnachfrage | ca. 6,93 |
| Empfohlene Bestellmenge Q | 274 |
| Meldebestand R | 66 |
| Sicherheitsbestand | 6 |
| Jährliche relevante Gesamtkosten | 14.007,86 € |
