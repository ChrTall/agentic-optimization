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
ganze Produkteinheiten empfohlen. Der Solver durchsucht dazu alle zulässigen
ganzzahligen Bestellmengen \(Q\) und prüft für jede die benachbarten
ganzzahligen Meldebestände \(R\) um das stetige Optimum. Empfohlen wird die
Kombination mit den geringsten erwarteten Kosten; es wird also nicht pauschal
auf- oder abgerundet. Statistische Eingaben und erwartete Kosten dürfen
weiterhin Dezimalwerte enthalten.

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
Aus den drei Quantilen bildet die Anwendung eine stückweise lineare
Nachfrageverteilung mit 100 gleich wahrscheinlichen Szenarien (Mittelwert etwa
268,75). Ohne Höchstbestellmenge läge die optimale Bestellmenge bei etwa 323
Bäumen (kritisches Verhältnis \((25 - 10) / (25 - 3) \approx 0{,}68\)); mit
der Grenze von 250 ist sie bindend:

- **Empfohlene Bestellmenge:** 250 Bäume
- **Erwarteter Gewinn:** 3.195,46 €

Die angegebene erwartete Nachfrage (276) wird mit dem Mittelwert der
Quantilsverteilung abgeglichen und im Ergebnis neben ihm angezeigt. Weichen
beide um mehr als 25 % voneinander ab, weist die Validierung auf
widersprüchliche Angaben hin.

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

### Formular 2: Problembeschreibung

Folgenden Text vollständig in das Freitextfeld kopieren:

> Ein Campusladen überwacht den Bestand eines T-Shirt-Modells laufend und
> möchte seine jährlichen Bestell-, Lagerhaltungs- und Fehlmengenkosten
> minimieren. Sobald die Bestandsposition den Meldebestand erreicht, wird eine
> feste Menge nachbestellt. Die Lieferzeit beträgt drei Tage. Für diese drei
> Tage erwarten wir eine Nachfrage von 60 T-Shirts; die Nachfrageprognose
> bezieht sich also auf einen Zeitraum von drei Tagen. Für denselben Zeitraum
> liegen folgende Nachfragequantile vor: 0,1 mit 51,12; 0,5 mit 60; 0,9 mit
> 68,88 T-Shirts. Jede Bestellung kostet 250 Euro, die
> Lagerhaltung 50 Euro je T-Shirt und Jahr und eine Fehlmenge 10 Euro je
> T-Shirt. Der Lieferant akzeptiert Bestellmengen von mindestens 250 und
> höchstens 300 T-Shirts. Der Meldebestand soll zwischen 60 und 80 T-Shirts
> liegen. Im Lager darf die konservativ gerechnete Bestandsposition aus
> Bestellmenge und Meldebestand zusammen höchstens 400 T-Shirts betragen.

Zusammen mit den oben beschriebenen Q–R-Auswahlen in Formular 1 sollte die
Anwendung die **Q–R Policy** erkennen und zu **Angaben zur Berechnung**
wechseln. Die Pflichtfelder sollten aus dem Text mit den folgenden
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

Die Anwendung erwartet die Nachfrage als Prognose (Erwartungswert, Quantile,
Prognosezeitraum). Die Quantile beziehen sich auf denselben Zeitraum wie der
Erwartungswert, hier also auf die drei Tage der Lieferzeit. Daraus leitet die
Anwendung den Jahresbedarf \(D = 60 \cdot 365 / 3 = 7300\) und
\(\sigma_L \approx 6{,}929\) selbst ab.

| Feld | Eingabe |
| --- | ---: |
| Erwartete Nachfrage | 60 |
| Nachfragequantile | `0.1: 51.12; 0.5: 60; 0.9: 68.88` |
| Prognosezeitraum | 3 Tage |
| Lieferzeit | 3 Tage |
| Fixe Bestellkosten | 250 |
| Jährliche Lagerhaltungskosten | 50 |
| Fehlmengenkosten | 10 |
| Betriebliche Rahmenbedingungen berücksichtigen? | Ja |
| Mindestbestellmenge Q | 250 |
| Höchstbestellmenge Q | 300 |
| Mindestmeldebestand R | 60 |
| Höchstmeldebestand R | 80 |
| Maximale Bestandsposition Q + R | 400 |

Erwartet wird das Grundmodell **Kontinuierliche Bestandsüberprüfung (Q,R)**,
das sowohl eine Bestellmenge als auch einen Meldebestand empfiehlt.

### Erwartetes Ergebnis

Die Cornell-Musterrechnung liefert vor der Ganzzahlrundung ungefähr
\(Q = 274{,}012\) und \(R = 66{,}142\) T-Shirts. Mit allen fünf
nichtbindenden Demo-Nebenbedingungen liefert die WebUI:

- **Empfohlene Bestellmenge Q:** 274 Einheiten
- **Meldebestand R:** 66 Einheiten
- **Sicherheitsbestand:** 6 Einheiten
- **Erwartete jährliche Kosten:** 14.007,86 €
- **Solverstatus:** Optimal

Der Solver verwendet wie die Cornell-Musterrechnung
\(F(R) = 1 - hQ/(pD)\). Dieselbe Näherung wird auch im
[LINDO-Kapitel zum Q–R-Modell](https://www.lindo.com/downloads/LINGO_text/Chapter17.pdf)
angegeben. Weil die ganzzahlige Suche die Kosten von \(R = 66\) und
\(R = 67\) direkt vergleicht, liegt das Ergebnis etwas unter dem einer
pauschalen Aufrundung (\(R = 67\): 14.011,42 €).
