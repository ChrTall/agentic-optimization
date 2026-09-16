# Referenzfälle für den neuen Geschäftseinstieg

Diese Fälle beschreiben den Weg von der Sprache eines Sachbearbeiters
bis zur Auswahl eines Grundmodells. Modellparameter wie Preise oder
Kosten werden erst nach der Grundmodellauswahl vervollständigt.

## 1. Einmaliger Saisonartikel

**Eingabe**

> Wir verkaufen Weihnachtsbäume nur im Dezember. Ende November
> bestellen wir einmal. Danach können wir nicht mehr nachbestellen.
> Wir möchten wissen, wie viele Bäume wir bestellen sollen.

**Unterlagen:** Q-Foundry-Forecast mit Erwartungswert, Quantilen und
30 Tagen Horizont sowie eine Preisliste.

**Erkannt:** Bestellmenge, einmalige Beschaffung, keine spätere
Nachbestellung, ein Produkt und Forecast mit Unsicherheitsinformationen.

**Noch zu klären:** Nur das wirtschaftliche Hauptziel, falls es nicht
beschrieben wurde.

**Ergebnis:** Einperiodiges Newsvendor-Grundmodell. Verkaufs-, Einkaufs-
und gegebenenfalls Restwert werden später vervollständigt.

## 2. Vierzig Saisonprodukte mit gemeinsamem Budget

**Eingabe**

> Für unsere Saisonartikel müssen die Bestellmengen vor Saisonbeginn
> festgelegt werden. Während der Saison können wir nicht nachbestellen.
> Insgesamt stehen höchstens 50.000 Euro zur Verfügung.

**Unterlagen:** Excel-Datei mit 40 Produkten, Forecasts und Preisen sowie
ein Lieferantenvertrag mit produktspezifischen Mindestmengen.

**Erkannt:** Bestellmengen, einmalige Beschaffung, keine spätere
Nachbestellung, 40 Produkte, Forecast je Produkt, Gesamtbudget und
Mindestbestellmengen.

**Noch zu klären:** Sollen alle 40 Produkte gemeinsam betrachtet werden?

**Ergebnis:** Einperiodiges Newsvendor-Grundmodell mit Budget- und
Mindestmengenbausteinen.

## 3. Regelmäßiger Verbrauch von Verpackungsmaterial

**Eingabe**

> Wir benötigen Verpackungsmaterial das ganze Jahr über und bestellen
> regelmäßig nach. Wir möchten wissen, wie groß eine Bestellung
> sinnvollerweise sein sollte.

**Unterlagen:** Q-Foundry-Forecast mit erwartetem Jahresbedarf.

**Erkannt:** Bestellmenge, regelmäßige Beschaffung, langfristig relativ
gleichmäßiger Verbrauch, ein Produkt und erwarteter Bedarf mit Zeitraum.

**Noch zu klären:** Keine Frage zur Bestandsüberprüfung, weil kein
Nachbestellzeitpunkt bestimmt werden soll.

**Ergebnis:** EOQ-Grundmodell. Bestell- und Lagerhaltungskosten werden
später vervollständigt.

## 4. Regelmäßige Beschaffung mit unklarer Bedarfssituation

**Eingabe**

> Wir bestellen den Artikel regelmäßig und möchten eine sinnvolle
> Bestellmenge berechnen lassen.

**Unterlagen:** Forecast mit stark auseinanderliegenden Quantilen.

**Erkannt:** Bestellmenge, regelmäßige Beschaffung und Forecast mit
Unsicherheitsinformationen. Der gewünschte Umgang mit Schwankungen ist
noch nicht geklärt.

**Adaptive Rückfrage**

> Soll die Empfehlung langfristig mit dem durchschnittlichen Bedarf
> rechnen oder sollen Nachfrageschwankungen und Lieferfähigkeit
> ausdrücklich berücksichtigt werden?

**Ergebnis:** Das Grundmodell ist erst nach der Antwort sicher auswählbar.

## 5. Laufende Bestandsüberwachung

**Eingabe**

> Das Produkt wird dauerhaft verkauft und unser Lagerbestand ist
> jederzeit bekannt. Wir möchten wissen, wann und wie viel wir
> nachbestellen sollen.

**Unterlagen:** Q-Foundry-Forecast mit Erwartungswert und Quantilen sowie
ein Lieferantenvertrag mit Lieferzeit.

**Erkannt:** Bestellmenge und Nachbestellzeitpunkt, regelmäßige
Beschaffung, laufende Bestandsüberprüfung, Forecast mit
Unsicherheitsinformationen und Lieferzeit.

**Noch zu klären:** Nur das wirtschaftliche Hauptziel, falls es nicht
beschrieben wurde.

**Ergebnis:** Grundmodell für kontinuierliche Bestandsüberprüfung (Q,R).
Bestell-, Lagerhaltungs- und Fehlmengenkosten werden später ergänzt.

## 6. Mehrere dauerhaft geführte Produkte

**Eingabe**

> Für alle Artikel soll automatisch bestimmt werden, wann und wie viel
> nachbestellt wird. Der Lagerbestand wird laufend aktualisiert.

**Unterlagen:** Excel-Datei mit 40 Produkten, Forecast je Produkt und
gemeinsamer Lagerkapazität.

**Erkannt:** Bestellmengen und Nachbestellzeitpunkte, regelmäßige
Beschaffung, laufende Bestandsüberprüfung, 40 Produkte und eine gemeinsame
Kapazitätsvorgabe.

**Noch zu klären:** Produktumfang bestätigen und abweichende
Beschaffungssituationen einzelner Produkte erkennen.

**Ergebnis:** Möglicher Kandidat ist Q,R. Das Grundmodell darf nur gewählt
werden, wenn der registrierte Kern Produktumfang und Kapazität unterstützt.

## 7. Unvollständige Beschreibung

**Eingabe**

> Wir möchten unsere Lagerhaltung verbessern und bestellen regelmäßig.

**Erkannt:** Regelmäßige Beschaffung. Entscheidung, Bestandsüberprüfung
und Ziel sind noch nicht eindeutig.

**Adaptive Rückfragen**

1. Soll die Bestellmenge, der Nachbestellzeitpunkt oder beides bestimmt werden?
2. Falls ein Nachbestellzeitpunkt gewünscht ist: Wird der Bestand laufend
   oder nur zu bestimmten Zeitpunkten geprüft?
3. Was soll hauptsächlich verbessert werden?

**Ergebnis:** Keine Grundmodellauswahl vor der Klärung.

## 8. Nicht unterstützte periodische Bestandsüberprüfung

**Eingabe**

> Wir prüfen den Bestand jeden Freitag und möchten wissen, wie viel wir
> dann jeweils bis zum nächsten Prüftermin bestellen sollen.

**Erkannt:** Regelmäßige Beschaffung, periodische Bestandsüberprüfung und
gewünschte Bestellmenge beziehungsweise Bestellzeitpunkt.

**Ergebnis:** Derzeit nicht unterstützt. Das System erklärt die erkannte
periodische Bestandspolitik und erzwingt keine Zuordnung zu EOQ oder Q,R.

## 9. Widersprüchliche Quellen

**Formularauswahl:** einmalige Beschaffung.

**Beschreibung oder Dokument**

> Das Produkt wird dauerhaft geführt und jeden Monat nachbestellt.

**Ergebnis:** Das Merkmal erhält den Status Konflikt. Beide Aussagen
werden mit ihren Quellen angezeigt und müssen bestätigt werden. Vorher
erfolgt keine Grundmodellauswahl.

## Abgeleitete Anforderungen an das Geschäftsprofil

Das Profil muss mindestens folgende Informationen aufnehmen können:

- gewünschte Empfehlung
- einmalige oder regelmäßige Beschaffung
- Möglichkeit einer späteren Nachbestellung
- Art der Bestandsüberprüfung
- Produktumfang und konkrete Produkt-IDs
- wirtschaftliches Hauptziel
- Forecast-Daten getrennt nach Produkt und Prognosezeitraum
- betriebliche Vorgaben
- Wert, Status und Quellenbelege je erkannter Information

Ein unbekannter Wert ist kein eigener fachlicher Wert. Stattdessen wird
gespeichert, ob die Information fehlt, vom Nutzer nicht gewusst wird,
aus einer Quelle abgeleitet wurde, bestätigt ist, mehrdeutig ist oder im
Widerspruch zu einer anderen Quelle steht.
