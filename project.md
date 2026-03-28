# win-acp-java – Produktanforderungen

## 1. Zweck

`win-acp-java` ist eine technische Laufzeitkomponente für Windows 11, die einen lokal gestarteten ACP-Agenten in Java 21 bereitstellt.

Die Komponente soll drei Probleme lösen:

1. Einen **ACP-kompatiblen Agentenprozess** für JetBrains und andere ACP-Clients bereitstellen.
2. Das **Agentenverhalten von außen konfigurierbar** machen, ohne beliebigen Java-Code als Konfiguration auszuführen.
3. **MCP-Tools und MCP-Ressourcen** als standardisierte Fähigkeiten des Agenten akzeptieren und in die Agentenlogik integrieren.

Zusätzlich soll die Komponente lokale, performante Inferenz unter Windows ermöglichen, ohne auf externe EXEs oder Drittanbieter-DLLs angewiesen zu sein.

---

## 2. Produktvision

`win-acp-java` ist ein Windows-orientierter Java-Agent-Host, der:

* nach außen **ACP** spricht,
* nach innen **MCP-Tools** konsumiert,
* das **Agentenverhalten deklarativ konfigurierbar** macht,
* und lokale Inferenz über einen Windows-nativen Low-Level-Pfad ermöglicht.

Der Nutzer der Komponente soll sich primär um folgende Dinge kümmern müssen:

* Agentenprofil,
* Verhaltenskonfiguration,
* verfügbare MCP-Tools,
* Freigaben und Policies,
* Prompting und Domänenregeln.

Der Nutzer soll sich **nicht** um folgende Dinge kümmern müssen:

* ACP-Transport,
* JSON-RPC-Grundverdrahtung,
* Prozesshülle,
* Windows-Native-Interop,
* direkte FFM-Signaturen,
* Low-Level-Initialisierung der Windows-ML-Laufzeit.

---

## 3. Zielbild

### 3.1 Außen

Ein ACP-Client startet `win-acp-java` als Subprozess.

### 3.2 Innen

Der Agent verarbeitet ACP-Anfragen, verwaltet seinen Zustand, entscheidet anhand einer konfigurierbaren Verhaltensbeschreibung über nächste Schritte und nutzt bei Bedarf MCP-Tools.

### 3.3 Unten

Die Inferenzschicht läuft lokal auf Windows 11 und nutzt Java 21 mit FFM sowie direkt angesprochene Windows-DLLs.

---

## 4. Technischer Zielstack

### 4.1 Hülle

* ACP Java SDK
* Java 21

### 4.2 Verhaltensschicht

* LangGraph4j
* extern geladene Verhaltenskonfiguration

### 4.3 Tool-Integration

* MCP-Client-Fähigkeit
* Unterstützung mehrerer MCP-Server

### 4.4 Inferenzschicht

* Java 21 FFM
* `--enable-native-access=ALL-UNNAMED`
* `jextract` zur Erzeugung einer internen Windows-Interop-Lib
* Windows SDK Header
* Windows 11 DLLs
* Low-Level-Pfad über DXGI + D3D12 + DirectML

---

## 5. Nicht-Ziele

Folgende Themen sind ausdrücklich **nicht** Teil der ersten Version:

* Spring-basierte Laufzeit oder Spring Boot
* Unterstützung für Linux oder macOS
* Vollständige Ollama-Kompatibilität
* Eigene Modellverwaltung mit Download, Pull oder Registry
* Ausführung beliebigen Java-Codes aus Konfiguration
* Direkte Unterstützung für beliebige Drittanbieter-DLLs
* Cloud-First-Architektur
* Generisches Workflow-System außerhalb des Agentenkontexts

---

## 6. Zielgruppen

### Primäre Zielgruppe

* Java-Entwickler und Teams, die einen lokal laufenden ACP-Agenten unter Windows betreiben wollen
* Teams mit Java-8-Hauptanwendung und Java-21-Sidecar
* Teams, die interne Tools und Intranet-Systeme per MCP anbinden wollen

### Sekundäre Zielgruppe

* Produktteams, die einen steuerbaren Unternehmensagenten auf JVM-Basis bereitstellen wollen
* Entwickler, die Windows-native lokale Inferenz nutzen möchten, ohne JNI/JNA von Hand zu schreiben

---

## 7. Hauptnutzen

`win-acp-java` bietet folgenden Mehrwert:

* standardisierte Anbindung an ACP-Clients,
* konfigurierbares Agentenverhalten ohne Code-Injektion,
* standardisierte Integration von MCP-Tools,
* lokale Inferenz unter Windows mit nativer Performance,
* klare Trennung zwischen Protokoll, Verhalten, Tools und Inferenz.

---

## 8. Zentrale Produktanforderungen

## 8.1 ACP-Agent als Produktkern

Das System muss einen ACP-Agentenprozess bereitstellen, der als externer Agent von einem ACP-Client gestartet werden kann.

### Muss-Anforderungen

* Der Agent muss als eigenständiger Java-21-Prozess startbar sein.
* Der Agent muss ACP-konform mit dem aufrufenden Client kommunizieren.
* Der Agent muss über Prozessparameter und Umgebungsvariablen konfigurierbar sein.
* Der Agent muss ohne zusätzliche manuelle Initialisierung nach Start betriebsbereit sein.

### Soll-Anforderungen

* Der Agent soll als lokale technische Komponente ohne Installer nutzbar sein.
* Der Agent soll mehrere Profile oder Konfigurationsdateien unterstützen.

---

## 8.2 Konfigurierbares Agentenverhalten

Das Verhalten des Agenten darf nicht fest nur im Java-Code verdrahtet sein. Es muss von außen beschreibbar sein.

### Ziel

Das Verhalten soll **einstellbar**, aber nicht **beliebig programmierbar** sein.

### Muss-Anforderungen

* Das Agentenverhalten muss extern konfigurierbar sein.
* Die Konfiguration darf keinen beliebigen Java-Code enthalten.
* Die Konfiguration muss auf einem festen Satz unterstützter Verhaltensbausteine basieren.
* Der Agent muss diese Konfiguration zur Laufzeit laden und anwenden können.
* Der Agent muss ungültige oder unsichere Konfigurationen sauber ablehnen.

### Konfigurierbare Aspekte

Folgende Aspekte müssen mindestens konfigurierbar sein:

* Systemrolle / Grundverhalten des Agenten
* aktive Nodes
* erlaubte Übergänge zwischen Nodes
* Startknoten
* Abbruchbedingungen
* Retry-Verhalten
* Freigabe- bzw. Approval-Regeln
* Tool-Nutzungsregeln
* Auswahlstrategie für Tools
* Fehlerbehandlung
* Verhalten bei fehlendem Ergebnis
* Verhalten bei mehreren passenden Tools
* Umgang mit Nutzer-Rückfragen
* Grenzen für autonome Aktionen

### Soll-Anforderungen

* Mehrere vordefinierte Agentenprofile sollen unterstützt werden.
* Konfigurationsdateien sollen versioniert werden können.
* Verhalten soll später durch weitere Node-Typen erweiterbar sein, ohne das Konfigurationsmodell zu brechen.

---

## 8.3 LangGraph4j als Verhaltens-Engine

LangGraph4j soll als technische Verhaltens-Engine dienen.

### Muss-Anforderungen

* Das System muss Agentenverhalten intern über State, Nodes und Edges abbilden.
* Es muss einen stabilen internen Satz von Node-Typen geben.
* Es muss einen stabilen internen Satz von Edge-/Routing-Typen geben.
* Die externe Konfiguration darf nur erlaubte Node- und Edge-Typen referenzieren.

### Vordefinierte Node-Kategorien

Die erste Version soll mindestens folgende Node-Kategorien unterstützen:

* Eingabe analysieren
* Ziel bestimmen
* Kontext laden
* verfügbares Tool auswählen
* Tool ausführen
* Tool-Ergebnis bewerten
* Rückfrage erzeugen
* Antwort formulieren
* Abschluss erzeugen
* Fehler behandeln
* Freigabe einholen

### Vordefinierte Routing-Kategorien

Die erste Version soll mindestens folgende Routing-Kriterien unterstützen:

* Tool erforderlich ja/nein
* Tool erlaubt ja/nein
* Freigabe erforderlich ja/nein
* Ergebnis ausreichend ja/nein
* Rückfrage erforderlich ja/nein
* maximale Versuche erreicht ja/nein
* Abschluss erreicht ja/nein

---

## 8.4 MCP-Tools akzeptieren und nutzen

Das System muss MCP-Tools und MCP-Ressourcen als standardisierte Fähigkeiten des Agenten akzeptieren.

### Muss-Anforderungen

* Der Agent muss MCP-Server anbinden können.
* Der Agent muss mehrere MCP-Server gleichzeitig unterstützen können.
* Der Agent muss verfügbare MCP-Tools erkennen und intern als nutzbare Tool-Kandidaten modellieren.
* Der Agent muss konfigurierbar steuern können, welche MCP-Tools erlaubt sind.
* Der Agent muss Tool-Aufrufe über seine Verhaltenslogik anstoßen können.
* Der Agent muss Tool-Ergebnisse in seinen Zustand übernehmen können.

### Konfigurierbare Aspekte

Folgende MCP-relevanten Aspekte müssen von außen konfigurierbar sein:

* registrierte MCP-Server
* Transportparameter
* Authentisierung
* Timeouts
* Tool-Whitelist / Tool-Blacklist
* Tool-Gruppen
* Priorisierung von Tools
* erlaubte Tools pro Agentenprofil
* erlaubte Tools pro Node oder Status
* Freigabepflicht pro Tool oder Tool-Gruppe

### Soll-Anforderungen

* Der Agent soll MCP-Ressourcen und Tools getrennt behandeln können.
* Der Agent soll Tool-Metadaten für bessere Tool-Auswahl verwenden können.
* Der Agent soll Tool-Aufrufe nachvollziehbar protokollieren können.

---

## 8.5 Lokale Inferenz unter Windows

Die Komponente muss lokale Inferenz unter Windows 11 ermöglichen.

### Muss-Anforderungen

* Die Inferenzschicht muss auf Windows 11 lauffähig sein.
* Die Inferenzschicht muss mit Java 21 und FFM nutzbar sein.
* Es dürfen keine externen EXEs erforderlich sein.
* Es sollen nur Windows-eigene DLLs verwendet werden.
* Die Low-Level-Anbindung muss über eine interne Java-Interop-Lib gekapselt werden.
* Direkte manuelle FFM-Aufrufe sollen nicht in der Agentenlogik verteilt werden.

### Technische Leitplanken

* `jextract` wird verwendet, um Windows-Bindings zu erzeugen.
* Das Windows SDK liefert die Header.
* Die erzeugten Bindings werden in einer internen Bibliothek gekapselt.
* Darüber wird eine fachliche Inferenzschicht bereitgestellt.

### Soll-Anforderungen

* Die Inferenzschicht soll austauschbar gekapselt sein.
* Der Rest des Systems soll die Inferenz über ein Interface konsumieren.

---

## 8.6 Konfiguration von außen

Das System muss von außen über Konfigurationsdateien und Startparameter steuerbar sein.

### Muss-Anforderungen

* Der Agent muss eine externe Hauptkonfigurationsdatei laden können.
* Der Pfad zur Konfiguration muss per Argument oder Umgebungsvariable übergeben werden können.
* Konfigurationsfehler müssen klar gemeldet werden.
* Die Konfiguration muss validiert werden, bevor der Agent in den Betriebsmodus geht.

### Konfigurationsbereiche

Mindestens folgende Bereiche müssen in der Konfiguration vorgesehen werden:

* Agent-Metadaten
* Modell- und Inferenzparameter
* Graph-/Verhaltensdefinition
* MCP-Server und Tool-Regeln
* Logging und Observability
* Sicherheits- und Freigaberichtlinien
* Laufzeitparameter

---

## 8.7 Sicherheit und Kontrolle

Der Agent darf nicht unkontrolliert handeln.

### Muss-Anforderungen

* Der Agent muss Freigabe- und Sicherheitsregeln berücksichtigen.
* Unsichere oder nicht erlaubte Tool-Aufrufe müssen blockiert werden können.
* Die Konfiguration darf keine Ausführung beliebigen Benutzer-Codes erlauben.
* Tool-Aufrufe müssen nachvollziehbar bleiben.
* Fehler in MCP-Servern dürfen den Agenten nicht unkontrolliert abstürzen lassen.

### Soll-Anforderungen

* Es soll einen sicheren Default-Modus geben.
* Es soll einen Modus für restriktive Unternehmensumgebungen geben.

---

## 8.8 Beobachtbarkeit

### Muss-Anforderungen

* Agentenstart, Konfigurationsladevorgang und Tool-Registrierung müssen protokolliert werden.
* Ausgewählte Nodes, Übergänge und Tool-Aufrufe müssen nachvollziehbar protokolliert werden.
* Fehler in Inferenz, Tool-Aufruf und Konfigurationsauswertung müssen getrennt erkennbar sein.

### Soll-Anforderungen

* Es soll ein Debug-Modus für Graph- und Zustandsverfolgung geben.
* Es soll eine lesbare Ausführungsspur pro Agentenlauf geben.

---

## 9. Fachliche Kernobjekte

Die erste Fassung des Domänenmodells soll mindestens folgende Konzepte enthalten:

* AgentProfile
* AgentBehaviorDefinition
* AgentState
* NodeDefinition
* EdgeDefinition
* ToolPolicy
* McpServerDefinition
* ToolExecutionRequest
* ToolExecutionResult
* InferenceRequest
* InferenceResult
* ApprovalPolicy
* RuntimeConfiguration

---

## 10. Qualitätsanforderungen

### Wartbarkeit

* Strikte Trennung von ACP, Verhalten, MCP und Inferenz
* Klare Interfaces zwischen den Schichten
* Kein Verteilen nativer Details über den ganzen Code

### Erweiterbarkeit

* Neue Node-Typen sollen ergänzbar sein.
* Neue Routing-Typen sollen ergänzbar sein.
* Neue MCP-Servertypen sollen ergänzbar sein.
* Die Inferenzschicht soll austauschbar bleiben.

### Testbarkeit

* Verhalten muss unabhängig von echter Windows-Inferenz testbar sein.
* MCP-Tools müssen mockbar sein.
* Konfigurationsvalidierung muss separat testbar sein.

### Robustheit

* Fehler in einzelnen Tools dürfen das Gesamtsystem nicht unbrauchbar machen.
* Fehlerhafte Konfiguration darf nicht zu undefiniertem Verhalten führen.

---

## 11. Modulzuschnitt

Ein möglicher erster Modulzuschnitt:

* `win-acp-java-acp`
* `win-acp-java-graph`
* `win-acp-java-mcp`
* `win-acp-java-config`
* `win-acp-java-windows-bindings`
* `win-acp-java-inference`
* `win-acp-java-runtime`

---

## 12. Erste fachliche Use Cases

### UC1 – Agent beantwortet eine Frage ohne Tool

Der Nutzer stellt eine Frage. Der Agent erkennt, dass kein Tool benötigt wird, und erzeugt direkt eine Antwort.

### UC2 – Agent verwendet ein MCP-Suchtool

Der Nutzer stellt eine Frage zum Intranet. Der Agent erkennt passenden Tool-Bedarf, wählt ein erlaubtes MCP-Suchtool, ruft es auf, bewertet das Ergebnis und antwortet.

### UC3 – Agent fordert Freigabe an

Der Nutzer fragt nach einer Aktion mit erhöhtem Risiko. Der Agent erkennt die Approval-Pflicht und stoppt bis zur Freigabe.

### UC4 – Tool ist verfügbar, aber nicht erlaubt

Ein passendes Tool ist vorhanden, aber durch Policy gesperrt. Der Agent muss alternative Wege prüfen oder sauber begründen, warum er nicht fortfahren kann.

### UC5 – Mehrere Tools sind verfügbar

Der Agent bewertet mehrere mögliche MCP-Tools, priorisiert sie gemäß Konfiguration und wählt das geeignete Tool aus.

### UC6 – Tool liefert unzureichende Ergebnisse

Der Agent erkennt unvollständige Ergebnisse, fragt nach, versucht einen alternativen Pfad oder beendet sauber.

---

## 13. Akzeptanzkriterien für die erste Produktversion

Die erste Version ist erfolgreich, wenn:

* ein ACP-Client den Agenten als Subprozess starten kann,
* eine externe Konfiguration geladen und validiert wird,
* das Agentenverhalten nicht nur fest im Code verdrahtet ist,
* mindestens ein konfigurierbarer Graph erfolgreich ausgeführt werden kann,
* mindestens ein MCP-Tool erkannt, ausgewählt und genutzt werden kann,
* Tool-Nutzung durch Policies begrenzt werden kann,
* die Inferenzschicht lokal unter Windows 11 lauffähig ist,
* die Low-Level-Windows-Anbindung intern gekapselt bleibt,
* kein beliebiger Java-Code aus Konfiguration ausgeführt wird.

---

## 14. Offene Produktentscheidungen

Folgende Punkte müssen noch entschieden werden:

1. Format der externen Konfiguration

    * YAML
    * JSON
    * beide

2. Umfang der ersten Verhaltens-DSL

    * nur Graph-Struktur
    * Graph + Policies
    * Graph + Policies + Tool-Regeln

3. Umgang mit MCP-Ressourcen zusätzlich zu Tools

4. Persistenzstrategie für AgentState und Checkpoints

5. Welche Approval-Modelle in V1 enthalten sind

6. Wie stark die Inferenzschicht in V1 abstrahiert wird

---

## 15. Produktentscheidung für die nächste Phase

Die nächste Phase soll auf folgende Ziele fokussieren:

1. Minimalen ACP-Agenten lauffähig machen
2. Konfigurationsmodell definieren
3. Verhaltens-DSL auf Basis fester Node-/Edge-Typen definieren
4. MCP-Client-Integration für mindestens einen Tool-Server bereitstellen
5. Windows-Interop-Layer mit `jextract` vorbereiten
6. Erste Inferenzabstraktion über den Windows-Low-Level-Pfad definieren

---

## 16. Zusammenfassung

`win-acp-java` ist keine allgemeine Agentenplattform, sondern eine technische Windows-Komponente mit klarer Verantwortung:

* ACP außen,
* konfigurierbares Verhalten in der Mitte,
* MCP-Tools als Fähigkeiten,
* lokale Windows-Inferenz unten.

Die Komponente soll dem Nutzer nicht das Agentenkonzept abnehmen, sondern eine kontrollierbare, konfigurierbare und lokal lauffähige technische Basis dafür bereitstellen.

---

## 17. Folgephase – Lokale Embeddings und lokales Reranking

### 17.1 Ziel der Folgephase

Nach Fertigstellung des aktuellen Phi-/Driver-Schwerpunkts soll `win-acp-java` zusätzlich lokale Embedding- und Rerank-Modelle unterstützen.

Diese Folgephase verfolgt ausdrücklich **kein Cloud-Ziel**. Es werden nur lokal lauffähige Modelle berücksichtigt.

### 17.2 Produktentscheidung

Für diese Folgephase gilt:

* Es werden **nur lokale Modelle** unterstützt.
* Cloud-Modelle werden nicht berücksichtigt.
* Die Umsetzung erfolgt **ONNX-first**.
* Es wird **keine generische ONNX-Runtime für beliebige Modelle** gebaut.
* Unterstützt werden nur explizit freigegebene Modellfamilien.

### 17.3 Unterstützte Modellliste der Folgephase

### Lokale Embedding-Modelle

* `all-minilm`
* `nomic-embed-text`

### Lokale Rerank-Modelle

* `BAAI/bge-reranker-base`
* `BAAI/bge-reranker-large`
* `BAAI/bge-reranker-v2-m3`
* `jinaai/jina-reranker-v2-base-multilingual`

### Ausdrücklich nicht Teil von V1 dieser Folgephase

* `BAAI/bge-reranker-v2-gemma`
* alle Cloud-Embedding-Modelle
* alle Cloud-Rerank-Modelle
* allgemeine LLM-Inferenz für Embeddings/Reranking in dieser Phase

### 17.3.1 Konkrete Modellquellen

Die folgenden Modellquellen sind verbindliche Referenzen für die Implementierung.

#### Embeddings

##### `all-minilm`

Primäre Modellquelle:

* Modellseite: `sentence-transformers/all-MiniLM-L6-v2`
* ONNX-Dateien: `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main/onnx`
* Root-Modellseite: `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`

Alternative kompakte ONNX-Port-Quelle:

* `https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx`

Verwendung in V1:

* ONNX-first
* Root-Modellquelle für Tokenizer/Metadaten bevorzugen
* ONNX-Dateien aus dem offiziellen `onnx/`-Ordner oder dem kompakten ONNX-Port verwenden

##### `nomic-embed-text`

Modellquelle:

* Modellseite: `nomic-ai/nomic-embed-text-v1.5`
* ONNX-Dateien: `https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/onnx`
* Root-Modellseite: `https://huggingface.co/nomic-ai/nomic-embed-text-v1.5`

Verwendung in V1:

* ONNX-first
* Task-Präfixe für Query/Dokument müssen unterstützt werden

#### Reranker

##### `BAAI/bge-reranker-base`

Modellquelle:

* Modellseite: `BAAI/bge-reranker-base`
* ONNX-Dateien: `https://huggingface.co/BAAI/bge-reranker-base/tree/main/onnx`
* Root-Modellseite: `https://huggingface.co/BAAI/bge-reranker-base`

##### `BAAI/bge-reranker-large`

Modellquelle:

* Modellseite: `BAAI/bge-reranker-large`
* ONNX-Dateien: `https://huggingface.co/BAAI/bge-reranker-large/tree/main/onnx`
* Root-Modellseite: `https://huggingface.co/BAAI/bge-reranker-large`

##### `BAAI/bge-reranker-v2-m3`

Offizielle Modellquelle:

* Root-Modellseite: `https://huggingface.co/BAAI/bge-reranker-v2-m3`

ONNX-Quelle für V1:

* Community-ONNX-Port: `https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX`
* ONNX-Dateien: `https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/tree/main/onnx`

Verwendung in V1:

* Für V1 ist der Community-ONNX-Port zulässig
* In der Dokumentation muss kenntlich gemacht werden, dass die ONNX-Dateien nicht aus dem offiziellen BAAI-Repository stammen

##### `jinaai/jina-reranker-v2-base-multilingual`

Modellquelle:

* Modellseite: `jinaai/jina-reranker-v2-base-multilingual`
* ONNX-Dateien: `https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/tree/main/onnx`
* Root-Modellseite: `https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual`

##### `BAAI/bge-reranker-v2-gemma` (nicht V1)

Referenzquelle für spätere Phase:

* Root-Modellseite: `https://huggingface.co/BAAI/bge-reranker-v2-gemma`

Verwendung:

* Nicht Teil von V1
* Nur Referenz für spätere Evaluierung

### 17.4 Zielarchitektur für die Folgephase

Die Folgephase soll auf einem gemeinsamen lokalen ONNX-Unterbau aufsetzen.

#### Geplanter Schichtenzuschnitt

* Modell-Registry
* Tokenizer-Laufzeit
* Tensor-Building
* ONNX-Ausführungsadapter
* DirectML-/Windows-Ausführungsbackend
* Embedding-Pipeline
* Rerank-Pipeline

Die Trennung zwischen Embeddings und Reranking muss fachlich klar bleiben.

### 17.5 Fachliche Anforderungen – Embeddings

Ein lokales Embedding-Modell muss einen Eingabetext in einen numerischen Vektor überführen.

#### Muss-Anforderungen

* Das System muss Text lokal tokenisieren können.
* Das System muss ONNX-Encoder-Modelle für Embeddings lokal ausführen können.
* Das System muss `input_ids` und `attention_mask` korrekt erzeugen können.
* Das System muss modellabhängig den finalen Embedding-Vektor bestimmen können.
* Das System muss Embeddings als `float[]` bereitstellen können.
* Es darf kein Cloud-Fallback existieren.

#### Modellbesonderheiten

##### `all-minilm`

* Das Modell muss als lokales Embedding-Modell unterstützt werden.
* Der erforderliche Pooling-/Output-Pfad muss korrekt implementiert werden.

##### `nomic-embed-text`

* Das Modell muss lokal unterstützt werden.
* Task-Präfixe wie Suchanfrage und Suchdokument müssen konfigurierbar berücksichtigt werden.
* Das System muss zwischen Query- und Dokument-Embedding unterscheiden können.

### 17.6 Fachliche Anforderungen – Reranking

Ein lokales Rerank-Modell muss Query und Dokument gemeinsam bewerten und einen Relevanz-Score liefern.

#### Muss-Anforderungen

* Das System muss Query und Dokument gemeinsam tokenisieren können.
* Das System muss ONNX-Cross-Encoder lokal ausführen können.
* Das System muss pro Query/Dokument-Paar einen Score als `float` liefern.
* Das System muss mehrere Kandidatendokumente für eine Query bewerten können.
* Das System darf keine Embedding-Pipeline mit einer Rerank-Pipeline vermischen.

#### Unterstützte Modelle dieser Phase

* `BAAI/bge-reranker-base`
* `BAAI/bge-reranker-large`
* `BAAI/bge-reranker-v2-m3`
* `jinaai/jina-reranker-v2-base-multilingual`

### 17.7 Gemeinsamer ONNX-Unterbau

Für Embedding- und Rerank-Modelle soll ein gemeinsamer technischer Unterbau geschaffen werden.

#### Muss-Anforderungen

* Der Unterbau muss lokale ONNX-Modelle laden können.
* Der Unterbau muss Eingabetensoren für Textmodelle aufbauen können.
* Der Unterbau muss die Modellausführung vom Rest des Systems kapseln.
* Direkte Low-Level-Details dürfen nicht in die Fachlogik von Embedding und Reranking auslaufen.

#### Soll-Anforderungen

* Der Unterbau soll für mehrere freigegebene Modellfamilien wiederverwendbar sein.
* Modellformatspezifische Unterschiede sollen in dedizierten Adapterklassen gekapselt werden.

### 17.8 Modell-Registry

Die Folgephase soll eine explizite Modell-Registry einführen.

#### Muss-Anforderungen

Für jedes freigegebene Modell müssen mindestens folgende Informationen hinterlegt werden können:

* technischer Modellname
* fachliche Modellfamilie
* Typ: `EMBEDDING` oder `RERANK`
* Format: `ONNX`
* Tokenizer-Typ
* Output-Verhalten
* erforderliche Präfix-Regeln
* lokale Modellpfade

#### Ziel

Die Registry soll verhindern, dass Modellwissen unkontrolliert im Code verteilt wird.

### 17.9 Öffentliche APIs der Folgephase

#### Embedding-API

Die Folgephase soll eine fachlich klare Embedding-API bereitstellen.

Beispielhafter Zielzuschnitt:

* Text rein
* Embedding-Vektor raus

#### Rerank-API

Die Folgephase soll eine fachlich klare Rerank-API bereitstellen.

Beispielhafter Zielzuschnitt:

* Query + Dokument rein
* Score raus

### 17.10 Qualitätsanforderungen der Folgephase

#### Muss-Anforderungen

* Alle unterstützten Modelle müssen lokal lauffähig sein.
* Alle unterstützten Modelle müssen reproduzierbar testbar sein.
* Modellpfade und Konfiguration müssen mit dem Repo konsistent sein.
* Cloud-Abhängigkeiten dürfen nicht versehentlich aktiviert werden.

#### Soll-Anforderungen

* Embedding- und Rerank-Pfade sollen getrennt benchmarkbar sein.
* Die Modell-Registry soll einfach erweiterbar bleiben.
* Der gemeinsame ONNX-Unterbau soll nicht unnötig generisch werden.

### 17.11 Testanforderungen

#### Embedding-Tests

Für jedes unterstützte Embedding-Modell sollen mindestens folgende Tests vorhanden sein:

* Modell parsebar
* Modell ladbar
* Einbettung eines einfachen Beispieltexts
* deterministische Wiederholung desselben Inputs
* plausibler Dimensions-Check des Embedding-Vektors

#### Rerank-Tests

Für jedes unterstützte Rerank-Modell sollen mindestens folgende Tests vorhanden sein:

* Modell parsebar
* Modell ladbar
* Score-Berechnung für Query/Dokument-Paar
* deterministische Wiederholung desselben Inputs
* Ranking mehrerer Kandidaten in einer einfachen Referenzsituation

### 17.12 Reihenfolge der Umsetzung

Die Folgephase soll in dieser Reihenfolge umgesetzt werden:

1. `all-minilm`
2. `nomic-embed-text`
3. `BAAI/bge-reranker-base`
4. `BAAI/bge-reranker-v2-m3`
5. `jinaai/jina-reranker-v2-base-multilingual`
6. `BAAI/bge-reranker-large`

### 17.13 Definition of Done für die Folgephase

Die Folgephase ist erfolgreich, wenn:

* alle unterstützten lokalen Embedding-Modelle lokal lauffähig sind,
* alle unterstützten lokalen Rerank-Modelle lokal lauffähig sind,
* kein Cloud-Modell mehr Teil des Umfangs ist,
* eine Modell-Registry vorhanden ist,
* Embedding und Reranking fachlich getrennte APIs haben,
* der gemeinsame ONNX-Unterbau produktiv nutzbar ist,
* für alle unterstützten Modelle Tests vorhanden sind.
