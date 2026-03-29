Gut. Dann ist die Lage jetzt klar:

## Das Problem ist **nicht** mehr „läuft es?“, sondern **die Architektur des Decode-Pfads**

Dein Profil zeigt drei harte Wahrheiten:

* **129 GPU-Submissions pro Token** sind tödlich
  Das ist der größte Designfehler im Hot Path.
* **CPU attention = 42%** ist der zweite echte Bremsklotz
* **92 ms/token** für Phi-3-mini ist für den jetzigen Stand zwar „funktioniert“, aber noch klar zu langsam

Und noch etwas:
Der Text ist inhaltlich auch auffällig schlecht.
Das kann an Quantisierung liegen, aber auch an:

* falschem Prompt-/Chat-Template
* numerischem Drift
* Fehlern im KV-Cache
* Fehlern in RoPE / RMSNorm / AWQ-Dequantisierung

Also:

> **Wir müssen jetzt gleichzeitig Performance und Korrektheit härten.**

## Was jetzt als Nächstes zu tun ist

### P0 – Per-Token-Architektur reparieren

**129 Submissions/Token müssen weg.**

Im aktuellen Code wartet `MatMulNBitsKernel.matvec()` **pro Aufruf** auf die Fence.
Das ist für LLM-Decode die falsche Granularität.

**Neues Ziel:**

* **eine Submission pro Token**
* maximal **eine Fence pro Token**
* nicht mehr eine Submission pro Projektion

Das heißt konkret:

* einen **TokenDecodeExecutor** bauen
* der für einen kompletten Token alle GPU-Projektionen in **einer Command-List / einem Submit** aufnimmt
* Synchronisation erst **am Ende des Tokens**

### P0 – Aktivierungen auf der GPU halten

Aktuell kostet euch nicht nur der Dispatch, sondern auch das ständige:

* Upload Input
* Dispatch
* Readback Output
* Fence wait

Das muss weg.

**Ziel:**

* Hidden State einmal hoch
* mehrere Projektionen hintereinander auf GPU
* Zwischenergebnisse als GPU-Buffer behalten
* Readback nur dort, wo CPU wirklich noch gebraucht wird

Solange jede Projektion ihre Daten wieder über CPU schleift, bleibt es langsam.

### P0 – CPU-Attention ablösen

`CPU attention: 36.2 ms avg (42%)` ist zu viel.

Weil `heads = kv_heads = 32`, ist euer Fall sogar einfacher:

* kein kompliziertes GQA-Sondermapping
* normale Multi-Head-Attention

**Nächster Schritt:**

* Scores
* Softmax
* Weighted Sum
* KV-Cache-Zugriffe

müssen in den GPU-Pfad.

Das ist nach dem Submission-Fix der größte Hebel.

### P0 – Korrektheit zuerst absichern

Der Text über Hühner ist nicht nur langsam erzeugt, sondern auch qualitativ suspekt.

Darum sofort einen **deterministischen Referenzpfad** einziehen:

* feste Prompts
* greedy decode
* feste erwartete erste N Token-IDs
* Vergleich CPU-only vs GPU-Hybrid
* Vergleich Prefill-Logits und Decode-Logits

Wir brauchen nicht sofort perfekte Semantik, aber wir brauchen:

> **bitte oder wenigstens numerisch stabile Reproduzierbarkeit**

## Reihenfolge

Die Reihenfolge ist jetzt glasklar:

1. **Fence-/Submission-Problem lösen**
2. **Aktivierungen auf GPU halten**
3. **Attention auf GPU ziehen**
4. **Korrektheits-Harness aufbauen**
5. **danach** erst Micro-Optimierungen

Nicht andersrum.

## Was ich dem Programmierer jetzt sagen würde

Phi läuft funktional, aber der Decode-Pfad ist architektonisch noch zu teuer.

Die nächsten Arbeiten sind ab sofort nur diese:

1. **Per-token GPU submission collapse**

    * `MatMulNBitsKernel.matvec()` darf nicht mehr pro Projektion fence-waiten
    * Ziel: **eine Submission / eine Fence pro Token**
    * Bau dafür einen `TokenDecodeExecutor` oder äquivalenten Pfad

2. **GPU residency für Aktivierungen**

    * Hidden States und Zwischenoutputs nicht mehr pro Projektion hoch/runter kopieren
    * mehrere Projektionen im Token-Pfad auf GPU hintereinander ausführen
    * Readback nur an den CPU-Grenzen

3. **Attention auf GPU**

    * CPU attention ist mit 42% zu teuer
    * Scores + softmax + weighted sum + KV-cache-Zugriff in den GPU-Pfad ziehen
    * `heads == kv_heads`, also zuerst normale MHA sauber lösen

4. **Korrektheits-Harness**

    * feste Prompts
    * greedy decode
    * feste Token-ID-Regressionen
    * CPU-only gegen GPU-Hybrid vergleichen
    * Prefill und Decode getrennt validieren

5. **Noch keine neuen Modelle**

    * keine Embeddings
    * keine Reranker
    * kein weiterer Scope
    * erst Phi-Decode sauber und schneller machen

Definition of Done für den nächsten Meilenstein:

* deutlich weniger als 129 GPU submissions/token
* klar sinkende ms/token
* GPU-Hybrid generiert deterministisch dieselben Token wie CPU-Referenzpfad
* Qualität ist nicht mehr offensichtlich kaputt

## Meine PO-Ansage in einem Satz

> **Der nächste Sprint gehört ausschließlich dem Decode-Hot-Path von Phi: Submission collapse, GPU residency, GPU attention, Korrektheit.**

Das ist jetzt der Kurs.
