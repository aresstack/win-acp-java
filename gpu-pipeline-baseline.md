# GPU Pipeline Sprint — Baseline & Fortschritt

## Sprint-Ziel

**129 GPU-Submissions pro Token → 1**

Alle GEMM-Projektionen, Attention (Score/Softmax/V-Sum), RoPE, KV-Cache-Updates,
RMSNorm, SwiGLU und Residual-Adds in EINER D3D12-Command-List mit EINEM Fence-Wait.

---

## Baseline (vor V3.0)

| Metrik | V1.x (per-kernel) | V2.0 (MLP batch) |
|--------|-------------------|-------------------|
| Submissions/Token | 129 | 65 |
| Fence-Waits/Token | 129 | 65 |
| CPU→GPU Transfers/Token | 129 upload + 129 readback | 32 QKV + 32 MLP + 1 lmHead |
| Hidden States | CPU-resident | CPU-resident (GPU nur innerhalb MLP batch) |
| KV-Cache | CPU-only | CPU-only |
| Attention | CPU (parallel, 4-acc ILP) | CPU (parallel, 4-acc ILP) |
| RoPE | CPU | CPU |
| RMSNorm | CPU (+ GPU in MLP batch) | CPU (+ GPU in MLP batch) |
| SwiGLU | CPU (+ GPU in MLP batch) | CPU (+ GPU in MLP batch) |

### Benchmark-Kommando

```
/benchmark
```

In der `Phi3ChatCLI` ausführen. Misst 3 Runs à 32 Tokens nach 2 Warmup-Runs.

---

## V3.0 Full GPU Pipeline — Implementierung

### Status: IMPLEMENTIERT — Validierung ausstehend

Die V3.0-Implementierung ist vollständig in folgenden Dateien:

| Datei | Inhalt |
|-------|--------|
| `Phi3GpuPipeline.java` | `decodeTokenFullGpu()` — 1 Command-List für alle 32 Layers + LM head |
| `Phi3ComputeShaders.java` | HLSL Compute Shader: RoPE, KV-Cache-Store, AttnScore, AttnSoftmax, AttnVsum |
| `GpuComputeKernel.java` | Generic D3D12 compute shader Wrapper |
| `Phi3Runtime.java` | `decodeFast()` mit V3.0-Pfad-Erkennung und automatischem Fallback |

### Architektur

```
pipeline.begin()
  Upload: embedding[3072] → hiddenBufA

  for layer 0..31:
    RMSNorm(currentHidden → qkvKernel.input)          [compute]
    QKV GEMM(qkvKernel.input → qkvKernel.output)      [DML]
    RoPE(qkvKernel.output, cos/sin, in-place)          [compute]
    KV-Cache-Store(qkvKernel.output → K/V cache[l])    [compute]
    AttnScore(Q, K_cache → scores)                     [compute]
    AttnSoftmax(scores, in-place)                      [compute]
    AttnVSum(scores, V_cache → attnOut)                [compute]
    Scale(attnOut × attnOutScale → oKernel.input)      [compute]
    O GEMM(oKernel.input → oKernel.output)             [DML]
    Add(currentHidden + oKernel.output → residualBuf)  [compute]
    RMSNorm(residualBuf → guKernel.input)              [compute]
    GateUp GEMM(guKernel.input → guKernel.output)      [DML]
    SwiGLU(guKernel.output → downKernel.input)         [compute]
    Down GEMM(downKernel.input → downKernel.output)    [DML]
    Add(residualBuf + downKernel.output → nextHidden)  [compute]

  RMSNorm(finalHidden → lmHeadKernel.input)            [compute]
  LM Head GEMM(lmHeadKernel.input → logits)            [DML]
  Readback: logits[32064]

pipeline.submitAndWait()    ← EIN Fence-Wait
readback(logits → CPU)
argmax(logits)
```

### Neue HLSL Compute Shader

| Shader | UAVs | Constants | Funktion |
|--------|------|-----------|----------|
| `rope` | 3 (QKV, Cos, Sin) | headDim, numQHeads, numKvHeads, pos | RoPE in-place auf Q+K |
| `kv_cache_store` | 3 (QKV, Kcache, Vcache) | hidden, pos | K/V → GPU KV-Cache |
| `attn_score` | 3 (Q, Kcache, Scores) | headDim, numHeads, seqLen, scale | Q·K^T dot products |
| `attn_softmax` | 1 (Scores) | seqLen, numHeads | Per-Head Softmax |
| `attn_vsum` | 3 (Scores, Vcache, Output) | headDim, numHeads, seqLen | Gewichtete V-Summe |
| `scale` | 3 (X, Scale, Out) | count | Elementweises Multiply |
| `rms_norm` | 3 (In, Weight, Out) | dim, eps | RMSNorm mit Group-Shared Reduction |
| `element_add` | 3 (A, B, C) | count | Elementweise Addition |
| `swiglu` | 3 (GateUp, Scale, Out) | intermediate | SwiGLU + Scale fused |

### GPU-Speicher-Budget

| Komponente | Größe |
|------------|-------|
| FP32 Weights (32 layers) | ~14.5 GB |
| LM Head | ~394 MB |
| GPU KV-Cache (32 layers × 2048 pos) | ~768 MB |
| Scores Buffer | ~256 KB |
| Zwischenpuffer (hidden A/B, attnOut, residual) | ~48 KB |
| **Gesamt** | **~15.7 GB** (16 GB VRAM) |

### Submission-Vergleich

| Version | Submissions/Token | Fence-Waits | CPU↔GPU Transfers |
|---------|-------------------|-------------|-------------------|
| V1.x | 129 | 129 | 258 (upload+readback) |
| V2.0 | 65 | 65 | ~100 |
| **V3.0** | **1** | **1** | **2** (embed upload + logit readback) |

---

## GPU-Ressourcen-State-Management

### Implizite Promotion + Decay

- KV-Cache-Puffer: UAV via compute shader → **auto-decay** nach Execute
- Intermediate-Puffer (hiddenBufB, attnOut, residual): **auto-decay**
- hiddenBufA: explizite Transition (COPY_DEST→UAV) → **cleanup-Barrier** am Ende

### Barrier-Übersicht pro Layer

| # | Barrier-Typ | Zwischen |
|---|-------------|----------|
| 1-15 | UAV Barrier | Jede Operation |
| +1 | COPY_DEST→UAV | hiddenBufA nach Upload (nur 1× pro Token) |
| +1 | UAV→COMMON | hiddenBufA Cleanup (1× pro Token) |
| +1 | UAV→COPY_SOURCE | lmHead Output für Readback (1× pro Token) |

---

## Bekanntes Issue: CPU-KV-Cache-Synchronisation

**Kritisch für Fallback**: Wenn V3.0 aktiv ist (`pos < maxGpuPos`), werden
K/V-Werte nur im GPU-KV-Cache gespeichert. Der CPU-KV-Cache wird nicht
synchronisiert. Falls der pos-Wert `maxGpuPos` überschreitet und das System
auf V2.0 zurückfällt, fehlen alle GPU-decodierten Positionen im CPU-Cache.

**Lösung V1**: `maxGpuPos = min(maxPositionEmbeddings, 2048)` begrenzt die
GPU-KV-Cache-Größe. Über 2048 Positionen fällt das System auf V2.0 zurück.
In der Praxis sind die meisten Phi-3-mini-Konversationen deutlich kürzer.

**Zukünftige Lösung**: Bei Fallback-Schwellwert GPU→CPU Readback des
gesamten KV-Cache triggern (einmalig, ~768 MB).

---

## Validierungs-Checkliste

- [ ] V2.0 Baseline-Benchmark festhalten (`/benchmark`)
- [ ] V3.0 auf Hardware testen — Shader-Kompilierung OK?
- [ ] V3.0 VRAM-Verbrauch messen (erwarteter Peak ~15.7 GB)
- [ ] Numerische Korrektheit: V3.0 vs V2.0 Logits vergleichen (Top-5 match?)
- [ ] Performance V3.0 vs V2.0 messen (ms/token)
- [ ] Multi-Turn-Stabilität: 10+ Turns ohne Fehler
- [ ] Fallback testen: pos > maxGpuPos → V2.0 automatisch
- [ ] KV-Cache-Reuse über Turns validieren
- [ ] GPU-Ressourcen-Cleanup nach close() validieren (keine Leaks)

---

## Test-Kommandos

### Benchmark (CLI)
```
/benchmark
```

### Manueller Vergleich V2.0 vs V3.0
```bash
# V2.0 erzwingen (V3.0 Shader deaktivieren):
java -Dphi3.gpu.v3=false --enable-native-access=ALL-UNNAMED -Xmx4g -cp ... Phi3ChatCLI

# V3.0 (default, wenn Shader kompilieren):
java --enable-native-access=ALL-UNNAMED -Xmx4g -cp ... Phi3ChatCLI
```
