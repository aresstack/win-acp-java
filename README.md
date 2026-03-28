# win-acp-java

> **⚠️ Status: `0.1.0-SNAPSHOT` – Technical Preview / Alpha**
>
> This project is under active development and **not production-ready**.
> APIs, configuration format, and module boundaries may change without notice.

A modular Java 21 agent host for Windows, implementing the
[Agent Client Protocol (ACP)](https://agentclientprotocol.com) with
[LangGraph4j](https://github.com/bsorrentino/langgraph4j)-powered behavior graphs
and [MCP](https://modelcontextprotocol.io) tool integration via stdio.

---

## V1 Scope – Small 28×28 Grayscale CNN · DirectML · Vertical Slice

**V1 scope:** small 28×28 grayscale CNN vertical slice, validated with
`mnist-12.onnx`, `mnist-12-int8.onnx`, and `mnist_emnist_blank_cnn_v1.onnx`.

| Model | Architecture | Output | Status |
|---|---|---|---|
| `mnist-12.onnx` | 2×Conv+ReLU, 2×MaxPool, Gemm (opset 12, float32) | 10 logits (digits 0–9) | ✅ |
| `mnist-12-int8.onnx` | Same, int8 quantized (dequantize-first) | 10 logits (digits 0–9) | ✅ |
| `mnist-8.onnx` | Same (opset 8, float32) | 10 logits (digits 0–9) | ✅ |
| `mnist_emnist_blank_cnn_v1.onnx` | 3×Conv+ReLU, 2×MaxPool, Gemm+ReLU (BN folded), Gemm | 11 logits (digits 0–9 + blank) | ✅ |

The EMNIST+blank model proves the pipeline handles a **second architecture**
with different layer topology (3 convolutions, batch normalization in the
classifier head, 11-class output). BatchNorm is **supported via inference-mode
fusion** (folded into FC weights at load time), not as a separate general-purpose
runtime operator. Dropout layers are absent in the ONNX graph (eliminated
during PyTorch eval-mode export).

The int8 model is supported via **dequantize-first**: quantized INT8 weights
are dequantized to float32 at load time, then processed through the same
DirectML operator pipeline.

The entire stack – from Java 21 FFM calls through DXGI → D3D12 → DirectML →
operator dispatch → argmax – is proven end-to-end with this model family.
**No other ONNX model architectures are supported in V1.** Generalized ONNX
operator coverage and text-generation / chat model support are future
milestones that will be tackled *after* the native layer is hardened and stable.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     win-acp-java-runtime                    │
│   Main entry point · wires all layers · starts ACP server   │
├──────────┬──────────┬──────────┬───────────┬────────────────┤
│   acp    │  graph   │   mcp    │ inference │ windows-bind.  │
│ ACP JSON │ LangGr.  │ MCP std  │ MNIST     │ FFM bindings   │
│  -RPC /  │  4j be-  │  io cli- │ DirectML  │ for dxgi.dll   │
│  stdio   │  havior  │  ent     │ engine    │ d3d12.dll      │
│  server  │  engine  │          │ (V1)      │ DirectML.dll   │
├──────────┴──────────┴──────────┴───────────┴────────────────┤
│                     win-acp-java-config                     │
│  YAML config loading · validation · domain model            │
└─────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Purpose | Status |
|---|---|---|
| **config** | YAML configuration, validation, domain model (Jackson) | ✅ Implemented |
| **acp** | ACP server: `initialize`, `session/new`, `session/prompt`, `session/cancel`, `session/update` over JSON-RPC 2.0 / stdio | ✅ Implemented |
| **graph** | LangGraph4j `StateGraph` – configurable agent behavior graph | ✅ Implemented |
| **mcp** | MCP stdio client: `initialize`, `tools/list`, `tools/call` | ✅ Implemented (happy path) |
| **inference** | MNIST digit classification via DirectML (V1 vertical slice) | ✅ `MnistDirectMlEngine` working end-to-end (mnist-12) |
| **windows-bindings** | Hand-written FFM bindings for `dxgi.dll`, `d3d12.dll`, `DirectML.dll` – calls Windows SDK DLLs directly via Java 21 Foreign Function & Memory API | ✅ **MNIST inference via DirectML – float32 + int8 models working** |
| **runtime** | Main entry point, wires all layers, `application` plugin | ✅ Implemented |

---

## Prerequisites

- **Java 21+** (Zulu / Temurin recommended)
- **Gradle 9.x** (wrapper included)
- **Windows 11** with **Windows SDK 10.0.26100+** installed (for DirectML native bindings)
- JVM flag `--enable-native-access=ALL-UNNAMED` and `--enable-preview` (set automatically by Gradle)

## Quick Start

```bash
# Clone
git clone https://github.com/aresstack/win-acp-java.git
cd win-acp-java

# Build + test
./gradlew clean build

# Run the agent (uses agent.example.yaml by default)
./gradlew :win-acp-java-runtime:run

# Or explicitly pass a config
./gradlew :win-acp-java-runtime:run --args="--config agent.example.yaml"

# Or directly via JAR
java --enable-native-access=ALL-UNNAMED --enable-preview \
     -jar win-acp-java-runtime/build/libs/win-acp-java-runtime-0.1.0-SNAPSHOT.jar
```

On **Windows** use `gradlew.bat` instead of `./gradlew`.

## Configuration

The agent is configured via a YAML file. The repo ships
[`agent.example.yaml`](agent.example.yaml) which works out of the box.
To customize, copy it to `application.yml` (gitignored) and edit as needed.

```yaml
profile:
  id: default
  name: My Agent
  systemRole: "You are a helpful assistant."

behavior:
  startNode: analyze
  maxIterations: 20
  nodes:
    - { id: analyze, type: ANALYZE_INPUT }
    - { id: goal,    type: DETERMINE_GOAL }
    - { id: respond, type: FORMULATE_RESPONSE }
    - { id: finalize,type: FINALIZE }
  edges:
    - { from: analyze, to: goal,     condition: ALWAYS }
    - { from: goal,    to: respond,  condition: TOOL_NOT_REQUIRED }
    - { from: respond, to: finalize, condition: ALWAYS }

# V1: MNIST-family CNN models supported (currently validated with mnist-12.onnx)
inference:
  modelPath: model/mnist-12.onnx
  backend: directml
```

Configuration resolution order:
1. `--config <path>` CLI argument
2. `WIN_ACP_CONFIG` environment variable
3. `application.yml` in working directory (local override, gitignored)
4. `agent.example.yaml` in working directory (shipped with repo)

## ACP Protocol

This server implements the [Agent Client Protocol](https://agentclientprotocol.com)
baseline over JSON-RPC 2.0 / stdio:

| Method | Direction | Description |
|---|---|---|
| `initialize` | client → agent | Capability negotiation |
| `session/new` | client → agent | Create a new session |
| `session/prompt` | client → agent | Send a user prompt |
| `session/cancel` | client → agent | Cancel a running prompt |
| `session/update` | agent → client | Streaming progress notifications |

> **Note:** This is a custom ACP implementation, not built on top of an
> official ACP Java SDK. The wire protocol is ACP-conformant, but the
> internal implementation is independent.

## MCP Tool Integration

The agent can connect to any [MCP](https://modelcontextprotocol.io)-compliant
tool server over stdio. The MCP client implements:

- `initialize` – handshake with the tool server
- `tools/list` – discover available tools
- `tools/call` – invoke a tool and return its result

Tool servers are configured in the YAML under `mcp.servers`.

## Project Structure

```
win-acp-java/
├── build.gradle                  # Root build: allprojects config, publishing
├── settings.gradle               # Module includes
├── agent.example.yaml            # Default configuration (shipped, works out of the box)
├── model/
│   ├── mnist-12.onnx             # MNIST model – V1 primary (float32, opset 12, 26 KB)
│   ├── mnist-12-int8.onnx        # MNIST model – int8 quantized (opset 12, 11 KB)
│   └── mnist-8.onnx              # MNIST model – also supported (float32, opset 8, 26 KB)
├── win-acp-java-config/          # Configuration & domain model
├── win-acp-java-acp/             # ACP JSON-RPC server
├── win-acp-java-graph/           # LangGraph4j behavior engine
├── win-acp-java-mcp/             # MCP stdio client
├── win-acp-java-inference/       # MNIST inference engine (DirectML + stub fallback)
├── win-acp-java-windows-bindings/# FFM bindings: DXGI, D3D12, DirectML – MNIST pipeline
└── win-acp-java-runtime/         # Main entry point + application plugin
```

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| [LangGraph4j](https://github.com/bsorrentino/langgraph4j) | 1.8.10 | Agent behavior graph engine |
| [Jackson](https://github.com/FasterXML/jackson) | 2.17.1 | JSON / YAML parsing |
| [SLF4J](https://www.slf4j.org/) + [Logback](https://logback.qos.ch/) | 2.0.13 / 1.5.6 | Logging |
| [JUnit 5](https://junit.org/junit5/) | 5.10.2 | Testing |
| Java 21 FFM (Preview) | JDK built-in | Foreign Function & Memory API for Windows SDK DLL calls |

### Windows SDK DLLs (loaded via FFM at runtime)

| DLL | Purpose |
|---|---|
| `dxgi.dll` | DXGI Factory, GPU adapter enumeration |
| `d3d12.dll` | Direct3D 12 device, command queues, buffers, descriptor heaps, fences |
| `DirectML.dll` | DirectML operator creation, compilation, dispatch, binding tables |

## MNIST DirectML Pipeline (V1 – MNIST-family CNN vertical slice)

The `windows-bindings` module implements a complete GPU inference pipeline for
MNIST-family CNN models using **only** Java 21 FFM calls to Windows system DLLs:

- **Float32**: `mnist-12.onnx`, `mnist-8.onnx` – weights parsed directly from ONNX
- **Int8 quantized**: `mnist-12-int8.onnx` – QLinearConv/QLinearMatMul weights
  dequantized to float32 at load time, then processed through the same DML pipeline

```
Input (784 floats, 28×28)
  ↓ Conv(8,1,5,5) + ReLU  → (1,8,28,28)
  ↓ MaxPool(2×2)           → (1,8,14,14)
  ↓ Conv(16,8,5,5) + ReLU → (1,16,14,14)
  ↓ MaxPool(3×3)           → (1,16,4,4)
  ↓ Gemm(256→10)           → (1,10)
  ↓ argmax
Output: predicted digit 0–9
```

**Key properties:**
- **No ORT, no JNA, no JNI** – zero third-party native dependencies
- All COM vtable calls via `java.lang.foreign.Linker.nativeLinker().downcallHandle()`
- Weights parsed from ONNX protobuf, uploaded to GPU via D3D12 upload buffers
- 5 DirectML operators compiled and dispatched per inference
- Deterministic: same input always produces same output

## Roadmap

### ✅ Done (0.1.0-SNAPSHOT)
- ACP baseline protocol (`initialize`, `session/*`)
- LangGraph4j-based behavior graph with configurable nodes and edges
- MCP stdio client (`initialize`, `tools/list`, `tools/call`)
- YAML configuration loading and validation
- Inference engine abstraction with explicit stub fallback
- **FFM bindings for Windows SDK**: hand-written `Linker`/`FunctionDescriptor` calls to `dxgi.dll`, `d3d12.dll`, `DirectML.dll`
- **MNIST end-to-end GPU inference**: MNIST-family models (float32 + int8 quantized) parsed, operators created, compiled and dispatched entirely via DirectML – Conv+Relu → MaxPool → Conv+Relu → MaxPool → Gemm → argmax
- **Int8 quantized model support**: `mnist-12-int8.onnx` loaded via dequantize-first pipeline – QLinearConv/QLinearMatMul/QLinearAdd weights dequantized to float32, same DML operator chain reused
- **DirectML inference engine**: DXGI→D3D12→DirectML device creation, descriptor heaps, binding tables, command recording, GPU synchronization
- Deterministic multi-run consistency verified
- CI pipeline (GitHub Actions)
- Gradle wrapper, multi-module build, Maven Central publishing skeleton

> **V1 scope**: MNIST-family CNN vertical slice, validated with
> `mnist-12.onnx` (float32) and `mnist-12-int8.onnx` (int8 quantized).
> This is a deliberate vertical slice to prove the pure-Java FFM → DirectML
> stack end-to-end, including quantized model support.

### 🔧 Hardening (current focus)
- [ ] COM lifecycle hardening (double-close guards, null-safe release)
- [ ] HRESULT diagnostic messages (known error codes)
- [ ] Fence/sync robustness (timeout, event-based wait)
- [ ] Descriptor/buffer resource leak prevention
- [ ] Multi-run stability tests (N-iteration, load-model-infer-close cycles)

### 🔜 Next (after hardening)
- [ ] Wire ACP `session/prompt` → LangGraph `INFER` → MNIST → argmax → answer (full vertical)
- [ ] MCP robustness: error recovery, timeouts, reconnect
- [ ] ACP streaming support (`session/update` with partial results)
- [ ] jextract-generated FFM bindings to supplement hand-written ones
- [ ] Integration tests with real MCP tool servers

### 🔮 Future (decide one at a time)
- [ ] Next supported model type **OR** first real ONNX operator generalization – not both at once
- [ ] GPU enumeration and selection
- [ ] Multi-model support
- [ ] Agent-to-agent communication
- [ ] Plugin system for custom node types

## Contributing

This project is in early alpha. Contributions are welcome, but please
open an issue first to discuss larger changes.

## License

[MIT](LICENSE) © 2026 AresStack

