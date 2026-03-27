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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     win-acp-java-runtime                    │
│   Main entry point · wires all layers · starts ACP server   │
├──────────┬──────────┬──────────┬───────────┬────────────────┤
│   acp    │  graph   │   mcp    │ inference │ windows-bind.  │
│ ACP JSON │ LangGr.  │ MCP std  │ Inference │ jextract FFM   │
│  -RPC /  │  4j be-  │  io cli- │ engine    │ bindings for   │
│  stdio   │  havior  │  ent     │ (stub)    │ DirectML       │
│  server  │  engine  │          │           │ (placeholder)  │
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
| **inference** | Local inference engine abstraction | ⚠️ **Stub only** |
| **windows-bindings** | jextract / FFM bindings for DXGI, D3D12, DirectML | ⚠️ **Placeholder only** |
| **runtime** | Main entry point, wires all layers, `application` plugin | ✅ Implemented |

---

## Prerequisites

- **Java 21+** (Temurin recommended)
- **Gradle 8.x** (wrapper included)
- Windows 11 (for future DirectML features; builds and tests run on any OS)

## Quick Start

```bash
# Clone
git clone https://github.com/aresstack/win-acp-java.git
cd win-acp-java

# Build + test
./gradlew clean build

# Run the agent (listens on stdio for ACP JSON-RPC)
./gradlew :win-acp-java-runtime:run --args="--config application.yml"

# Or directly via JAR
java --enable-native-access=ALL-UNNAMED \
     -jar win-acp-java-runtime/build/libs/win-acp-java-runtime-0.1.0-SNAPSHOT.jar \
     --config application.yml
```

On **Windows** use `gradlew.bat` instead of `./gradlew`.

## Configuration

The agent is configured via a single YAML file. See [`agent.example.yaml`](agent.example.yaml)
for a fully annotated example.

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

mcp:
  servers:
    - name: my-tools
      command: "npx -y @my/mcp-server"
      env: {}
```

Configuration can be passed via:
- `--config <path>` CLI argument
- `WIN_ACP_CONFIG` environment variable
- Falls back to `application.yml` in the working directory

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
├── application.yml               # Default agent configuration
├── agent.example.yaml            # Annotated example configuration
├── win-acp-java-config/          # Configuration & domain model
├── win-acp-java-acp/             # ACP JSON-RPC server
├── win-acp-java-graph/           # LangGraph4j behavior engine
├── win-acp-java-mcp/             # MCP stdio client
├── win-acp-java-inference/       # Inference abstraction (stub)
├── win-acp-java-windows-bindings/# FFM/jextract bindings (placeholder)
└── win-acp-java-runtime/         # Main entry point + application plugin
```

## Key Dependencies

| Library | Version | Purpose |
|---|---|---|
| [LangGraph4j](https://github.com/bsorrentino/langgraph4j) | 1.8.10 | Agent behavior graph engine |
| [Jackson](https://github.com/FasterXML/jackson) | 2.17.1 | JSON / YAML parsing |
| [SLF4J](https://www.slf4j.org/) + [Logback](https://logback.qos.ch/) | 2.0.13 / 1.5.6 | Logging |
| [JUnit 5](https://junit.org/junit5/) | 5.10.2 | Testing |

## Roadmap

### ✅ Done (0.1.0-SNAPSHOT)
- ACP baseline protocol (`initialize`, `session/*`)
- LangGraph4j-based behavior graph with configurable nodes and edges
- MCP stdio client (`initialize`, `tools/list`, `tools/call`)
- YAML configuration loading and validation
- Inference engine abstraction with explicit stub
- CI pipeline (GitHub Actions)
- Gradle wrapper, multi-module build, Maven Central publishing skeleton

### 🔜 Next (planned)
- [ ] Windows-Bindings: jextract-generated FFM bindings for DXGI / D3D12 / DirectML
- [ ] Real inference engine implementation (DirectML backend)
- [ ] MCP robustness: error recovery, timeouts, reconnect
- [ ] ACP streaming support (`session/update` with partial results)
- [ ] ACP Java SDK integration (when available / stable)
- [ ] Integration tests with real MCP tool servers
- [ ] GGUF / ONNX model loading

### 🔮 Future
- [ ] GPU enumeration and selection
- [ ] Multi-model support
- [ ] Agent-to-agent communication
- [ ] Plugin system for custom node types

## Contributing

This project is in early alpha. Contributions are welcome, but please
open an issue first to discuss larger changes.

## License

[MIT](LICENSE) © 2026 AresStack

