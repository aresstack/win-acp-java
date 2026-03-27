package com.aresstack.winacp.acp;

import com.aresstack.winacp.graph.AgentGraphRunner;
import com.aresstack.winacp.graph.AgentState;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * ACP-compatible agent server using JSON-RPC 2.0 over stdio.
 * <p>
 * Reads requests from stdin (Content-Length framed), dispatches to
 * the {@link AgentGraphRunner}, and writes responses to stdout.
 */
public class AcpAgentServer {

    private static final Logger log = LoggerFactory.getLogger(AcpAgentServer.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    private final AgentGraphRunner graphRunner;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final InputStream input;
    private final OutputStream output;

    public AcpAgentServer(AgentGraphRunner graphRunner) {
        this(graphRunner, System.in, System.out);
    }

    /** Testable constructor with custom streams. */
    public AcpAgentServer(AgentGraphRunner graphRunner, InputStream input, OutputStream output) {
        this.graphRunner = graphRunner;
        this.input = input;
        this.output = output;
    }

    /**
     * Start the JSON-RPC message loop. Blocks until shutdown or EOF.
     */
    public void start() {
        running.set(true);
        log.info("ACP Agent Server started (stdio JSON-RPC 2.0)");

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8))) {
            while (running.get()) {
                String rawMessage = readMessage(reader);
                if (rawMessage == null) {
                    log.info("stdin closed – stopping");
                    break;
                }
                handleRawMessage(rawMessage);
            }
        } catch (IOException e) {
            if (running.get()) {
                log.error("IO error in message loop", e);
            }
        }
    }

    /**
     * Handle a single user message directly (for testing).
     */
    public String handleRequest(String userMessage) {
        AgentState state = new AgentState();
        state.setUserInput(userMessage);

        AgentState result = graphRunner.run(state);

        return result.getPendingResponse() != null
                ? result.getPendingResponse()
                : "(no response generated)";
    }

    public void shutdown() {
        running.set(false);
        log.info("ACP Agent Server shutting down");
    }

    public boolean isRunning() {
        return running.get();
    }

    // ---- JSON-RPC 2.0 wire protocol ----

    private String readMessage(BufferedReader reader) throws IOException {
        // Read headers until empty line
        int contentLength = -1;
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.isEmpty()) break;
            if (line.toLowerCase().startsWith("content-length:")) {
                contentLength = Integer.parseInt(line.substring(15).trim());
            }
        }
        if (line == null || contentLength < 0) return null;

        // Read body
        char[] body = new char[contentLength];
        int read = 0;
        while (read < contentLength) {
            int n = reader.read(body, read, contentLength - read);
            if (n == -1) return null;
            read += n;
        }
        return new String(body);
    }

    private void handleRawMessage(String rawJson) {
        try {
            JsonNode request = mapper.readTree(rawJson);
            String method = request.path("method").asText("");
            JsonNode id = request.get("id");

            log.info("Received JSON-RPC method: {}", method);

            ObjectNode result = switch (method) {
                case "initialize" -> handleInitialize(request);
                case "agent/run"  -> handleAgentRun(request);
                case "shutdown"   -> { shutdown(); yield resultNode("ok"); }
                default           -> errorNode(-32601, "Method not found: " + method);
            };

            if (id != null) {
                sendResponse(id, result, null);
            }
        } catch (Exception e) {
            log.error("Error handling message", e);
            try { sendResponse(null, null, errorNode(-32603, e.getMessage())); }
            catch (IOException ignored) {}
        }
    }

    private ObjectNode handleInitialize(JsonNode request) {
        ObjectNode result = mapper.createObjectNode();
        result.put("name", "win-acp-java");
        result.put("version", "0.1.0-SNAPSHOT");
        result.putObject("capabilities");
        return result;
    }

    private ObjectNode handleAgentRun(JsonNode request) {
        JsonNode params = request.path("params");
        String userMessage = params.path("message").asText(
                params.path("content").asText(""));

        String response = handleRequest(userMessage);

        ObjectNode result = mapper.createObjectNode();
        result.put("content", response);
        return result;
    }

    private void sendResponse(JsonNode id, ObjectNode result, ObjectNode error) throws IOException {
        ObjectNode response = mapper.createObjectNode();
        response.put("jsonrpc", "2.0");
        if (id != null) response.set("id", id);
        if (error != null) response.set("error", error);
        else response.set("result", result != null ? result : mapper.createObjectNode());

        byte[] bytes = mapper.writeValueAsBytes(response);
        String header = "Content-Length: " + bytes.length + "\r\n\r\n";

        synchronized (output) {
            output.write(header.getBytes(StandardCharsets.UTF_8));
            output.write(bytes);
            output.flush();
        }
    }

    private ObjectNode resultNode(String value) {
        ObjectNode node = mapper.createObjectNode();
        node.put("status", value);
        return node;
    }

    private ObjectNode errorNode(int code, String message) {
        ObjectNode node = mapper.createObjectNode();
        node.put("code", code);
        node.put("message", message);
        return node;
    }
}
