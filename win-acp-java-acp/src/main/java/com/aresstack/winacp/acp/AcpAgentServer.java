package com.aresstack.winacp.acp;

import com.aresstack.winacp.graph.AgentState;
import com.aresstack.winacp.graph.LangGraphAgentRunner;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * ACP-conformant agent server implementing the
 * <a href="https://agentclientprotocol.com">Agent Client Protocol</a>
 * over JSON-RPC 2.0 / stdio.
 * <p>
 * Implements the baseline ACP methods:
 * <ul>
 *   <li>{@code initialize} – capability negotiation</li>
 *   <li>{@code session/new} – create a session</li>
 *   <li>{@code session/prompt} – process a user prompt</li>
 *   <li>{@code session/cancel} – cancel a running prompt (notification)</li>
 * </ul>
 * Sends {@code session/update} notifications back to the client during
 * prompt processing.
 */
public class AcpAgentServer {

    private static final Logger log = LoggerFactory.getLogger(AcpAgentServer.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    /** ACP protocol version (uint16 per spec). */
    static final int PROTOCOL_VERSION = 1;

    private final LangGraphAgentRunner graphRunner;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final InputStream input;
    private final OutputStream output;

    /** Active sessions keyed by session ID. */
    private final Map<String, SessionState> sessions = new ConcurrentHashMap<>();

    public AcpAgentServer(LangGraphAgentRunner graphRunner) {
        this(graphRunner, System.in, System.out);
    }

    /** Testable constructor with custom streams. */
    public AcpAgentServer(LangGraphAgentRunner graphRunner, InputStream input, OutputStream output) {
        this.graphRunner = graphRunner;
        this.input = input;
        this.output = output;
    }

    // ---- lifecycle ----

    /** Start the JSON-RPC message loop. Blocks until shutdown or EOF. */
    public void start() {
        running.set(true);
        log.info("ACP Agent Server started (stdio JSON-RPC 2.0, protocolVersion={})", PROTOCOL_VERSION);

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

    public void shutdown() {
        running.set(false);
        log.info("ACP Agent Server shutting down");
    }

    public boolean isRunning() {
        return running.get();
    }

    // ---- public API for direct testing ----

    /**
     * Process a user message directly through the graph (bypasses JSON-RPC framing).
     */
    public String handleRequest(String userMessage) {
        AgentState state = new AgentState();
        state.setUserInput(userMessage);
        AgentState result = graphRunner.run(state);
        return result.getPendingResponse() != null
                ? result.getPendingResponse()
                : "(no response generated)";
    }

    // ---- JSON-RPC 2.0 wire protocol ----

    private String readMessage(BufferedReader reader) throws IOException {
        int contentLength = -1;
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.isEmpty()) break;
            if (line.toLowerCase().startsWith("content-length:")) {
                contentLength = Integer.parseInt(line.substring(15).trim());
            }
        }
        if (line == null || contentLength < 0) return null;

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

            // Notifications (no id expected in response)
            if ("session/cancel".equals(method)) {
                handleSessionCancel(request);
                return;
            }

            // Requests (id required)
            ObjectNode result = switch (method) {
                case "initialize"      -> handleInitialize(request);
                case "session/new"     -> handleSessionNew(request);
                case "session/prompt"  -> handleSessionPrompt(request);
                default                -> errorNode(-32601, "Method not found: " + method);
            };

            if (id != null) {
                boolean isError = result.has("code");
                sendResponse(id, isError ? null : result, isError ? result : null);
            }
        } catch (Exception e) {
            log.error("Error handling message", e);
            try { sendResponse(null, null, errorNode(-32603, e.getMessage())); }
            catch (IOException ignored) {}
        }
    }

    // ---- ACP method handlers ----

    /**
     * {@code initialize} – Capability negotiation per ACP spec.
     * <p>
     * Response contains {@code protocolVersion}, {@code agentInfo},
     * {@code agentCapabilities} and {@code authMethods}.
     *
     * @see <a href="https://agentclientprotocol.com/protocol/initialization">ACP Initialization</a>
     */
    private ObjectNode handleInitialize(JsonNode request) {
        ObjectNode result = mapper.createObjectNode();

        // protocolVersion (required, uint16)
        result.put("protocolVersion", PROTOCOL_VERSION);

        // agentInfo (Implementation object: name, version)
        ObjectNode agentInfo = mapper.createObjectNode();
        agentInfo.put("name", "win-acp-java");
        agentInfo.put("version", "0.1.0-SNAPSHOT");
        agentInfo.put("title", "Windows ACP Agent Host for Java 21");
        result.set("agentInfo", agentInfo);

        // agentCapabilities
        ObjectNode capabilities = mapper.createObjectNode();
        capabilities.put("loadSession", false);

        ObjectNode promptCaps = mapper.createObjectNode();
        promptCaps.put("image", false);
        promptCaps.put("audio", false);
        promptCaps.put("embeddedContext", false);
        capabilities.set("promptCapabilities", promptCaps);

        ObjectNode mcpCaps = mapper.createObjectNode();
        mcpCaps.put("http", false);
        mcpCaps.put("sse", false);
        capabilities.set("mcpCapabilities", mcpCaps);

        capabilities.putObject("sessionCapabilities");

        result.set("agentCapabilities", capabilities);

        // authMethods (empty – no auth required)
        result.putArray("authMethods");

        log.info("initialize: protocolVersion={}, agent=win-acp-java/0.1.0-SNAPSHOT", PROTOCOL_VERSION);
        return result;
    }

    /**
     * {@code session/new} – Create a new conversation session.
     * <p>
     * Accepts {@code cwd} and {@code mcpServers}, returns a unique {@code sessionId}.
     *
     * @see <a href="https://agentclientprotocol.com/protocol/session-setup">ACP Session Setup</a>
     */
    private ObjectNode handleSessionNew(JsonNode request) {
        JsonNode params = request.path("params");
        String cwd = params.path("cwd").asText(".");

        String sessionId = UUID.randomUUID().toString();
        sessions.put(sessionId, new SessionState(sessionId, cwd));

        ObjectNode result = mapper.createObjectNode();
        result.put("sessionId", sessionId);
        // modes: null (we don't support session modes yet)

        log.info("session/new: sessionId={}, cwd={}", sessionId, cwd);
        return result;
    }

    /**
     * {@code session/prompt} – Process a user prompt within a session.
     * <p>
     * Sends {@code session/update} notifications during processing,
     * then returns the final {@code PromptResponse} with a {@code stopReason}.
     *
     * @see <a href="https://agentclientprotocol.com/protocol/prompt-turn">ACP Prompt Turn</a>
     */
    private ObjectNode handleSessionPrompt(JsonNode request) {
        JsonNode params = request.path("params");
        String sessionId = params.path("sessionId").asText("");

        SessionState session = sessions.get(sessionId);
        if (session == null) {
            return errorNode(-32602, "Unknown session: " + sessionId);
        }

        if (session.isCancelled()) {
            session.resetCancelled();
            return promptResponse("cancelled");
        }

        // Extract text from prompt content blocks
        String userText = extractPromptText(params);
        log.info("session/prompt: sessionId={}, text='{}...'", sessionId,
                userText.length() > 60 ? userText.substring(0, 60) : userText);

        // Execute through LangGraph4j
        AgentState state = new AgentState();
        state.setUserInput(userText);

        AgentState result = graphRunner.run(state);

        // Send session/update notification with the agent's response
        String responseText = result.getPendingResponse() != null
                ? result.getPendingResponse()
                : "(no response generated)";

        try {
            sendSessionUpdate(sessionId, responseText);
        } catch (IOException e) {
            log.error("Failed to send session/update notification", e);
        }

        // Check for cancellation during execution
        if (session.isCancelled()) {
            session.resetCancelled();
            return promptResponse("cancelled");
        }

        return promptResponse("end_turn");
    }

    /**
     * {@code session/cancel} – Cancel ongoing operations for a session (notification).
     *
     * @see <a href="https://agentclientprotocol.com/protocol/prompt-turn#cancellation">ACP Cancellation</a>
     */
    private void handleSessionCancel(JsonNode request) {
        JsonNode params = request.path("params");
        String sessionId = params.path("sessionId").asText("");
        SessionState session = sessions.get(sessionId);
        if (session != null) {
            session.cancel();
            log.info("session/cancel: sessionId={}", sessionId);
        } else {
            log.warn("session/cancel: unknown sessionId={}", sessionId);
        }
    }

    // ---- Notifications (agent → client) ----

    /**
     * Send a {@code session/update} notification with an agent_message_chunk.
     */
    private void sendSessionUpdate(String sessionId, String text) throws IOException {
        ObjectNode notification = mapper.createObjectNode();
        notification.put("jsonrpc", "2.0");
        notification.put("method", "session/update");

        ObjectNode params = mapper.createObjectNode();
        params.put("sessionId", sessionId);

        ObjectNode update = mapper.createObjectNode();
        update.put("sessionUpdate", "agent_message_chunk");
        ObjectNode content = mapper.createObjectNode();
        content.put("type", "text");
        content.put("text", text);
        update.set("content", content);

        params.set("update", update);
        notification.set("params", params);

        byte[] bytes = mapper.writeValueAsBytes(notification);
        String header = "Content-Length: " + bytes.length + "\r\n\r\n";
        synchronized (output) {
            output.write(header.getBytes(StandardCharsets.UTF_8));
            output.write(bytes);
            output.flush();
        }
    }

    // ---- helpers ----

    /**
     * Extract plain text from ACP prompt content blocks.
     * Supports both text blocks ({@code [{"type":"text","text":"..."}]})
     * and a simple string fallback.
     */
    private String extractPromptText(JsonNode params) {
        JsonNode prompt = params.get("prompt");
        if (prompt == null) {
            return params.path("message").asText(
                    params.path("content").asText(""));
        }

        if (prompt.isArray()) {
            StringBuilder sb = new StringBuilder();
            for (JsonNode block : prompt) {
                if ("text".equals(block.path("type").asText())) {
                    if (!sb.isEmpty()) sb.append('\n');
                    sb.append(block.path("text").asText(""));
                }
            }
            return sb.toString();
        }

        return prompt.asText("");
    }

    private ObjectNode promptResponse(String stopReason) {
        ObjectNode result = mapper.createObjectNode();
        result.put("stopReason", stopReason);
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

    private ObjectNode errorNode(int code, String message) {
        ObjectNode node = mapper.createObjectNode();
        node.put("code", code);
        node.put("message", message);
        return node;
    }

    // ---- Session state ----

    static class SessionState {
        private final String sessionId;
        private final String cwd;
        private volatile boolean cancelled;

        SessionState(String sessionId, String cwd) {
            this.sessionId = sessionId;
            this.cwd = cwd;
        }

        String getSessionId() { return sessionId; }
        String getCwd() { return cwd; }

        void cancel() { cancelled = true; }
        boolean isCancelled() { return cancelled; }
        void resetCancelled() { cancelled = false; }
    }
}
