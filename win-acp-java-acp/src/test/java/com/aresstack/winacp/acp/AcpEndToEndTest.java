package com.aresstack.winacp.acp;

import com.aresstack.winacp.config.*;
import com.aresstack.winacp.graph.LangGraphAgentRunner;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end test: ACP client → Agent → LangGraph4j graph → response.
 * <p>
 * Simulates a real ACP client sending the complete protocol flow over stdio
 * and verifying each response against the ACP specification.
 */
class AcpEndToEndTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    private LangGraphAgentRunner graphRunner;

    @BeforeEach
    void setUp() {
        graphRunner = new LangGraphAgentRunner(fullBehavior());
        graphRunner.registerDefaults();
    }

    /**
     * Full ACP protocol flow:
     * <ol>
     *   <li>{@code initialize} – negotiate capabilities</li>
     *   <li>{@code session/new} – create session</li>
     *   <li>{@code session/prompt} – send prompt, receive update + response</li>
     * </ol>
     */
    @Test
    void fullProtocolFlow_initializeToPromptResponse() throws Exception {
        // --- Build the ACP message stream ---
        String initReq = jsonRpc(1, "initialize",
                "{\"protocolVersion\":1," +
                "\"clientInfo\":{\"name\":\"test-client\",\"version\":\"1.0\"}," +
                "\"clientCapabilities\":{\"fs\":{\"readTextFile\":false,\"writeTextFile\":false},\"terminal\":false}}");

        String sessionReq = jsonRpc(2, "session/new",
                "{\"cwd\":\"/project\",\"mcpServers\":[]}");

        // We don't know the sessionId yet – the prompt request will use a
        // placeholder that we replace after reading session/new's response.
        // Instead, we run the flow in two steps.

        // --- Step 1: initialize + session/new ---
        byte[] step1Input = concat(frameMessage(initReq), frameMessage(sessionReq));
        ByteArrayOutputStream step1Output = new ByteArrayOutputStream();
        AcpAgentServer server1 = new AcpAgentServer(graphRunner,
                new ByteArrayInputStream(step1Input), step1Output);
        server1.start();

        List<JsonNode> step1Responses = parseAllResponses(step1Output.toString(StandardCharsets.UTF_8));
        assertEquals(2, step1Responses.size(), "should have two responses (initialize + session/new)");

        // Validate initialize response
        JsonNode initResponse = step1Responses.get(0);
        assertEquals(1, initResponse.get("id").asInt());
        JsonNode initResult = initResponse.get("result");
        assertNotNull(initResult);
        assertTrue(initResult.has("protocolVersion"), "must have protocolVersion");
        assertTrue(initResult.has("agentInfo"), "must have agentInfo");
        assertTrue(initResult.has("agentCapabilities"), "must have agentCapabilities");
        assertEquals("win-acp-java", initResult.path("agentInfo").path("name").asText());

        // Validate session/new response
        JsonNode sessionResponse = step1Responses.get(1);
        assertEquals(2, sessionResponse.get("id").asInt());
        JsonNode sessionResult = sessionResponse.get("result");
        assertNotNull(sessionResult);
        assertTrue(sessionResult.has("sessionId"), "must have sessionId");
        String sessionId = sessionResult.get("sessionId").asText();
        assertFalse(sessionId.isEmpty());

        // --- Step 2: session/prompt with the real session ID ---
        String promptReq = jsonRpc(3, "session/prompt",
                "{\"sessionId\":\"" + sessionId + "\"," +
                "\"prompt\":[{\"type\":\"text\",\"text\":\"Hello, how are you?\"}]}");

        // We need a new server instance that shares the session state.
        // Since we can't share state between instances, send all 3 in one go.
        byte[] fullInput = concat(
                frameMessage(initReq),
                frameMessage(sessionReq),
                frameMessage(promptReq.replace(sessionId, "__SESS__"))
        );

        // Better approach: send all messages to one server and parse all outputs
        // But the sessionId is dynamic. Let's use a trick: send init + new first,
        // then construct the prompt with the real ID.

        // Actually the cleanest E2E test: pipe everything into one server,
        // using a PipedInputStream/PipedOutputStream pair.
        PipedOutputStream clientOut = new PipedOutputStream();
        PipedInputStream serverIn = new PipedInputStream(clientOut);
        ByteArrayOutputStream serverOut = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, serverIn, serverOut);

        // Run server in background thread
        Thread serverThread = new Thread(server::start, "e2e-server");
        serverThread.setDaemon(true);
        serverThread.start();

        // Client sends initialize
        clientOut.write(frameMessage(initReq));
        clientOut.flush();
        Thread.sleep(200);

        // Client sends session/new
        clientOut.write(frameMessage(sessionReq));
        clientOut.flush();
        Thread.sleep(200);

        // Read server output so far and extract sessionId
        String outputSoFar = serverOut.toString(StandardCharsets.UTF_8);
        List<JsonNode> responses = parseAllResponses(outputSoFar);
        assertTrue(responses.size() >= 2, "should have at least 2 responses by now");

        String realSessionId = responses.get(1).path("result").path("sessionId").asText();
        assertFalse(realSessionId.isEmpty(), "sessionId must not be empty");

        // Client sends session/prompt with real session ID
        String realPrompt = jsonRpc(3, "session/prompt",
                "{\"sessionId\":\"" + realSessionId + "\"," +
                "\"prompt\":[{\"type\":\"text\",\"text\":\"Hello, how are you?\"}]}");
        clientOut.write(frameMessage(realPrompt));
        clientOut.flush();
        Thread.sleep(500);

        // Close the stream to terminate the server
        clientOut.close();
        serverThread.join(3000);

        // --- Verify the full output ---
        String finalOutput = serverOut.toString(StandardCharsets.UTF_8);
        List<JsonNode> allMessages = parseAllJsonMessages(finalOutput);

        // Should have: init response, session/new response, session/update notification, prompt response
        assertTrue(allMessages.size() >= 3, "should have at least 3 messages (init, session/new, prompt response); got " + allMessages.size());

        // Find the session/prompt response (id=3)
        JsonNode promptResponse = allMessages.stream()
                .filter(m -> m.has("id") && m.get("id").asInt() == 3)
                .findFirst()
                .orElse(null);
        assertNotNull(promptResponse, "should have prompt response with id=3");

        JsonNode promptResult = promptResponse.get("result");
        assertNotNull(promptResult, "prompt response should have result");
        assertTrue(promptResult.has("stopReason"), "should have stopReason");
        assertEquals("end_turn", promptResult.get("stopReason").asText());

        // Find session/update notification
        JsonNode updateNotification = allMessages.stream()
                .filter(m -> "session/update".equals(m.path("method").asText()))
                .findFirst()
                .orElse(null);
        assertNotNull(updateNotification, "should have session/update notification");
        assertEquals(realSessionId,
                updateNotification.path("params").path("sessionId").asText(),
                "update should reference the correct session");
    }

    // ---- helpers ----

    private String jsonRpc(int id, String method, String params) {
        return "{\"jsonrpc\":\"2.0\",\"id\":" + id +
                ",\"method\":\"" + method +
                "\",\"params\":" + params + "}";
    }

    private byte[] frameMessage(String json) {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        String header = "Content-Length: " + body.length + "\r\n\r\n";
        byte[] h = header.getBytes(StandardCharsets.UTF_8);
        byte[] result = new byte[h.length + body.length];
        System.arraycopy(h, 0, result, 0, h.length);
        System.arraycopy(body, 0, result, h.length, body.length);
        return result;
    }

    private byte[] concat(byte[]... arrays) {
        int total = 0;
        for (byte[] a : arrays) total += a.length;
        byte[] result = new byte[total];
        int offset = 0;
        for (byte[] a : arrays) {
            System.arraycopy(a, 0, result, offset, a.length);
            offset += a.length;
        }
        return result;
    }

    private List<JsonNode> parseAllResponses(String raw) throws Exception {
        return parseAllJsonMessages(raw);
    }

    /**
     * Parse all Content-Length framed JSON messages from raw output.
     */
    private List<JsonNode> parseAllJsonMessages(String raw) throws Exception {
        List<JsonNode> messages = new ArrayList<>();
        int pos = 0;
        while (pos < raw.length()) {
            int clIdx = raw.indexOf("Content-Length:", pos);
            if (clIdx < 0) break;

            int eol = raw.indexOf("\r\n", clIdx);
            if (eol < 0) break;
            int contentLength = Integer.parseInt(raw.substring(clIdx + 15, eol).trim());

            int bodyStart = raw.indexOf("\r\n\r\n", clIdx);
            if (bodyStart < 0) break;
            bodyStart += 4;

            if (bodyStart + contentLength > raw.length()) break;
            String body = raw.substring(bodyStart, bodyStart + contentLength);
            messages.add(mapper.readTree(body));
            pos = bodyStart + contentLength;
        }
        return messages;
    }

    private AgentBehaviorDefinition fullBehavior() {
        var analyze = node("analyze", NodeType.ANALYZE_INPUT);
        var goal = node("goal", NodeType.DETERMINE_GOAL);
        var respond = node("respond", NodeType.FORMULATE_RESPONSE);
        var fin = node("finalize", NodeType.FINALIZE);

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(10);
        behavior.setNodes(new ArrayList<>(List.of(analyze, goal, respond, fin)));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "goal", RoutingCondition.ALWAYS),
                edge("goal", "respond", RoutingCondition.ALWAYS),
                edge("respond", "finalize", RoutingCondition.ALWAYS)
        )));
        return behavior;
    }

    private NodeDefinition node(String id, NodeType type) {
        var n = new NodeDefinition();
        n.setId(id);
        n.setType(type);
        return n;
    }

    private EdgeDefinition edge(String from, String to, RoutingCondition cond) {
        var e = new EdgeDefinition();
        e.setFrom(from);
        e.setTo(to);
        e.setCondition(cond);
        return e;
    }
}

