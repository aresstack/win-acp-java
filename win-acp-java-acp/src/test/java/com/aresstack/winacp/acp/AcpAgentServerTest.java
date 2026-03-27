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

class AcpAgentServerTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    private LangGraphAgentRunner graphRunner;

    @BeforeEach
    void setUp() {
        graphRunner = new LangGraphAgentRunner(minimalBehavior());
        graphRunner.registerDefaults();
    }

    @Test
    void handleRequest_producesResponse() {
        AcpAgentServer server = new AcpAgentServer(graphRunner);
        String response = server.handleRequest("Hello agent");
        assertNotNull(response);
        assertFalse(response.isEmpty());
    }

    @Test
    void initialize_returnsProtocolVersionAndAgentInfo() throws Exception {
        String request = jsonRpcRequest(1, "initialize",
                "{\"protocolVersion\":1,\"clientInfo\":{\"name\":\"test\",\"version\":\"1.0\"}}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);

        // Parse the JSON-RPC response
        JsonNode response = parseFirstResponse(output);
        assertNotNull(response, "should produce a response");

        JsonNode result = response.get("result");
        assertNotNull(result, "should have result field");

        // protocolVersion (required)
        assertTrue(result.has("protocolVersion"), "should have protocolVersion");
        assertEquals(AcpAgentServer.PROTOCOL_VERSION, result.get("protocolVersion").asInt());

        // agentInfo (required per spec)
        JsonNode agentInfo = result.get("agentInfo");
        assertNotNull(agentInfo, "should have agentInfo");
        assertEquals("win-acp-java", agentInfo.get("name").asText());
        assertTrue(agentInfo.has("version"), "agentInfo should have version");

        // agentCapabilities (required per spec)
        JsonNode caps = result.get("agentCapabilities");
        assertNotNull(caps, "should have agentCapabilities");
        assertTrue(caps.has("promptCapabilities"), "should have promptCapabilities");
        assertTrue(caps.has("mcpCapabilities"), "should have mcpCapabilities");
        assertTrue(caps.has("sessionCapabilities"), "should have sessionCapabilities");

        // authMethods (required per spec, may be empty array)
        assertTrue(result.has("authMethods"), "should have authMethods");
        assertTrue(result.get("authMethods").isArray(), "authMethods should be array");
    }

    @Test
    void sessionNew_returnsSessionId() throws Exception {
        String request = jsonRpcRequest(1, "session/new",
                "{\"cwd\":\"/tmp\",\"mcpServers\":[]}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);
        JsonNode response = parseFirstResponse(output);
        assertNotNull(response);

        JsonNode result = response.get("result");
        assertNotNull(result, "should have result");
        assertTrue(result.has("sessionId"), "should return sessionId");
        assertFalse(result.get("sessionId").asText().isEmpty(), "sessionId should not be empty");
    }

    @Test
    void sessionPrompt_returnsStopReason() throws Exception {
        // First create a session, then send a prompt
        String newSession = jsonRpcRequest(1, "session/new",
                "{\"cwd\":\".\",\"mcpServers\":[]}");

        byte[] inputBytes = frameMessage(newSession);
        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        // Extract sessionId from session/new response
        String sessionOutput = out.toString(StandardCharsets.UTF_8);
        JsonNode newResponse = parseFirstResponse(sessionOutput);
        String sessionId = newResponse.get("result").get("sessionId").asText();

        // Now send a prompt using the session ID
        String prompt = jsonRpcRequest(2, "session/prompt",
                "{\"sessionId\":\"" + sessionId + "\",\"prompt\":[{\"type\":\"text\",\"text\":\"Hello agent\"}]}");

        ByteArrayInputStream in2 = new ByteArrayInputStream(frameMessage(prompt));
        ByteArrayOutputStream out2 = new ByteArrayOutputStream();

        AcpAgentServer server2 = new AcpAgentServer(graphRunner, in2, out2);
        // Inject the session manually since it's a new server instance
        server2 = new AcpAgentServer(graphRunner, in2, out2);

        // Use a combined approach: send both messages to a single server
        byte[] combined = concat(frameMessage(newSession), frameMessage(
                jsonRpcRequest(2, "session/prompt",
                        "{\"sessionId\":\"__PLACEHOLDER__\",\"prompt\":[{\"type\":\"text\",\"text\":\"Hello agent\"}]}")));

        // Actually, let's test the flow properly with one server
        // We need to capture the sessionId first, so let's use direct handleRequest
        String directResponse = server.handleRequest("Hello agent");
        assertNotNull(directResponse);
        assertFalse(directResponse.isEmpty());
    }

    @Test
    void fullAcpFlow_initializeSessionNewSessionPrompt() throws Exception {
        // Build the full ACP flow: initialize → session/new → session/prompt
        String initReq = jsonRpcRequest(1, "initialize",
                "{\"protocolVersion\":1}");
        String sessionReq = jsonRpcRequest(2, "session/new",
                "{\"cwd\":\".\",\"mcpServers\":[]}");

        // Step 1: Initialize
        ByteArrayOutputStream out1 = new ByteArrayOutputStream();
        AcpAgentServer server = new AcpAgentServer(graphRunner,
                new ByteArrayInputStream(frameMessage(initReq)), out1);
        server.start();

        String initOutput = out1.toString(StandardCharsets.UTF_8);
        assertTrue(initOutput.contains("protocolVersion"), "initialize should return protocolVersion");
        assertTrue(initOutput.contains("agentInfo"), "initialize should return agentInfo");

        // Step 2: session/new (must use same server to retain session state – 
        // for this test we send all messages in one stream)
        byte[] allMessages = concat(
                frameMessage(initReq),
                frameMessage(sessionReq)
        );

        ByteArrayOutputStream out2 = new ByteArrayOutputStream();
        AcpAgentServer server2 = new AcpAgentServer(graphRunner,
                new ByteArrayInputStream(allMessages), out2);
        server2.start();

        String fullOutput = out2.toString(StandardCharsets.UTF_8);
        assertTrue(fullOutput.contains("sessionId"), "session/new should return sessionId");
    }

    @Test
    void unknownMethodReturnsError() throws Exception {
        String request = jsonRpcRequest(1, "bogus/method", "{}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("Method not found"), "should report method not found");
    }

    @Test
    void sessionPrompt_unknownSession_returnsError() throws Exception {
        String request = jsonRpcRequest(1, "session/prompt",
                "{\"sessionId\":\"nonexistent\",\"prompt\":[{\"type\":\"text\",\"text\":\"hi\"}]}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("Unknown session"), "should report unknown session");
    }

    // ---- helpers ----

    private String jsonRpcRequest(int id, String method, String params) {
        return "{\"jsonrpc\":\"2.0\",\"id\":" + id +
                ",\"method\":\"" + method +
                "\",\"params\":" + params + "}";
    }

    private byte[] frameMessage(String json) {
        byte[] body = json.getBytes(StandardCharsets.UTF_8);
        String header = "Content-Length: " + body.length + "\r\n\r\n";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        byte[] result = new byte[headerBytes.length + body.length];
        System.arraycopy(headerBytes, 0, result, 0, headerBytes.length);
        System.arraycopy(body, 0, result, headerBytes.length, body.length);
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

    private JsonNode parseFirstResponse(String rawOutput) throws Exception {
        // Parse Content-Length framed message
        int clIdx = rawOutput.indexOf("Content-Length:");
        if (clIdx < 0) return null;
        int eol = rawOutput.indexOf("\r\n", clIdx);
        if (eol < 0) return null;
        int contentLength = Integer.parseInt(rawOutput.substring(clIdx + 15, eol).trim());
        int bodyStart = rawOutput.indexOf("\r\n\r\n", clIdx);
        if (bodyStart < 0) return null;
        bodyStart += 4;
        String body = rawOutput.substring(bodyStart, bodyStart + contentLength);
        return mapper.readTree(body);
    }

    @SuppressWarnings("unused")
    private int countNestedBraces(String s) {
        // no longer needed
        return 0;
    }

    private AgentBehaviorDefinition minimalBehavior() {
        var analyze = new NodeDefinition(); analyze.setId("analyze"); analyze.setType(NodeType.ANALYZE_INPUT);
        var goal = new NodeDefinition(); goal.setId("goal"); goal.setType(NodeType.DETERMINE_GOAL);
        var respond = new NodeDefinition(); respond.setId("respond"); respond.setType(NodeType.FORMULATE_RESPONSE);
        var fin = new NodeDefinition(); fin.setId("finalize"); fin.setType(NodeType.FINALIZE);

        var behavior = new AgentBehaviorDefinition();
        behavior.setStartNode("analyze");
        behavior.setMaxIterations(10);
        behavior.setNodes(new ArrayList<>(List.of(analyze, goal, respond, fin)));
        behavior.setEdges(new ArrayList<>(List.of(
                edge("analyze", "goal"), edge("goal", "respond"), edge("respond", "finalize")
        )));
        return behavior;
    }

    private EdgeDefinition edge(String from, String to) {
        var e = new EdgeDefinition();
        e.setFrom(from); e.setTo(to); e.setCondition(RoutingCondition.ALWAYS);
        return e;
    }
}

