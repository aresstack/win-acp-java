package com.aresstack.winacp.acp;

import com.aresstack.winacp.config.*;
import com.aresstack.winacp.graph.AgentGraphRunner;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AcpAgentServerTest {

    private AgentGraphRunner graphRunner;

    @BeforeEach
    void setUp() {
        graphRunner = new AgentGraphRunner(minimalBehavior());
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
    void jsonRpc_initializeReturnsCapabilities() throws Exception {
        String request = jsonRpcRequest(1, "initialize", "{}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start(); // reads until EOF (ByteArrayInputStream)

        String output = out.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("win-acp-java"), "should contain agent name");
        assertTrue(output.contains("\"jsonrpc\":\"2.0\""), "should be JSON-RPC 2.0");
    }

    @Test
    void jsonRpc_agentRunReturnsContent() throws Exception {
        String request = jsonRpcRequest(1, "agent/run", "{\"message\":\"hi\"}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("\"content\""), "should contain content field");
    }

    @Test
    void jsonRpc_unknownMethodReturnsError() throws Exception {
        String request = jsonRpcRequest(1, "bogus/method", "{}");
        byte[] inputBytes = frameMessage(request);

        ByteArrayInputStream in = new ByteArrayInputStream(inputBytes);
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        AcpAgentServer server = new AcpAgentServer(graphRunner, in, out);
        server.start();

        String output = out.toString(StandardCharsets.UTF_8);
        assertTrue(output.contains("Method not found"), "should report method not found");
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

