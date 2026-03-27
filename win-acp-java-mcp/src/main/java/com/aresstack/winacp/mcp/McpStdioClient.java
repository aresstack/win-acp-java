package com.aresstack.winacp.mcp;

import com.aresstack.winacp.config.McpServerDefinition;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lightweight MCP client that communicates with an MCP server over stdio.
 * <p>
 * Starts the server as a subprocess, sends JSON-RPC 2.0 messages, and
 * parses responses. No Spring, no Reactor – just Process + Jackson.
 */
public class McpStdioClient implements Closeable {

    private static final Logger log = LoggerFactory.getLogger(McpStdioClient.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    private final McpServerDefinition serverDef;
    private final AtomicInteger nextId = new AtomicInteger(1);
    private Process process;
    private BufferedWriter writer;
    private BufferedReader reader;

    public McpStdioClient(McpServerDefinition serverDef) {
        this.serverDef = serverDef;
    }

    /** Start the MCP server subprocess and send initialize. */
    public void connect() throws IOException {
        log.info("Starting MCP server '{}': {}", serverDef.getName(), serverDef.getCommand());

        String[] parts = serverDef.getCommand().split("\\s+");
        ProcessBuilder pb = new ProcessBuilder(parts);
        pb.redirectErrorStream(false);
        serverDef.getEnv().forEach((k, v) -> pb.environment().put(k, v));

        process = pb.start();
        writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
        reader = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));

        // Send initialize
        JsonNode initResult = sendRequest("initialize", mapper.createObjectNode()
                .put("protocolVersion", "2024-11-05")
                .putObject("capabilities")
                .putObject("clientInfo").put("name", "win-acp-java").put("version", "0.1.0"));

        log.info("MCP server '{}' initialized: {}", serverDef.getName(),
                initResult != null ? initResult.path("serverInfo").path("name").asText("?") : "no response");

        // Send initialized notification
        sendNotification("notifications/initialized");
    }

    /** Discover available tools from the server. */
    public List<ToolDescriptor> discoverTools() throws IOException {
        JsonNode result = sendRequest("tools/list", mapper.createObjectNode());
        if (result == null) return List.of();

        List<ToolDescriptor> tools = new ArrayList<>();
        ArrayNode toolsArray = (ArrayNode) result.path("tools");
        if (toolsArray.isMissingNode()) return List.of();

        for (JsonNode toolNode : toolsArray) {
            List<String> params = new ArrayList<>();
            JsonNode schema = toolNode.path("inputSchema").path("properties");
            if (schema.isObject()) {
                schema.fieldNames().forEachRemaining(params::add);
            }
            tools.add(new ToolDescriptor(
                    serverDef.getId(),
                    toolNode.path("name").asText(),
                    toolNode.path("description").asText(""),
                    params
            ));
        }
        log.info("Discovered {} tools from '{}'", tools.size(), serverDef.getName());
        return tools;
    }

    /** Execute a tool call. */
    public ToolExecutionResult callTool(String toolName, Map<String, Object> arguments) throws IOException {
        ObjectNode params = mapper.createObjectNode();
        params.put("name", toolName);
        params.set("arguments", mapper.valueToTree(arguments));

        JsonNode result = sendRequest("tools/call", params);
        if (result == null) {
            return ToolExecutionResult.failure(toolName, "No response from MCP server");
        }
        if (result.has("isError") && result.get("isError").asBoolean()) {
            String errMsg = result.path("content").path(0).path("text").asText("Tool error");
            return ToolExecutionResult.failure(toolName, errMsg);
        }

        StringBuilder content = new StringBuilder();
        for (JsonNode c : result.path("content")) {
            if ("text".equals(c.path("type").asText())) {
                content.append(c.path("text").asText());
            }
        }
        return ToolExecutionResult.success(toolName, content.toString());
    }

    @Override
    public void close() {
        if (process != null && process.isAlive()) {
            log.info("Stopping MCP server '{}'", serverDef.getName());
            process.destroyForcibly();
        }
    }

    // ---- JSON-RPC wire protocol ----

    private JsonNode sendRequest(String method, JsonNode params) throws IOException {
        int id = nextId.getAndIncrement();
        ObjectNode request = mapper.createObjectNode();
        request.put("jsonrpc", "2.0");
        request.put("id", id);
        request.put("method", method);
        request.set("params", params);

        writeMessage(mapper.writeValueAsString(request));
        return readResponse();
    }

    private void sendNotification(String method) throws IOException {
        ObjectNode notif = mapper.createObjectNode();
        notif.put("jsonrpc", "2.0");
        notif.put("method", method);
        writeMessage(mapper.writeValueAsString(notif));
    }

    private void writeMessage(String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        writer.write("Content-Length: " + bytes.length + "\r\n\r\n");
        writer.write(json);
        writer.flush();
    }

    private JsonNode readResponse() throws IOException {
        int contentLength = -1;
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.isEmpty()) break;
            if (line.toLowerCase().startsWith("content-length:")) {
                contentLength = Integer.parseInt(line.substring(15).trim());
            }
        }
        if (contentLength < 0) return null;

        char[] body = new char[contentLength];
        int read = 0;
        while (read < contentLength) {
            int n = reader.read(body, read, contentLength - read);
            if (n == -1) return null;
            read += n;
        }

        JsonNode msg = mapper.readTree(new String(body));
        return msg.get("result");
    }
}

