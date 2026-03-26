package com.aresstack.winacp.mcp;

/**
 * The result of an MCP tool execution.
 */
public class ToolExecutionResult {

    private final String toolName;
    private final boolean success;
    private final String content;
    private final String errorMessage;

    private ToolExecutionResult(String toolName, boolean success, String content, String errorMessage) {
        this.toolName = toolName;
        this.success = success;
        this.content = content;
        this.errorMessage = errorMessage;
    }

    public static ToolExecutionResult success(String toolName, String content) {
        return new ToolExecutionResult(toolName, true, content, null);
    }

    public static ToolExecutionResult failure(String toolName, String errorMessage) {
        return new ToolExecutionResult(toolName, false, null, errorMessage);
    }

    public String getToolName() { return toolName; }
    public boolean isSuccess() { return success; }
    public String getContent() { return content; }
    public String getErrorMessage() { return errorMessage; }

    @Override
    public String toString() {
        return "ToolExecutionResult{tool='" + toolName + "', success=" + success + "}";
    }
}

