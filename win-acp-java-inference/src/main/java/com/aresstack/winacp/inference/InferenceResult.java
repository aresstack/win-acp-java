package com.aresstack.winacp.inference;

/**
 * The result produced by the local inference engine.
 * <p>
 * V1 fields: generated text, finish reason, and optional token usage.
 */
public class InferenceResult {

    private final String text;
    private final String finishReason;   // "end_turn" | "max_tokens" | "error"
    private final Usage usage;           // optional

    public InferenceResult(String text, String finishReason, Usage usage) {
        this.text = text;
        this.finishReason = finishReason;
        this.usage = usage;
    }

    /** Convenience constructor without usage. */
    public InferenceResult(String text, String finishReason) {
        this(text, finishReason, null);
    }

    public String getText() { return text; }
    public String getFinishReason() { return finishReason; }
    public Usage getUsage() { return usage; }

    /**
     * Token usage statistics.
     */
    public record Usage(int promptTokens, int completionTokens, int totalTokens) {}

    @Override
    public String toString() {
        return "InferenceResult{finishReason='" + finishReason +
                "', textLength=" + (text != null ? text.length() : 0) +
                (usage != null ? ", usage=" + usage : "") + "}";
    }
}
