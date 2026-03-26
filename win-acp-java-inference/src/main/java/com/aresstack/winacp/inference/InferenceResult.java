package com.aresstack.winacp.inference;

/**
 * The result produced by the local inference engine.
 */
public class InferenceResult {

    private final String text;
    private final boolean complete;
    private final int tokenCount;

    public InferenceResult(String text, boolean complete, int tokenCount) {
        this.text = text;
        this.complete = complete;
        this.tokenCount = tokenCount;
    }

    public String getText() { return text; }
    public boolean isComplete() { return complete; }
    public int getTokenCount() { return tokenCount; }
}

