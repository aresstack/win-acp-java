package com.aresstack.winacp.inference;

import java.util.List;

/**
 * A request to the local inference engine.
 */
public class InferenceRequest {

    private final String prompt;
    private final List<String> stopSequences;
    private final int maxTokens;
    private final float temperature;

    public InferenceRequest(String prompt, List<String> stopSequences, int maxTokens, float temperature) {
        this.prompt = prompt;
        this.stopSequences = stopSequences;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    public String getPrompt() { return prompt; }
    public List<String> getStopSequences() { return stopSequences; }
    public int getMaxTokens() { return maxTokens; }
    public float getTemperature() { return temperature; }
}

