package com.aresstack.winacp.inference;

import java.util.Objects;

/**
 * A request to the local inference engine.
 * <p>
 * V1 keeps this deliberately small: model identifier, prompts, and
 * sampling parameters. No stop-sequences, no streaming, no multi-turn
 * history – those belong to later versions.
 */
public class InferenceRequest {

    private final String modelId;
    private final String systemPrompt;
    private final String userPrompt;
    private final int maxTokens;
    private final float temperature;

    private InferenceRequest(Builder builder) {
        this.modelId = builder.modelId;
        this.systemPrompt = Objects.requireNonNull(builder.systemPrompt, "systemPrompt");
        this.userPrompt = Objects.requireNonNull(builder.userPrompt, "userPrompt");
        this.maxTokens = builder.maxTokens;
        this.temperature = builder.temperature;
    }

    public String getModelId() { return modelId; }
    public String getSystemPrompt() { return systemPrompt; }
    public String getUserPrompt() { return userPrompt; }
    public int getMaxTokens() { return maxTokens; }
    public float getTemperature() { return temperature; }

    /**
     * Convenience: full prompt as a single string (system + user).
     */
    public String toFullPrompt() {
        if (systemPrompt == null || systemPrompt.isBlank()) {
            return userPrompt;
        }
        return systemPrompt + "\n\n" + userPrompt;
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String modelId;
        private String systemPrompt = "";
        private String userPrompt = "";
        private int maxTokens = 256;
        private float temperature = 0.7f;

        public Builder modelId(String modelId) { this.modelId = modelId; return this; }
        public Builder systemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; return this; }
        public Builder userPrompt(String userPrompt) { this.userPrompt = userPrompt; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(float temperature) { this.temperature = temperature; return this; }

        public InferenceRequest build() { return new InferenceRequest(this); }
    }

    @Override
    public String toString() {
        return "InferenceRequest{modelId='" + modelId + "', userPrompt='" +
                (userPrompt.length() > 40 ? userPrompt.substring(0, 40) + "…" : userPrompt) +
                "', maxTokens=" + maxTokens + ", temperature=" + temperature + "}";
    }
}
