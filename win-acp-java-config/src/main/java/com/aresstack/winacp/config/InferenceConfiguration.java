package com.aresstack.winacp.config;

/**
 * Configuration for the local inference layer.
 */
public class InferenceConfiguration {

    private String modelPath;
    private String backend = "directml";   // "directml" | "cpu"
    private int maxTokens = 2048;
    private float temperature = 0.7f;

    public InferenceConfiguration() {}

    public String getModelPath() { return modelPath; }
    public void setModelPath(String modelPath) { this.modelPath = modelPath; }

    public String getBackend() { return backend; }
    public void setBackend(String backend) { this.backend = backend; }

    public int getMaxTokens() { return maxTokens; }
    public void setMaxTokens(int maxTokens) { this.maxTokens = maxTokens; }

    public float getTemperature() { return temperature; }
    public void setTemperature(float temperature) { this.temperature = temperature; }
}

