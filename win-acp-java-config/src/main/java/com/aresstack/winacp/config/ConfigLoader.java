package com.aresstack.winacp.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Loads {@link RuntimeConfiguration} from an external YAML or JSON file.
 */
public class ConfigLoader {

    private static final Logger log = LoggerFactory.getLogger(ConfigLoader.class);

    private static final ObjectMapper YAML_MAPPER = new ObjectMapper(new YAMLFactory());
    private static final ObjectMapper JSON_MAPPER = new ObjectMapper();

    /**
     * Load configuration from the given path.
     * File type is inferred from extension (.yaml/.yml → YAML, otherwise JSON).
     */
    public RuntimeConfiguration load(Path path) throws IOException {
        log.info("Loading configuration from {}", path);
        if (!Files.exists(path)) {
            throw new IOException("Configuration file not found: " + path);
        }

        String filename = path.getFileName().toString().toLowerCase();
        ObjectMapper mapper = (filename.endsWith(".yaml") || filename.endsWith(".yml"))
                ? YAML_MAPPER
                : JSON_MAPPER;

        RuntimeConfiguration config = mapper.readValue(path.toFile(), RuntimeConfiguration.class);
        log.info("Configuration loaded successfully");
        return config;
    }

    /**
     * Load configuration from a classpath resource.
     *
     * @param resourceName resource path, e.g. "agent-default.yaml"
     */
    public RuntimeConfiguration loadFromClasspath(String resourceName) throws IOException {
        log.info("Loading configuration from classpath resource: {}", resourceName);

        try (InputStream is = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(resourceName)) {
            if (is == null) {
                throw new IOException("Classpath resource not found: " + resourceName);
            }

            ObjectMapper mapper = (resourceName.endsWith(".yaml") || resourceName.endsWith(".yml"))
                    ? YAML_MAPPER
                    : JSON_MAPPER;

            RuntimeConfiguration config = mapper.readValue(is, RuntimeConfiguration.class);
            log.info("Configuration loaded successfully from classpath");
            return config;
        }
    }
}
