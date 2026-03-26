package com.aresstack.winacp.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
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
        log.info("Loading configuration from {}", path.toAbsolutePath());

        if (!Files.exists(path)) {
            throw new IOException(
                    "Configuration file not found: " + path.toAbsolutePath() + "\n" +
                    "\n" +
                    "Please provide a valid configuration file. Options:\n" +
                    "  1. Place an 'application.yml' in the working directory\n" +
                    "  2. Pass --config <path> as command-line argument\n" +
                    "  3. Set the environment variable WIN_ACP_CONFIG=<path>\n" +
                    "\n" +
                    "A sample configuration is available at: agent.example.yaml"
            );
        }

        String filename = path.getFileName().toString().toLowerCase();
        ObjectMapper mapper = (filename.endsWith(".yaml") || filename.endsWith(".yml"))
                ? YAML_MAPPER
                : JSON_MAPPER;

        RuntimeConfiguration config = mapper.readValue(path.toFile(), RuntimeConfiguration.class);
        log.info("Configuration loaded successfully from {}", path.toAbsolutePath());
        return config;
    }
}
