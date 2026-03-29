package com.aresstack.winacp.inference.app;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

final class Phi3ModelDirectoryResolver {

    private static final String RELATIVE_MODEL_DIR =
            "model/phi3-mini-directml-int4/directml/directml-int4-awq-block-128";

    private Phi3ModelDirectoryResolver() {
    }

    static Path resolve() {
        Path configuredModelDir = resolveConfiguredModelDir();
        if (configuredModelDir != null) {
            return configuredModelDir;
        }

        Path jarDirectory = resolveJarDirectory();
        Path workingDirectory = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();

        List<Path> candidates = new ArrayList<Path>();
        if (jarDirectory != null) {
            candidates.add(jarDirectory.resolve(RELATIVE_MODEL_DIR));
        }
        candidates.add(workingDirectory.resolve(RELATIVE_MODEL_DIR));

        Path parentDirectory = workingDirectory.getParent();
        if (parentDirectory != null) {
            candidates.add(parentDirectory.resolve(RELATIVE_MODEL_DIR));
        }

        for (Path candidate : candidates) {
            Path normalizedCandidate = candidate.toAbsolutePath().normalize();
            if (containsModel(normalizedCandidate)) {
                return normalizedCandidate;
            }
        }

        if (jarDirectory != null) {
            return jarDirectory.resolve(RELATIVE_MODEL_DIR).toAbsolutePath().normalize();
        }

        return workingDirectory.resolve(RELATIVE_MODEL_DIR).toAbsolutePath().normalize();
    }

    private static Path resolveConfiguredModelDir() {
        String configuredValue = System.getProperty("phi3.model.dir");
        if (configuredValue == null || configuredValue.trim().isEmpty()) {
            return null;
        }

        return Path.of(configuredValue.trim()).toAbsolutePath().normalize();
    }

    private static Path resolveJarDirectory() {
        try {
            URI location = Phi3ModelDirectoryResolver.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI();

            Path path = Path.of(location).toAbsolutePath().normalize();
            if (Files.isDirectory(path)) {
                return path;
            }

            return path.getParent();
        } catch (Exception ignored) {
            return null;
        }
    }

    private static boolean containsModel(Path directory) {
        return Files.isRegularFile(directory.resolve("config.json"))
                && Files.isRegularFile(directory.resolve("tokenizer.json"))
                && Files.isRegularFile(directory.resolve("model.onnx"))
                && Files.isRegularFile(directory.resolve("model.onnx.data"));
    }
}