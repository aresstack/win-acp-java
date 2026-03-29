package com.aresstack.winacp.inference.app;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;

final class Phi3ModelDirectoryResolver {

    private Phi3ModelDirectoryResolver() {
    }

    static Path resolve() {
        Path configuredModelDir = resolveConfiguredModelDir();
        if (configuredModelDir != null) {
            return configuredModelDir;
        }

        Path jarDirectory = resolveJarDirectory();
        if (jarDirectory != null && containsModelFiles(jarDirectory)) {
            return jarDirectory;
        }

        Path workingDirectory = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        if (containsModelFiles(workingDirectory)) {
            return workingDirectory;
        }

        if (jarDirectory != null) {
            return jarDirectory;
        }

        return workingDirectory;
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

            Path locationPath = Path.of(location).toAbsolutePath().normalize();
            if (Files.isDirectory(locationPath)) {
                return locationPath;
            }

            Path parent = locationPath.getParent();
            if (parent == null) {
                return locationPath;
            }

            return parent;
        } catch (Exception ignored) {
            return null;
        }
    }

    private static boolean containsModelFiles(Path directory) {
        return Files.isRegularFile(directory.resolve("config.json"))
                && Files.isRegularFile(directory.resolve("tokenizer.json"))
                && Files.isRegularFile(directory.resolve("model.onnx"))
                && Files.isRegularFile(directory.resolve("model.onnx.data"));
    }
}