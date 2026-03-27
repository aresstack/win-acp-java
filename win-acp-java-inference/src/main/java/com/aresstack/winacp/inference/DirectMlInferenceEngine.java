package com.aresstack.winacp.inference;

import com.aresstack.winacp.config.InferenceConfiguration;
import com.aresstack.winacp.windows.WindowsBindings;
import com.aresstack.winacp.windows.WindowsNativeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Inference engine backed by the Windows native stack (DXGI → D3D12 → DirectML).
 * <p>
 * Uses the {@link WindowsBindings} façade which calls directly into
 * {@code dxgi.dll}, {@code d3d12.dll}, and {@code DirectML.dll}
 * via Java 21 FFM (Foreign Function &amp; Memory API).
 * <p>
 * <b>No third-party inference runtime</b> (no ONNX Runtime, no wrapper libs).
 * All native interop goes through the {@code win-acp-java-windows-bindings} module.
 * <p>
 * V1 proves the vertical slice: device creation, adapter enumeration,
 * and DirectML device initialisation. Full operator dispatch (tensor
 * creation, operator compilation, binding tables, command recording)
 * is V2.
 * <p>
 * For V1, this engine proves:
 * <ol>
 *   <li>DXGI Factory created ✓</li>
 *   <li>GPU adapter enumerated ✓</li>
 *   <li>D3D12 device created ✓</li>
 *   <li>DirectML device created ✓</li>
 *   <li>Stack lifecycle (init/shutdown) ✓</li>
 * </ol>
 */
public class DirectMlInferenceEngine implements InferenceEngine {

    private static final Logger log = LoggerFactory.getLogger(DirectMlInferenceEngine.class);

    private final InferenceConfiguration config;
    private WindowsBindings bindings;
    private boolean ready = false;

    public DirectMlInferenceEngine(InferenceConfiguration config) {
        this.config = config;
    }

    @Override
    public void initialize() throws InferenceException {
        log.info("DirectMlInferenceEngine initializing (backend={})", config.getBackend());

        try {
            bindings = new WindowsBindings();
            bindings.init(config.getBackend());
            ready = true;

            log.info("DirectMlInferenceEngine ready (d3d12={}, directml={})",
                    bindings.getD3d12Device() != null,
                    bindings.hasDirectMl());

        } catch (WindowsNativeException e) {
            throw new InferenceException(
                    "Failed to initialize DirectML engine: " + e.getMessage(), e);
        }
    }

    @Override
    public InferenceResult generate(InferenceRequest request) throws InferenceException {
        if (!ready) {
            throw new InferenceException("Engine not initialized – call initialize() first");
        }

        log.debug("DirectMlInferenceEngine.generate: {}", request);

        // V1: DirectML device is initialised. Full operator dispatch
        // (DMLCreateOperator, DMLCompileOperator, IDMLCommandRecorder::RecordDispatch)
        // is V2 scope. For now, return a diagnostic result proving the stack works.
        String fullPrompt = request.toFullPrompt();
        int promptTokens = fullPrompt.split("\\s+").length;

        String resultText = String.format(
                "[DirectML engine active: d3d12=%s, dml=%s, backend=%s] " +
                        "Operator dispatch not yet implemented (V2). Prompt received: %d tokens.",
                bindings.getD3d12Device() != null,
                bindings.hasDirectMl(),
                config.getBackend(),
                promptTokens);

        return new InferenceResult(resultText, "end_turn",
                new InferenceResult.Usage(promptTokens, 0, promptTokens));
    }

    @Override
    public void shutdown() {
        ready = false;
        if (bindings != null) {
            bindings.close();
            bindings = null;
        }
        log.info("DirectMlInferenceEngine shut down");
    }

    @Override
    public boolean isReady() {
        return ready;
    }
}
