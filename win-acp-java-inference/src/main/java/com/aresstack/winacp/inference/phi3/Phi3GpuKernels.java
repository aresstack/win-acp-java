package com.aresstack.winacp.inference.phi3;

import com.aresstack.winacp.inference.phi3.Phi3Weights.LayerWeights;
import com.aresstack.winacp.inference.phi3.Phi3Weights.QuantizedWeight;
import com.aresstack.winacp.windows.MatMulNBitsKernel;
import com.aresstack.winacp.windows.WindowsBindings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU kernel pool for the Phi-3 decoder.
 * <p>
 * Creates one {@link MatMulNBitsKernel} per projection (q/k/v/o/gate_up/down)
 * per layer, plus an optional lm_head kernel. Each kernel dequantizes
 * INT4→FP32 at creation time and keeps the weight matrix resident on GPU.
 * <p>
 * <b>GPU memory budget (Phi-3-mini, FP32 dequantized):</b>
 * <pre>
 *   q/k/v/o:    4 × 3072² × 4 bytes      ≈ 151 MB
 *   gate_up:    16384 × 3072 × 4          ≈ 201 MB
 *   down:       3072 × 8192 × 4           ≈ 101 MB
 *   ─────────────────────────────────────────────
 *   Per layer:                             ≈ 453 MB
 *   32 layers:                             ≈ 14.5 GB
 *   lm_head:    32064 × 3072 × 4          ≈ 394 MB
 * </pre>
 * Use {@code gpuLayers} to limit GPU memory usage. On an 8 GB GPU,
 * ~14–16 layers fit comfortably.
 * <p>
 * System properties:
 * <ul>
 *   <li>{@code phi3.gpu.layers} – override number of GPU-accelerated layers (default: all)</li>
 *   <li>{@code phi3.gpu.lmhead} – put lm_head on GPU (default: true)</li>
 * </ul>
 */
public final class Phi3GpuKernels implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(Phi3GpuKernels.class);

    // Projection indices within layerKernels[l]
    static final int Q_PROJ = 0;
    static final int K_PROJ = 1;
    static final int V_PROJ = 2;
    static final int O_PROJ = 3;
    static final int GATE_UP_PROJ = 4;
    static final int DOWN_PROJ = 5;
    static final int PROJECTIONS_PER_LAYER = 6;

    private final int gpuLayers;
    private final MatMulNBitsKernel[][] layerKernels;   // [layer][projection]
    private final MatMulNBitsKernel lmHeadKernel;       // nullable
    private boolean closed = false;

    private Phi3GpuKernels(int gpuLayers,
                           MatMulNBitsKernel[][] layerKernels,
                           MatMulNBitsKernel lmHeadKernel) {
        this.gpuLayers = gpuLayers;
        this.layerKernels = layerKernels;
        this.lmHeadKernel = lmHeadKernel;
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /**
     * Create GPU kernels for the first {@code gpuLayers} decoder layers
     * and optionally the lm_head projection.
     *
     * @param wb         initialised WindowsBindings with DirectML device
     * @param weights    loaded Phi-3 weights
     * @param config     model configuration
     * @param gpuLayers  number of decoder layers to place on GPU (clamped to numHiddenLayers)
     * @param gpuLmHead  whether to put the lm_head on GPU
     * @return ready-to-use kernel pool
     */
    public static Phi3GpuKernels create(WindowsBindings wb, Phi3Weights weights,
                                         Phi3Config config, int gpuLayers,
                                         boolean gpuLmHead) {
        int layers = Math.min(Math.max(gpuLayers, 0), config.numHiddenLayers());
        log.info("Creating GPU kernels: {} / {} layers, lmHead={}",
                layers, config.numHiddenLayers(), gpuLmHead);
        long t0 = System.currentTimeMillis();

        MatMulNBitsKernel[][] kernels = new MatMulNBitsKernel[layers][];

        for (int l = 0; l < layers; l++) {
            LayerWeights lw = weights.layers[l];
            MatMulNBitsKernel[] k = new MatMulNBitsKernel[PROJECTIONS_PER_LAYER];
            k[Q_PROJ]       = createKernel(wb, lw.qProj(),      "layer." + l + ".q_proj");
            k[K_PROJ]       = createKernel(wb, lw.kProj(),      "layer." + l + ".k_proj");
            k[V_PROJ]       = createKernel(wb, lw.vProj(),      "layer." + l + ".v_proj");
            k[O_PROJ]       = createKernel(wb, lw.oProj(),      "layer." + l + ".o_proj");
            k[GATE_UP_PROJ] = createKernel(wb, lw.gateUpProj(), "layer." + l + ".gate_up");
            k[DOWN_PROJ]    = createKernel(wb, lw.downProj(),   "layer." + l + ".down");
            kernels[l] = k;

            if ((l + 1) % 4 == 0 || l == layers - 1) {
                log.info("GPU kernels: {}/{} layers created ({} ms)",
                        l + 1, layers, System.currentTimeMillis() - t0);
            }
        }

        MatMulNBitsKernel lmHead = null;
        if (gpuLmHead) {
            lmHead = createKernel(wb, weights.lmHead, "lm_head");
            log.info("GPU lm_head kernel created [{}, {}]",
                    weights.lmHead.N(), weights.lmHead.K());
        }

        long elapsed = System.currentTimeMillis() - t0;
        int totalKernels = layers * PROJECTIONS_PER_LAYER + (lmHead != null ? 1 : 0);
        log.info("Phi3GpuKernels ready: {} kernels in {} ms ({} ms/kernel)",
                totalKernels, elapsed, totalKernels > 0 ? elapsed / totalKernels : 0);

        return new Phi3GpuKernels(layers, kernels, lmHead);
    }

    private static MatMulNBitsKernel createKernel(WindowsBindings wb,
                                                   QuantizedWeight qw,
                                                   String name) {
        log.debug("Creating GPU kernel: {} [{}, {}]", name, qw.N(), qw.K());
        return new MatMulNBitsKernel(wb, qw.N(), qw.K(),
                qw.qWeight(), qw.scales(), qw.zeroPoints(), qw.blockSize());
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /** Whether layer {@code layerIdx} has GPU kernels. */
    public boolean hasLayer(int layerIdx) {
        return layerIdx >= 0 && layerIdx < gpuLayers;
    }

    /** Whether lm_head has a GPU kernel. */
    public boolean hasLmHead() { return lmHeadKernel != null; }

    /** Number of decoder layers on GPU. */
    public int getGpuLayers() { return gpuLayers; }

    public MatMulNBitsKernel qProj(int layer)      { return layerKernels[layer][Q_PROJ]; }
    public MatMulNBitsKernel kProj(int layer)      { return layerKernels[layer][K_PROJ]; }
    public MatMulNBitsKernel vProj(int layer)      { return layerKernels[layer][V_PROJ]; }
    public MatMulNBitsKernel oProj(int layer)      { return layerKernels[layer][O_PROJ]; }
    public MatMulNBitsKernel gateUpProj(int layer) { return layerKernels[layer][GATE_UP_PROJ]; }
    public MatMulNBitsKernel downProj(int layer)   { return layerKernels[layer][DOWN_PROJ]; }
    public MatMulNBitsKernel lmHead()              { return lmHeadKernel; }

    // ── AutoCloseable ────────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        int count = 0;
        for (MatMulNBitsKernel[] layer : layerKernels) {
            if (layer == null) continue;
            for (MatMulNBitsKernel k : layer) {
                if (k != null) { k.close(); count++; }
            }
        }
        if (lmHeadKernel != null) { lmHeadKernel.close(); count++; }

        log.info("Phi3GpuKernels closed ({} kernels, {} layers)", count, gpuLayers);
    }
}

