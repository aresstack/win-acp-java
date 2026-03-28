package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.*;
import java.util.Arrays;

/**
 * GPU-accelerated MatMulNBits kernel for AWQ INT4 block-128 quantized weights.
 * <p>
 * This is the <b>single most important kernel</b> in the Phi-3 driver.
 * Almost every projection in the model (q/k/v/o, gate_up, down, lm_head)
 * depends on this kernel.
 * <p>
 * <b>V1 strategy</b>: Dequantize INT4→FP32 once at preparation time, upload
 * the full FP32 weight matrix to GPU, then use DirectML GEMM for inference.
 * This trades GPU memory for implementation simplicity. A future V2 can
 * replace this with a custom D3D12 compute shader operating directly on
 * packed INT4 data.
 * <p>
 * <b>Kernel contract</b>:
 * <ul>
 *   <li>Input:  x ∈ FP32 [M, K]  (M=1 for decode, M=seqLen for prefill)</li>
 *   <li>Weight: packed INT4 uint8 + FP16 scales + uint4 zero-points, block=128</li>
 *   <li>Output: y ∈ FP32 [M, N]  where y = x @ W^T</li>
 * </ul>
 * <p>
 * No ONNX Runtime, no JNI, no JNA. Pure Java 21 FFM → D3D12 → DirectML.
 */
public final class MatMulNBitsKernel implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MatMulNBitsKernel.class);

    private final WindowsBindings wb;
    private final Arena arena;
    private final int N;  // output features (rows of weight matrix)
    private final int K;  // input features  (cols of weight matrix)

    // ── GPU resources ────────────────────────────────────────────────────
    private MemorySegment weightBuf;     // GPU default buffer: dequantized FP32 [N, K]
    private MemorySegment biasBuf;       // GPU default buffer: zero-bias [N] (GEMM requires C tensor)
    private MemorySegment inputBuf;      // GPU default buffer: input [1, K] (reused per call)
    private MemorySegment outputBuf;     // GPU default buffer: output [1, N]

    // ── DML compiled operator ────────────────────────────────────────────
    private MemorySegment compiledGemm;
    private MemorySegment descriptorHeap;
    private MemorySegment cmdRecorder;
    private int descriptorIncrement;

    // ── Binding properties ───────────────────────────────────────────────
    private int descCount;
    private long tempSize;
    private long persistSize;
    private MemorySegment tempBuf;
    private MemorySegment persistBuf;

    private boolean prepared = false;
    private boolean closed = false;

    /**
     * Create a MatMulNBits kernel for a specific weight matrix.
     *
     * @param wb     initialized WindowsBindings (D3D12 + DirectML devices)
     * @param N      output features (weight rows)
     * @param K      input features (weight cols)
     * @param qWeight   packed INT4 weights [N, K/blockSize, blockSize/2]
     * @param scales    per-block FP32 scales [N * K/blockSize]
     * @param zeroPoints packed uint4 zero points
     * @param blockSize  quantization block size (128)
     */
    public MatMulNBitsKernel(WindowsBindings wb, int N, int K,
                              byte[] qWeight, float[] scales, byte[] zeroPoints,
                              int blockSize) {
        this.wb = wb;
        this.arena = Arena.ofConfined();
        this.N = N;
        this.K = K;

        try {
            prepare(qWeight, scales, zeroPoints, blockSize);
        } catch (WindowsNativeException e) {
            arena.close();
            throw new RuntimeException("MatMulNBitsKernel preparation failed", e);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Preparation: dequantize → upload → compile
    // ══════════════════════════════════════════════════════════════════════

    private void prepare(byte[] qWeight, float[] scales, byte[] zeroPoints, int blockSize)
            throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        long t0 = System.nanoTime();

        // ── Step 1: Dequantize INT4 → FP32 on CPU ────────────────────
        // The dequantized weight matrix is [N, K] in row-major order.
        // Each weight = (nibble_value - zero_point) * scale
        float[] dequantized = dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);
        log.info("Dequantized [{}, {}] INT4→FP32 in {} ms",
                N, K, (System.nanoTime() - t0) / 1_000_000);

        // ── Step 2: Create GPU buffers ────────────────────────────────
        long weightBytes = (long) N * K * Float.BYTES;
        long inputBytes  = (long) K * Float.BYTES;        // M=1 for matvec
        long outputBytes = (long) N * Float.BYTES;
        long biasBytes   = (long) N * Float.BYTES;

        weightBuf = D3D12Bindings.createDefaultBuffer(dev, weightBytes, arena);
        biasBuf   = D3D12Bindings.createDefaultBuffer(dev, biasBytes, arena);
        inputBuf  = D3D12Bindings.createDefaultBuffer(dev, inputBytes, arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, outputBytes, arena);

        // ── Step 3: Upload weight data to GPU ─────────────────────────
        D3D12Bindings.uploadFloats(dev, queue, weightBuf, dequantized, arena);
        float[] zeroBias = new float[N]; // GEMM C tensor = zero bias
        D3D12Bindings.uploadFloats(dev, queue, biasBuf, zeroBias, arena);
        log.info("Uploaded weight [{}, {}] to GPU ({} MB)",
                N, K, weightBytes / (1024 * 1024));

        // ── Step 4: Create and compile DirectML GEMM operator ─────────
        // GEMM: Y = alpha * (A @ B^T) + beta * C
        //   A = input  [1, K]
        //   B = weight [N, K]  → with transB, this computes A @ B^T = [1, K] @ [K, N] = [1, N]
        //   C = bias   [1, N]
        //   Y = output [1, N]
        // alpha=1.0, beta=1.0 (bias is zero, so effectively Y = A @ B^T)

        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS,  0, td(new int[]{1, 1, 1, K}));       // A: [1,1,1,K]
        gemm.set(ValueLayout.ADDRESS,  8, td(new int[]{1, 1, N, K}));       // B: [1,1,N,K]
        gemm.set(ValueLayout.ADDRESS, 16, td(new int[]{1, 1, 1, N}));       // C: [1,1,1,N]
        gemm.set(ValueLayout.ADDRESS, 24, td(new int[]{1, 1, 1, N}));       // Y: [1,1,1,N]
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE);       // transA
        gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_TRANSPOSE);  // transB
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f);  // alpha
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 1.0f);  // beta
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL);  // no fused activation

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_GEMM, gemm);
        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        compiledGemm = DirectMlBindings.compileOperator(dml, op,
                DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);

        // ── Step 5: Query binding properties ──────────────────────────
        long[] props = DirectMlBindings.getBindingProperties(compiledGemm, arena);
        descCount   = Math.max((int) props[0], 1);
        tempSize    = props[1];
        persistSize = props[2];
        log.debug("GEMM binding: desc={}, temp={}, persist={}", descCount, tempSize, persistSize);

        // ── Step 6: Create descriptor heap ────────────────────────────
        // Need descriptors for: initialization + execution
        int totalDesc = descCount * 2 + 4; // extra headroom
        descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, totalDesc, arena);
        descriptorIncrement = D3D12Bindings.getDescriptorIncrementSize(dev);
        cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

        // ── Step 7: Allocate temp/persist buffers ─────────────────────
        if (tempSize > 0) {
            tempBuf = D3D12Bindings.createDefaultBuffer(dev, tempSize, arena);
        }
        if (persistSize > 0) {
            persistBuf = D3D12Bindings.createDefaultBuffer(dev, persistSize, arena);
        }

        // ── Step 8: Initialize the operator ───────────────────────────
        initializeOperator(dev, queue, dml);

        prepared = true;
        log.info("MatMulNBitsKernel ready: [{}, {}] on GPU", N, K);
    }

    private void initializeOperator(MemorySegment dev, MemorySegment queue,
                                     MemorySegment dml) throws WindowsNativeException {
        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(
                dml, new MemorySegment[]{compiledGemm}, arena);

        long[] initProps = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) initProps[0], 1);
        long initTempSize = initProps[1];

        long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);

        MemorySegment initBtDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuBase, gpuBase, initDescCount);
        MemorySegment initBt = DirectMlBindings.createBindingTable(dml, initBtDesc, arena);

        // Bind persistent resource to initializer output
        if (persistSize > 0 && persistBuf != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuf, 0, persistSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindOutputs(initBt, 1, bd);
        }

        // Bind temp resource for initialization
        MemorySegment initTempBuf = null;
        if (initTempSize > 0) {
            initTempBuf = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, initTempBuf, 0, initTempSize);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                    DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindTemporaryResource(initBt, bd);
        }

        // Record and execute initialization
        var alloc = D3D12Bindings.createCommandAllocator(dev,
                D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        MemorySegment cmdList = null;
        try {
            cmdList = D3D12Bindings.createCommandList(dev,
                    D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
            D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
            DirectMlBindings.recordDispatch(cmdRecorder, cmdList, initializer, initBt);
            D3D12Bindings.executeAndWait(dev, queue, cmdList, arena);
        } finally {
            if (cmdList != null) DxgiBindings.release(cmdList);
            DxgiBindings.release(alloc);
            DxgiBindings.release(initBt);
            DxgiBindings.release(initializer);
            if (initTempBuf != null) DxgiBindings.release(initTempBuf);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Inference: matvec on GPU
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Compute y = x @ W^T on GPU.
     * <p>
     * This is the hot path. Input vector x[K] is uploaded, GEMM executes
     * on GPU, result y[N] is read back.
     *
     * @param x input vector [K]
     * @return output vector [N]
     */
    public float[] matvec(float[] x) {
        if (!prepared) throw new IllegalStateException("Kernel not prepared");
        if (x.length != K) throw new IllegalArgumentException(
                "Input length " + x.length + " != K=" + K);

        var dev = wb.getD3d12Device();
        var queue = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        try {
            // Upload input
            D3D12Bindings.uploadFloats(dev, queue, inputBuf, x, arena);

            // Create binding table for execution
            long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
            long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
            // Use offset past initialization descriptors
            int descOff = descCount + 4;

            MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiledGemm,
                    cpuBase + (long) descOff * descriptorIncrement,
                    gpuBase + (long) descOff * descriptorIncrement, descCount);
            MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

            try {
                // Bind inputs: A=input, B=weight, C=bias
                long inputBytes  = (long) K * Float.BYTES;
                long weightBytes = (long) N * K * Float.BYTES;
                long biasBytes   = (long) N * Float.BYTES;
                long outputBytes = (long) N * Float.BYTES;

                MemorySegment inputs = arena.allocate(16L * 3, 8);
                setBufferBinding(inputs, 0, inputBuf, inputBytes);
                setBufferBinding(inputs, 1, weightBuf, weightBytes);
                setBufferBinding(inputs, 2, biasBuf, biasBytes);
                DirectMlBindings.bindInputs(bt, 3, inputs);

                // Bind output
                MemorySegment outputs = arena.allocate(16, 8);
                setBufferBinding(outputs, 0, outputBuf, outputBytes);
                DirectMlBindings.bindOutputs(bt, 1, outputs);

                // Bind temp/persist
                if (tempSize > 0 && tempBuf != null) {
                    MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, tempBuf, 0, tempSize);
                    MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
                    DirectMlBindings.bindTemporaryResource(bt, bd);
                }
                if (persistSize > 0 && persistBuf != null) {
                    MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBuf, 0, persistSize);
                    MemorySegment bd = DirectMlBindings.allocBindingDesc(arena,
                            DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
                    DirectMlBindings.bindPersistentResource(bt, bd);
                }

                // Record and execute
                var alloc = D3D12Bindings.createCommandAllocator(dev,
                        D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
                MemorySegment cmdList = null;
                try {
                    cmdList = D3D12Bindings.createCommandList(dev,
                            D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
                    D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
                    DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiledGemm, bt);
                    D3D12Bindings.executeAndWait(dev, queue, cmdList, arena);
                } finally {
                    if (cmdList != null) DxgiBindings.release(cmdList);
                    DxgiBindings.release(alloc);
                }
            } finally {
                DxgiBindings.release(bt);
            }

            // Readback result
            return D3D12Bindings.readbackFloats(dev, queue, outputBuf, N, arena);

        } catch (WindowsNativeException e) {
            throw new RuntimeException("MatMulNBitsKernel.matvec failed", e);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // INT4 dequantization (CPU, one-time)
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Dequantize INT4 AWQ block-128 packed weights to FP32.
     * <p>
     * Each byte in {@code qWeight} contains 2 uint4 values (low nibble first).
     * Weight value = (nibble - zero_point) * scale
     *
     * @return float[N * K] row-major weight matrix
     */
    static float[] dequantizeInt4(byte[] qWeight, float[] scales, byte[] zeroPoints,
                                   int N, int K, int blockSize) {
        float[] result = new float[N * K];
        int blocksPerRow = K / blockSize;

        for (int n = 0; n < N; n++) {
            int qOffset = n * blocksPerRow * (blockSize / 2);
            int scaleOffset = n * blocksPerRow;

            for (int blk = 0; blk < blocksPerRow; blk++) {
                float scale = scales[scaleOffset + blk];

                // Zero point: 2 per byte, low nibble first
                int zpIdx = n * blocksPerRow + blk;
                int zpByte = zeroPoints[zpIdx / 2] & 0xFF;
                int zp = (zpIdx % 2 == 0) ? (zpByte & 0xF) : (zpByte >>> 4);

                int kBase = blk * blockSize;
                int qBase = qOffset + blk * (blockSize / 2);
                int rowBase = n * K;

                for (int j = 0; j < blockSize / 2; j++) {
                    int packed = qWeight[qBase + j] & 0xFF;
                    int w0 = (packed & 0xF) - zp;
                    int w1 = (packed >>> 4) - zp;
                    result[rowBase + kBase + 2 * j]     = w0 * scale;
                    result[rowBase + kBase + 2 * j + 1] = w1 * scale;
                }
            }
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Helpers
    // ══════════════════════════════════════════════════════════════════════

    /** Build a DML_TENSOR_DESC for FP32 buffer tensor. */
    private MemorySegment td(int[] sizes) {
        int elems = 1;
        for (int s : sizes) elems *= s;
        long byteSize = (long) elems * Float.BYTES;
        var bufTD = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, null, byteSize);
        return DirectMlBindings.allocTensorDesc(arena, bufTD);
    }

    /**
     * Set a DML_BINDING_DESC (buffer type) into an array at the given index.
     * Each binding desc is 16 bytes: Type(4)+pad(4)+Desc*(8).
     */
    private void setBufferBinding(MemorySegment array, int index,
                                   MemorySegment buffer, long sizeBytes) {
        long off = (long) index * 16;
        MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, buffer, 0, sizeBytes);
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    // ══════════════════════════════════════════════════════════════════════
    // AutoCloseable
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Release COM objects in reverse order
        if (cmdRecorder != null) DxgiBindings.release(cmdRecorder);
        if (descriptorHeap != null) DxgiBindings.release(descriptorHeap);
        if (compiledGemm != null) DxgiBindings.release(compiledGemm);
        if (persistBuf != null) DxgiBindings.release(persistBuf);
        if (tempBuf != null) DxgiBindings.release(tempBuf);
        if (outputBuf != null) DxgiBindings.release(outputBuf);
        if (inputBuf != null) DxgiBindings.release(inputBuf);
        if (biasBuf != null) DxgiBindings.release(biasBuf);
        if (weightBuf != null) DxgiBindings.release(weightBuf);

        arena.close();
        log.debug("MatMulNBitsKernel closed [{}, {}]", N, K);
    }

    /** Output features (rows of weight matrix). */
    public int getN() { return N; }

    /** Input features (cols of weight matrix). */
    public int getK() { return K; }
}

