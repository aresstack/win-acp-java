package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.file.Path;
import java.util.*;

/**
 * MNIST-specific DirectML inference pipeline.
 * <p>
 * Loads {@code mnist-8.onnx}, creates DirectML operators for each layer,
 * and runs inference entirely on the GPU via D3D12 + DirectML.
 * <p>
 * Network: Input(1,1,28,28) → Conv+Relu → MaxPool → Conv+Relu → MaxPool → Gemm → Output(1,10)
 * <p>
 * No ONNX Runtime, no third-party libs. Pure FFM → Windows DLLs.
 */
public final class MnistPipeline implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(MnistPipeline.class);

    private final WindowsBindings wb;
    private Arena arena;

    // ── Parsed weights ───────────────────────────────────────────────────
    private float[] conv1Filter, conv1Bias;
    private float[] conv2Filter, conv2Bias;
    private float[] fcWeight, fcBias;

    // ── GPU buffers (D3D12 default-heap, UAV) ────────────────────────────
    private MemorySegment inputBuf;       // (1,1,28,28)  784
    private MemorySegment conv1FBuf;      // (8,1,5,5)    200
    private MemorySegment conv1BBuf;      // (1,8,1,1)      8
    private MemorySegment conv1Out;       // (1,8,28,28)  6272
    private MemorySegment pool1Out;       // (1,8,14,14)  1568
    private MemorySegment conv2FBuf;      // (16,8,5,5)   3200
    private MemorySegment conv2BBuf;      // (1,16,1,1)     16
    private MemorySegment conv2Out;       // (1,16,14,14) 3136
    private MemorySegment pool2Out;       // (1,16,4,4)    256
    private MemorySegment fcWBuf;         // (1,1,256,10) 2560
    private MemorySegment fcBBuf;         // (1,1,1,10)     10
    private MemorySegment outputBuf;      // (1,1,1,10)     10

    // ── Compiled DML operators ───────────────────────────────────────────
    private MemorySegment compiledConv1, compiledPool1, compiledConv2, compiledPool2, compiledGemm;
    private MemorySegment[] allCompiled;

    // ── DML infra ────────────────────────────────────────────────────────
    private MemorySegment descriptorHeap;
    private MemorySegment cmdRecorder;
    private int descriptorIncrement;
    private int totalDescriptors;

    // ── Temp / persistent resources per operator ─────────────────────────
    private long[] tempSizes;       // [5]
    private long[] persistSizes;    // [5]
    private int[] descCounts;       // [5] – cached from allocateBindingResources
    private MemorySegment[] tempBufs;
    private MemorySegment[] persistBufs;

    private boolean loaded = false;

    public MnistPipeline(WindowsBindings wb) {
        this.wb = Objects.requireNonNull(wb);
        this.arena = Arena.ofConfined();
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Load model
    // ══════════════════════════════════════════════════════════════════════

    public void loadModel(Path onnxFile) throws WindowsNativeException, IOException {
        log.info("MnistPipeline.loadModel({})", onnxFile);
        OnnxModelReader.OnnxGraph graph = OnnxModelReader.parse(onnxFile);

        extractWeights(graph);
        createGpuBuffers();
        uploadWeights();
        compileAllOperators();
        allocateBindingResources();
        initializeOperators();

        loaded = true;
        log.info("MnistPipeline ready – 5 DML operators compiled and initialized");
    }

    // ── Step 1: Extract weights from parsed ONNX graph ───────────────────

    private void extractWeights(OnnxModelReader.OnnxGraph graph) {
        Map<String, OnnxModelReader.OnnxTensor> inits = graph.initializers();

        // Build maps for tracing data flow
        Map<String, String> reshapeMap = new HashMap<>(); // reshape output → reshape input
        Map<String, String> outputToNode = new HashMap<>(); // tensor name → producing node's opType

        for (OnnxModelReader.OnnxNode node : graph.nodes()) {
            if ("Reshape".equals(node.opType()) && !node.inputs().isEmpty() && !node.outputs().isEmpty()) {
                reshapeMap.put(node.outputs().get(0), node.inputs().get(0));
            }
            for (String out : node.outputs()) {
                outputToNode.put(out, node.opType());
            }
        }

        // MNIST-8 graph structure (from CNTK):
        //   Conv(input, filter) → Add(_, bias) → Relu → MaxPool  (×2)
        //   Reshape → MatMul(_, weight) → Add(_, bias) → output
        // Conv nodes have NO bias input; bias is in the following Add node.

        List<float[]> convFilters = new ArrayList<>();
        List<float[]> convBiases = new ArrayList<>();
        float[] matMulWeight = null;
        float[] addBias = null;

        var nodes = graph.nodes();
        for (int i = 0; i < nodes.size(); i++) {
            var node = nodes.get(i);
            switch (node.opType()) {
                case "Conv" -> {
                    OnnxModelReader.OnnxTensor f = resolveInitializer(node.inputs().get(1), inits, reshapeMap);
                    convFilters.add(f != null ? f.data() : new float[0]);

                    // Look ahead for the Add node that provides the bias
                    float[] bias = null;
                    if (i + 1 < nodes.size() && "Add".equals(nodes.get(i + 1).opType())) {
                        var addNode = nodes.get(i + 1);
                        for (String in : addNode.inputs()) {
                            OnnxModelReader.OnnxTensor t = resolveInitializer(in, inits, reshapeMap);
                            if (t != null) { bias = t.data(); break; }
                        }
                    }
                    convBiases.add(bias != null ? bias : new float[0]);
                }
                case "MatMul" -> {
                    OnnxModelReader.OnnxTensor w = resolveInitializer(node.inputs().get(1), inits, reshapeMap);
                    if (w != null) matMulWeight = w.data();
                }
                case "Add" -> {
                    // Only capture the final Add (after MatMul) as FC bias
                    if (matMulWeight != null && addBias == null) {
                        for (String in : node.inputs()) {
                            OnnxModelReader.OnnxTensor t = resolveInitializer(in, inits, reshapeMap);
                            if (t != null) { addBias = t.data(); break; }
                        }
                    }
                }
            }
        }

        conv1Filter = convFilters.size() > 0 ? convFilters.get(0) : new float[200];
        conv1Bias   = convBiases.size() > 0 ? convBiases.get(0)  : new float[8];
        conv2Filter = convFilters.size() > 1 ? convFilters.get(1) : new float[3200];
        conv2Bias   = convBiases.size() > 1 ? convBiases.get(1)  : new float[16];
        fcWeight    = matMulWeight != null ? matMulWeight : new float[2560];
        fcBias      = addBias != null ? addBias : new float[10];

        log.info("Weights: conv1F={}, conv1B={}, conv2F={}, conv2B={}, fcW={}, fcB={}",
                conv1Filter.length, conv1Bias.length, conv2Filter.length,
                conv2Bias.length, fcWeight.length, fcBias.length);
    }

    // ── Step 2: Create GPU buffers ───────────────────────────────────────

    private void createGpuBuffers() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        inputBuf  = D3D12Bindings.createDefaultBuffer(dev, fb(784), arena);
        conv1FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Filter.length), arena);
        conv1BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv1Bias.length), arena);
        conv1Out  = D3D12Bindings.createDefaultBuffer(dev, fb(6272), arena);
        pool1Out  = D3D12Bindings.createDefaultBuffer(dev, fb(1568), arena);
        conv2FBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Filter.length), arena);
        conv2BBuf = D3D12Bindings.createDefaultBuffer(dev, fb(conv2Bias.length), arena);
        conv2Out  = D3D12Bindings.createDefaultBuffer(dev, fb(3136), arena);
        pool2Out  = D3D12Bindings.createDefaultBuffer(dev, fb(256), arena);
        fcWBuf    = D3D12Bindings.createDefaultBuffer(dev, fb(fcWeight.length), arena);
        fcBBuf    = D3D12Bindings.createDefaultBuffer(dev, fb(fcBias.length), arena);
        outputBuf = D3D12Bindings.createDefaultBuffer(dev, fb(10), arena);
        log.debug("GPU buffers created (12 default-heap UAV buffers)");
    }

    // ── Step 3: Upload weights ───────────────────────────────────────────

    private void uploadWeights() throws WindowsNativeException {
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        D3D12Bindings.uploadFloats(dev, q, conv1FBuf, conv1Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv1BBuf, conv1Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2FBuf, conv2Filter, arena);
        D3D12Bindings.uploadFloats(dev, q, conv2BBuf, conv2Bias, arena);
        D3D12Bindings.uploadFloats(dev, q, fcWBuf, fcWeight, arena);
        D3D12Bindings.uploadFloats(dev, q, fcBBuf, fcBias, arena);
        log.debug("Weights uploaded to GPU");
    }

    // ── Step 4: Compile DML operators ────────────────────────────────────

    private void compileAllOperators() throws WindowsNativeException {
        var dml = wb.getDmlDevice();

        // Conv1 + fused Relu: (1,1,28,28)→(1,8,28,28), filter(8,1,5,5), bias(1,8,1,1), pad=2
        compiledConv1 = createAndCompileConv(
                new int[]{1, 1, 28, 28}, new int[]{8, 1, 5, 5}, new int[]{1, 8, 1, 1},
                new int[]{1, 8, 28, 28}, new int[]{1, 1}, new int[]{2, 2}, new int[]{2, 2}, true, true);

        // MaxPool1: (1,8,28,28)→(1,8,14,14), window=2, stride=2
        compiledPool1 = createAndCompilePool(
                new int[]{1, 8, 28, 28}, new int[]{1, 8, 14, 14},
                new int[]{2, 2}, new int[]{2, 2});

        // Conv2 + fused Relu: (1,8,14,14)→(1,16,14,14), filter(16,8,5,5), bias(1,16,1,1), pad=2
        compiledConv2 = createAndCompileConv(
                new int[]{1, 8, 14, 14}, new int[]{16, 8, 5, 5}, new int[]{1, 16, 1, 1},
                new int[]{1, 16, 14, 14}, new int[]{1, 1}, new int[]{2, 2}, new int[]{2, 2}, true, true);

        // MaxPool2: (1,16,14,14)→(1,16,4,4), window=3, stride=3
        compiledPool2 = createAndCompilePool(
                new int[]{1, 16, 14, 14}, new int[]{1, 16, 4, 4},
                new int[]{3, 3}, new int[]{3, 3});

        // Gemm: A(1,1,1,256) × B(1,1,256,10) + C(1,1,1,10) → (1,1,1,10)
        compiledGemm = createAndCompileGemm(
                new int[]{1, 1, 1, 256}, new int[]{1, 1, 256, 10},
                new int[]{1, 1, 1, 10}, new int[]{1, 1, 1, 10});

        allCompiled = new MemorySegment[]{compiledConv1, compiledPool1, compiledConv2, compiledPool2, compiledGemm};
        log.info("All 5 DML operators compiled");
    }

    // ── Step 5: Allocate binding resources ───────────────────────────────

    private void allocateBindingResources() throws WindowsNativeException {
        var dml = wb.getDmlDevice();
        var dev = wb.getD3d12Device();

        tempSizes = new long[5];
        persistSizes = new long[5];
        descCounts = new int[5];
        tempBufs = new MemorySegment[5];
        persistBufs = new MemorySegment[5];
        totalDescriptors = 0;

        for (int i = 0; i < 5; i++) {
            long[] bp = DirectMlBindings.getBindingProperties(allCompiled[i], arena);

            descCounts[i] = (int) bp[0];
            tempSizes[i] = bp[1];
            persistSizes[i] = bp[2];
            totalDescriptors += Math.max(descCounts[i], 1); // dispatch uses min 1

            if (tempSizes[i] > 0) {
                tempBufs[i] = D3D12Bindings.createDefaultBuffer(dev, tempSizes[i], arena);
            }
            if (persistSizes[i] > 0) {
                persistBufs[i] = D3D12Bindings.createDefaultBuffer(dev, persistSizes[i], arena);
            }
            log.debug("Op[{}]: desc={}, temp={}, persist={}", i, descCounts[i], tempSizes[i], persistSizes[i]);
        }

        // Add some slack for initializer
        totalDescriptors = Math.max(totalDescriptors, 32) + 32;
        descriptorHeap = D3D12Bindings.createDescriptorHeap(dev, totalDescriptors, arena);
        descriptorIncrement = D3D12Bindings.getDescriptorIncrementSize(dev);
        cmdRecorder = DirectMlBindings.createCommandRecorder(dml, arena);

        log.debug("Descriptor heap: {} descriptors, increment={}", totalDescriptors, descriptorIncrement);
    }

    // ── Step 6: Initialize operators (upload persistent resources) ───────

    private void initializeOperators() throws WindowsNativeException {
        boolean anyPersist = false;
        for (long ps : persistSizes) if (ps > 0) { anyPersist = true; break; }
        if (!anyPersist) {
            log.info("No persistent resources needed – skipping operator initialization");
            return;
        }

        var dml = wb.getDmlDevice();
        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();

        MemorySegment initializer = DirectMlBindings.createOperatorInitializer(dml, allCompiled, arena);
        long[] initBp = DirectMlBindings.getBindingProperties(initializer, arena);
        int initDescCount = Math.max((int) initBp[0], 1); // DML needs at least 1 descriptor
        long initTempSize = initBp[1];

        long cpuStart = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuStart = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, initializer,
                cpuStart, gpuStart, initDescCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        // Bind inputs (weight data for each compiled op, or NONE)
        MemorySegment inputBindings = arena.allocate(16L * 5, 8);
        for (int i = 0; i < 5; i++) {
            if (persistSizes[i] > 0) {
                // TODO: bind weight upload buffers for initialization
                setBindingDesc(inputBindings, i, DirectMlBindings.DML_BINDING_TYPE_NONE, MemorySegment.NULL);
            } else {
                setBindingDesc(inputBindings, i, DirectMlBindings.DML_BINDING_TYPE_NONE, MemorySegment.NULL);
            }
        }
        DirectMlBindings.bindInputs(bt, 5, inputBindings);

        // Bind temp
        if (initTempSize > 0) {
            MemorySegment initTmp = D3D12Bindings.createDefaultBuffer(dev, initTempSize, arena);
            MemorySegment tmpBb = DirectMlBindings.allocBufferBinding(arena, initTmp, 0, initTempSize);
            MemorySegment tmpBd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, tmpBb);
            DirectMlBindings.bindTemporaryResource(bt, tmpBd);
        }

        // Record and execute
        var alloc = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        var cmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
        D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, initializer, bt);
        D3D12Bindings.executeAndWait(dev, q, cmdList, arena);

        DxgiBindings.release(cmdList);
        DxgiBindings.release(alloc);
        DxgiBindings.release(bt);
        DxgiBindings.release(initializer);
        log.info("Operator initialization complete");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Inference
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Run MNIST inference. Input: 784 floats (28×28 grayscale, 0.0–1.0). Output: 10 logits.
     */
    public float[] infer(float[] input) throws WindowsNativeException {
        if (!loaded) throw new WindowsNativeException("Pipeline not loaded – call loadModel() first");
        if (input.length != 784) throw new WindowsNativeException("Expected 784 floats, got " + input.length);

        var dev = wb.getD3d12Device();
        var q = wb.getCommandQueue();
        var dml = wb.getDmlDevice();

        // Upload input
        log.debug("Uploading input to GPU...");
        D3D12Bindings.uploadFloats(dev, q, inputBuf, input, arena);

        // Create command infrastructure for this dispatch
        log.debug("Creating command infrastructure...");
        var alloc = D3D12Bindings.createCommandAllocator(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, arena);
        var cmdList = D3D12Bindings.createCommandList(dev, D3D12Bindings.D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, arena);
        D3D12Bindings.setDescriptorHeaps(cmdList, descriptorHeap, arena);

        long cpuBase = D3D12Bindings.getCpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        long gpuBase = D3D12Bindings.getGpuDescriptorHandleForHeapStart(descriptorHeap, arena);
        log.debug("INFER: cpuBase=0x{}, gpuBase=0x{}", Long.toHexString(cpuBase), Long.toHexString(gpuBase));
        int descOffset = 0;

        // ── Op 0: Conv1 + Relu ───────────────────────────────────────────
        log.debug("Dispatching Op0 (Conv1)...");
        descOffset = dispatchConv(dml, cmdList, compiledConv1, 0,
                inputBuf, fb(784), conv1FBuf, fb(conv1Filter.length),
                conv1BBuf, fb(conv1Bias.length), conv1Out, fb(6272),
                true, cpuBase, gpuBase, descOffset);
        log.debug("Op0 done, descOffset={}", descOffset);
        D3D12Bindings.uavBarrier(cmdList, arena);

        // ── Op 1: MaxPool1 ──────────────────────────────────────────────
        log.debug("Dispatching Op1 (Pool1)...");
        descOffset = dispatchPool(dml, cmdList, compiledPool1, 1,
                conv1Out, fb(6272), pool1Out, fb(1568),
                cpuBase, gpuBase, descOffset);
        log.debug("Op1 done, descOffset={}", descOffset);
        D3D12Bindings.uavBarrier(cmdList, arena);

        // ── Op 2: Conv2 + Relu ───────────────────────────────────────────
        log.debug("Dispatching Op2 (Conv2)...");
        descOffset = dispatchConv(dml, cmdList, compiledConv2, 2,
                pool1Out, fb(1568), conv2FBuf, fb(conv2Filter.length),
                conv2BBuf, fb(conv2Bias.length), conv2Out, fb(3136),
                true, cpuBase, gpuBase, descOffset);
        log.debug("Op2 done, descOffset={}", descOffset);
        D3D12Bindings.uavBarrier(cmdList, arena);

        // ── Op 3: MaxPool2 ──────────────────────────────────────────────
        log.debug("Dispatching Op3 (Pool2)...");
        descOffset = dispatchPool(dml, cmdList, compiledPool2, 3,
                conv2Out, fb(3136), pool2Out, fb(256),
                cpuBase, gpuBase, descOffset);
        log.debug("Op3 done, descOffset={}", descOffset);
        D3D12Bindings.uavBarrier(cmdList, arena);

        // ── Op 4: Gemm ──────────────────────────────────────────────────
        log.debug("Dispatching Op4 (Gemm)...");
        descOffset = dispatchGemm(dml, cmdList, compiledGemm, 4,
                pool2Out, fb(256), fcWBuf, fb(fcWeight.length),
                fcBBuf, fb(fcBias.length), outputBuf, fb(10),
                cpuBase, gpuBase, descOffset);
        log.debug("Op4 done, executing...");

        // Execute all dispatches
        D3D12Bindings.executeAndWait(dev, q, cmdList, arena);

        // Read back
        float[] result = D3D12Bindings.readbackFloats(dev, q, outputBuf, 10, arena);

        DxgiBindings.release(cmdList);
        DxgiBindings.release(alloc);

        log.debug("Inference complete: {}", Arrays.toString(result));
        return result;
    }

    /** Compute argmax of output logits. */
    public static int argmax(float[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Dispatch helpers
    // ══════════════════════════════════════════════════════════════════════

    private int dispatchConv(MemorySegment dml, MemorySegment cmdList,
                              MemorySegment compiled, int opIdx,
                              MemorySegment inBuf, long inSize,
                              MemorySegment filterBuf, long filterSize,
                              MemorySegment biasBuf, long biasSize,
                              MemorySegment outBuf, long outSize,
                              boolean hasBias,
                              long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1); // DML needs at least 1 descriptor
        log.debug("dispatchConv[{}]: descCount={}", opIdx, descCount);

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);

        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        if (hasBias) {
            // 3 inputs: data, filter, bias
            MemorySegment inputs = arena.allocate(16L * 3, 8);
            setBufferBinding(inputs, 0, inBuf, inSize);
            setBufferBinding(inputs, 1, filterBuf, filterSize);
            setBufferBinding(inputs, 2, biasBuf, biasSize);
            DirectMlBindings.bindInputs(bt, 3, inputs);
        } else {
            // 2 inputs: data, filter (no bias)
            MemorySegment inputs = arena.allocate(16L * 2, 8);
            setBufferBinding(inputs, 0, inBuf, inSize);
            setBufferBinding(inputs, 1, filterBuf, filterSize);
            DirectMlBindings.bindInputs(bt, 2, inputs);
        }

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }

    private int dispatchPool(MemorySegment dml, MemorySegment cmdList,
                              MemorySegment compiled, int opIdx,
                              MemorySegment inBuf, long inSize,
                              MemorySegment outBuf, long outSize,
                              long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1); // DML needs at least 1 descriptor

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        MemorySegment inputs = arena.allocate(16, 8);
        setBufferBinding(inputs, 0, inBuf, inSize);
        DirectMlBindings.bindInputs(bt, 1, inputs);

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }

    private int dispatchGemm(MemorySegment dml, MemorySegment cmdList,
                              MemorySegment compiled, int opIdx,
                              MemorySegment aBuf, long aSize,
                              MemorySegment bBuf, long bSize,
                              MemorySegment cBuf, long cSize,
                              MemorySegment outBuf, long outSize,
                              long cpuBase, long gpuBase, int descOff)
            throws WindowsNativeException {
        int descCount = Math.max(descCounts[opIdx], 1); // DML needs at least 1 descriptor

        MemorySegment btDesc = DirectMlBindings.allocBindingTableDesc(arena, compiled,
                cpuBase + (long) descOff * descriptorIncrement,
                gpuBase + (long) descOff * descriptorIncrement, descCount);
        MemorySegment bt = DirectMlBindings.createBindingTable(dml, btDesc, arena);

        MemorySegment inputs = arena.allocate(16L * 3, 8);
        setBufferBinding(inputs, 0, aBuf, aSize);
        setBufferBinding(inputs, 1, bBuf, bSize);
        setBufferBinding(inputs, 2, cBuf, cSize);
        DirectMlBindings.bindInputs(bt, 3, inputs);

        MemorySegment outputs = arena.allocate(16, 8);
        setBufferBinding(outputs, 0, outBuf, outSize);
        DirectMlBindings.bindOutputs(bt, 1, outputs);

        bindTempAndPersist(bt, opIdx);
        DirectMlBindings.recordDispatch(cmdRecorder, cmdList, compiled, bt);
        DxgiBindings.release(bt);
        return descOff + descCount;
    }

    private void bindTempAndPersist(MemorySegment bt, int opIdx) {
        if (tempSizes[opIdx] > 0 && tempBufs[opIdx] != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, tempBufs[opIdx], 0, tempSizes[opIdx]);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindTemporaryResource(bt, bd);
        }
        if (persistSizes[opIdx] > 0 && persistBufs[opIdx] != null) {
            MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, persistBufs[opIdx], 0, persistSizes[opIdx]);
            MemorySegment bd = DirectMlBindings.allocBindingDesc(arena, DirectMlBindings.DML_BINDING_TYPE_BUFFER, bb);
            DirectMlBindings.bindPersistentResource(bt, bd);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Operator creation helpers
    // ══════════════════════════════════════════════════════════════════════

    /**
     * Create and compile a DML Convolution operator (optionally with fused ReLU).
     */
    private MemorySegment createAndCompileConv(int[] inShape, int[] filterShape, int[] biasShape,
                                                int[] outShape, int[] strides, int[] padStart,
                                                int[] padEnd, boolean fuseRelu,
                                                boolean hasBias)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();

        var inTD = td(inShape);
        var filterTD = td(filterShape);
        var outTD = td(outShape);

        // DML_CONVOLUTION_OPERATOR_DESC: 104 bytes
        MemorySegment conv = arena.allocate(104, 8);
        conv.set(ValueLayout.ADDRESS, 0, inTD);          // InputTensor
        conv.set(ValueLayout.ADDRESS, 8, filterTD);       // FilterTensor
        if (hasBias) {
            var biasTD = td(biasShape);
            conv.set(ValueLayout.ADDRESS, 16, biasTD);    // BiasTensor
        } else {
            conv.set(ValueLayout.ADDRESS, 16, MemorySegment.NULL); // No bias
        }
        conv.set(ValueLayout.ADDRESS, 24, outTD);          // OutputTensor
        conv.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_CONVOLUTION_MODE_CROSS_CORRELATION);
        conv.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_CONVOLUTION_DIRECTION_FORWARD);
        conv.set(ValueLayout.JAVA_INT, 40, 2);            // DimensionCount (spatial)

        MemorySegment stridesP = allocInts(strides);
        MemorySegment dilations = allocInts(new int[]{1, 1});
        MemorySegment padStartP = allocInts(padStart);
        MemorySegment padEndP = allocInts(padEnd);
        MemorySegment outPadP = allocInts(new int[]{0, 0});

        conv.set(ValueLayout.ADDRESS, 48, stridesP);
        conv.set(ValueLayout.ADDRESS, 56, dilations);
        conv.set(ValueLayout.ADDRESS, 64, padStartP);
        conv.set(ValueLayout.ADDRESS, 72, padEndP);
        conv.set(ValueLayout.ADDRESS, 80, outPadP);
        conv.set(ValueLayout.JAVA_INT, 88, 1);            // GroupCount

        if (fuseRelu) {
            // DML_ACTIVATION_RELU_OPERATOR_DESC: InputTensor=NULL, OutputTensor=NULL (fused)
            MemorySegment reluDesc = arena.allocate(16, 8); // both pointers NULL
            MemorySegment fusedAct = DirectMlBindings.allocOperatorDesc(arena,
                    DirectMlBindings.DML_OPERATOR_ACTIVATION_RELU, reluDesc);
            conv.set(ValueLayout.ADDRESS, 96, fusedAct);
        } else {
            conv.set(ValueLayout.ADDRESS, 96, MemorySegment.NULL);
        }

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_CONVOLUTION, conv);

        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }

    /**
     * Create and compile a DML MaxPooling operator.
     */
    private MemorySegment createAndCompilePool(int[] inShape, int[] outShape,
                                                int[] windowSize, int[] strides)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();

        var inTD = td(inShape);
        var outTD = td(outShape);

        // DML_MAX_POOLING_OPERATOR_DESC: 56 bytes
        MemorySegment pool = arena.allocate(56, 8);
        pool.set(ValueLayout.ADDRESS, 0, inTD);
        pool.set(ValueLayout.ADDRESS, 8, outTD);
        pool.set(ValueLayout.JAVA_INT, 16, 2); // DimensionCount (spatial)

        pool.set(ValueLayout.ADDRESS, 24, allocInts(strides));
        pool.set(ValueLayout.ADDRESS, 32, allocInts(windowSize));
        pool.set(ValueLayout.ADDRESS, 40, allocInts(new int[]{0, 0})); // StartPadding
        pool.set(ValueLayout.ADDRESS, 48, allocInts(new int[]{0, 0})); // EndPadding

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_MAX_POOLING, pool);

        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }

    /**
     * Create and compile a DML Gemm operator (MatMul + Add).
     */
    private MemorySegment createAndCompileGemm(int[] aShape, int[] bShape,
                                                int[] cShape, int[] outShape)
            throws WindowsNativeException {
        var dml = wb.getDmlDevice();

        var aTD = td(aShape);
        var bTD = td(bShape);
        var cTD = td(cShape);
        var outTD = td(outShape);

        // DML_GEMM_OPERATOR_DESC: 56 bytes
        MemorySegment gemm = arena.allocate(56, 8);
        gemm.set(ValueLayout.ADDRESS, 0, aTD);
        gemm.set(ValueLayout.ADDRESS, 8, bTD);
        gemm.set(ValueLayout.ADDRESS, 16, cTD);
        gemm.set(ValueLayout.ADDRESS, 24, outTD);
        gemm.set(ValueLayout.JAVA_INT, 32, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE); // TransA
        gemm.set(ValueLayout.JAVA_INT, 36, DirectMlBindings.DML_MATRIX_TRANSFORM_NONE); // TransB
        gemm.set(ValueLayout.JAVA_FLOAT, 40, 1.0f); // Alpha
        gemm.set(ValueLayout.JAVA_FLOAT, 44, 1.0f); // Beta
        gemm.set(ValueLayout.ADDRESS, 48, MemorySegment.NULL); // FusedActivation

        MemorySegment opDesc = DirectMlBindings.allocOperatorDesc(arena,
                DirectMlBindings.DML_OPERATOR_GEMM, gemm);

        MemorySegment op = DirectMlBindings.createOperator(dml, opDesc, arena);
        MemorySegment compiled = DirectMlBindings.compileOperator(dml, op, DirectMlBindings.DML_EXECUTION_FLAG_NONE, arena);
        DxgiBindings.release(op);
        return compiled;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Internal helpers
    // ══════════════════════════════════════════════════════════════════════

    /** Build a DML_TENSOR_DESC for a 4D float32 tensor. */
    private MemorySegment td(int[] sizes) {
        int elems = 1;
        for (int s : sizes) elems *= s;
        long byteSize = (long) elems * Float.BYTES;
        var bufTD = DirectMlBindings.allocBufferTensorDesc(arena,
                DirectMlBindings.DML_TENSOR_DATA_TYPE_FLOAT32, sizes, null, byteSize);
        return DirectMlBindings.allocTensorDesc(arena, bufTD);
    }

    /** Allocate a native int[] array. */
    private MemorySegment allocInts(int[] values) {
        MemorySegment seg = arena.allocate((long) values.length * ValueLayout.JAVA_INT.byteSize(), 4);
        for (int i = 0; i < values.length; i++) seg.setAtIndex(ValueLayout.JAVA_INT, i, values[i]);
        return seg;
    }

    /** Resolve a tensor name through Reshape chain to find the actual initializer. */
    private static OnnxModelReader.OnnxTensor resolveInitializer(
            String name, Map<String, OnnxModelReader.OnnxTensor> inits,
            Map<String, String> reshapeMap) {
        // Direct lookup first
        OnnxModelReader.OnnxTensor t = inits.get(name);
        if (t != null) return t;
        // Trace through Reshape outputs → inputs
        String traced = name;
        for (int i = 0; i < 5 && traced != null; i++) {
            traced = reshapeMap.get(traced);
            if (traced != null) {
                t = inits.get(traced);
                if (t != null) return t;
            }
        }
        return null;
    }

    /** Float byte size. */
    private static long fb(int elementCount) {
        return (long) elementCount * Float.BYTES;
    }

    /** Write a DML_BINDING_DESC (buffer type) into a binding desc array at index. */
    private void setBufferBinding(MemorySegment array, int index, MemorySegment buffer, long size) {
        MemorySegment bb = DirectMlBindings.allocBufferBinding(arena, buffer, 0, size);
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, DirectMlBindings.DML_BINDING_TYPE_BUFFER);
        array.set(ValueLayout.ADDRESS, off + 8, bb);
    }

    /** Write a DML_BINDING_DESC at index. */
    private void setBindingDesc(MemorySegment array, int index, int type, MemorySegment desc) {
        long off = (long) index * 16;
        array.set(ValueLayout.JAVA_INT, off, type);
        array.set(ValueLayout.ADDRESS, off + 8, desc);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Cleanup
    // ══════════════════════════════════════════════════════════════════════

    @Override
    public void close() {
        loaded = false;
        if (cmdRecorder != null) DxgiBindings.release(cmdRecorder);
        if (descriptorHeap != null) DxgiBindings.release(descriptorHeap);
        if (allCompiled != null) {
            for (var c : allCompiled) if (c != null) DxgiBindings.release(c);
        }
        // Release GPU buffers
        for (var buf : new MemorySegment[]{inputBuf, conv1FBuf, conv1BBuf, conv1Out, pool1Out,
                conv2FBuf, conv2BBuf, conv2Out, pool2Out, fcWBuf, fcBBuf, outputBuf}) {
            if (buf != null) DxgiBindings.release(buf);
        }
        if (tempBufs != null) for (var b : tempBufs) if (b != null) DxgiBindings.release(b);
        if (persistBufs != null) for (var b : persistBufs) if (b != null) DxgiBindings.release(b);
        if (arena != null) arena.close();
        log.info("MnistPipeline closed");
    }
}

