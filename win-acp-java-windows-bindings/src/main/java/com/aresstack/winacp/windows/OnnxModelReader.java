package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Minimal ONNX protobuf reader – parses what is needed for MNIST-family models
 * (validated with {@code mnist-12.onnx} and {@code mnist-8.onnx}).
 * <p>
 * No dependency on Google Protobuf or any ONNX library.
 * Hand-written decoder for the ONNX wire format (protobuf3).
 * <p>
 * Only supports: ModelProto → GraphProto → NodeProto, TensorProto (float32 / raw_data).
 */
public final class OnnxModelReader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelReader.class);

    private OnnxModelReader() {}

    // ── Public data types ────────────────────────────────────────────────

    public record OnnxTensor(String name, long[] dims, int dataType, float[] data) {
        public int elementCount() {
            int n = 1;
            for (long d : dims) n *= (int) d;
            return n;
        }
    }

    public record OnnxNode(String opType, List<String> inputs, List<String> outputs,
                            Map<String, Object> attrs) {
        @SuppressWarnings("unchecked")
        public List<Long> getInts(String name) {
            Object v = attrs.get(name);
            return v instanceof List<?> ? (List<Long>) v : List.of();
        }
        public long getInt(String name, long def) {
            Object v = attrs.get(name);
            return v instanceof Long l ? l : def;
        }
        public String getString(String name, String def) {
            Object v = attrs.get(name);
            return v instanceof String s ? s : def;
        }
    }

    public record OnnxGraph(String name, List<OnnxNode> nodes,
                             Map<String, OnnxTensor> initializers,
                             List<String> inputNames, List<String> outputNames) {}

    // ── Entry point ──────────────────────────────────────────────────────

    public static OnnxGraph parse(Path onnxFile) throws IOException {
        byte[] bytes = Files.readAllBytes(onnxFile);
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        log.info("Parsing ONNX model: {} ({} bytes)", onnxFile.getFileName(), bytes.length);

        OnnxGraph[] graph = new OnnxGraph[1];

        // ModelProto: field 7 = graph (GraphProto)
        while (buf.hasRemaining()) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 7 && wireType == 2) {
                int len = readVarint32(buf);
                int end = buf.position() + len;
                graph[0] = parseGraph(buf, end);
                buf.position(end);
            } else {
                skipField(buf, wireType);
            }
        }

        if (graph[0] == null) throw new IOException("No graph found in ONNX model");
        log.info("Parsed ONNX graph '{}': {} nodes, {} initializers",
                graph[0].name, graph[0].nodes.size(), graph[0].initializers.size());
        return graph[0];
    }

    // ── GraphProto ───────────────────────────────────────────────────────

    private static OnnxGraph parseGraph(ByteBuffer buf, int end) {
        List<OnnxNode> nodes = new ArrayList<>();
        Map<String, OnnxTensor> initializers = new LinkedHashMap<>();
        List<String> inputNames = new ArrayList<>();
        List<String> outputNames = new ArrayList<>();
        String name = "";

        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { // repeated NodeProto node
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int nodeEnd = buf.position() + len;
                        nodes.add(parseNode(buf, nodeEnd));
                        buf.position(nodeEnd);
                    } else skipField(buf, wireType);
                }
                case 2 -> { // string name
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 5 -> { // repeated TensorProto initializer
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int tEnd = buf.position() + len;
                        OnnxTensor t = parseTensor(buf, tEnd);
                        buf.position(tEnd);
                        if (t != null) initializers.put(t.name, t);
                    } else skipField(buf, wireType);
                }
                case 11 -> { // repeated ValueInfoProto input
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int viEnd = buf.position() + len;
                        String n = parseValueInfoName(buf, viEnd);
                        buf.position(viEnd);
                        if (n != null) inputNames.add(n);
                    } else skipField(buf, wireType);
                }
                case 12 -> { // repeated ValueInfoProto output
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int viEnd = buf.position() + len;
                        String n = parseValueInfoName(buf, viEnd);
                        buf.position(viEnd);
                        if (n != null) outputNames.add(n);
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
        }
        return new OnnxGraph(name, nodes, initializers, inputNames, outputNames);
    }

    // ── NodeProto ────────────────────────────────────────────────────────

    private static OnnxNode parseNode(ByteBuffer buf, int end) {
        List<String> inputs = new ArrayList<>();
        List<String> outputs = new ArrayList<>();
        String name = "";
        String opType = "";
        Map<String, Object> attrs = new LinkedHashMap<>();

        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { if (wireType == 2) inputs.add(readString(buf)); else skipField(buf, wireType); }
                case 2 -> { if (wireType == 2) outputs.add(readString(buf)); else skipField(buf, wireType); }
                case 3 -> { if (wireType == 2) name = readString(buf); else skipField(buf, wireType); }
                case 4 -> { if (wireType == 2) opType = readString(buf); else skipField(buf, wireType); }
                case 5 -> { // AttributeProto
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int aEnd = buf.position() + len;
                        parseAttribute(buf, aEnd, attrs);
                        buf.position(aEnd);
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
        }
        return new OnnxNode(opType, inputs, outputs, attrs);
    }

    // ── AttributeProto ───────────────────────────────────────────────────

    private static void parseAttribute(ByteBuffer buf, int end, Map<String, Object> attrs) {
        String attrName = "";
        Long intVal = null;
        Float floatVal = null;
        String strVal = null;
        List<Long> ints = new ArrayList<>();

        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { if (wireType == 2) attrName = readString(buf); else skipField(buf, wireType); }
                case 2 -> { if (wireType == 0) intVal = readVarint64(buf); else skipField(buf, wireType); }
                case 3 -> { if (wireType == 2) strVal = readString(buf); else skipField(buf, wireType); }
                case 4 -> { if (wireType == 5) floatVal = Float.intBitsToFloat(buf.getInt()); else skipField(buf, wireType); }
                case 7 -> { // repeated float — skip for now
                    skipField(buf, wireType);
                }
                case 8 -> { // repeated int64 ints (packed or repeated)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = buf.position() + len;
                        while (buf.position() < pEnd) ints.add(readVarint64(buf));
                    } else if (wireType == 0) {
                        ints.add(readVarint64(buf));
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
        }

        if (!attrName.isEmpty()) {
            if (!ints.isEmpty()) attrs.put(attrName, ints);
            else if (intVal != null) attrs.put(attrName, intVal);
            else if (floatVal != null) attrs.put(attrName, floatVal);
            else if (strVal != null) attrs.put(attrName, strVal);
        }
    }

    // ── TensorProto ──────────────────────────────────────────────────────

    private static OnnxTensor parseTensor(ByteBuffer buf, int end) {
        List<Long> dims = new ArrayList<>();
        int dataType = 0;
        String name = "";
        float[] floatData = null;
        byte[] rawData = null;

        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNum) {
                case 1 -> { // repeated int64 dims (packed or repeated)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int pEnd = buf.position() + len;
                        while (buf.position() < pEnd) dims.add(readVarint64(buf));
                    } else if (wireType == 0) {
                        dims.add(readVarint64(buf));
                    } else skipField(buf, wireType);
                }
                case 2 -> { // int32 data_type
                    if (wireType == 0) dataType = readVarint32(buf);
                    else skipField(buf, wireType);
                }
                case 4 -> { // repeated float float_data (packed)
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        int count = len / 4;
                        floatData = new float[count];
                        for (int i = 0; i < count; i++) floatData[i] = Float.intBitsToFloat(buf.getInt());
                    } else if (wireType == 5) {
                        // Single float (non-packed) — rare for tensors
                        floatData = new float[]{ Float.intBitsToFloat(buf.getInt()) };
                    } else skipField(buf, wireType);
                }
                case 8 -> { // string name — wait, TensorProto.name is field 8
                    if (wireType == 2) name = readString(buf);
                    else skipField(buf, wireType);
                }
                case 13 -> { // bytes raw_data
                    if (wireType == 2) {
                        int len = readVarint32(buf);
                        rawData = new byte[len];
                        buf.get(rawData);
                    } else skipField(buf, wireType);
                }
                default -> skipField(buf, wireType);
            }
        }

        // Convert raw_data to float[] if needed
        float[] data;
        if (floatData != null) {
            data = floatData;
        } else if (rawData != null && dataType == 1) { // FLOAT
            ByteBuffer rb = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);
            data = new float[rawData.length / 4];
            for (int i = 0; i < data.length; i++) data[i] = rb.getFloat();
        } else {
            data = new float[0];
        }

        long[] dimsArr = dims.stream().mapToLong(Long::longValue).toArray();
        log.debug("Tensor '{}': dims={}, dataType={}, elements={}", name, Arrays.toString(dimsArr), dataType, data.length);
        return new OnnxTensor(name, dimsArr, dataType, data);
    }

    // ── ValueInfoProto (just extract name) ───────────────────────────────

    private static String parseValueInfoName(ByteBuffer buf, int end) {
        String name = null;
        while (buf.position() < end) {
            int tag = readVarint32(buf);
            int fieldNum = tag >>> 3;
            int wireType = tag & 0x7;
            if (fieldNum == 1 && wireType == 2) {
                name = readString(buf);
            } else {
                skipField(buf, wireType);
            }
        }
        return name;
    }

    // ── Protobuf primitives ──────────────────────────────────────────────

    private static int readVarint32(ByteBuffer buf) {
        int result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static long readVarint64(ByteBuffer buf) {
        long result = 0;
        int shift = 0;
        while (buf.hasRemaining()) {
            byte b = buf.get();
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
        }
        return result;
    }

    private static String readString(ByteBuffer buf) {
        int len = readVarint32(buf);
        byte[] bytes = new byte[len];
        buf.get(bytes);
        return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static void skipField(ByteBuffer buf, int wireType) {
        switch (wireType) {
            case 0 -> readVarint64(buf);          // varint
            case 1 -> buf.position(buf.position() + 8);  // 64-bit
            case 2 -> {                            // length-delimited
                int len = readVarint32(buf);
                buf.position(buf.position() + len);
            }
            case 5 -> buf.position(buf.position() + 4);  // 32-bit
            default -> throw new RuntimeException("Unknown wire type: " + wireType);
        }
    }
}

