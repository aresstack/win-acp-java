package com.aresstack.winacp.windows;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link MatMulNBitsKernel} — the core GPU kernel for Phi-3.
 * <p>
 * Validates that GPU-computed y = x @ W^T matches the CPU reference
 * for INT4 AWQ block-128 quantized weights.
 */
@EnabledOnOs(OS.WINDOWS)
class MatMulNBitsKernelTest {

    // ══════════════════════════════════════════════════════════════════════
    // Unit test: CPU dequantization logic (no GPU required)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    void dequantize_int4_basic() {
        // Simple 2x4 weight matrix with block_size=4, scale=1.0, zp=0
        // 2 rows, 4 cols, block_size=4 → 1 block per row
        // Each block: 4 values packed in 2 bytes (4/2)
        int N = 2, K = 4, blockSize = 4;
        int blocksPerRow = K / blockSize; // 1

        // qWeight: [N * blocksPerRow * (blockSize/2)] = [2 * 1 * 2] = 4 bytes
        // Row 0: values [1, 2, 3, 4] → packed: byte0 = (2<<4)|1 = 0x21, byte1 = (4<<4)|3 = 0x43
        // Row 1: values [5, 6, 7, 8] → packed: byte0 = (6<<4)|5 = 0x65, byte1 = (8<<4)|7 = 0x87
        byte[] qWeight = new byte[]{0x21, 0x43, 0x65, (byte) 0x87};

        // scales: [N * blocksPerRow] = [2] → both 1.0
        float[] scales = new float[]{1.0f, 1.0f};

        // zeroPoints: packed uint4, [ceil(N*blocksPerRow/2)] = 1 byte
        // zp=0 for both: 0x00
        byte[] zeroPoints = new byte[]{0x00};

        float[] result = MatMulNBitsKernel.dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);

        assertEquals(N * K, result.length);
        // Row 0: [1, 2, 3, 4]
        assertEquals(1.0f, result[0], 1e-6f);
        assertEquals(2.0f, result[1], 1e-6f);
        assertEquals(3.0f, result[2], 1e-6f);
        assertEquals(4.0f, result[3], 1e-6f);
        // Row 1: [5, 6, 7, 8]
        assertEquals(5.0f, result[4], 1e-6f);
        assertEquals(6.0f, result[5], 1e-6f);
        assertEquals(7.0f, result[6], 1e-6f);
        assertEquals(8.0f, result[7], 1e-6f);
    }

    @Test
    void dequantize_int4_with_zero_point_and_scale() {
        // 1x4, block_size=4, scale=2.0, zp=8 (centered INT4: range [0,15] → [-8,7])
        int N = 1, K = 4, blockSize = 4;
        // Values [8, 9, 10, 11] → after zp=8: [0, 1, 2, 3] → after scale=2: [0, 2, 4, 6]
        // packed: byte0 = (9<<4)|8 = 0x98, byte1 = (11<<4)|10 = 0xBA
        byte[] qWeight = new byte[]{(byte) 0x98, (byte) 0xBA};
        float[] scales = new float[]{2.0f};
        // zp=8 packed: low nibble = 8 → 0x08
        byte[] zeroPoints = new byte[]{0x08};

        float[] result = MatMulNBitsKernel.dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);
        assertEquals(0.0f, result[0], 1e-6f);
        assertEquals(2.0f, result[1], 1e-6f);
        assertEquals(4.0f, result[2], 1e-6f);
        assertEquals(6.0f, result[3], 1e-6f);
    }

    // ══════════════════════════════════════════════════════════════════════
    // GPU integration test: CPU reference vs GPU GEMM
    // ══════════════════════════════════════════════════════════════════════

    @Test
    void gpu_matvec_matches_cpu_reference() throws Exception {
        if (!DirectMlBindings.isAvailable()) {
            System.out.println("DirectML not available, skipping GPU test");
            return;
        }

        // Create a synthetic quantized weight matrix [N=64, K=128], block_size=128
        int N = 64, K = 128, blockSize = 128;
        int blocksPerRow = K / blockSize; // 1
        int qWeightSize = N * blocksPerRow * (blockSize / 2); // 64 * 1 * 64 = 4096
        byte[] qWeight = new byte[qWeightSize];
        float[] scales = new float[N * blocksPerRow];
        byte[] zeroPoints = new byte[(N * blocksPerRow + 1) / 2];

        // Fill with deterministic test data
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < qWeight.length; i++) {
            qWeight[i] = (byte) rng.nextInt(256);
        }
        for (int i = 0; i < scales.length; i++) {
            scales[i] = 0.01f + rng.nextFloat() * 0.02f; // scale in [0.01, 0.03]
        }
        for (int i = 0; i < zeroPoints.length; i++) {
            zeroPoints[i] = (byte) 0x88; // zp=8 (symmetric)
        }

        // Input vector
        float[] x = new float[K];
        for (int i = 0; i < K; i++) {
            x[i] = (rng.nextFloat() - 0.5f) * 2.0f;
        }

        // ── CPU reference: dequantize + manual matvec ────────────────
        float[] dequantized = MatMulNBitsKernel.dequantizeInt4(
                qWeight, scales, zeroPoints, N, K, blockSize);
        float[] cpuResult = new float[N];
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += x[k] * dequantized[n * K + k];
            }
            cpuResult[n] = sum;
        }

        // ── GPU: MatMulNBitsKernel ───────────────────────────────────
        try (var wb = new WindowsBindings()) {
            wb.init("directml");

            try (var kernel = new MatMulNBitsKernel(wb, N, K, qWeight, scales, zeroPoints, blockSize)) {
                float[] gpuResult = kernel.matvec(x);

                // Verify dimensions
                assertEquals(N, gpuResult.length);

                // Verify values match within tolerance
                // FP32 GPU arithmetic may differ slightly due to operation ordering
                float maxAbsErr = 0;
                float maxRelErr = 0;
                for (int i = 0; i < N; i++) {
                    float absErr = Math.abs(gpuResult[i] - cpuResult[i]);
                    float relErr = Math.abs(cpuResult[i]) > 1e-6f
                            ? absErr / Math.abs(cpuResult[i]) : absErr;
                    maxAbsErr = Math.max(maxAbsErr, absErr);
                    maxRelErr = Math.max(maxRelErr, relErr);
                }

                System.out.printf("CPU vs GPU: maxAbsErr=%.6f, maxRelErr=%.6f%n",
                        maxAbsErr, maxRelErr);

                // GPU GEMM should be very close to CPU for FP32
                for (int i = 0; i < N; i++) {
                    assertEquals(cpuResult[i], gpuResult[i], 0.01f,
                            String.format("Mismatch at [%d]: cpu=%.6f, gpu=%.6f",
                                    i, cpuResult[i], gpuResult[i]));
                }
            }
        }
    }

    @Test
    void gpu_matvec_realistic_dimensions() throws Exception {
        if (!DirectMlBindings.isAvailable()) {
            System.out.println("DirectML not available, skipping GPU test");
            return;
        }

        // Phi-3 q_proj dimensions: [3072, 3072], block=128
        // For test: use [256, 256] to keep memory reasonable
        int N = 256, K = 256, blockSize = 128;
        int blocksPerRow = K / blockSize; // 2
        int qWeightSize = N * blocksPerRow * (blockSize / 2);
        byte[] qWeight = new byte[qWeightSize];
        float[] scales = new float[N * blocksPerRow];
        byte[] zeroPoints = new byte[(N * blocksPerRow + 1) / 2];

        java.util.Random rng = new java.util.Random(123);
        for (int i = 0; i < qWeight.length; i++) qWeight[i] = (byte) rng.nextInt(256);
        for (int i = 0; i < scales.length; i++) scales[i] = 0.005f + rng.nextFloat() * 0.01f;
        java.util.Arrays.fill(zeroPoints, (byte) 0x88);

        float[] x = new float[K];
        for (int i = 0; i < K; i++) x[i] = (rng.nextFloat() - 0.5f);

        // CPU reference
        float[] deq = MatMulNBitsKernel.dequantizeInt4(qWeight, scales, zeroPoints, N, K, blockSize);
        float[] cpuResult = new float[N];
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += x[k] * deq[n * K + k];
            cpuResult[n] = sum;
        }

        // GPU
        try (var wb = new WindowsBindings()) {
            wb.init("directml");
            try (var kernel = new MatMulNBitsKernel(wb, N, K, qWeight, scales, zeroPoints, blockSize)) {
                float[] gpuResult = kernel.matvec(x);

                float maxErr = 0;
                for (int i = 0; i < N; i++) {
                    maxErr = Math.max(maxErr, Math.abs(gpuResult[i] - cpuResult[i]));
                }
                System.out.printf("[%d,%d] CPU vs GPU maxAbsErr=%.6f%n", N, K, maxErr);
                assertTrue(maxErr < 0.05f, "Max error too large: " + maxErr);
            }
        }
    }
}

