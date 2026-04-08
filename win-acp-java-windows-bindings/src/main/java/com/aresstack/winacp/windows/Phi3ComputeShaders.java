package com.aresstack.winacp.windows;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Pre-defined HLSL compute shaders for the Phi-3 GPU pipeline.
 * <p>
 * Each shader operates on UAV root descriptors (no descriptor tables)
 * and is dispatched via {@link GpuComputeKernel}.
 * <p>
 * <b>V3.0 additions</b>: RoPE, KV-cache store, attention score/softmax/V-sum
 * shaders enable a <b>full GPU decode pipeline</b> — all 32 layers + lm_head
 * recorded into ONE command list with ONE fence wait per token.
 */
public final class Phi3ComputeShaders {

    private static final Logger log = LoggerFactory.getLogger(Phi3ComputeShaders.class);

    private Phi3ComputeShaders() {}

    /** Thread group size for all element-wise shaders. */
    public static final int GROUP_SIZE = 256;

    // ═══════════════════════════════════════════════════════════════════
    // Element-wise Add: C[i] = A[i] + B[i]
    // Root params: u0=A, u1=B, u2=C, b0={count}
    // ═══════════════════════════════════════════════════════════════════

    public static final String ELEMENT_ADD_HLSL = """
            RWByteAddressBuffer A : register(u0);
            RWByteAddressBuffer B : register(u1);
            RWByteAddressBuffer C : register(u2);
            cbuffer CB : register(b0) { uint count; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                if (dtid.x < count) {
                    uint addr = dtid.x * 4;
                    C.Store(addr, asuint(asfloat(A.Load(addr)) + asfloat(B.Load(addr))));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // RMSNorm: out[i] = x[i] * weight[i] * rsqrt(mean(x²) + eps)
    // Root params: u0=Input, u1=Weight, u2=Output, b0={dim, eps_bits}
    // ═══════════════════════════════════════════════════════════════════

    public static final String RMSNORM_HLSL = """
            RWByteAddressBuffer Input  : register(u0);
            RWByteAddressBuffer Weight : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint dim; uint eps_bits; };
            
            groupshared float gs_sum[256];
            
            // Single-group RMSNorm: entire vector processed by one group.
            // Requires dim <= 256 * elements_per_thread.
            // For hidden=3072: each thread handles ceil(3072/256) = 12 elements.
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
                float eps = asfloat(eps_bits);
                
                // Phase 1: Compute partial sum of squares
                float partial = 0.0;
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    partial += v * v;
                }
                gs_sum[gi] = partial;
                GroupMemoryBarrierWithGroupSync();
                
                // Phase 2: Tree reduction
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) {
                        gs_sum[gi] += gs_sum[gi + stride];
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                
                // Phase 3: Normalize — rms_inv = rsqrt(sum_sq / dim + eps)
                float rms_inv = rsqrt(gs_sum[0] / (float)dim + eps);
                
                // Phase 4: Apply normalization + weight
                for (uint i = gi; i < dim; i += 256) {
                    float v = asfloat(Input.Load(i * 4));
                    float w = asfloat(Weight.Load(i * 4));
                    Output.Store(i * 4, asuint(v * rms_inv * w));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // SwiGLU + Scale: out[i] = gateUp[i+inter] * gateUp[i] * sigmoid(gateUp[i]) * scale[i]
    // Root params: u0=GateUp, u1=Scale, u2=Output, b0={intermediate}
    // ═══════════════════════════════════════════════════════════════════

    public static final String SWIGLU_HLSL = """
            RWByteAddressBuffer GateUp : register(u0);
            RWByteAddressBuffer Scale  : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint intermediate; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i < intermediate) {
                    float gate = asfloat(GateUp.Load(i * 4));
                    float up   = asfloat(GateUp.Load((intermediate + i) * 4));
                    float sc   = asfloat(Scale.Load(i * 4));
                    float sigmoid = 1.0 / (1.0 + exp(-gate));
                    Output.Store(i * 4, asuint(up * gate * sigmoid * sc));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Scale: out[i] = x[i] * scale[i]
    // Root params: u0=X, u1=Scale, u2=Output, b0={count}
    // ═══════════════════════════════════════════════════════════════════

    public static final String SCALE_HLSL = """
            RWByteAddressBuffer X     : register(u0);
            RWByteAddressBuffer Scale : register(u1);
            RWByteAddressBuffer Out   : register(u2);
            cbuffer CB : register(b0) { uint count; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                if (dtid.x < count) {
                    uint addr = dtid.x * 4;
                    Out.Store(addr, asuint(asfloat(X.Load(addr)) * asfloat(Scale.Load(addr))));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: RoPE — Rotary Position Embedding (in-place on QKV buffer)
    // Applies cos/sin rotation to Q and K heads in the QKV output buffer.
    // Root params: u0=QKV, u1=Cos, u2=Sin, b0={headDim, numQHeads, numKvHeads, pos}
    // Each thread processes one (head, element) pair.
    // Total threads: (numQHeads + numKvHeads) * halfDim
    // ═══════════════════════════════════════════════════════════════════

    public static final String ROPE_HLSL = """
            RWByteAddressBuffer QKV : register(u0);
            RWByteAddressBuffer Cos : register(u1);
            RWByteAddressBuffer Sin : register(u2);
            cbuffer CB : register(b0) { uint headDim; uint numQHeads; uint numKvHeads; uint pos; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint idx = dtid.x;
                uint halfDim = headDim / 2;
                uint totalPairs = (numQHeads + numKvHeads) * halfDim;
                if (idx >= totalPairs) return;
                
                uint totalQPairs = numQHeads * halfDim;
                uint headLocal, elemIdx, baseOffset;
                
                if (idx < totalQPairs) {
                    headLocal = idx / halfDim;
                    elemIdx = idx % halfDim;
                    baseOffset = headLocal * headDim;
                } else {
                    uint kIdx = idx - totalQPairs;
                    headLocal = kIdx / halfDim;
                    elemIdx = kIdx % halfDim;
                    uint hidden = numQHeads * headDim;
                    baseOffset = hidden + headLocal * headDim;
                }
                
                uint csIdx = pos * halfDim + elemIdx;
                float cosVal = asfloat(Cos.Load(csIdx * 4));
                float sinVal = asfloat(Sin.Load(csIdx * 4));
                
                uint x0Addr = (baseOffset + elemIdx) * 4;
                uint x1Addr = (baseOffset + halfDim + elemIdx) * 4;
                float x0 = asfloat(QKV.Load(x0Addr));
                float x1 = asfloat(QKV.Load(x1Addr));
                
                QKV.Store(x0Addr, asuint(x0 * cosVal - x1 * sinVal));
                QKV.Store(x1Addr, asuint(x0 * sinVal + x1 * cosVal));
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: KV Cache Store — copy K and V from QKV output to KV cache
    // Root params: u0=QKV, u1=Kcache, u2=Vcache, b0={hidden, pos}
    // Each thread copies one float for K and one for V.
    // ═══════════════════════════════════════════════════════════════════

    public static final String KV_CACHE_STORE_HLSL = """
            RWByteAddressBuffer QKV    : register(u0);
            RWByteAddressBuffer Kcache : register(u1);
            RWByteAddressBuffer Vcache : register(u2);
            cbuffer CB : register(b0) { uint hidden; uint pos; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint i = dtid.x;
                if (i >= hidden) return;
                uint kSrcAddr = (hidden + i) * 4;
                uint vSrcAddr = (2 * hidden + i) * 4;
                uint dstAddr = (pos * hidden + i) * 4;
                Kcache.Store(dstAddr, QKV.Load(kSrcAddr));
                Vcache.Store(dstAddr, QKV.Load(vSrcAddr));
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: Attention Score — compute Q·K^T scaled dot products
    // One thread per (head, position) pair.
    // Root params: u0=Q, u1=Kcache, u2=Scores, b0={headDim, numHeads, seqLen, scale_bits}
    // Q layout: [numHeads * headDim] (first portion of QKV output)
    // Kcache layout: [seqLen * hidden] where hidden = numHeads * headDim
    // Scores layout: [numHeads * seqLen]
    // ═══════════════════════════════════════════════════════════════════

    public static final String ATTN_SCORE_HLSL = """
            RWByteAddressBuffer Q      : register(u0);
            RWByteAddressBuffer Kcache : register(u1);
            RWByteAddressBuffer Scores : register(u2);
            cbuffer CB : register(b0) { uint headDim; uint numHeads; uint seqLen; uint scale_bits; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 dtid : SV_DispatchThreadID) {
                uint idx = dtid.x;
                uint totalWork = numHeads * seqLen;
                if (idx >= totalWork) return;
                
                uint head = idx / seqLen;
                uint pos = idx % seqLen;
                uint hidden = numHeads * headDim;
                float scale = asfloat(scale_bits);
                
                float dot = 0.0;
                uint qBase = head * headDim * 4;
                uint kBase = (pos * hidden + head * headDim) * 4;
                
                for (uint d = 0; d < headDim; d++) {
                    dot += asfloat(Q.Load(qBase + d * 4)) * asfloat(Kcache.Load(kBase + d * 4));
                }
                
                Scores.Store((head * seqLen + pos) * 4, asuint(dot * scale));
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: Attention Softmax — per-head in-place softmax over scores
    // One thread group per head. Uses shared memory for tree reduction.
    // Root params: u0=Scores (in/out), b0={seqLen, numHeads}
    // ═══════════════════════════════════════════════════════════════════

    public static final String ATTN_SOFTMAX_HLSL = """
            RWByteAddressBuffer Scores : register(u0);
            cbuffer CB : register(b0) { uint seqLen; uint numHeads; };
            
            groupshared float gs_data[256];
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                uint head = gid.x;
                if (head >= numHeads) return;
                uint baseOff = head * seqLen;
                
                // Phase 1: Find max
                float threadMax = -3.402823466e+38;
                for (uint i = gi; i < seqLen; i += 256) {
                    float v = asfloat(Scores.Load((baseOff + i) * 4));
                    threadMax = max(threadMax, v);
                }
                gs_data[gi] = threadMax;
                GroupMemoryBarrierWithGroupSync();
                
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) gs_data[gi] = max(gs_data[gi], gs_data[gi + stride]);
                    GroupMemoryBarrierWithGroupSync();
                }
                float globalMax = gs_data[0];
                
                // Phase 2: exp and sum
                float threadSum = 0.0;
                for (uint i = gi; i < seqLen; i += 256) {
                    float v = exp(asfloat(Scores.Load((baseOff + i) * 4)) - globalMax);
                    Scores.Store((baseOff + i) * 4, asuint(v));
                    threadSum += v;
                }
                gs_data[gi] = threadSum;
                GroupMemoryBarrierWithGroupSync();
                
                for (uint stride = 128; stride > 0; stride >>= 1) {
                    if (gi < stride) gs_data[gi] += gs_data[gi + stride];
                    GroupMemoryBarrierWithGroupSync();
                }
                float globalSum = gs_data[0];
                
                // Phase 3: Normalize
                float invSum = 1.0 / globalSum;
                for (uint i = gi; i < seqLen; i += 256) {
                    float v = asfloat(Scores.Load((baseOff + i) * 4)) * invSum;
                    Scores.Store((baseOff + i) * 4, asuint(v));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // V3.0: Attention V-Sum — weighted sum of V cache with softmax scores
    // One thread group per head. Each thread handles 1+ output dimensions.
    // Root params: u0=Scores, u1=Vcache, u2=Output, b0={headDim, numHeads, seqLen}
    // Output layout: [numHeads * headDim] = [hidden]
    // ═══════════════════════════════════════════════════════════════════

    public static final String ATTN_VSUM_HLSL = """
            RWByteAddressBuffer Scores : register(u0);
            RWByteAddressBuffer Vcache : register(u1);
            RWByteAddressBuffer Output : register(u2);
            cbuffer CB : register(b0) { uint headDim; uint numHeads; uint seqLen; };
            
            [numthreads(256, 1, 1)]
            void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
                uint head = gid.x;
                if (head >= numHeads) return;
                uint hidden = numHeads * headDim;
                
                for (uint d = gi; d < headDim; d += 256) {
                    float acc = 0.0;
                    uint scoreBase = head * seqLen;
                    uint vDimAddr = (head * headDim + d) * 4;
                    
                    for (uint p = 0; p < seqLen; p++) {
                        float w = asfloat(Scores.Load((scoreBase + p) * 4));
                        float v = asfloat(Vcache.Load((p * hidden + head * headDim + d) * 4));
                        acc += w * v;
                    }
                    
                    Output.Store((head * headDim + d) * 4, asuint(acc));
                }
            }
            """;

    // ═══════════════════════════════════════════════════════════════════
    // Factory
    // ═══════════════════════════════════════════════════════════════════

    /** Create all compute kernels needed for the Phi-3 GPU pipeline. */
    public static ComputeKernelSet createAll(WindowsBindings wb, java.lang.foreign.MemorySegment cmdList)
            throws WindowsNativeException {
        long t0 = System.currentTimeMillis();

        GpuComputeKernel addKernel = new GpuComputeKernel(wb, cmdList,
                ELEMENT_ADD_HLSL, "element_add", 3, 1, GROUP_SIZE);
        GpuComputeKernel rmsNormKernel = new GpuComputeKernel(wb, cmdList,
                RMSNORM_HLSL, "rms_norm", 3, 2, GROUP_SIZE);
        GpuComputeKernel swigluKernel = new GpuComputeKernel(wb, cmdList,
                SWIGLU_HLSL, "swiglu", 3, 1, GROUP_SIZE);
        GpuComputeKernel scaleKernel = new GpuComputeKernel(wb, cmdList,
                SCALE_HLSL, "scale", 3, 1, GROUP_SIZE);

        log.info("Phi3 compute shaders (V2.0) compiled in {} ms", System.currentTimeMillis() - t0);
        return new ComputeKernelSet(addKernel, rmsNormKernel, swigluKernel, scaleKernel,
                null, null, null, null, null);
    }

    /**
     * Create ALL compute kernels including V3.0 attention/RoPE/KV-cache shaders.
     * If V3.0 shader compilation fails, returns a set with V2.0 shaders only.
     */
    public static ComputeKernelSet createAllV3(WindowsBindings wb, java.lang.foreign.MemorySegment cmdList)
            throws WindowsNativeException {
        long t0 = System.currentTimeMillis();

        // V2.0 shaders
        GpuComputeKernel addKernel = new GpuComputeKernel(wb, cmdList,
                ELEMENT_ADD_HLSL, "element_add", 3, 1, GROUP_SIZE);
        GpuComputeKernel rmsNormKernel = new GpuComputeKernel(wb, cmdList,
                RMSNORM_HLSL, "rms_norm", 3, 2, GROUP_SIZE);
        GpuComputeKernel swigluKernel = new GpuComputeKernel(wb, cmdList,
                SWIGLU_HLSL, "swiglu", 3, 1, GROUP_SIZE);
        GpuComputeKernel scaleKernel = new GpuComputeKernel(wb, cmdList,
                SCALE_HLSL, "scale", 3, 1, GROUP_SIZE);

        // V3.0 shaders
        GpuComputeKernel ropeKernel = new GpuComputeKernel(wb, cmdList,
                ROPE_HLSL, "rope", 3, 4, GROUP_SIZE);
        GpuComputeKernel kvCacheStoreKernel = new GpuComputeKernel(wb, cmdList,
                KV_CACHE_STORE_HLSL, "kv_cache_store", 3, 2, GROUP_SIZE);
        GpuComputeKernel attnScoreKernel = new GpuComputeKernel(wb, cmdList,
                ATTN_SCORE_HLSL, "attn_score", 3, 4, GROUP_SIZE);
        GpuComputeKernel attnSoftmaxKernel = new GpuComputeKernel(wb, cmdList,
                ATTN_SOFTMAX_HLSL, "attn_softmax", 1, 2, GROUP_SIZE);
        GpuComputeKernel attnVsumKernel = new GpuComputeKernel(wb, cmdList,
                ATTN_VSUM_HLSL, "attn_vsum", 3, 3, GROUP_SIZE);

        log.info("Phi3 compute shaders (V3.0 full GPU) compiled in {} ms",
                System.currentTimeMillis() - t0);
        return new ComputeKernelSet(addKernel, rmsNormKernel, swigluKernel, scaleKernel,
                ropeKernel, kvCacheStoreKernel, attnScoreKernel, attnSoftmaxKernel, attnVsumKernel);
    }

    /**
     * Bundle of all compute kernels for Phi-3.
     * V3.0 fields are nullable (null = V3.0 full GPU not available).
     */
    public record ComputeKernelSet(
            GpuComputeKernel add,
            GpuComputeKernel rmsNorm,
            GpuComputeKernel swiglu,
            GpuComputeKernel scale,
            // V3.0 additions (nullable)
            GpuComputeKernel rope,
            GpuComputeKernel kvCacheStore,
            GpuComputeKernel attnScore,
            GpuComputeKernel attnSoftmax,
            GpuComputeKernel attnVsum
    ) implements AutoCloseable {

        /** Whether V3.0 full GPU shaders are available. */
        public boolean hasV3() {
            return rope != null && kvCacheStore != null
                    && attnScore != null && attnSoftmax != null && attnVsum != null;
        }

        @Override
        public void close() {
            add.close();
            rmsNorm.close();
            swiglu.close();
            scale.close();
            if (rope != null) rope.close();
            if (kvCacheStore != null) kvCacheStore.close();
            if (attnScore != null) attnScore.close();
            if (attnSoftmax != null) attnSoftmax.close();
            if (attnVsum != null) attnVsum.close();
        }
    }
}

