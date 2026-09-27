// Precision-arena kernels for the FP16-vs-INT8 baseline on M3.
//
// This file is an INDEPENDENT experiment. It is not AMD FSR arithmetic, not a
// port of any vendor operator, and contains no model weights. The comparison is
// between our own synthetic operator formulations on identical shapes, on this
// exact host, so that "an FP16 path may beat a scalar integer implementation"
// is answered by measurement rather than by bit count.
//
// Families of the SAME logical convolution:
//   conv_f16_direct    : FP16 multiply-accumulate, half accumulator.
//   conv_f16_f32acc    : identical products, FP32 accumulator, half output.
//   conv_f32_direct    : FP32 reference formulation.
//   conv_f16_simd_1x1  : FP16 1x1 as an 8x8 simdgroup matrix multiply.
// The existing signed INT8 path lives in conv_i8.metal / epilogue.metal and is
// timed by PrecisionLab on the same shapes for a like-for-like comparison.
//
// Correctness: every variant is compared against a host oracle written
// independently of the shader arithmetic. FP16 error is REPORTED, never
// silently accepted, and half arithmetic is explicitly distinguished from
// half storage.

#include <metal_stdlib>
using namespace metal;

struct F16Params {
    uint height, width, channels, outputs, kernelSize, stride, padding;
    uint outHeight, outWidth;
};

constant uint kSimdTile = 8;

kernel void conv_f16_direct(
    device const half* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant F16Params& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.outHeight * p.outWidth * p.outputs) return;
    uint oc = index % p.outputs;
    uint x = (index / p.outputs) % p.outWidth;
    uint y = index / (p.outputs * p.outWidth);
    half acc = half(0.0);
    for (uint ky = 0; ky < p.kernelSize; ++ky) {
        int iy = int(y * p.stride + ky) - int(p.padding);
        for (uint kx = 0; kx < p.kernelSize; ++kx) {
            int ix = int(x * p.stride + kx) - int(p.padding);
            if (iy < 0 || ix < 0 || iy >= int(p.height) || ix >= int(p.width)) continue;
            uint a = (uint(iy) * p.width + uint(ix)) * p.channels;
            uint b = ((oc * p.kernelSize + ky) * p.kernelSize + kx) * p.channels;
            for (uint c = 0; c < p.channels; ++c)
                acc += input[a + c] * weights[b + c];
        }
    }
    output[index] = acc;
}

kernel void conv_f16_f32acc(
    device const half* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant F16Params& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.outHeight * p.outWidth * p.outputs) return;
    uint oc = index % p.outputs;
    uint x = (index / p.outputs) % p.outWidth;
    uint y = index / (p.outputs * p.outWidth);
    float acc = 0.0f;
    for (uint ky = 0; ky < p.kernelSize; ++ky) {
        int iy = int(y * p.stride + ky) - int(p.padding);
        for (uint kx = 0; kx < p.kernelSize; ++kx) {
            int ix = int(x * p.stride + kx) - int(p.padding);
            if (iy < 0 || ix < 0 || iy >= int(p.height) || ix >= int(p.width)) continue;
            uint a = (uint(iy) * p.width + uint(ix)) * p.channels;
            uint b = ((oc * p.kernelSize + ky) * p.kernelSize + kx) * p.channels;
            for (uint c = 0; c < p.channels; ++c)
                acc += float(input[a + c]) * float(weights[b + c]);
        }
    }
    output[index] = half(acc);
}

kernel void conv_f32_direct(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant F16Params& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.outHeight * p.outWidth * p.outputs) return;
    uint oc = index % p.outputs;
    uint x = (index / p.outputs) % p.outWidth;
    uint y = index / (p.outputs * p.outWidth);
    float acc = 0.0f;
    for (uint ky = 0; ky < p.kernelSize; ++ky) {
        int iy = int(y * p.stride + ky) - int(p.padding);
        for (uint kx = 0; kx < p.kernelSize; ++kx) {
            int ix = int(x * p.stride + kx) - int(p.padding);
            if (iy < 0 || ix < 0 || iy >= int(p.height) || ix >= int(p.width)) continue;
            uint a = (uint(iy) * p.width + uint(ix)) * p.channels;
            uint b = ((oc * p.kernelSize + ky) * p.kernelSize + kx) * p.channels;
            for (uint c = 0; c < p.channels; ++c)
                acc += input[a + c] * weights[b + c];
        }
    }
    output[index] = acc;
}

// FP16 1x1 convolution as an 8x8 simdgroup matrix multiply. A 1x1 convolution
// over NHWC is a GEMM: C[N*O, C] = A[N, C] * B[C, O]. Each threadgroup walks K
// in steps of 8 and accumulates an 8x8 output tile.
//
// Orientation contract, verified on this host with an identity-matrix probe:
// simdgroup `a * b` computes an ordinary row-major C = A * B. So `b` must be
// supplied as B[k][o] = W[o][k] (the host pre-transposes the weight tile);
// passing W directly with a transposed load produces a column-shifted result.
kernel void conv_f16_simd_1x1(
    device const half* input [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant F16Params& p [[buffer(3)]],
    constant uint& spatial [[buffer(4)]],
    uint3 tile [[threadgroup_position_in_grid]],
    uint3 lane [[thread_position_in_threadgroup]]) {
    uint n0 = tile.y * kSimdTile;
    uint o0 = tile.x * kSimdTile;

    simdgroup_half8x8 acc;
    acc.thread_elements() = half(0.0);
    for (uint k0 = 0; k0 + kSimdTile <= p.channels; k0 += kSimdTile) {
        simdgroup_half8x8 a, b;
        simdgroup_load(a, input + n0 * p.channels + k0, p.channels);
        // weights is pre-transposed on the host: element [k0+t][o0+s] = W[o0+s][k0+t]
        simdgroup_load(b, weights + k0 * p.outputs + o0, p.outputs);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    // simdgroup_store writes the whole 8x8 tile in the compiler's fragment
    // layout. Reconstructing the tile by hand via thread_elements() indexing
    // is incorrect; the free function is the supported path.
    uint rows = min(kSimdTile, spatial - n0);
    uint cols = min(kSimdTile, p.outputs - o0);
    if (rows == 0 || cols == 0) return;
    if (rows == kSimdTile && cols == kSimdTile) {
        simdgroup_store(acc, output + n0 * p.outputs + o0, p.outputs);
    } else {
        // Edge tiles are staged through threadgroup memory so the store never
        // runs past the end of the output row.
        threadgroup half staged[kSimdTile * kSimdTile];
        simdgroup_store(acc, staged, kSimdTile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < rows; ++r) {
            for (uint c = lane.x; c < cols; c += 8)
                output[(n0 + r) * p.outputs + o0 + c] = staged[r * kSimdTile + c];
        }
    }
}
