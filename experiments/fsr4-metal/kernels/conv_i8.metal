// Independent arithmetic prototype. No AMD source or model weights included.
// This is NOT an FSR upscaler, a quantizer, or an accelerated matrix intrinsic.
#include <metal_stdlib>
using namespace metal;

struct ConvParams {
    uint height, width, channels, outputs, kernelSize, stride, padding;
    uint outHeight, outWidth, paddedChannels;
};

kernel void conv_scalar(
    device const char* input [[buffer(0)]],
    device const char* weights [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.outHeight * p.outWidth * p.outputs) return;
    uint oc = index % p.outputs;
    uint x = (index / p.outputs) % p.outWidth;
    uint y = index / (p.outputs * p.outWidth);
    int sum = 0;
    for (uint ky = 0; ky < p.kernelSize; ++ky) {
        int iy = int(y * p.stride + ky) - int(p.padding);
        for (uint kx = 0; kx < p.kernelSize; ++kx) {
            int ix = int(x * p.stride + kx) - int(p.padding);
            if (iy < 0 || ix < 0 || iy >= int(p.height) || ix >= int(p.width)) continue;
            uint a = (uint(iy) * p.width + uint(ix)) * p.paddedChannels;
            uint b = ((oc * p.kernelSize + ky) * p.kernelSize + kx) * p.paddedChannels;
            for (uint c = 0; c < p.channels; ++c)
                sum += int(input[a + c]) * int(weights[b + c]);
        }
    }
    output[index] = sum;
}

kernel void conv_packed4(
    device const char4* input [[buffer(0)]],
    device const char4* weights [[buffer(1)]],
    device int* output [[buffer(2)]],
    constant ConvParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.outHeight * p.outWidth * p.outputs) return;
    uint oc = index % p.outputs;
    uint x = (index / p.outputs) % p.outWidth;
    uint y = index / (p.outputs * p.outWidth);
    uint groups = p.paddedChannels / 4;
    int sum = 0;
    for (uint ky = 0; ky < p.kernelSize; ++ky) {
        int iy = int(y * p.stride + ky) - int(p.padding);
        for (uint kx = 0; kx < p.kernelSize; ++kx) {
            int ix = int(x * p.stride + kx) - int(p.padding);
            if (iy < 0 || ix < 0 || iy >= int(p.height) || ix >= int(p.width)) continue;
            uint a = (uint(iy) * p.width + uint(ix)) * groups;
            uint b = ((oc * p.kernelSize + ky) * p.kernelSize + kx) * groups;
            for (uint c = 0; c < groups; ++c) {
                int4 av = int4(input[a + c]);
                int4 bv = int4(weights[b + c]);
                sum += av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
            }
        }
    }
    output[index] = sum;
}
