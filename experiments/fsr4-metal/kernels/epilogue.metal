// Independent dyadic quantization experiment, NOT verified AMD FSR arithmetic.
// ConvParams and conv_packed4 are supplied by conv_i8.metal in the same library.
// Host proves |(sum+bias)*multiplier| < INT_MAX and shift in [1,30].
inline char finish_i8(int sum, int bias, int multiplier, uint shift, int zero, int lower) {
    int v = (sum + bias) * multiplier;
    uint magnitude = v < 0 ? uint(-v) : uint(v);
    uint q = magnitude >> shift;
    uint remainder = magnitude & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    q += uint(remainder > midpoint || (remainder == midpoint && (q & 1u)));
    int signed_q = v < 0 ? -int(q) : int(q);
    return char(clamp(signed_q + zero, lower, 127));
}

kernel void requantize_i8(
    device const int* sums [[buffer(0)]], device const int* bias [[buffer(1)]],
    device const int* mult [[buffer(2)]], device const uint* shifts [[buffer(3)]],
    device const int* zeros [[buffer(4)]], device const int* lower [[buffer(5)]],
    device char* output [[buffer(6)]], constant uint& channels [[buffer(7)]],
    constant uint& count [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint c = i % channels;
    output[i] = finish_i8(sums[i], bias[c], mult[c], shifts[c], zeros[c], lower[c]);
}

kernel void conv_fused_i8(
    device const char4* input [[buffer(0)]], device const char4* weights [[buffer(1)]],
    device char* output [[buffer(2)]], constant ConvParams& p [[buffer(3)]],
    device const int* bias [[buffer(4)]], device const int* mult [[buffer(5)]],
    device const uint* shifts [[buffer(6)]], device const int* zeros [[buffer(7)]],
    device const int* lower [[buffer(8)]], uint index [[thread_position_in_grid]]) {
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
                int4 av = int4(input[a+c]);
                int4 bv = int4(weights[b+c]);
                sum += av.x*bv.x + av.y*bv.y + av.z*bv.z + av.w*bv.w;
            }
        }
    }
    output[index] = finish_i8(sum, bias[oc], mult[oc], shifts[oc], zeros[oc], lower[oc]);
}
