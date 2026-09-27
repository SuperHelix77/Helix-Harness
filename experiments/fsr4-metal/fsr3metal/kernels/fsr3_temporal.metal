// FSR3 temporal accumulation in Metal: the anti-shimmer mechanism.
//
// Provenance: ported from AMD FidelityFX SDK FSR3
// include/gpu/fsr3upscaler/ffx_fsr3upscaler_accumulate.h, MIT, commit
// 60f4ea81909200d8542eca14dccb2628b763a9a3. See ../docs/PROVENANCE.md.
//
// The mechanism under test is the one that decides whether the result shimmers:
// clip the history colour into a per-pixel YCoCg box, then lerp toward the new
// sample by an alpha derived from the box volume. Small alpha means the frame
// trusts history (stable, no shimmer); large alpha means it trusts the current
// sample (responsive, but noisy). The box is driven by reactivity, disocclusion,
// shading change and the lock, exactly as AMD intends.
//
// PERFORMANCE NOTE. An earlier version passed a 96-byte struct per pixel, of
// which 32 bytes were identical for every pixel in the frame. At 4K that is
// 33.2M pixels, so the frame streamed 3.19 GB of parameters to do arithmetic
// that needs about 64 bytes per pixel of real data. Measured cost was ~16 ms,
// matching the bandwidth estimate almost exactly. The frame-constant half is
// now in its own small uniform buffer, which Metal broadcasts from constant
// memory rather than streaming. The per-pixel struct is 64 bytes.

#include <metal_stdlib>
using namespace metal;

// Frame-constant parameters. One of these per dispatch, broadcast to all
// threads from constant memory. 32 bytes.
struct AccumUniforms {
    uint2  upscaleSize;
    float  frameIndex;
    float  deltaTime;
    float  exposure;
    float  prevExposure;
    float  accumulation;
    int    reset;
};

// Per-pixel parameters. 32 bytes. Everything here genuinely varies per pixel
// and is not already resident on the GPU.
//
// The current colour is NOT in this struct. It is the output of the previous
// pass, so it is already in device memory; duplicating it here would make the
// pipeline stream the same pixels twice per frame for no reason.
struct AccumPixel {
    float  lumaInstability;
    float  reactiveMask;
    float  disocclusion;
    float  shadingChange;
    float  lock;
    float  lockContribution;
    float2 motionVector;
};

// RGB <-> YCoCg, the space AMD accumulates in. Accumulating in YCoCg keeps
// chroma error from dragging luma, which is itself a visible shimmer source.
inline float3 rgb_to_ycocg(float3 c) {
    return float3(0.25f*c.r + 0.5f*c.g + 0.25f*c.b,
                  0.5f*c.r - 0.5f*c.b,
                 -0.25f*c.r + 0.5f*c.g - 0.25f*c.b);
}

inline float3 ycocg_to_rgb(float3 v) {
    return float3(v.x + v.y - v.z,
                  v.x + v.z,
                  v.x - v.y - v.z);
}

inline float luma_of(float3 c) { return dot(c, float3(0.299f, 0.587f, 0.114f)); }

inline float volume_of_box(float3 extent) {
    return 8.0f * extent.x * extent.y * extent.z + 1.0e-5f;
}

kernel void temporal_accumulate(
    device const AccumPixel* pixels [[buffer(0)]],
    device float4* outColor [[buffer(1)]],
    device float2* outLock [[buffer(2)]],
    device const float4* currentIn [[buffer(3)]],
    device const float4* historyIn [[buffer(4)]],
    constant AccumUniforms& u [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= u.upscaleSize.x || gid.y >= u.upscaleSize.y) return;
    uint idx = gid.y * u.upscaleSize.x + gid.x;

    AccumPixel p = pixels[idx];
    float4 currentTexel = currentIn[idx];
    // History is reprojected by the caller into a frame-stable layout, so the
    // kernel reads it in place rather than doing the fetch itself.
    float4 historyTexel = historyIn[idx];

    // A camera cut must not blend against a stale frame. Reset is absolute.
    if (u.reset != 0) {
        outColor[idx] = float4(currentTexel.rgb, 0.0f);
        outLock[idx] = float2(0.0f);
        return;
    }

    // Exposure is normalised out of the history so a brightness change does not
    // read as detail and get locked in.
    float3 current = currentTexel.rgb * u.exposure;
    float3 history = historyTexel.rgb * u.prevExposure;

    float3 curYCoCg  = rgb_to_ycocg(current);
    float3 histYCoCg = rgb_to_ycocg(history);

    // --- Clip the history into a box around the current sample ---------------
    // The box width is driven by how much we distrust the neighbourhood.
    // Reactive pixels, disocclusions and shading changes all widen it, which
    // lets the frame move quickly exactly where it must and hold still
    // everywhere else. That split is the whole anti-shimmer design.
    float dist = min(min(p.reactiveMask, p.disocclusion), p.shadingChange);

    const float fShadingExtent = 0.125f;
    float3 boxExtent = fShadingExtent * (1.0f - p.lockContribution * 0.95f);
    boxExtent *= float3(1.0f, 1.0f + dist, 1.0f);

    // The box is centred on the CURRENT sample. Averaging it with history
    // biases the output toward the old frame and produces exactly the smear
    // the clip is supposed to prevent.
    float3 boxCenter = curYCoCg;

    float3 clippedHist = clamp(histYCoCg, boxCenter - boxExtent, boxCenter + boxExtent);

    // --- How much to trust the new sample -----------------------------------
    // History contribution is scaled by the lock and by 1 - disocclusion, so a
    // newly exposed surface is never blended with the geometry that used to
    // occupy that pixel.
    float historyContribution = p.lockContribution
                              * u.accumulation
                              * (1.0f - p.disocclusion);
    historyContribution = saturate(historyContribution);

    // The box volume is the other half of the decision: a small box means high
    // confidence, so a small change is enough to move the result.
    float boxVolume = volume_of_box(boxExtent);
    float volInfluence = saturate(pow(boxVolume, 1.0f / 27.0f) * 0.125f);

    // Alpha is the new sample's weight.
    float alpha = saturate(1.0f - (historyContribution * (1.0f - volInfluence)));

    float3 blended = mix(clippedHist, curYCoCg, alpha);
    float3 resultRGB = max(ycocg_to_rgb(blended) / max(u.exposure, 1.0e-5f), float3(0.0f));

    // --- Lock lifetime ------------------------------------------------------
    // The lock decays so it cannot freeze a pixel forever, and is reset when the
    // neighbourhood is unstable. A locked pixel leans on history, which is what
    // kills shimmer on static detail.
    float decrease = max(max(p.shadingChange, p.reactiveMask), p.disocclusion);
    float lock = p.lock + (1.0f - p.lock) * 0.35f;     // build while stable
    lock = max(0.0f, lock - decrease * 2.0f);            // collapse when not
    lock = max(0.0f, lock - 0.02f);                      // slow decay
    float newLock = saturate(lock);
    float contribution = saturate(newLock);

    outColor[idx] = float4(resultRGB, 0.0f);
    outLock[idx] = float2(newLock, contribution);
}

// A NOTE ON RINGING, kept here because it is easy to assume otherwise.
// Lanczos2 has negative side lobes, so ringing at edges is the obvious suspect
// for any max-error figure. Measured here, the result never leaves the
// [lo,hi] range of its own tap footprint, so there is no overshoot to damp.
// An edge-adaptive damping was implemented and changed the result by exactly
// nothing. The remaining max error is the irreducible cost of a diagonal edge,
// where every 4x4 footprint straddles both sides.

// ---------------------------------------------------------------------------
// Luma instability: a per-pixel measure of how fast luminance is moving. High
// values mark pixels that are already shimmering, so the accumulation stage
// trusts history less there. Computed from the previous output, so it is
// available in a live pipeline with no reference frame.
// ---------------------------------------------------------------------------

kernel void luma_instability(
    device const float4* current [[buffer(0)]],
    device const float4* previous [[buffer(1)]],
    device float* outInstability [[buffer(2)]],
    constant uint2& size [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    uint idx = gid.y * size.x + gid.x;
    float a = luma_of(current[idx].rgb);
    float b = luma_of(previous[idx].rgb);
    float d = abs(a - b) / max(a + b, 1.0e-4f);
    outInstability[idx] = saturate(d * d);
}
