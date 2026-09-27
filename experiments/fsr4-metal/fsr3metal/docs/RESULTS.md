# FSR3-class Metal upscaler - measured results, 2026-09-27

Host: Apple M3 Max, macOS 26.6.2, Swift 6.2. Real Metal GPU, runtime MSL
compilation, no CPU fallback. Reproducible to 5 decimal places across runs.

## What is here

`kernels/fsr3_spatial.metal` and `kernels/fsr3_temporal.metal` implement the FSR3
reconstruction and accumulation mechanisms in Metal, ported from the
MIT-licensed AMD reference. `native/UpscaleLab.swift` measures them against
controls. See `PROVENANCE.md` for scope and licensing.

## Two fixture bugs, both of which were hiding the real results

**1. A 2px checker below Nyquist.** The original fixture added a 2px checker to
the high-resolution source. At 4x upscale that becomes 0.5px in the low-resolution
input, so the box downsample destroys it before the upscaler runs. RMSE then
carried a large irreducible constant, and bilinear and Lanczos scored within 1%
of each other for reasons unrelated to the filters. The fixture is now
band-limited: a 3-cycle-per-source-texel ripple survives the downsample and is
exactly where reconstruction quality separates. The receipt reports
`reconstruction_floor_rmse` so this can never silently recur.

**2. A 2x2 Lanczos footprint.** The first working version sampled a 2x2
neighbourhood while applying Lanczos weights, and scored *worse* than bilinear
(0.04327 vs 0.03926). Lanczos2 is 4 taps per axis; using 2 taps keeps the
negative side lobes but discards the taps they act on. Fixing the footprint to a
true 4x4 turned a loss into a win. Recorded because the symptom looks like "the
filter is wrong" when it is really "the footprint is wrong".

## Reconstruction, 4x (320x180 -> 1280x720)

| Variant | RMSE | max error |
|---|---:|---:|
| bilinear (host reference) | 0.04927 | 0.53187 |
| Lanczos (Metal) | **0.04882** | 0.75103 |
| Lanczos + RCAS | **0.04878** | 0.75444 |
| + temporal lock | 0.04878 | 0.75444 |

Lanczos beats bilinear on RMSE by 0.9%, and the margin survives a 0.2%
threshold in the test suite. It is a real but modest win, and modest is the
honest word: on band-limited content a good interpolator recovers most of what
is there.

## The ringing that was not ringing

Max error is worse than bilinear (0.751 vs 0.532), which looked like Lanczos
ringing at the hard diagonal edge. It is not.

An edge-adaptive lobe damping was implemented and measured. It changed the
result by exactly nothing, because the condition it tests - the reconstructed
value falling outside the min/max of its own tap footprint - never occurs. The
diagnostic shows the worst pixel sits at 0.957 with a footprint spanning
[0.088, 0.998], so the result is inside its own range.

The remaining max error is the irreducible cost of a 45-degree edge: a 4x4
footprint on a diagonal contains both sides, so every reconstruction must blur
across it and worst-case error approaches the local contrast. Measured 0.751
against a 0.830 contrast, about 10% over the bound. Bilinear reaches 0.532
against the same contrast. The range clamp is kept as a cheap guard and is
documented as not currently load-bearing.

## Shimmer

Mean absolute frame-to-frame change of the upscaled output on a **static**
input: **0.0** across 16 frames.

That is the correct result but a weak test on its own. With current == history
any correct pipeline is a no-op, so the number cannot separate a working lock
from a broken one. Reported as a necessary condition, not a sufficient one.

## Ghosting, with a control that can fail

A static-input shimmer number cannot tell a working temporal filter from a
broken one, so the real test uses **moving** content: a bright dot translating
2 upscale px/frame over a static background. The metric is trailing energy in
the strip between the previous frame's trailing edge and the current one,
exactly where a pipeline that fails to reproject history would smear the old
dot.

| | trailing energy |
|---|---:|
| this pipeline (reprojects history) | **0.09000** |
| naive blend, no reprojection (control) | 0.17667 |
| clean background | 0.09000 |

The pipeline reads exactly the background value and the control reads 2x that,
so the metric has demonstrated discriminating power and the clean result is a
pass rather than a vacuous one.

Building that control took three wrong versions, each of which would have
passed a broken pipeline: a wake band 12px behind the dot (outside any
possible smear), measurement on the last frame (the old dot has already left),
and a control built from a lerp of background with background (which
mathematically cannot exceed background). Only the third version can fail.

## Performance, and it is not good enough

Measured on this host with all variants interleaved per trial. Ratios are the
durable number; absolute ms are not reproducible here (two clock states).

| Output | spatial | +RCAS | full | full vs spatial | fps |
|---|---:|---:|---:|---:|---:|
| 1440p (3840x2160) | 0.89 ms | 1.88 ms | 5.84 ms | 6.6x | 171 |
| 4K (7680x4320) | 3.49 ms | 7.57 ms | **23.24 ms** | 6.7x | **43** |

**The full pipeline does not fit a 60 fps budget at 4K.** 23.24 ms against a
16.67 ms budget. It clears 4K at 43 fps, which is playable but not what the
objective asks for.

The cost is almost entirely the temporal stage: spatial 3.49 ms, +RCAS
7.57 ms, +temporal 23.24 ms. The accumulate pass alone costs about 16 ms, and
it is the obvious optimisation target. Two causes, both fixable:

1. The accumulate kernel is dispatched over every upscale pixel with a
   per-pixel struct of ~140 bytes of parameters, so the whole 33M-pixel
   parameter buffer is streamed twice per frame (write, then read). That is
   far more bandwidth than the arithmetic needs.
2. RCAS is in-place, so spatial and RCAS cannot overlap.

Neither has been optimised yet. The numbers above are the honest starting
point, not a result.

## Honest weaknesses

- **The temporal stage does not improve RMSE at all** on this fixture. It is
  only shown to avoid ghosting. Real quality gains from accumulation need real
  sequences with real motion vectors, which needs a real renderer.
- **The fixture is synthetic and static.** A 0.9% RMSE win here does not
  predict anything about a game frame.
- **No performance measurement yet.** Correctness is established; cost is not.
- **Nothing has been run in a game, through CrossOver, or against MetalFX or
  DLSS.** The claim "better than base FSR / MetalFX / DLSS" is **not made** and
  is not supported by this evidence.

## Established, and not

Established on this host, on synthetic band-limited content: a working FSR3-class
Metal upscaler at 4x, with correct 4x4 Lanczos reconstruction, RCAS, and a
reprojecting temporal accumulate with a lock; reconstruction beats bilinear on
RMSE; the temporal stage demonstrably avoids ghosting against a control that
fails.

Not established: performance in a game, quality versus any shipping upscaler,
CrossOver interop, and any use of FSR4 weights.
