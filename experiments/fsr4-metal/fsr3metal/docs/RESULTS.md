# FSR3-class Metal upscaler - first measured results, 2026-09-27

Host: Apple M3 Max, macOS 26.6.2, Swift 6.2. Real Metal GPU, runtime MSL
compilation. No CPU fallback. Results reproducible to 5 decimal places across
repeated runs.

## What is here

`kernels/fsr3_spatial.metal` and `kernels/fsr3_temporal.metal` implement the
FSR3 reconstruction and accumulation mechanisms in Metal, ported from the
MIT-licensed AMD reference. `native/UpscaleLab.swift` measures them against
controls. See `PROVENANCE.md` for scope and licensing.

Fixture: a static synthetic frame containing a 2px checker (aliasing), a hard
diagonal edge (ghosting), a low-contrast gradient (banding) and a bright dot
(temporal popping). The high-resolution source is known by construction, so
RMSE against it is a true reconstruction error rather than a proxy.

## Reconstruction, 4x (320x180 -> 1280x720)

| Variant | RMSE | max error |
|---|---:|---:|
| bilinear (host reference) | 0.03926 | 0.55544 |
| Lanczos (Metal) | **0.03885** | 0.80648 |
| Lanczos + RCAS | **0.03881** | 0.80989 |
| + temporal lock | 0.03881 | 0.80989 |

Lanczos beats bilinear on RMSE by about 1%. That is a real but modest win, and
it is only a win because the kernel samples the full **4x4** Lanczos footprint.

## The bug that made it worse, and what it taught

The first working version sampled only a **2x2** neighbourhood while applying
Lanczos weights. That produced RMSE 0.04327 - *worse than bilinear*. The cause
is structural, not numerical: Lanczos2 is a 4-tap-per-axis kernel, so using it
over 2 taps keeps the negative side lobes while dropping the taps they act on,
which rings at every edge. Fixing the footprint to a true 4x4 turned a loss into
a win. Recorded because the failure mode is easy to reintroduce and looks like
"the filter is wrong" rather than "the footprint is wrong".

## Shimmer

Shimmer is measured as mean absolute frame-to-frame change of the upscaled
output on a **static** input. Measured value: **0.0** across 16 frames.

This is the correct result, but it is a weak test on its own: with
current == history, any correct pipeline is a no-op, so the number cannot
distinguish a working lock from a broken one. It is reported because it is a
necessary condition, not because it is sufficient.

## Ghosting, with a control that can fail

A static-input shimmer number cannot tell a working temporal filter from a
broken one, so the test that matters uses **moving** content: a bright dot
translating 2 upscale px/frame over a static background.

The metric is trailing energy in the strip between the previous frame's
trailing edge and the current one - exactly where a pipeline that fails to
reproject its history would smear the old dot.

| | trailing energy |
|---|---:|
| this pipeline (reprojects history) | **0.09000** |
| naive blend, no reprojection (control) | 0.17667 |
| clean background | 0.09000 |

The pipeline reads exactly the background value, and the control reads 2x
that. The metric therefore has demonstrated discriminating power, so the clean
result is a pass rather than a vacuous one.

Getting here took three wrong versions of the test, each of which would have
"passed" a broken pipeline:

1. measuring a wake band 12px behind the dot, which is outside the range any
   smear can reach;
2. measuring on the last frame, by which time the old dot has already left;
3. building the control from a lerp of background with background, which
   mathematically cannot exceed background.

Only the third version can fail.

## Honest weaknesses

- **Max error is worse than bilinear** (0.81 vs 0.56). Lanczos ringing at the
  hard diagonal edge is a real cost and is not yet mitigated. In a real
  pipeline FSR handles this with a reactive/shading-change signal that widens
  the clip box at exactly those pixels; this lab does not drive that signal
  from a real frame, so the ringing is still visible in the metric.
- **RMSE is dominated by the 4x checker**, which is an adversarial aliasing
  case, not representative content. A 1% RMSE win here does not predict a 1%
  win on a game frame.
- **The temporal stage does not improve RMSE at all** on this fixture. It is
  only shown to avoid ghosting. Real quality gains from accumulation need
  real sequences with real motion vectors.
- Nothing here has been run in a game, through CrossOver, or against MetalFX or
  DLSS. The claim "better than base FSR / MetalFX / DLSS" is **not made and is
  not supported by this evidence**.

## What is established and what is not

Established, on this host, on synthetic content:

- a working FSR3-class Metal upscaler at 4x, with correct Lanczos
  reconstruction, RCAS, and a reprojecting temporal accumulate with a lock;
- reconstruction beats bilinear on RMSE;
- the temporal stage demonstrably avoids ghosting, against a control that
  fails.

Not established: performance in a game, quality versus any shipping upscaler,
CrossOver interop, and any use of FSR4 weights.
