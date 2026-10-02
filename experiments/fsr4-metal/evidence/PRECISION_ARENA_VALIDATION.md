# FP16 vs INT8 precision arena - validation receipt, 2026-09-27

This closes the first sentence of the capsule's own "Next executable research
boundary": **run FP16 and INT8 native baselines on identical shapes.** The
question it asks is one the research plan had left open in two places:
*"Compare INT8 and FP16/matrix paths on this exact M3; never choose a format
merely because it has fewer bits"* and *"An M3 FP16 matrix path may beat a
scalar integer implementation; this must be measured."* Nothing in the capsule
measured FP16 before this run.

Host: Apple M3 Max, macOS 26.6.2 (25G83), Swift 6.2 Command Line Tools. GPU
executed on the real Metal device; the standalone `metal` executable is
unavailable on this host, so runtime MSL compilation is used, as elsewhere. No
CPU fallback was used. No model weights, no game, no picture-quality claim.

Portable validation: `python3 -m unittest discover -s tests -v` - **58 tests
passed** (45 pre-existing, 13 new), zero failures. Transcript:
`portable-tests-20260927.txt`.

## What was measured

Five operator shapes, each fed the **same underlying random values** quantised
into each format, so no path was handed an easier or harder problem:

| Variant | Arithmetic | Output |
|---|---|---|
| `int8_split` | existing packed-four INT8 conv + separate dyadic requantisation | int8 |
| `fp32` | float32 accumulate | float32 |
| `fp16_halfacc` | half accumulator; every product and partial sum rounded to half | half |
| `fp16_f32acc` | half storage, float32 accumulator | half |
| `fp16_simd8x8` | 1x1 only: 8x8 `simdgroup` matrix multiply over the flattened spatial axis | half |

INT8 is checked exactly against an independent Int64/Double dyadic oracle.
FP16 and the simdgroup path are checked against their own oracles and their
error is **reported**, never claimed to be zero. All output tail sentinels
pass on every path. Every variant is timed inside the same trial loop with the
order alternated per trial, so a clock or power shift cannot land inside one
variant only. Receipts record min/median/p95/max, not a bare median.

## Result: precision is not a single answer, and neither is shape

**Absolute milliseconds on this host are not reproducible and must not be
compared across sessions.** The same binary, run 10 times, lands in two
discrete speed states roughly 2-4x apart, and *every variant moves together
within a run*, which identifies the cause as a whole-machine GPU clock state
rather than anything about a kernel. The 3x3 and stride-two shapes were
almost perfectly repeatable (1.00-1.01x spread); the 1x1 shapes swung 2-4x.
This is the same instability the earlier `fusion` receipts documented, and it
is why the comparisons below are **ratios to INT8 on the same shape within the
same run**, not absolute milliseconds.

Median ratio to the INT8 path, over 10 runs of one binary (lower is faster):

| Shape | fp32 | fp16+f32acc | simdgroup 8x8 | ratio stability |
|---|---:|---:|---:|---|
| 7x9x5, 3x3 tail | 0.95 | 0.87 | n/a, not 1x1 | 1.4x |
| 128x72x32, 1x1 | 0.79 | 0.79 | **0.39** | 1.2-1.4x |
| 128x72x32, 3x3 | 1.78 | 1.90 | n/a, not 1x1 | 1.00x |
| 129x73x32, 2x2 s2 | 1.02 | 1.07 | n/a, not 1x1 | 1.06x |
| 64x64x64, 1x1 | 0.89 | 0.95 | **0.43** | 1.1x |

Three findings, on our own synthetic operators on this host:

1. **The simdgroup FP16 matrix path is the only clear, repeatable speed win, and
   only on 1x1:** about **2.3-2.6x faster than INT8** (ratio 0.39-0.43), and
   this ratio is the most stable result in the table. On 3x3 the same FP16
   units are **1.8-1.9x slower** than INT8 (ratio 1.78-1.90, the most
   reproducible numbers measured, 1.00x spread over 10 runs), because the 8x8
   tiling only amortises for the 1x1-as-GEMM shape while the scalar loops carry
   everything else. "Use FP16" is a per-shape decision, not a global one.
2. **Bit count does not predict this host's speed.** At the same 3x3 shape INT8
   (8 bits) beats both FP32 and FP16 by ~1.8-1.9x. At the same 1x1 shape FP16
   slightly beats INT8 without matrix units, and beats it decisively with them.
   The format has to be chosen per shape from measurement.
3. **Half accumulation is numerically dangerous; half *storage* with float32
   accumulation is not.** `fp16_halfacc` reached max relative error up to
   ~1434x on the 3x3 shape, while `fp16_f32acc` stayed within 0.016 absolute on
   every shape. A naive "FP16 GEMM" that accumulates in half is not a safe
   drop-in for a real network. This is a result about arithmetic order, not
   about storage width, and it is the single most important numerical finding
   here.

## What this does and does not establish

This is a **measurement on synthetic shapes**, not a model decision:

- It does **not** choose a precision for a student model. That needs operator
  sensitivity against real activations and image quality, still open.
- It does **not** reproduce AMD FSR operators, real weights, or picture quality.
- The simdgroup result applies to **1x1 only** in this lab. Extending it to 3x3
  via im2col/GEMM is plausible and untested here.
- The timing interval excludes compile, allocation, packing, CPU reference and
  readback. It is not a full game frame.
- Absolute milliseconds on this host are not stable enough to publish; only
  same-run ratios are.

The finding that most affects planning: **a per-shape precision/specialisation
table is the right target, and FP16 buys real speed only where a 1x1 GEMM shape
lets the matrix units engage.** INT8 remains the strong default for 3x3 and
strided layers on this host, and its exactness is still an advantage. Before
any of this informs a real model, the *real* layer shapes and activation
statistics must replace the synthetic ones.

## Two real bugs this work caught (retained as method notes)

**1. A fast but wrong matrix path.** The first two attempts to run the
simdgroup path produced results that were quick but **incorrect** (max absolute
error ~11-27, many outputs zero) - specifically, a column shift, because
`simdgroup_matrix` `a * b` computes an ordinary row-major `C = A*B` (verified
on-device with an identity probe) and the conv weight tile `W[o][k]` was being
passed with a transposed load. The fix is to pre-transpose on the host to
`B[k][o] = W[o][k]` and load plainly. The failure signature - a
plausible-looking shift, not a crash - is why the lab now treats the
simdgroup path as a hard failure unless it is within a half-ULP budget of the
exact half product-sum.

**2. Block-timed variants were contaminated by machine state.** The first
implementation of the harness timed each variant in its own block of trials.
That let a GPU clock shift land entirely inside one variant, producing
internally impossible numbers (an FP16 kernel appearing 4x faster than FP32 on
the same work). The harness now interleaves every variant inside each trial
with alternating order, and records min/p50/p95/max. The lesson generalises: on
a machine with variable GPU clocks, only same-run ratios are trustworthy.
