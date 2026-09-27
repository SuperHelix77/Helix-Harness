# Quality-first FSR compression and optimization on M3

Activation: 2026-09-27. Scope: the user's request to transfer Q38's methods to
FSR, make the Mac path faster, and prioritize picture quality even in Performance
and Ultra Performance. This extends the bootstrap; it does not claim a full port.

## Decision

Use Q38's representation/execution co-design, exact accounting, actual-artifact
round trips, layer-specific search and held-out quality discipline. Do not adopt
its extreme byte target as an assumed quality-preserving compression ratio.

The inspected Q38 continuity snapshot is
`SuperHelix77/q38@c8a72f95122050624517be6ca7f2b5e0ef77706f`, branch
`astra/q38-continuity-20260926`. Its C128 fresh-evaluation report records the
93,950-byte artifact at 26.48% versus a 26.17% trivial length control on 2,919
fresh cases (p=0.8125). The artifact exists; equivalent retained capability is
not established. That is a result about that candidate, not an impossibility
claim about small models or a rejection of Q38's exact runtime work.

Q38's `AGENTS.md` sections 10–12 explicitly make quality a hard admissibility
constraint and distinguish archive/resident/repack/scratch bytes. Those are the
right rules for this project. The local attention-successor snapshot also uses
one stored block four times: sharing weights reduces storage, not automatically
the four evaluations. No Q38 code or private model/data artifacts are copied here.

## What transfers, and what changes

| Q38 mechanism | FSR adaptation | Necessary qualification |
|---|---|---|
| Compact core plus counted residuals | Lossless weight representation, then learned low-rank/core corrections if justified | Count exceptions, scales, indexes, expanded weights and decode work |
| Serialize/reload the real candidate | Run the deployed quantized graph, not just a float training model | Layer dumps and full sequence agreement |
| Per-shape kernel selection | Specialized Metal conv/matrix paths, fused epilogues, bounded scratch | Actual M3 measurements, not FLOP counts |
| Teacher supervision/QAT | Recurrent SR student, mixed precision, temporal and disocclusion losses | Independent held-out game/scene sequences, all seeds |
| Canonical state and reuse | Validated history, motion/depth reprojection and reset on discontinuities | Image reuse is approximate; unlike exact token-prefix identity |
| Adversarial controls | FSR/MetalFX/native references plus blur, sharpen and history-only controls | Averages cannot conceal worse wires, UI, motion or rare artifacts |

### Three budgets, not one compression ratio

1. **Stored weights:** lossless coding can improve package size and loading. If
   weights are expanded once at load, inference working-set size is unchanged.
2. **GPU execution and activations:** fusion, layout specialization, fewer global
   intermediates, lifetime reuse and suitable matrix arithmetic can reduce work
   without changing the model. AMD describes FSR4 runtime as largely dependent
   on output resolution; simply making the input smaller is not the whole win.
3. **Learned computation:** narrower channels, low-rank factorization, sharing,
   conditional residual refinement and distillation can reduce the network's
   work, but must earn admission with image/temporal quality. Each introduces a
   new candidate model, not a claim of unchanged FSR arithmetic.

The lossless codec added here attacks budget 1. The fused Metal operator attacks
budget 2. Budget 3 remains a proposed training campaign, not an implemented win.

## Picture quality at aggressive scaling

Freeze actual render and output sizes. Performance is 2x per dimension (one
quarter as many input pixels); Ultra Performance is 3x (one ninth). At 4K output
these are 1920x1080 and 1280x720. At 1080p, Ultra Performance is only 640x360.
These are sampling counts, not the fraction of recoverable temporal information.

The target is faithful detail, stable motion, preserved text/UI and correct
lighting at those SAME inputs, not a hidden resolution increase, invented detail
or a frame-generation multiplier. A universally native-identical result from
missing current/history evidence is not established. Strong improvement over
the stock low-input mode remains an explicit research objective.

Proposed trainable path: low-resolution feature extraction -> depth/motion-aware
history aggregation -> compact recurrent reconstruction -> limited expensive
detail correction for difficult regions. Preserve sensitive output, confidence,
reprojection and color operations at higher precision when lower precision hurts.
Do not apply INT4/INT8 uniformly because the files are smaller. An M3 FP16 matrix
path may beat a scalar integer implementation; this must be measured.

Training should use high-resolution rendered targets and the exact low-resolution
input contract. Compare reference-only supervision with permitted teacher output
and feature supervision. Use reconstruction/edge losses, temporal residual error,
disocclusion-weighted loss and a restrained perceptual loss. Teacher agreement
alone cannot certify improvement over the teacher. Do not use adversarial texture
fabrication as a shortcut to claiming faithful scene recovery.

Game-generated motion, exposure, jitter, depth, reactive masks and material/LOD
choices matter. A mod cannot assume engine-only high-resolution G-buffers are
available. Tile/feature reuse needs confidence, receptive-field halos, and
invalidation for camera cuts, disocclusions, reflections, animated materials and
exposure changes. The validator must measure rejected/recomputed work and the
cost of selection, not just the easy reused tiles.

The frozen quality set must cover Performance and Ultra Performance at 1080p,
1440p and 4K, with wires, hair, foliage, particles, transparency, specular motion,
HUD/text, scene cuts and HDR. Split by game/scene/camera trajectory, not adjacent
frames. Retain every candidate/seed and keep tuning and final evaluation separate.
Assess per-sequence spatial, worst-pixel, edge, perceptual and reference-warped
temporal errors, plus visual review. A better average PSNR cannot hide rare large
errors or increased ghosting. Overall frame time, p95 latency and input latency
must include renderer, bridge, synchronization and SR on the shared GPU.

## What the DLSS ports actually contribute

**DLSS 5 Neural Rendering is not simply FSR-like super-resolution.** NVIDIA's
research describes a generative appearance stage; it can change lighting and
materials. Keep this distinct from faithful reconstruction. No NR restyling is
enabled in our proposed default path.

**MLX-DLSS (`iamwavecut/MLX-DLSS`) is the closest code reference.** Its native
experimental SR path reconstructs preset K / SDK310.7.0 / LDR / 2x with eleven
transformer blocks, preprocessing and recurrent history. Its authors report
sequence agreement against the Linux implementation, but rare channel errors
reach 0.19; no universal bitwise or Windows/game parity is claimed. Other scales,
HDR and game integration are explicitly unsupported. Do not call it a ready
Ultra Performance game plugin.

Its embedding layer separates host tensors from a GPU-resident video path,
supports IOSurface-backed buffers, explicit sequence ownership and resets, and
uses fixed-shape/fused graph alternatives. These are useful engineering patterns.
IOSurface video import is not proof that CrossOver exposes an importable game
texture. No third-party model weights have been copied or executed here.

**DLSSMac (`Mappsnet7/DLSSMac`) informs bridge architecture, not M3 readiness.**
Its documented shared-resource route is GameHub Wine Proton11 + GPTK4beta2;
CrossOver DX12 uses TCP. Its shipping FP8 path requires macOS27 and uses M5 lane
assumptions not validated on M1–M4. Reuse concepts such as a separate x86_64 Wine
helper, ARM64 service, explicit resource ownership and bounded queues only after
code/license review and M3 qualification. More buffering adds latency; it does
not make the network faster. Do not install its M5 release on this Mac as a test.

**DLSS-NR-on-AMD (`danielblnc/DLSS-NR-on-AMD`) is evidence of a different-runtime
approach.** The author describes a HIP reimplementation for RDNA4, with RDNA3
expected; this is not a portable Apple backend. The inspected repository lists
documentation/releases rather than a reusable HIP source tree, so do not assume
its runtime can be copied under an open-source grant. Reported performance is
the author's evidence, not ours. Its September12 release also moves NR before
upscaling and delegates temporal accumulation to FSR: useful evidence that
pipeline placement matters, but neither FSR compression nor an equivalent
quality result on M3. No binaries or vendor DLLs were downloaded.

Older-NVIDIA unlockers can replace hardware-specific instructions or enable
features while retaining NVIDIA runtime dependencies. Their existence does not
give Metal the same instructions. The transferable lesson is to replace the
hardware-specific execution path, not to spoof a device and expect parity.

## New implementation in this capsule

`weight_codec.py`: deterministic lossless nibble core + exact INT8 exceptions,
raw fallback, integrity checks, complete storage ledger. Synthetic fixtures only;
fully expanded INT8 inference memory is unchanged.

`kernels/epilogue.metal` / `native/FusionLab.swift`: bias, per-channel dyadic
scaling, round-to-nearest-even, zero point, saturation and optional quantized
ReLU, comparing split conv+epilogue to one fused dispatch. Exact CPU/GPU checks
cover rounding and spatial/channel tails. This numerical contract is independent;
it has NOT been checked against AMD's model operators or real weights.

`sequence_quality.py`: finite normalized-SDR reference metrics and per-frame
non-inferiority tests with reference-side integer correspondences and HUD/
disocclusion masks. Rejects changed declared capture/scale/frame populations,
empty temporal coverage, train/dev/holdout scene overlap, and hidden worst-pixel
or flicker regressions. It is a smoke test, not complete perceptual/HDR or live
capture attestation; production-quality approval remains false.

## Next executable research boundary

Obtain a provenance-approved model/reference package and real layer/sequence
fixtures. Run FP16 and INT8 native baselines on identical shapes; replace our
synthetic dyadic epilogue with the actual documented model contract. Perform
operator sensitivity and activation-lifetime profiling before choosing a
student's width/precision. Build the smallest real resource/fence bridge proof
independently. Never postpone all numerical work behind the bridge, or label
native numerical success as proof that the bridge works.

## Primary sources

- Q38 [quality/accounting contract](https://github.com/SuperHelix77/q38/blob/c8a72f95122050624517be6ca7f2b5e0ef77706f/AGENTS.md) and [fresh-evaluation C128](https://github.com/SuperHelix77/q38/blob/c8a72f95122050624517be6ca7f2b5e0ef77706f/docs/q38-continuity-20260926/C128_FRESH_EVALUATION_RESULT.md).
- AMD [FSR4.1.1 integration, scaling and performance](https://gpuopen.com/manuals/fsr_sdk/techniques/super-resolution-ml/).
- NVIDIA [DLSS5 generative-rendering research](https://research.nvidia.com/labs/adlr/DLSS5/).
- MLX-DLSS [SR contract](https://github.com/iamwavecut/MLX-DLSS/blob/main/docs/super-resolution.md), inspected blob `023fe0768d6c4515c33d780aa5134b5e347e3500`; [embedding](https://github.com/iamwavecut/MLX-DLSS/blob/main/docs/embedding.md), blob `4b6b2f38955b506137ecae90d704f22ef894e009`.
- DLSSMac [architecture](https://github.com/Mappsnet7/DLSSMac/blob/main/docs/ARCHITECTURE.md), inspected blob `8ba727e820113e496312f91ab6ddb0860accb2d6`.
- DLSS-NR-on-AMD [author's scope](https://github.com/danielblnc/DLSS-NR-on-AMD) and [release notes](https://github.com/danielblnc/DLSS-NR-on-AMD/releases).

Public upstream claims are cited as upstream claims, not independent replication.
