# FSR 4 -> Metal -> CrossOver research decision

Date: 2026-09-27. Status: active, primitive-only bootstrap.

## 1. What is established, and what is not

AMD's current technique documentation identifies FSR Upscaling **4.1.1**,
HLSL CS_6_6, and integration through the FSR API and signed binary distribution
[1]. Therefore the shipping Windows DLL is not a portable Metal source package.
Do not treat an older 4.0.2 archive as the current 4.1.1 implementation.

An archival repository advertises older `fsr4_model_v07_i8_*` and FP8 variants
[2]. It is a candidate source reference, not provenance approval. Before using
any code or weights, record an exact commit, original upstream lineage,
per-file notices and initializer hashes. A mirror README or MIT-licensed sample
does not establish rights for every binary/model in a different release.
This bootstrap includes only independent code and public file-list metadata.

Apple documents Metal 4 tensor resources, ML command encoders and in-shader
matrix/convolution facilities [3]. These are candidates for native inference
and fusion on a GPU timeline. They do not guarantee that an AMD FP8 intrinsic
has a one-to-one accelerated equivalent on M3, nor that INT8 beats FP16.
The bootstrap uses ordinary Metal compute, not the Metal 4 ML encoder yet.

CodeWeavers lists CrossOver 26.3.0 and, in its 26.0 notes, D3DMetal 3.0 and
DXMT 0.72 [4]. This says nothing about a verified FSR4/native-Metal bridge in
our installed games. DXMT's upstream scope is D3D10/D3D11 [5]; do not assume
the previously proposed DXMT12 path is an available DX12 solution.

OptiScaler is a candidate API-hook/reference layer, not the native engine.
Its current release notes discuss FSR4.1.1 / SDK2.3 and warn about unsupported
GPU paths and fallback [6]. Its GPL-3.0 obligations must be considered if code
is integrated. Merely loading a DLL does not prove which upscaler executed.

## 2. Integration architecture and hard unknown

```text
Game DX12 frame: color + depth + motion + jitter/exposure/reset
                   |
           FSR/DLSS/XeSS hook (one selected API first)
                   |
          Wine/D3DMetal-compatible native boundary
                   |
        PROVEN resource import + GPU synchronization
                   |
       Metal prepass -> neural graph -> temporal postpass
                   |
            PROVEN output resource and completion
                   |
                  game
```

The central integration gate is resource identity, ownership, lifetime and
GPU synchronization, not calling a native function from Windows code. Never
reinterpret an `ID3D12Resource*` as an `MTLTexture*`. Never silently fall back to
screen capture and call that a temporal FSR implementation. An extra GPU copy
may be acceptable if measured, but CPU readback/upload is not the default plan.

`utmapp/d3dmetal-native` is an especially relevant investigation target: its
native GFXT host implements D3D texture/buffer/fence sharing, and its README
says the corresponding D3DMetal facilities are otherwise stubbed [7]. It is a
separate non-Wine host, not demonstrated stock CrossOver interoperability. It
also specifies an x86_64 process for its D3DMetal framework. Investigate an
architecture-matched native shim versus an ARM64 helper with a measured
cross-process sharing path; do not attempt to load an ARM64-only library into
an x86_64 process. Apple frameworks are not redistributed in this repository.

Minimum bridge proof: tagged color, depth and motion resources; matching device;
producer GPU fence -> consumer dispatch -> completion -> game resource; zero
CPU frame readbacks; clear ownership on resize/destroy; no stale frame after
camera cut; deterministic failure instead of an unreported fallback. Run this
in a disposable test harness before a test bottle, and preserve the user's
existing bottles and anti-cheat/protected games.

## 3. How to make it less demanding

**First: preserve the reference model, reduce implementation overhead.**
Benchmark each real layer's shape and layout. Specialize small kernels, pack
weights once, keep intermediate features on GPU, reuse heap memory, fuse
quantization/activation/skip operations only after equivalence tests, and
reduce dispatches. Compare INT8 and FP16/matrix paths on this exact M3; never
choose a format merely because it has fewer bits. Two bootstrap kernels both
use INT8, so their comparison is a packing/layout experiment, not INT8 vs FP16.

The INT8 vs FP16/matrix comparison has since been run on this exact M3
(`evidence/PRECISION_ARENA_VALIDATION.md`): on identical shapes and values the
8x8 `simdgroup` FP16 matrix path wins on 1x1 (~2.3-2.6x) and loses on 3x3
(~1.8-1.9x), so specialization is by shape, and half accumulation is
numerically unsafe while half storage with float32 accumulation is acceptable.
Absolute milliseconds on this host are not stable run to run (two GPU clock
states 2-4x apart), so only same-run ratios are quoted. That is a
synthetic-operator result; real layer shapes and activations are still needed
before any of it informs a model.

**Second: reduce the model itself, only if the faithful implementation misses
the frame budget.** Train/calibrate a narrower or pruned student against a
proven teacher and temporal sequences. Preserve disocclusion handling and
history resets. Width reduction may reduce internal convolution work, but
input/output layers, memory traffic and dispatch costs do not scale equally.
Call the result a distinct Helix lightweight derivative, not an unchanged
AMD FSR4 implementation. No student, training set or quality result exists yet.

**Do not re-open frame generation, FEX or neural texture compression in this
first lane.** Super-resolution must first be correct and net beneficial.

## 4. Measurement and acceptance gates

| Gate | Required proof | Bootstrap state |
|---|---|---|
| G0 provenance | exact permitted source + model lineage and hashes | open |
| G1 arithmetic | signed INT8, padding, tails, stride and exact accumulation | primitive fixtures pass |
| G2 model operators | bias, scales/zero points, rounding/clamping, activations, reshape/skip parity | open; our own dyadic contract passes, AMD's is unverified |
| G3 native graph | complete pre/model/post path vs approved reference sequences | open |
| G4 bridge | genuine game resources + GPU fence/lifetime proof | open |
| G5 quality/speed | held-out sequence quality and total frame-time improvement | open |

Provisional engineering targets, **not observed results**: begin with a 1080p
output workload; aim for native SR p95 <= 2 ms and bridge overhead p95 <= 0.5 ms.
Revise these only with documented evidence. Report total rendered-frame time,
1% lows, GPU contention, allocations/working set and sustained thermals. A
kernel-only speedup or generated-frame multiplier cannot satisfy G5.

Quality comparisons should include approved FSR reference, FSR3, MetalFX where
accessible, and native/TAA. Use held-out motion sequences, not only screenshots:
thin geometry, particles, disocclusion, camera cuts, foliage, UI and HDR exposure
changes. Pair spatial metrics with temporal flicker/ghosting measurements and
visual review. Keep tuning sequences separate from acceptance sequences.

`lab_contract.py` specifies color/depth/motion layout, jitter and exposure,
convention changes, duplicate/stale frames, and success-only history commit.
It is a reference metadata state machine, not real native handle validation.

## 5. Immediate next implementation capsule

Recover the original ledger if it exists on an uninspected branch or checkout.
Resolve G0 for the selected INT8 reference. Build a complete operator manifest
and per-stage reference dumps. Add bias and exact requantization with extreme
rounding/saturation tests, followed by a real recorded layer fixture. In
parallel, once worker coordination is available, investigate G4 in a separate
host-only adapter worktree. Do not let either lane claim the other's gate.

## Sources consulted

1. AMD, [FSR Upscaling 4.1.1 technique and integration documentation](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/docs/techniques/super-resolution-ml.md).
2. [FSR 4.0.2 source archive description](https://github.com/Rolaand-Jayz/FSR-4.0.2-reference). Third-party archive; provenance/licensing still to be verified.
3. Apple, [Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/).
4. CodeWeavers, [CrossOver change log](https://www.codeweavers.com/crossover/changelog).
5. DXMT upstream, [project scope](https://github.com/3Shain/dxmt).
6. OptiScaler upstream, [releases](https://github.com/optiscaler/OptiScaler/releases) and [repository](https://github.com/OptiScaler/OptiScaler).
7. UTM, [d3dmetal-native architecture, sharing, requirements and licensing](https://github.com/utmapp/d3dmetal-native).

Sources describe upstream capabilities, not tests performed by this project.
