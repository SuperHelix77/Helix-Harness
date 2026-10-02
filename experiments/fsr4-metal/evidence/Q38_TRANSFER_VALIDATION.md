# Q38-transfer implementation receipt — 2026-09-27

This capsule adds independent research code. It is not a full FSR implementation,
a DLSS port, a trained student, or a working CrossOver bridge.

Portable validation: `python3 -m unittest discover -s tests -v` — **45 tests
passed**, zero failures. Full transcript: `q38-transfer-tests.txt`.

## Lossless representation

`lossless-codec-20260927-a01.json`: nine exact round trips, three distributions
and three seeds. Each input has 32,768 signed INT8 weight bytes. The identical
raw-container header is 52 bytes, so the raw comparison container is 32,820 bytes.

| Synthetic distribution | Archived bytes including header | Expanded INT8 bytes |
|---|---:|---:|
| Values entirely in signed nibble range | 20,532 | 32,768 |
| Same small values with approximately 1% exact extreme outliers | 20,860 | 32,768 |
| Uniform INT8 | 32,820 (raw fallback) | 32,768 |

The first two synthetic distributions reduce the equivalent archive by about
37.4% and 36.4%. This is NOT an FSR compression ratio. All decoded bytes are
identical; inference weights remain fully expanded. No inference speedup or
runtime-memory reduction follows from this codec result.

## Real Metal execution of synthetic operators

Three receipts: `fusion-20260927-a01.json`, `fusion-20260927-a02.json`,
`fusion-20260927-a03.json`. Each records eight exact CPU/GPU output comparisons
(four shapes x split/fused) and 2,063 exact epilogue cases with independent CPU
rounding logic. Output and scratch tail sentinels pass. Numerical ties,
saturation, bias, per-channel dyadic scales/zero points and quantized ReLU are
covered. All executions used the Apple M3 Max, macOS26.6.2 Metal GPU.

The implementation fuses convolution and epilogue into one dispatch instead of
two. For a 128x72 feature map with 32 output channels, the logical full-tensor
INT32 intermediate falls from 1,179,648 bytes to zero. Registers and scratch
inside a GPU kernel still exist. Both variants' allocations coexist in the
benchmark, so this is NOT a measured reduction in process RSS or total game VRAM.

### Retained timing evidence

| Workload | Run 2 split / fused median (ms) | Run 3 split / fused median (ms) |
|---|---:|---:|
| Small padded channel-tail 3x3 | 0.027500 / 0.021542 | 0.027208 / 0.021542 |
| 128x72 C32->32, 1x1 | 0.230875 / 0.184000 | 0.231000 / 0.184125 |
| 128x72 C32->32, 3x3 | 0.466375 / 0.442167 | 0.419542 / 0.396000 |
| 129x73 C32->32, stride-two 2x2 | 0.051875 / 0.045750 | 0.052042 / 0.045708 |

The repeat medians suggest approximately 20% lower GPU time for the tested 1x1
pipeline, 5–6% for the 3x3 pipeline and 12% for stride-two. Those are comparisons
against OUR split synthetic operator pipeline, not AMD FSR or a game. The raw
receipts record GPU and wall-clock samples, median/p95, source/binary hashes,
10 warmups, 31 measured samples and alternating variant order.

Run 1 is retained as unstable evidence, not quietly discarded: its 1x1 split/
fused medians were 2.073250/0.745750 ms and split p95 was 6.754292 ms. The cause
of the variance was not established. Both later runs are much less variable in
that workload, but not a controlled idle-host or sustained-thermal experiment;
run 2's tiny fused tail case still has a large p95 outlier. No stable global
speed multiplier is promoted. Other processes were not stopped.

The GPU interval excludes compilation, allocation, CPU packing/reference and
readback. Wall time covers command construction through GPU completion, not
the complete game frame. This dyadic quantization contract has not been checked
against AMD's real weights, scales, rounding or activations. No full model or
image-quality result is implied by exact synthetic operator outputs.

## Reference sequence evaluation

New tests distinguish a rare large pixel defect from a better average MSE and
detect flicker when per-frame spatial error is unchanged. They check reference
integer motion maps, reset boundaries, HUD/disocclusion masks, declared capture
and scale identity, frame populations, nonfinite data and scene-split overlap.
An all-reset sequence is rejected as lacking temporal coverage.

These are normalized-SDR metric smoke tests. They do not prove subpixel motion,
HDR, semantic/perceptual quality, live capture authenticity or visual acceptance.
The report always leaves production-quality admission false. Real held-out
game sequences and actual FSR reference outputs are still required.

## Preservation

Q38 files, existing games, bottles, installed applications and the macOS kernel
were not changed. No vendor weights, DLLs or third-party executable releases
were downloaded. Subagent coordination failed twice with lost conversation
identity; no subagents ran. Research code stays on the existing draft branch.
