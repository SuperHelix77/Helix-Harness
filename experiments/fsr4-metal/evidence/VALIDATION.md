# Bootstrap validation receipt

Host observed: Apple M3 Max, unified memory, macOS 26.6.2 (25G83), Swift 6.2
Command Line Tools. The offline `metal` executable was unavailable; runtime
MSL compilation succeeded. No CPU fallback was used.

## Portable checks

`python3 -m unittest discover -s tests -v`: **20 passed**, zero failures.
These test the offline contract/state machine and source inventory utility;
they do not validate CrossOver textures or a complete FSR graph.

## Native GPU checks

Two complete runs, each **14 exact GPU/CPU comparisons** (7 fixtures x 2 kernels),
zero maximum absolute error and all output tail guards intact. Each variant
has 3 warm-up dispatches and 21 measured dispatches per fixture, alternated in
order. The second run additionally checks the CPU oracle's signed-extrema
result against the independently calculated value 32514, and records Swift,
MSL and executable SHA-256 digests.

| Synthetic feature workload | Scalar median, run 1 / run 2 (ms) | Packed-four median, run 1 / run 2 (ms) |
|---|---:|---:|
| 128x72, 32->32, 1x1 | 0.170375 / 0.170417 | 0.181542 / 0.181458 |
| 128x72, 32->32, 3x3 padded | 0.647625 / 0.722625 | 0.395917 / 0.442167 |
| 128x72, 32->32, 2x2 stride 2 | 0.058625 / 0.061833 | 0.045625 / 0.048292 |

Interpretation: packing regresses 1x1 by about 6.5%, improves the tested 3x3
primitive by about 1.63x, and improves tested downsampling by about 1.28x.
Absolute timings vary with host/GPU state. This is evidence to specialize by
layer shape, not evidence of an FSR or game speedup. Neither kernel uses an
AMD intrinsic or a Metal matrix acceleration API.

The raw receipts retain every sample, including outliers. These brief runs do
not establish sustained thermals, power efficiency or idle-host reproducibility.
The interval includes a single GPU command buffer/dispatch, and excludes
compilation, allocation, CPU packing and readback. Synthetic dimensions are
not asserted to match an FSR layer. No latency or throughput figure here is a
complete 1080p upscaler time.

## Source and integration status

A public reference repository tree listing was retrieved (183 FSR4-related
files; tree SHA `c5e34b4b7128ffeb152f80cbdb6d715d577ec289`). This is a **tree SHA,
not a commit SHA**. Shader/body and license retrieval was subsequently blocked;
no third-party shader bodies or weights were downloaded into this lab. The
original source lineage and permission for any future code/weight reuse remain
unresolved. The inventory utility has been tested on fixtures, not that full
upstream source tree.

Full FSR graph: not implemented. Native Metal 4 ML graph: not implemented.
CrossOver resource bridge: not implemented or tested. Game quality/performance:
not tested. Reduced/distilled model: not trained. Existing games/bottles:
unchanged. Subagents: unavailable due conversation identity failure.

Files: `native-kernels.json` is the initial run, before the extra CPU sanity
assertion/provenance fields; `native-kernels-repeat.json` is the final native
runner's repeat run. The Metal kernel is identical between those runs.
