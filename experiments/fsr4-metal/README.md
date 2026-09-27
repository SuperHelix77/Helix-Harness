# Helix FSR-Metal — research bootstrap

**Goal:** a lower-cost FSR 4-family temporal super-resolution implementation on
Apple Silicon, reached through a verified CrossOver/Metal integration path.

**Current implementation: native INT8 convolution primitives and offline
contracts, not a complete FSR port or a game mod.** No model weights are loaded.
No game, CrossOver bottle, installed app or macOS kernel has been modified.

## Run

On the tested macOS 26.6.2 / M3 Max host with Swift 6.2 Command Line Tools:

```sh
python3 -m unittest discover -s tests -v
sh scripts/run_native.sh evidence/native-kernels-new-run.json
```

The shader is compiled with `MTLDevice.makeLibrary(source:options:)`; the
standalone `metal` executable is not required. GPU absence or failure is an
error, never a CPU fallback that gets reported as Metal performance.

`native/KernelLab.swift` compares two independently written Metal kernels with
an Int64 CPU oracle (range checked before Int32 output). Seven fixtures cover
signed extrema, padded channel tails, spatial borders, odd dimensions, 1x1 and
3x3 convolution, and 2x2 stride-two downsampling. It checks every output and an
out-of-bounds sentinel region. Bias, activation and requantization are not yet
implemented. `char4` is packed storage plus integer arithmetic, not a claim of
Apple hardware dot-product acceleration.

Reports include all GPU samples, median/p95, device, OS, source/binary hashes,
toolchain, warm-up count and explicit scope. Synthetic feature maps are not
FSR layer traces. Compilation, allocation, CPU packing and readback are outside
the GPU interval. That is useful for kernel work, not an end-to-end claim.

## Files

| Path | Purpose |
|---|---|
| `kernels/conv_i8.metal` | Scalar and packed-four signed INT8 convolution |
| `native/KernelLab.swift` | Real Metal correctness and timing runner |
| `lab_contract.py` | Offline temporal input/history lifecycle specification |
| `scripts/audit_sources.py` | Read-only hash and lexical inventory of separately obtained source |
| `tests/` | Portable metadata, failure, history and inventory tests |
| `docs/RESEARCH.md` | Source-grounded design, alternatives and acceptance gates |
| `evidence/` | Actual receipts; negative results are retained |

Source audit, for a separately obtained and reviewed source tree:

```sh
python3 scripts/audit_sources.py /path/to/source \
  --revision FULL_40_CHARACTER_COMMIT --output new-inventory.json
```

An inventory is **not** license clearance, an executable graph, or proof of an
upstream revision. No third-party source, binaries, models or Apple frameworks
are vendored. The source-fetch attempt in the bootstrap was blocked; only
public source-tree metadata was retrieved, not the shader bodies or weights.

## Integration boundary

The future pipeline is game temporal inputs -> upscaler hook -> verified GPU
resource/timeline bridge -> native Metal graph -> game output resource.
`lab_contract.py` currently accepts only synthetic `native-fixture` metadata.
It deliberately rejects CrossOver/D3DMetal/DXMT descriptors and rejects all
end-to-end performance promotion. It does not validate real handles and is not
wired into the convolution runner. This keeps a specification test from being
mistaken for a working native bridge.

This isolated lab is carried on a research branch of `SuperHelix77/Helix-Harness`
under `experiments/fsr4-metal/`; it is not part of the Harness production build.
The older game-runtime ledger was named in a prior handoff but was not found in
the inspected current `main`/`v3-release` snapshots. This lab does not pretend
to restore or overwrite that missing original.
