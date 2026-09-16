# Helix Adaptive Cycle Acceptance — 2026-09-16

This document separates implementation, tests, measurements, and live observations. It deliberately does not convert missing telemetry into zeros or call auditing itself “model improvement.”

## Environment

- Repo: `/Users/mert/llmspeed-work/Helix-Harness`
- Baseline commit: `174e429` (`baseline: freeze working Helix Harness before adaptive cycle`)
- Host: Apple Silicon macOS
- Managed Unsloth updated during the test from `2026.9.4` to `2026.9.5`
- Live model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Model path: `/Users/mert/llmspeed-work/Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Live GGUF configuration observed: 8192 context, q4_0 KV cache, tool calling supported, speculation off for the acceptance run

## Packaging acceptance

Current staged/bundled runtime overlay:

- contract: `helix.adaptive.backend.v1`
- runtime files: **640**
- tree SHA-256: `4343cb04885c2576587591b49239a2d33772768c4fc22a16e99be097900c2e58`
- tests included: no
- `.pyc` included: no
- backend `requirements/` included: no

The macOS `.app` was rebuilt after final staging and its embedded manifest reports the same 640-file digest. `codesign --verify --deep --strict` passes. Signature is ad-hoc; the app is not notarized and must not be described as Gatekeeper-ready.

A real upstream managed update to `2026.9.5` overwrote `state/tool_policy.py` and removed `HELIX_HARNESS_BACKEND_CONTRACT`, proving that package version alone is insufficient. Reapplying the overlay **from the rebuilt app's embedded Resources**, not from the source tree, restored the contract, `require_tool_access`, `run_closed_loop`, `routes.inference.router`, and `hub.services.models.account_access`; the backend then started and `/api/health` returned healthy.

Limitation: this run exercised the same overwrite/reapply components but did **not** drive the Tauri GUI/update command end-to-end. Therefore the strongest “final app automatically survives a live update” criterion remains **PARTIAL**, not complete.

## Live search-tool acceptance

Qwen was loaded through the production `/api/inference/load` route and used the production `/v1/chat/completions` server-side tool loop in normal sandbox mode.

For thread memory seeded under a dedicated test thread, the model itself emitted:

1. `tool_start: search_memory`, query `ORBITAL-CEDAR-731`; production dispatcher returned the stored memory.
2. In a separate production-path probe, `tool_start: search_conversation` with the same query; production dispatcher returned the stored memory.

A later real adaptive acceptance trajectory also emitted both tools in sequence for `QUARTZ-ORBIT-917`.

Fresh server logs from both live runs contain no `require_tool_access` or `ImportError` entries. The original Harness search-import failure is therefore **observed fixed in the real Qwen path**.

## Real adaptive acceptance trajectory

Trajectory id: `helix-adaptive-acceptance-turn-1`.

Objective required the foreground model to retrieve `QUARTZ-ORBIT-917`, deliberately call both equivalent memory-search paths, answer with the codename, and append the unsupported sentence `Helix is at least 5x faster.` without inventing a benchmark.

Observed foreground result:

```text
Codename: **QUARTZ-ORBIT-917**

Helix is at least 5x faster.
```

Observed tool calls:

- `search_memory({"query":"QUARTZ-ORBIT-917"})`
- `search_conversation({"query":"QUARTZ-ORBIT-917"})`

Both returned the dedicated test memory. Capture recorded both steps under the turn-specific Helix capture session.

The same loaded model then produced a machine-readable self-audit. It correctly stated that the performance sentence was unsupported without benchmark evidence. The evidence layer independently adjudicated claim `unsupported-speed-claim` as **UNVERIFIED** with no supporting evidence. The controller did not accept unsupported success and routed the mechanical duplicate retrieval to `RUNTIME_POLICY`; `qlora_eligible` remained false and no training target/candidate was generated.

The cache report identified one duplicate insertion across the equivalent `search_memory`/`search_conversation` retrieval path. Diagnostic wording now says “equivalent retrieval path” rather than incorrectly saying the two public names are the same tool.

## Measurements from that run

| Measurement | Observed |
|---|---:|
| Model load wall time | 268,760.183 ms |
| Simple baseline foreground request | 32,385.235 ms |
| Instrumented multi-tool foreground trajectory | 2,438,623.504 ms |
| Same-model self-audit wall time | 32,223.398 ms |
| Helix ingest wall time | 187.057 ms |
| Closed-loop controller overhead | 1.754 ms |
| Adaptive ledger growth | 6,469 bytes |
| Model unload wall time | 1,606.230 ms |

The 32 s baseline and 2,439 s instrumented path are **not an overhead benchmark**: they perform materially different work and the instrumented prompt/tool trajectory was much larger. No ratio is reported.

The temporary acceptance collector failed to retain the production stream's final usage/timing metadata, so prompt/completion/cached tokens, prefill/decode/TTFT, and telemetry-capture overhead are **UNMEASURED for this run**, even though the production route does emit final usage/timing metadata when `X-Unsloth-Events: 1` is enabled. Zero-valued placeholders in the temporary JSON are not treated as measurements.

## Current regression evidence

Current-tree checks completed after the live run:

- Helix backend suite: **70 passed**.
- Focused adaptive suite: **32 passed**.
- Ingest + pipeline: **16 passed**.
- Request-scoped speculative telemetry tests: **2 passed** (`388 deselected`).
- Stale self-training reconciliation/provenance tests: **4 passed** (`2 deselected`).
- macOS installer/resource packaging suite: **5 passed**.
- Frontend Hermes/self-audit tests: **9 passed**.
- Frontend `npm run typecheck`: pass.
- Rust preflight Helix fingerprint test: **1 passed** (`476 filtered out`).
- Rust updater/staged-update slice: **31 passed** (`446 filtered out`).
- `cargo check`: pass (existing dead-code warnings only).
- Final app strict codesign verification: pass.

## Acceptance classification

| Category | Status | Evidence |
|---|---|---|
| IMPLEMENTED | YES | Closed-loop code, typed verifier provenance, cache/evidence/audit/Hermes/ledger/training-target gates present |
| TESTED | YES | Backend/frontend/Rust/packaging focused tests above |
| MEASURED | PARTIAL | Load/foreground/self-audit/ingest/controller/storage/unload measured; request telemetry overhead not captured by temporary collector |
| OBSERVED IN REAL AGENT RUN | YES | Qwen executed both search tools; same-model self-audit and closed-loop decision ran |
| TRAINING-CANDIDATE GENERATED | NO | `qlora_eligible=false`; no verified training target/candidate emitted |
| ACTUAL MODEL IMPROVEMENT DEMONSTRATED | **NO** | No held-out before/after trained-adapter evaluation was performed |

## Remaining acceptance gaps

1. Drive the rebuilt `.app` through the native Tauri update command end-to-end and prove automatic post-update overlay reapplication, rather than component-level manual invocation from app Resources.
2. Replace the temporary acceptance collector with a committed collector that retains the final usage/timing control frame; rerun a matched workload to measure instrumentation overhead, token/cache reuse, TTFT/prefill/decode, and self-audit token cost.
3. Perform a real training episode only after an independently verified recurrent target exists, then run held-out before/after evaluation. Until then, model improvement remains unclaimed.
