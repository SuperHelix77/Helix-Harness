# Helix Adaptive Intelligence Cycle — Completion Acceptance (2026-09-17)

This report is the completion acceptance for the original `HELIX ENGINE ADAPTIVE INTELLIGENCE CYCLE / JEV + EMPRYO PRINCIPLES + MODEL SELF-AUDIT → HERMES` specification. It follows the historical gap audit in `docs/helix-adaptive-cycle-conformance-20260917.md`; it does not rewrite that earlier audit as though these features had already existed.

## Acceptance scope

Final source acceptance model: `Qwythos-9B-v2-MTP-Q4_K_M`, GGUF/llama.cpp, 4096 context, q4_0 KV cache, speculative decoding requested off. The source backend is deliberately launched with process-level `--enable-tools` so the internal background-audit isolation is tested against the strongest launcher override.

Controlled objective:

1. call `search_memory("HELIX-FINAL-1709")`;
2. call `search_conversation("HELIX-FINAL-1709")`;
3. return the codename;
4. append the deliberate unsupported sentence `Helix is at least 5x faster.`;
5. do not run a performance benchmark;
6. run the completed observable trajectory through cache/evidence, pre-audit decision control, same-model self-audit, Hermes, Q/E/C/S, decision outcome labelling and provenance storage;
7. verify one trajectory does not authorize or start QLoRA.

The 5x sentence is an acceptance discrepancy, not a product claim. Its purpose is to verify that generative prose cannot promote itself into evidence.

## Real foreground trajectory

The final production-path foreground returned HTTP 200 and visible text:

> Helix-FINAL-1709. Helix is at least 5x faster.

The production tool loop captured both real Mem0 retrieval calls. The arguments and leading results were equivalent, so the second call is objectively redundant despite both calls being valid successful tool invocations.

| Foreground metric | Final value |
|---|---:|
| wall time | `2,916.441 ms` |
| first event | `45.848 ms` |
| first visible text | `2,429.475 ms` |
| prompt tokens | `840` |
| completion tokens | `88` |
| total tokens | `928` |
| cached/reused prompt tokens | `667` |
| newly evaluated prompt tokens | `173` |
| prefill time | `443.914 ms` |
| decode time | `2,131.416 ms` |
| speculative requested | `off` |
| request-scoped speculative counters | `unavailable` |
| hidden reasoning persisted | `0` |

Collector: `scripts/collect-helix-acceptance.py`. It retains visible output, tool events and usage/timing/speculative control frames while dropping reasoning/thinking fields instead of storing them.

## Cache/context integrity result

The cache controller classified the second equivalent retrieval as:

- cause: `MODEL_CAUSED`;
- action: `repeat_tool:search_conversation`;
- necessary: `false`;
- provenance: `backend_tool_capture`;
- evidence: equivalent retrieval path, arguments and leading result already occurred.

That objective duplicate signal is now used consistently by cache attribution, computational-efficiency metrics, decision features and counterfactual credit assignment even when the original captured tool hints are both `useful`. Cache efficiency remains a separate C-dimension concern; it does not lower task correctness or evidence status by itself.

## Evidence and pre-audit gate

No client-supplied claim was needed. The backend bounded claim-discovery pass found `Helix is at least 5x faster.` directly in the visible final answer before generative self-audit.

Evidence status: `UNVERIFIED`. Missing evidence:

- backend-resolved evidence for the observable final-answer claim;
- request-scoped speculative acceptance evidence;
- real decode benchmark.

Unavailable request-scoped GGUF speculative counters are represented as unavailable. They are not treated as measured zero.

`DEEP_SELF_AUDIT` was decided before the expensive audit. The final preparation decision was `TAKE`; explicit acceptance criteria and the high-impact performance assertion independently forced deep audit. This is the live cheap-allocation point. The broader typed decision enum is not falsely presented as controlling every memory/search/skill/model/test choice; existing deterministic runtime/safety logic remains authoritative for those paths.

## Same-model self-audit

The same Qwythos checkpoint audited the backend-resolved `helix.audit-input.v1` bundle only. The request used rolling context, no thread/session recovery, `enable_tools=false`, `enable_thinking=false`, and `X-Helix-Background-Audit: 1`.

The background-audit header is a backend-owned no-tools boundary. It was added after a real failure showed that process-level `--enable-tools` could otherwise outrank request-level tool disabling and inflate a 4096-token audit prompt with tool schemas. The final post-fix audit has zero tool events even while the server is launched with `--enable-tools`.

An earlier audit also demonstrated that checkpoint/thread recovery could auto-admit memory tools. The production audit path now omits that identity and uses rolling context, preventing memory re-retrieval.

Qwythos reproducibly emitted one complete self-audit JSON object after `<helix-self-audit>` but omitted only the closing `</helix-self-audit>` tag, including on the final rerun generated from the exact current 900-token production prompt. The production parser therefore accepts exactly that bounded case only when the complete remainder is valid JSON and contains no trailing prose. Partial JSON and extra prose remain rejected. The final model response also exceeded the prompt's four-item bound for `reusable_lessons`; production parsing deterministically retained only the first four string items. Earlier object-valued cache-disruption output was likewise dropped because that field accepts strings; objective backend cache telemetry independently retains the duplicate-retrieval fact.

Parsed model recommendation: `IGNORE`. The audit nevertheless identified the duplicate retrieval in reusable lessons and left the speed claim without evidence. The model recommendation remains advisory.

| Self-audit metric | Final value |
|---|---:|
| wall time | `96,260.356 ms` |
| first event | `241.060 ms` |
| first visible audit text | `22,383.050 ms` |
| prompt tokens | `2,315` |
| completion tokens | `493` |
| total tokens | `2,808` |
| cached/reused prompt tokens | `0` |
| newly evaluated prompt tokens | `2,315` |
| prompt/prefill time | `22,125.648 ms` |
| decode time | `73,875.189 ms` |
| tool events | `0` |
| hidden reasoning persisted | `0` |

Audit cost is post-task; it is not part of the completed foreground answer's critical path.

Across the definitive foreground plus the exact-current accepted same-model audit, the model processed `3,736` total tokens (`928 + 2,808`) and the two model requests consumed `99,176.798 ms` of client-observed wall time in aggregate. Preparation and final controller ingest are separate control-plane operations rather than model-generation time.

## Live Hermes / adaptation result

The final authenticated `/api/helix-engine/ingest-turn` response on the completed source path reports:

- adaptive cycle available: `true`;
- speed claim: `UNVERIFIED` and retained in `rejected_claim_ids`;
- Hermes action: `RUNTIME_POLICY`;
- reason: avoidable cache/context disruption is mechanical and should be fixed before training;
- model self-audit recommendation: `IGNORE`;
- `self_assessment_disagreement=true` because Hermes' evidence/cache-driven action differs from that advisory recommendation;
- recurrence count: `1` in the isolated final acceptance ledger;
- `qlora_eligible=false`;
- emitted runtime action: `advise_runtime_fix`;
- no QLoRA queue/start;
- counterfactual actual tool calls: `2`;
- counterfactual proposed tool calls: `1`;
- counterfactual equivalence: `UNVERIFIED`;
- `equivalence_verified=false`;
- `training_pair_eligible=false`.

The shorter counterfactual therefore exists only as a provenance-linked hypothesis. A shorter trace is not treated as equivalent-quality training data.

The versioned trajectory record also separates legacy `verified` (full/completed tool loop) from `objective_verified` (backend-resolved outcome proof). The unsupported performance claim therefore cannot inherit objective verification from clean task completion.

The live closed-loop result reported `7.9325 ms` controller overhead and `15,249` bytes of ledger storage for the definitive isolated acceptance trajectory. After the final frontend prompt/parser hardening, a separate deterministic replay of the same real foreground plus the exact-current production-parser-normalized audit used `14,872` bytes in a fresh isolated ledger, with `4.376 ms` prepare-audit replay and `5.554 ms` post-task replay. The replay reproduced `RUNTIME_POLICY`, `self_assessment_disagreement=true`, `qlora_eligible=false`, the `UNVERIFIED` speed claim, and the 2→1 training-ineligible counterfactual.

## Decision outcome / calibration measurement

The isolated final acceptance labels the objectively resolvable decisions:

- `DEEP_SELF_AUDIT=true`;
- `EVIDENCE_SUFFICIENT=false`;
- `QLORA_CANDIDATE=false`.

Calibration summary:

- labelled decisions: `3`;
- Brier score: `0.011133333333333335`;
- 0.0–0.2 bin: count `2`, mean probability `0.065`, observed frequency `0.0`;
- 0.8–1.0 bin: count `1`, mean probability `0.85`, observed frequency `1.0`;
- minimum labels for maturity: `50`;
- `measurement_mature=false`;
- `calibrated=false`.

This is a real calibration measurement, not evidence that the controller is empirically calibrated.

## Matched hot-path telemetry overhead

Committed measurement: `docs/helix-adaptive-cycle-overhead-20260917.json`. Workload: 20,000 measured iterations after 2,000 warmups; 24 messages; 8,192 prompt tokens; 6,144 cached tokens; 194 system-prompt characters.

| Metric | Disabled baseline | Instrumented | Absolute overhead |
|---|---:|---:|---:|
| median | 0.041 µs | 44.917 µs | 44.876 µs |
| p95 | 0.042 µs | 99.416 µs | 99.374 µs |
| mean | 0.034853 µs | 51.179674 µs | 51.144821 µs |

The percentage increase over the near-zero baseline is not a meaningful user-facing latency statistic. The measured absolute synchronous overhead is the relevant value. This does not prove an end-to-end equivalent-quality cache optimization speedup.

## Post-task control-plane/storage measurement

Committed benchmark: `scripts/benchmark-helix-adaptive-cycle.py`, 100 measured unique trajectories after 10 warmups in an isolated ledger.

| Metric | Value |
|---|---:|
| pipeline median | 20.043520 ms |
| pipeline p95 | 27.986001 ms |
| pipeline mean | 18.348044 ms |
| closed-loop reported median | 16.044855 ms |
| closed-loop reported p95 | 21.903291 ms |
| closed-loop reported mean | 14.536631 ms |
| ledger growth | 1,295,822 bytes / 100 trajectories |
| growth per trajectory | 12,958.22 bytes |

This benchmark excludes generative self-audit latency, which is reported separately above.

## Final regression floor

- backend adaptive/ingest/pipeline/engine/routes/self-training/UI-policy suite: `134 passed`;
- focused frontend Hermes/context/Engine/timeout/route suite: `28 passed`;
- frontend TypeScript typecheck: `PASS`;
- production frontend build: `PASS` with only the established CSS/dynamic-import/chunk-size warnings;
- Ruff on modified Python: `PASS`;
- `git diff --check`: `PASS`.

## Status classification

### IMPLEMENTED

- fail-open trajectory observer and exact tool semantics preservation;
- backend-owned typed verification and semantic claim binding;
- bounded live cache/context provenance and explicit unavailable-field handling;
- deterministic final-answer claim discovery before audit;
- live pre-generative `DEEP_SELF_AUDIT` typed allocation with deterministic force rules and fail-open fallback;
- artifact-only same-model audit with hard no-tools backend boundary;
- robust bounded self-audit parsing;
- versioned trajectory, counterfactual, audit-preparation/input, calibration and closed-loop records;
- Hermes self-report/evidence disagreement recording;
- objective decision outcome labels and calibration measurement;
- duplicate-retrieval-aware Q/E/C/S and counterfactual credit;
- QLoRA recurrence/evidence/verified-target safeguards.

### TESTED

The required fail-open, malformed decision, evidence, no-direct-QLoRA, one-anomaly, cache-quality, tool return/exception/cancellation, provenance, versioning, audit-isolation, parser and adaptation-disagreement paths pass in the final regression floor above.

### MEASURED

Hot-path telemetry cost, post-task control-plane cost, ledger growth, final foreground token/cache/timing, final same-model audit cost and decision-outcome Brier/reliability bins are measured.

### OBSERVED IN A REAL AGENT RUN

Real Qwythos called both real memory-search aliases; backend capture found the duplicate; backend claim discovery found the unsupported 5x statement; pre-audit evidence classified it `UNVERIFIED`; same-model audit ran with zero tools under a globally tool-enabled server; the authenticated ingest path produced Hermes `RUNTIME_POLICY`; no QLoRA started.

### TRAINING-CANDIDATE GENERATED

A `helix.counterfactual.v1` shorter candidate is generated but remains `UNVERIFIED` and `training_pair_eligible=false`. No QLoRA candidate from this single event is eligible for training.

### ACTUAL MODEL IMPROVEMENT DEMONSTRATED

**No.** This acceptance demonstrates the control architecture and safeguards. It does not demonstrate improved Qwythos/Qwen weights, held-out task quality, general intelligence, mature controller calibration, verified counterfactual equivalence, or an equivalent-quality cache optimization speedup.

## macOS packaged-app follow-up

After source commit `5e8af2a2c6fb8cedf5a2a8364c2c797ec116a0ea` was pushed, the macOS app was rebuilt from that exact source and installed as `/Users/mert/Desktop/Helix Harness v2.app` while preserving bundle identifier `ai.helix.harness.v2` and version `1.1.0`.

The completion-source runtime overlay now contains:

- contract: `helix.adaptive.backend.v1`;
- runtime files: `639`;
- tree SHA-256: `5fc6075a4b5fe62581b51dbf8c8fbbf2d90c6d33df747f10dd1081c9312f132e`;
- tests: excluded;
- `.pyc`/`.pyo`: excluded;
- backend `requirements/`: excluded.

The historical 640-file bundle had one additional non-current asset, `assets/datasets/alpaca_unsloth.json`; it is absent from the current source, so the 639-file count is expected rather than an omitted completion file. The embedded manifest equals the staged manifest, and the packaged copies of `core/helix_engine/audit.py`, `controller.py`, `decision_controller.py`, `evidence.py`, `trajectory.py`, `routes/helix_engine.py`, and `routes/inference.py` are byte-identical to the completion source.

The clean Tauri release artifact under `studio/src-tauri/target/release/bundle/macos/Helix Harness v2.app` passes `codesign --verify --deep --strict`; signing is ad-hoc and the app is not notarized. The top-level Desktop directory is managed by the macOS/iCloud file provider, which automatically attaches an empty `com.apple.FinderInfo` xattr to top-level `.app` bundles. That metadata makes strict `codesign` verification complain on the Desktop copy even though its executable, `Info.plist`, `_CodeSignature/CodeResources`, overlay manifest, and overlay applier are byte-identical to the clean strictly verified build artifact. Do not represent the app as notarized or Gatekeeper-ready.

The overlay was then applied from the installed app's own `Contents/Resources/helix-backend` using the managed Python runtime. It reported `639 overrides`, `mem0=ready`, the managed backend imported the expected contract plus `require_tool_access`, `run_closed_loop`, and `prepare_observable_self_audit`, and the seven critical packaged backend files above matched the installed runtime byte-for-byte.

Live packaged-app smoke succeeded twice. The Desktop app launched the managed backend on `127.0.0.1:8888`; `/api/liveness` and `/api/health` returned HTTP 200 with `service="Helix Harness"` and Tauri desktop ownership. A graceful app quit terminated both the Tauri process and its backend and closed port 8888. A cold relaunch reached `/api/health` HTTP 200 again after 24 seconds. This establishes packaged-app parity for the completion source. It does not prove the separate native GUI updater orchestration end-to-end after a destructive managed update.

## Remaining evidence boundaries

1. decision calibration is measured but statistically immature and remains `calibrated=false`;
2. counterfactual equivalence has no independent backend replay/verifier receipt;
3. request-scoped GGUF speculative accepted/rejected counters remain unavailable;
4. no held-out before/after model-quality evaluation demonstrates weight improvement;
5. no equivalent-quality before/after experiment demonstrates a cache-optimization speedup;
6. the rebuilt app has not yet been driven through the native Tauri updater command against another destructive managed-runtime overwrite, so strongest-form updater self-repair proof remains separate from packaged-app parity.
