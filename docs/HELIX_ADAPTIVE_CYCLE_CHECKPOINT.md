# Helix Harness — Adaptive Intelligence Cycle Checkpoint

Date: 2026-09-16
Authority: the current working `/Users/mert/llmspeed-work/Helix-Harness` filesystem is the regression floor for this cycle.

## Baseline state

This repository has no prior Git commit. It is an imported Unsloth Studio baseline plus the current Helix Harness additions. The cycle must preserve working chat, tool execution, model loading, DFlash/GDN, memory, QLoRA, and existing Helix Engine trajectory capture.

Focused pre-change regression command:

```sh
cd studio/backend
PYTHONPATH=. /Users/mert/.unsloth/studio/unsloth_studio/bin/python -m pytest -q \
  tests/test_helix_engine.py \
  tests/test_helix_pipeline.py \
  tests/test_helix_ingest.py \
  tests/test_helix_engine_routes.py \
  tests/test_helix_speed_policy.py \
  tests/test_helix_occupancy.py \
  tests/test_helix_product_strings.py
```

Result: **34 passed**.

## Already implemented

- `core/helix_engine/capture.py`: bounded in-process tool trajectory capture.
- `core/inference/tools.py`: fail-open successful-tool capture wrapper; captured telemetry cannot change successful return values.
- `core/helix_engine/trajectory.py`: trajectory/tool/correction records.
- `core/helix_engine/credit.py`: useful-subsequence credit filtering.
- `core/helix_engine/compress.py`: simple counterfactual compression candidate.
- `core/helix_engine/critic.py`: typed compact per-turn critic plus deterministic fallback.
- `core/helix_engine/routing.py`: discard/Hermes/QLoRA routing; successful work alone never authorizes weight updates.
- `core/helix_engine/pipeline.py`: observe→extract→deduplicate→attribute→critic→counterfactual→holdout→route→regression→promote gate chain.
- `core/helix_engine/ingest.py`: post-turn ingestion, Hermes staging, advisory self-training recommendations.
- `routes/learning.py`: governed local learning ledger and skill approval/autonomous permission gates.
- `core/memory/mem0_store.py`: optional local Mem0/Qdrant memory, fail-open.
- `routes/self_training.py`: bounded self-QLoRA examples, deterministic held-out benchmark, reversible adapter promotion/rollback.
- frontend `scheduleSelfReflect`: same-model post-turn critic is already asynchronous/fire-and-forget and cannot break chat.
- chat frontend already receives server `usage` and `timings`, including prompt tokens, cached tokens, prompt/prefill milliseconds, decode milliseconds/rate, and measured first-token time.
- DFlash/GDN/llama.cpp runtime has substantial existing support and must not be destabilized.

## Gaps for this cycle

1. Current critic is too narrow: it does not audit the observable tool trajectory, cache telemetry, evidence, tests, claims, or acceptance criteria.
2. No versioned `SelfAuditReport`, evidence record, cache-integrity report, decision record, quality vector, or adaptation-decision schema exists.
3. No evidence-sufficiency controller distinguishes implementation from experimentally demonstrated objectives.
4. No calibrated typed decision-controller API/ledger exists. Existing routing is deterministic but not represented as probabilistic advisory decisions with outcome/calibration metadata.
5. Cache data exists in the chat/runtime path but is not associated with Helix trajectories or attributed to necessary/user/tool/harness/model causes.
6. Failed tool calls are not captured at the `execute_tool` wrapper because capture happens only after a successful return.
7. Counterfactual compression exists, but has no confidence/provenance/evidence linkage.
8. Hermes does not yet explicitly adjudicate self-audit claims against objective evidence or track disagreement/calibration.
9. Autonomous self-QLoRA collection can accumulate ordinary completed turns; autonomous training admission must be tightened so a single anomalous/self-critical trajectory cannot become a training trigger.
10. Current DFlash speed policy wires Auto→DFlash for Darwin Qwen3.8 GGUF, while `helix_speculation_fail_open`, accepted-draft proof, and `write_speed_json` are not wired into the runtime path. Therefore DFlash performance remains an evidence claim, not a demonstrated objective.

## Integration constraints

- Instrumentation and all optional controllers fail open.
- The exact tool return or exception/cancellation semantics are authoritative.
- No hidden/private reasoning is collected; self-audit receives observable artifacts only.
- Deterministic safety/reliability gates remain authoritative over probabilistic decisions.
- QLoRA remains downstream of Hermes/evidence/recurrence gates and never starts because one self-audit requested it.
- Cache efficiency is secondary to result quality/evidentiary completeness.
- Heavy audit/Hermes work remains post-task; hot-path capture must stay bounded and cheap.
