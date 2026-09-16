# Helix Harness Restart Handoff

Updated: 2026-09-16

## Authoritative workspace

Work only in:

```text
/Users/mert/llmspeed-work/Helix-Harness
```

Do not use `/Users/mert/llmspeed-work/unsloth-studio-v1.1`. Preserve the dirty tree; do not reset/clean. Baseline commit is `174e429`.

## Current state

The Adaptive Intelligence Cycle is implemented in the working tree and focused regressions are green. The two original search tools are now live-proven with Qwen after an actual upstream managed update.

Managed runtime was updated to Unsloth `2026.9.5`; that update erased the Helix contract as expected. The 640-file overlay embedded in the rebuilt `Helix Harness.app` restored the contract and backend dependencies. No backend or `llama-server` process was left running after the live acceptance/unload.

Current overlay identity:

```text
contract: helix.adaptive.backend.v1
files:    640
sha256:   4343cb04885c2576587591b49239a2d33772768c4fc22a16e99be097900c2e58
```

The app is ad-hoc signed and passes strict codesign. It is not notarized/Gatekeeper-ready.

## What is now closed

- `require_tool_access` is a real policy gate, not a no-op.
- Preflight contract probe/fingerprint prevents a same-version overwrite from hiding behind Ready cache.
- Final bundle contains the dependency-closed 640-file overlay, not the obsolete 25-file overlay.
- Real Qwen called both `search_memory` and `search_conversation`; fresh logs contain no `require_tool_access` ImportError.
- Tool capture preserves return/exception/cancellation semantics and carries backend-only typed verification provenance.
- Evidence authority is not derived from tool/command names.
- Passed verifier evidence must be exactly claim-bound before it supports a claim.
- Frontend self-audit JSON no longer truncates oversized telemetry into invalid JSON.
- Tool slicing preserves evidence indexes.
- Self-audit evidence refs are grammar-filtered to backend-resolvable refs.
- A self-audit QLoRA recommendation cannot re-enter the legacy critic bypass.
- GGUF global speculative counters are not represented as request-scoped evidence under unsafe attribution; MLX preserves request-scoped speculative statistics.
- One event does not trigger QLoRA; model self-audit is not a training target.
- Real acceptance marked an unsupported `>=5x` claim `UNVERIFIED` and chose a runtime-policy response, not QLoRA.

## Last current-tree checks

- Helix backend suite: 70 passed.
- Adaptive suite: 32 passed.
- Ingest/pipeline: 16 passed.
- Speculative telemetry focused tests: 2 passed.
- Self-training stale-reconciliation tests: 4 passed.
- Frontend Hermes/self-audit tests: 9 passed.
- Frontend typecheck: pass.
- macOS installer/resource packaging suite: 5 passed.
- Rust Helix managed-fingerprint test: pass.
- Rust updater/staged-update slice: 31 passed.
- `cargo check`: pass.
- Final app `codesign --verify --deep --strict`: pass.

## Remaining work, in order

1. Exercise the **native Tauri updater command path** with the rebuilt app. Reproduce upstream overwrite, let `update.rs` automatically reapply the bundled overlay, then prove preflight Ready/backend health. The component behavior is live-proven, but this exact GUI/Tauri orchestration is not yet live-proven.
2. Commit a repeatable real-acceptance collector that retains the final request usage/timing frame. Rerun a **matched** baseline vs instrumented workload so instrumentation overhead is measurable rather than inferred from incomparable requests.
3. Capture prompt/completion/cached tokens, prefill/decode/TTFT, telemetry overhead, and self-audit token cost in that matched run.
4. Run final full regression selections and `git diff --check` after any further edits.
5. Review the entire diff and commit intentional source/docs only. Do not commit generated `studio/src-tauri/artifacts/`, frontend `node_modules`, runtime databases/logs, or temporary acceptance state.
6. Do not claim actual model improvement unless a real trained adapter is evaluated before/after on held-out data.

## Important prohibitions

- no reset/clean;
- no obsolete 25-file overlay;
- no full backend copy including `requirements/`;
- no package-version-only compatibility proof;
- no hidden chain-of-thought capture;
- no self-audit-as-training-truth;
- no one-event QLoRA;
- no verifier authority from strings/names;
- no global GGUF speculative counters presented as per-request proof;
- no unsupported performance claim;
- no “self-improving” claim merely because the loop audits itself;
- no claim that packaging durability is fully complete until native Tauri update orchestration is live-proven;
- `ACTUAL MODEL IMPROVEMENT DEMONSTRATED` remains NO until held-out before/after evidence exists.
