# Helix Harness — Agentic Loop Analysis and Hardening (2026-09-17)

This note analyzes the foreground agent loop after the adaptive-intelligence-cycle completion. It is intentionally separate from post-task Hermes adaptation: the question here is what happens **while the model is still acting**, before the final answer exists.

## Current loop

The production loop is not a single implementation. Three execution paths share the same `ToolLoopController`:

1. GGUF / llama.cpp (`core/inference/llama_cpp.py`);
2. native safetensors / Transformers (`core/inference/safetensors_agentic.py`);
3. external-provider models using local Studio tools (`core/inference/studio_tool_loop.py`).

Each path performs roughly:

```text
context / memory / RAG preflight
  -> model generation
  -> parse / heal typed tool call
  -> ToolLoopController pre-execution gate
  -> permission / confirmation gate
  -> real tool execution
  -> result-budget / context handling
  -> ToolLoopController result ledger
  -> model continuation
  -> bounded final-answer pass
  -> post-task Helix evidence / self-audit / Hermes
```

The shared controller already provided exact-success duplicate suppression, one-shot tool retirement, workspace invalidation, disabled/forced-tool handling and bounded recovery nudges. GGUF additionally had a result-level no-progress guard, while the other paths relied more heavily on call budgets and the shared exact-call ledger.

## Concrete gaps found

### 1. A known semantic duplicate was detected only after paying for it

`search_memory` and `search_conversation` are public aliases that dispatch through the same conversation-memory implementation. The real adaptive acceptance deliberately called both with the same arguments and result. Post-task cache/Hermes correctly classified the second call as avoidable `MODEL_CAUSED` retrieval waste, but the live foreground controller did not know the two names were equivalent, so it executed both.

This was the clearest example of a post-task lesson that should become a cheap online invariant.

### 2. Exact failed calls could be retried for the whole tool budget

The controller correctly did **not** mark failures as successful duplicates, which allowed recovery from transient failures. However there was no upper bound on retrying the exact same tool with the exact same arguments after repeated failure. On a high or unlimited call budget this could become a costly non-progress loop.

### 3. Successful edits created no explicit verification obligation

After `edit_file` succeeded, the model received the raw edit result and was free to immediately claim completion. Post-task evidence could later notice missing tests, but the foreground loop did not remind the acting model to verify the mutation while verification tools were still available.

### 4. Live retry telemetry existed in the schema but not in capture

`ToolStep.retry` was versioned and exposed to audit, but `record_tool_execution()` always left it at zero. A same-call retry was therefore indistinguishable from a first attempt in the observable trajectory.

### 5. Safetensors violated the optional-instrumentation compatibility boundary

The safetensors loop treats `execute_tool` as injectable and already feature-detects optional keyword arguments. `helix_turn_id` was nevertheless passed unconditionally. A compatible executor without that keyword failed before the tool body ran. This is exactly the kind of failure optional Helix instrumentation is not allowed to introduce.

### 6. Objective online progress was not surfaced as a bounded controller snapshot

The controller had enough internal state to know executions, failures and suppressions, but no compact objective progress summary was attached to tool-event provenance. Diagnostics therefore had to reconstruct the loop from raw events.

## Implemented in this pass

### Proven semantic-equivalence suppression

The shared controller now has a deliberately tiny allowlist of **proven read-only equivalence families**. It currently contains only:

```text
search_memory
search_conversation
  -> thread_memory_search
```

The second alias is suppressed **before execution** only when the normalized arguments are exactly equal and a sibling alias already succeeded in the same response. After the first successful alias, the unused sibling is also removed from subsequent advertised tool catalogs so the model is less likely to spend another generation choosing an equivalent spelling; the successful alias remains available for different queries. Explicitly forced calls are not semantically suppressed. A tool name literally requested by the user is also exempt, so a diagnostic/acceptance request such as “call `search_memory` then `search_conversation`” keeps both names advertised and is not silently optimized away. No fuzzy tool-name or result-similarity guessing is used.

The model receives a hidden recovery nudge explaining that the equivalent read-only evidence is already present and that it should either materially change the query/strategy or answer. External-provider tool cards close cleanly with an explicit skipped result instead of spinning.

### Bounded exact-failure retries

The controller now tracks consecutive failures by exact canonical tool + arguments. The default permits:

- initial execution;
- retry 1;
- retry 2;
- then suppresses an unchanged fourth attempt.

This leaves room for transient recovery without allowing an exact failure loop to consume an arbitrarily large budget. Changed arguments or a different tool remain available. For workspace operations, a novel workspace-changing call invalidates the stale failure guard, so the normal `test -> edit -> same test` cycle remains valid.

Two repeated controller suppressions transition to the existing no-tools final-answer recovery path, matching the established exact-duplicate behavior.

### Model-only post-edit verification reminder

A successful `edit_file` result remains **byte-for-byte unchanged in the visible tool event**. Only the model-facing continuation receives a short instruction to run the most relevant available verification when practical, or explicitly state that verification was unavailable/out of scope rather than implying it passed.

This is a quality reminder, not an authority mechanism. A simple conservative terminal-command detector records whether a recognizable targeted verification (`pytest`, typecheck/build/lint, `cargo test/check`, `git diff --check`, etc.) happened after the edit. That observation is exposed as `verification_pending` / `verification_runs` in loop progress. It does **not** create `SUPPORTED` evidence; typed backend verifier receipts remain authoritative.

### Objective loop-progress provenance

Tool-end provenance now carries a bounded `loop_progress` snapshot derived from controller state:

- executed calls;
- successful calls;
- failed calls;
- exact duplicate suppressions;
- equivalent duplicate suppressions;
- repeated-failure suppressions;
- verification pending/runs;
- force-final state.

No hidden reasoning or model self-assessment is used to produce these counters.

Controller no-ops are also persisted separately as versioned `helix.tool-control.v1` events when a real Helix turn id exists. This closes the learning blind spot created by successful prevention: Hermes can now observe that the model *attempted* an exact duplicate, proven semantic duplicate, repeated unchanged failure, disabled call, forced-choice mismatch, or spent one-shot call without representing the prevented action as an executed `ToolStep`. Observer failure is fail-open.

### Real retry index in Helix capture

Live `record_tool_execution()` now sets `ToolStep.retry` from prior executions of the same normalized tool + arguments in that exact capture scope. First attempt is `0`, first retry is `1`, and so on. Changed arguments start a new retry sequence.

### Safetensors fail-open instrumentation fix

`helix_turn_id` is now forwarded only when the injectable `execute_tool` accepts that keyword (or `**kwargs`), matching the existing compatibility contract for other optional execution metadata. Helix capture can no longer make a valid safetensors tool executor fail merely because it lacks that optional parameter.

## What remains missing

### A. Most typed decision kinds are still shadow/advisory, not online allocators

The typed Helix controller supports memory retrieval, repository search, another test, retry, model escalation, local/Astra choice, speculation and other decisions. Only `DEEP_SELF_AUDIT` currently controls a real pre-generative cost allocation point. This is deliberate: the probability model has very little objective calibration data and remains `calibrated=false`.

The next safe step is to shadow these decisions beside the deterministic foreground loop, collect objective outcomes, and promote only decisions with enough held-out calibration evidence. Hard-wiring today would replace known deterministic reliability behavior with an immature probability estimate.

### B. Prevented-call outcomes still need longitudinal policy evaluation

Controller suppressions are now first-class versioned control events and can participate in objective recurrence fingerprints. What is still missing is a held-out longitudinal study answering whether each online policy improves equivalent-quality outcomes: avoided calls, latency/tokens saved, false suppressions, recovery rate after a changed strategy, and final task/evidence quality. Runtime prevention existing is not by itself proof that every threshold is optimal.

### C. Verification pending is a heuristic, not verifier evidence

The edit reminder improves behavior, and the command detector is useful telemetry, but a command looking like `pytest` is not proof by itself. Promotion from "verification attempted" to "claim supported" must continue to require backend-owned typed verification provenance and semantic claim binding.

### D. There is no explicit typed task/goal DAG in the foreground loop

The model still owns decomposition and plan state in natural-language context. The runtime has action history and progress counters but no versioned `Goal -> subgoal -> evidence -> status` object. Such a graph could improve long tasks, but adding one without reliable completion semantics would create another self-reported plan rather than an objective controller.

### E. Result-level no-progress behavior is not fully unified across all backends

GGUF has additional protection against repeated identical tool results (including context-window starvation notices). The shared controller now handles semantic alias duplication and repeated exact failures across all backends, but a generalized result-equivalence/no-progress mechanism still requires care: identical short outputs such as `OK` can be legitimate progress for different mutations. Any shared implementation must distinguish read-only evidence from side-effecting work.

### F. Counterfactual compression still lacks independent equivalence verification

Helix can propose a shorter trajectory, but the candidate remains `UNVERIFIED` and training-ineligible. An online controller should not learn "fewer tools is better" until a backend-owned replay/equivalence verifier demonstrates equal result quality and evidence.

## Validation

Final source checks for this pass:

- shared controller + external Studio loop: `106 passed`;
- full native safetensors agent loop: `366 passed`;
- external-provider hosted/local-tool selection + exception contracts + focused Codex loop: `98 passed`;
- GGUF result-level no-progress suite: `12 passed`;
- adaptive/ingest/pipeline/engine/routes/self-training/UI-policy suite: `136 passed`;
- focused frontend Hermes/context/Engine/timeout/route suite: `28 passed`;
- frontend TypeScript typecheck: `PASS`;
- frontend production build: `PASS` with only the established `::highlight`, ineffective dynamic-import and chunk-size warnings;
- Ruff over all modified Python files: `PASS`;
- `git diff --check`: `PASS`.

The agent-loop tests include real loop integration, not only direct controller unit calls: external-provider calls are observed before execution, semantic duplicates close tool cards without running the alias, real failure executions remain bounded, and optional Helix observers are verified fail-open.

## Priority next architecture

1. Measure the new `ToolControlEvent` policies longitudinally: avoided calls/cost versus false-suppression and final-quality rates.
2. Shadow online typed decisions (`RETRY_FAILED_TOOL`, `RUN_ANOTHER_TEST`, `EVIDENCE_SUFFICIENT`, memory/skill/repository search) and attach objective retrospective labels.
3. Add claim-bound verification debt: important final claims should know which backend verifier receipt, if any, discharged them before the model is allowed to describe them as verified.
4. Build an independent counterfactual replay/equivalence verifier; only then use shorter trajectories as efficiency-training pairs.
5. Run a held-out calibration study before allowing probabilistic decisions to suppress deterministic actions.
6. Consider a typed goal/subgoal graph only after completion/evidence semantics are backend-resolvable; do not store model planning prose as truth.

The governing invariant remains: **the generative model proposes, deterministic runtime invariants protect execution, objective telemetry observes, evidence constrains claims, and Hermes decides what may be learned.**
