# v3.0.1 — verification record

This records what was independently re-verified for the v3.0.1 release
candidate, what was corrected, and what remains blocked. It is written to be
read by someone who does not trust the release notes.

## Baseline

| Field | Value |
|---|---|
| Source tree | `/Users/mert/Desktop/Helix-Harness-v3-public` |
| Branch | `v3` |
| HEAD at start of v3.0.1 | `9e1fd8c6838d3b3a4aae549ec3f43801c7781ffb` |
| Published release | `v3.0.0` (tag `cfcc5d6`) on the **private** repo `SuperHelix77/Helix-Harness` |

## Defect found and fixed: detached approval still killed two streams

`wait_tool_decision` gained a `fail_closed` parameter, but only **one of four**
call sites used it. The other three still let `ToolApprovalDetached` escape
after the loop had already written a status and a start event to the SSE wire
— the exact shape that leaves the client with a permanently empty thinking box
and no error frame.

| Call site | Before | After |
|---|---|---|
| `core/inference/safetensors_agentic.py:1686` | `try/except` → `deny` | unchanged (already correct) |
| `core/inference/llama_cpp.py:34613` | raised → killed stream | `fail_closed=True` |
| `core/inference/studio_tool_loop.py:1684` | raised → killed stream | `fail_closed=True` |
| `core/inference/durable_tool_journal.py:648,684` | raised → killed stream | `fail_closed=True` |

`llama_cpp.py` is the llama.cpp/Qwen inference path, i.e. the backend used for
the Qwen3.8-27B field report.

### Invariant verified, not assumed

Direct test of `wait_tool_decision` on the durable (`run_id`) path — the only
path where detach raises:

1. default (`fail_closed=False`) still raises `ToolApprovalDetached`, so
   supervisors and recovery passes still learn a detach happened;
2. `fail_closed=True` returns `"deny"` instead of raising;
3. **neither** path writes a denial into the slot;
4. **neither** path calls `expire_tool_approval`.

So the durable approval stays resolvable by an operator, and the model-visible
loop continues with the same conservative answer a user pressing Stop produces.
All four properties pass. All five touched files parse.

## Privacy scrub

Full-content scan across all refs (`git grep` over `git rev-list --all`), not
filename-only.

- **Real secrets: none.** No AWS keys, GitHub tokens, private keys, or live
  credentials in any ref.
- **Placeholders: left alone.** The AWS example key `AKIAIOSFODNN7EXAMPLE` and
  the dummy `ghp_…`/JWT strings live in redaction and security tests; deleting
  them would weaken the very tests that guard against leaks.
- **Fixed:** `core/inference/github_context.py:145` hardcoded
  `home / "llmspeed-work"` as a workspace-discovery root. Removed — it leaked a
  local scratch directory name into a public tree and probed a path that does
  not exist for every user.
- **Publishing constraint:** the repo has 14 commits across 3 refs. `origin/main`
  and 10 intermediate commits contain `/Users/mert` paths in 7 documents
  (`baselines/FROZEN_UNSLOTH_V1.1.txt`, `docs/HELIX_STUCK_P0.md`,
  `docs/HELIX_ADAPTIVE_CYCLE_CHECKPOINT.md`, `docs/helix-harness-restart-handoff.md`,
  `docs/helix-adaptive-cycle.md`, and two acceptance files). Those refs must
  **not** be published. Push the `v3` ref only — never `--all` or `--mirror`.

## README truth corrections

The published v3.0.0 evidence file records:

```
gate_status: qualified_static_identity
gate_checks_passed: 11 / 12
source_dirty: true
distribution_trust_qualified: false
```

and states that static qualification "does not execute or behaviorally validate
the app". The README nonetheless described a frozen, hostile-audited v3
baseline as the release gate. Only `docs/helix-reliability-baseline-v2.1-public.md`
exists, frozen at the v2.0.0-rc.0.1 commit. Seven claims were corrected to
distinguish **v2.1's frozen hostile-audited baseline** (which v3 carries
forward) from **v3's own static package qualification**:

`L16` "real, tested, and load-bearing" → "real and load-bearing" + gate scope.
`L28` v3 lineage row → carries the v2.1 baseline forward; static qualification added.
`L30` "adversarial testing before shipping" → "qualifying that artifact before release".
`L85` "implemented and tested" → "implemented, and carried forward from the v2.1 baseline".
`L107` attributes the hostile testing to the v2.1 baseline explicitly.
`L109` baseline named as v2.1; v3's static gate described separately.
`L124` adversarial test classes labelled as the v2.1 baseline's.

Verified after edit: 8/8 relative links resolve; Unsloth and Buy Me a Coffee
links return HTTP 200. No "years-equivalent" or world-first phrasing remains.

## Producer-claim fencing gap (found and closed)

The working tree mixed two source generations: the uncommitted
`tests/test_durable_tool_approvals.py` imported `append_tool_proposal` and
`current_durable_run_claim` from `core.inference.durable_tool_journal`, and
defined neither — in **no** ref (`v3`, `origin/v3-release`, `origin/main`, disk).
The suite failed at import, before a single assertion.

The storage layer already accepted a `producer_claim` on every relevant call
(`append_events_with_tool_proposal`, `get_worker_run`, `claim_tool_execution`,
`claim_ungated_tool_execution`, `mark_tool_execution_started`,
`mark_ungated_tool_execution_started`, `finish_tool_execution`,
`finish_ungated_tool_execution`). The journal simply never threaded it through.
That is a **fencing gap, not only a missing symbol**: without the claim, a
superseded physical producer could read-modify-write a run it no longer owned.

Reconstructed from the tests' explicit contract:

- `_PRODUCER_CLAIM` contextvar; `durable_tool_run` accepts `producer_claim=` or
  bare `producer_epoch=` / `producer_token=`;
- `current_durable_run_claim()` — backend-only accessor, never projected to the
  public run representation or the frontend event stream;
- `append_tool_proposal()` — commits output, the public `tool_start`, and its
  private checkpoint in one writer transaction, forwarding the **exact** claim
  object (a reconstructed equivalent would let a superseded producer append);
- `DurableExecutionHandle` carries `producer_epoch` / `producer_token`;
- the claim is forwarded on claim, start, and finish — approved and ungated;
- `_require_current_producer()` refuses locally when a handle's producer claim
  no longer matches the active durable run, **before** any storage call, so a
  stale producer cannot reach a mutation even to discover the fence.

`producer_claim` is also embedded in the proposal envelope as a four-part
record, so the approval card carries the fencing identity with it.

### Result

```
tests/test_durable_tool_approvals.py + tests/test_computer_use.py
44 passed, 0 failed  (3.0s)
```

Wider durable-tool surface:

```
test_durable_tool_approvals, test_durable_tool_execution_receipts,
test_durable_agent_recovery, test_tool_policy_state, test_computer_use
129 passed, 10 failed
```

The 10 failures are **pre-existing and identical with or without these
changes** — verified by control: restoring the `HEAD` journal and re-running the
same suite produces the *same* 10 failures (67 passed there, because the
approval suite then fails at import). `diff` of the two failure sets is empty.
This change introduces **zero** new failures. The 10 stem from the same
generation mismatch (`ResultBudgetExposure.__init__() got an unexpected keyword
argument 'request_binding_id'`, `ObservationSeed` never constructed) and still
need their source generation reconciled.

## Receipt/pricing binding gap (8 of the 10 closed)

`tools.py` passed six fields to `ResultBudgetExposure`, which declared three, so
every bounded producer fit raised `TypeError` and the observation seed was never
built. `storage/chat_generation_runs_db.py` already treated
`request_binding_id` / `budget_epoch` / `serving_generation_id` as a coherent
admission tuple, and required all three to be nonempty strings.

Reconciled to that contract rather than to the narrower dataclass:

- `ResultBudgetExposure` gained `request_binding_id`, `budget_epoch`,
  `serving_generation_id`, and a never-serialized `exact_projection_pricer`,
  all defaulted so a plain fit that never had request-scoped pricing still
  validates;
- `is_valid()` rejects a **half-present** binding tuple — if any part of the
  request-scoped identity is set, all of it must be a nonempty string — and
  rejects a non-callable pricer;
- `ObservationSeed` gained `preserved_hint`, and `observation_seed_for_text()`
  accepts and forwards it. The hint is the producer's own recovery pointer
  (how much was kept, how much dropped, how to reach the rest). Discarding it
  would have published a truncated head that looks complete — the same
  evidence-loss shape as the page-truncation fix.

`tests/test_durable_tool_execution_receipts.py`: **10 failed → 2 failed**
(75 passed).

### The last 2 are a specification conflict, not a defect to code around

`test_claimed_row_takeover_keeps_execution_and_checkpoint_identity` and
`test_double_restart_advances_sibling_without_duplicate_end_or_rerun` call
`db.get_worker_run("run-receipt")` with no `worker_token` and no
`producer_claim`. `_require_producer_authority_locked` deliberately refuses that
once a run has been claimed:

> Direct legacy database callers remain usable only while the row is still
> unclaimed (`epoch=0` and no token). The first production claim closes that
> compatibility window: subsequent writes must carry the exact tuple.

Both fail identically on unmodified `HEAD`, so this is not a regression from the
changes above. Weakening the guard to make them pass would re-open the
superseded-producer hole the fencing exists to close, so the guard stands and
the conflict is recorded instead.

The real remaining gap is on the recovery side, and it is worth stating plainly:
`RecoveryPlan` carries `expected_worker_token` but **not** the producer claim, so
`requeue_planned_runs` → `requeue_run_for_restart(expected_producer_claim=…)`
cannot re-authorize across a restart. `get_recovery_snapshot()` already supports
`include_producer_claim=True` for exactly this; the planner does not yet use it.
That is a follow-up, not something to fake in a test.

## Cancellation during the controller phase (closed)

`llama_cpp.py` already passed `cancel_event=` into `settle_controller_completion`,
which did not accept it. Threaded through, and given meaning rather than being
swallowed: a Stop that lands while the controller is still running now raises
before the completion receipt commits, so the tool result stays unrecorded and
the ambiguous/unknown-outcome path owns it. Silently dropping the argument would
have let a cancelled run write a terminal receipt the user had already
cancelled.

`tests/test_external_tool_truncated_and_budget.py` + `test_observation_pack_run_storage.py`:
**6 failed → 2 failed**.

## Combined result

```
test_durable_tool_approvals
test_durable_tool_execution_receipts
test_durable_agent_recovery
test_tool_policy_state
test_computer_use
test_observation_pack
test_observation_pack_run_storage
test_external_tool_truncated_and_budget
test_pricing
test_pricing_edge
test_generation_budget
test_chat_generation_runs

350 passed, 4 failed, 9 skipped  (8.6s)
```

Every one of the 4 remaining failures was confirmed to fail **identically on
unmodified `HEAD`**, by restoring the committed files and re-running. All four
are the same specification conflict: the tests call
`db.get_worker_run(run_id)` with no lease and no producer claim, which
`_require_producer_authority_locked` refuses by design once a run has been
claimed. The guard is load-bearing for the superseded-producer fencing, so it
stands; the tests encode the pre-fencing contract and need updating to pass a
claim, not the guard weakening to satisfy them.

## Environment note

The host's iCloud Drive is still syncing `~/Desktop` and `~/Documents`
(`CloudDocs/Desktop -> /Users/mert/Desktop`). `bird` and `fileproviderd` run at
~95% CPU, and `git status`, `rsync`, `grep -r`, and `find` over the synced trees
time out (60s–240s). Workarounds used: `git archive` (≈1s) and a non-synced
copy under `/tmp` for test execution.

The earlier "TypeScript install is broken" diagnosis in the v3.0.1 fixes note is
**incorrect**. `tsc --version` returns 5.9.3; the hang was host I/O starvation.
