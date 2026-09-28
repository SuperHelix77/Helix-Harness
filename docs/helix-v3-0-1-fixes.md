# v3.0.1 — four fixes after the v3.0.0 release

Branch `v3-release`, following the published v3.0.0. All four were found by field
use and by the research swarm, not by a failing test.

## 1. A detached tool-approval worker killed the whole answer

Field report: prompting Qwen3.8-27B left the thinking box empty and produced no
answer, after a long wait.

Cause, from the backend log: `ToolApprovalDetached("durable tool approval worker
detached before a decision")` was raised inside `wait_tool_decision` and caught
nowhere. The tool loop raises it *after* it has already put a status and a start
event on the SSE wire, so the exception killed the stream and the client kept a
permanently empty thinking box with no error frame to explain it.

Fix: `wait_tool_decision` takes `fail_closed`. It still writes no denial and
still expires nothing — the durable approval is left for an operator, exactly as
`test_shutdown_detaches_without_denial_or_expiry_change` requires. What changed
is only what a *model-visible* caller does next: the streaming loop passes
`fail_closed=True` and receives `"deny"`, the same answer a user pressing Stop
produces. The loop continues and the model gets its turn.

`ToolApprovalDetached` remains the default, so recovery and supervision paths
still learn that a detach happened.

Tests: `test_durable_tool_approvals.py` 41 passed, including a new
`test_fail_closed_waiter_denies_instead_of_raising`.

## 2. Fetched pages were truncated into a dead end

Field report from the harness itself: a documentation lookup returned
"truncated before I could read it", and the model concluded the results were
unreadable.

Cause: `_truncate_page_text` returned `text[:max_chars] + "(truncated, N chars
total)"`. The tail was discarded with nothing naming where it went. Unlike the
terminal path, this one has no spill and no resume hint, so the only recovery was
to re-fetch — which returned the same head, truncated identically. The model was
in a loop with no exit.

Fix: the notice now states how much was kept, how much was dropped, and that
narrowing the request (a specific path, section, or search term) reaches the
unread part. This spends a few characters of the existing budget and adds no
context. The terminal path already had the better answer — a spill file plus an
exact `sed` resume command — and the page path now at least stops implying there
is nothing else.

No headroom is added anywhere.

## 3. Computer use had never actually run

Field test on the operator's Mac, calling `use_computer` directly:

    screenshot -> TypeError: account_path() takes 1 positional argument but 2 were given

`computer_use._session_dir` called `account_path("computer", token)`; the helper
takes one relative path. Every screenshot raised before reaching `screencapture`,
so the read-only half of the tool had never executed on any machine. No test
covered it: the existing tests assert the tool's schema and its approval
classification, never that it runs.

Fix: fold the session token into the relative path
(`account_path(f"computer/{token}")`), which is what the signature intends and
which also keeps concurrent sessions in separate directories.

Verified on the operator's Mac after the fix: a real 4112x2658 PNG, plus live
`open_app`, `click`, `key`, `type`, and `scroll`; and clean validation errors for
missing coordinates, empty or oversized text, and unknown actions.

Tests: `test_computer_use.py` 3 passed, including a new
`test_session_dir_builds_a_real_account_scoped_path`.

## 4. curl/wget could not fetch

Shipped in v3.0.0; listed here so the set reads complete. The capability broker
already classified both as network-shaped, so the blocklist was refusing
dependency acquisition while `pip`/`git`/`npm` stayed open. Authority is
unchanged: the sandbox still denies `network-outbound` without a grant.

## Environment note — what is NOT a product defect

The frontend typecheck and the Tauri build could not be re-run for this commit.
`node -e 'console.log(...)'` runs, but `tsc --version` — which does no work at
all — hangs indefinitely at 0.0% CPU with no output. At the time `fileproviderd`
was at 157% CPU and iCloud's `bird` at 99%, with load average 8.5-10.9 on 14
cores. The TypeScript toolchain is starved by host I/O pressure, not broken by
source.

The frontend change in this commit was therefore verified by inspection (both
call sites import the policy flag and gate on it) rather than by `tsc`. It is
three lines across three small files and typechecked green before the host
saturated; it has not been re-verified by the compiler since.

## Not done

- No re-release. These fixes are on `v3-release`; v3.0.0 still points at the
  earlier qualified bytes, which remain downloadable and verifiable.
- The `v2` Tauri flavor configs and `build_helix_macos_v2*.sh` were deliberately
  left in the tree: the branding and deep-link contract tests still reference
  them, and removing them is a source change beyond this fix.
