# P0: Helix conversation-stuck bug — fix record

Updated: 2026-09-16T06:15Z.

## Newest stuck thread
- Thread: `__LOCALID_jJ6N4wN` ("Please, fix the damn thing.")
- Model at hang: empero-ai/Qwythos-9B-v2-GGUF Q4_K_M
- Last failed user: `Qq0L0nv` @ 2026-09-15T08:16:21.991Z "server restarted, continue"
- Orphan assistants (empty `[]`, same parent) before surgery: `kIakGPz`, `Gam5qZz`, `HdSQ5di`

Older same-signature threads (not newest): `__LOCALID_CxYzeOc`, `__LOCALID_ZyPdYuP` / `lcLWXar`.

## Evidence paths
- Live DB: `/Users/mert/.unsloth/studio/studio.db`
- Snapshot: `/Users/mert/llmspeed-work/p0-forensics/db-snapshots/studio-before-p0-20260916T054403Z.db`
- Backend log: `/Users/mert/.unsloth/studio/logs/backend-backend-1789460169134-3-s01.log`
- Live Python: site-packages `studio` under `~/.unsloth/studio/unsloth_studio` (checkout `/Users/mert/llmspeed-work/unsloth-studio-v1.1` via pth; frontend dist from that tree)
- Frontend source: Unsloth-Helix `studio/frontend/src/features/chat/`

## Verified first bad state (`jJ6N4wN` @ 08:16–08:19)

Do NOT assume learning-context hang. Logs prove:

1. `GET /api/learning/context` → **404 in ~1ms** (version skew; not a hang)
2. Optimistic assistant `[]` persisted via `PUT .../messages`
3. Send 1: durable chat-run `0220816c` events stream ~125s then cancel; empty content
4. Send 2: `POST /v1/chat/completions` 200 in 38s (`inference.reload_cancelled_generations`); empty content; engine `running:1` / `gen_tok_s:0`
5. Send 3 (`HdSQ5di`): learning 404, autosave loop, **no `request_completed` for completions** — first token never arrives; catch/finally never run; busy + empty bubble stick

First stage that does not complete for the terminal stuck send: **backend.first_token** (and for send 3, `backend.request` never completes). Learning 404 is not causal. Empty placeholders plus uncleared busy are the persisted UI stuck state.

## Architectural invariant
CORE CHAT MUST WORK EVEN IF MEM0, HERMES, QLORA, LEARNING CONTEXT, AND SPECULATIVE ACCELERATION ARE ALL BROKEN.

## Changes
1. `learning-api.ts`: 4s AbortController on `getLearningContext`; `jsonOrThrow`; outer `.catch` degrades to `{enabled:false, instruction:""}`.
2. `memory-api.ts`: 4s abort; failures return `{available:false}` so Mem0 cannot break chat.
3. `outbound-tool-sanitize.ts`: drop unmatched tool_calls before completions; stub empty tool-only turns. Wired into send and token-count paths. No context-length change.
4. `chat-adapter.ts`: SendTraceSession events (`send.received` … `ui.terminal`); 10s race on first-save; skip empty yield + `DELETE` orphan placeholder when generation never produces text; `finally` still clears busy.
5. `runtime-provider.tsx`: empty no-run-id assistants restore as interrupted, not running/Generating.
6. Backend `DELETE /api/chat/threads/{id}/messages/{id}` + `delete_orphan_assistant_placeholder` (assistant + empty + no children + no active run). Applied to Helix checkout, v1.1 checkout, and live site-packages studio.
7. Startup `reconcile_orphaned_runs` already existed; left as-is.

## Surgery
Deleted 21 unreferenced empty assistant rows (including `HdSQ5di` / `lcLWXar` / `vZcIwbS`). Last message on `__LOCALID_jJ6N4wN` became user `Qq0L0nv`.

## Tests
- Frontend: `tests/outbound-tool-sanitize.test.ts`, `tests/learning-context-timeout.test.ts`, `tests/cancelled-turn-history-prune.test.ts` — 23 passed.
- Backend: `tests/test_orphan_assistant_placeholder.py` — 3 passed (Helix + v1.1).

## Acceptance (live)
1. Rebuilt Helix frontend dist; rsynced to v1.1 `studio/frontend/dist` (chat bundle `chat-BCLx718s.js` contains traces + sanitizer).
2. Restarted `unsloth studio --api-only -H 127.0.0.1 -p 8888`. DELETE route registered (`openapi` methods `delete,get,put`). `/api/learning/context` still 404 in 2ms.
3. Loaded `empero-ai/Qwythos-9B-v2-GGUF` `Q4_K_M`.
4. **Stuck thread `__LOCALID_jJ6N4wN` streamed non-empty `HELIX_P0_OK`** (persisted `lFNfzDr`).
5. **Restarted Helix again; same thread streamed `HELIX_P0_OK2`** (persisted `5rcu2Q9`). Zero empty `[]` assistant rows remain on that thread.

## Next action
Open the Unsloth / Helix desktop app against this backend so the new dist is what the UI serves. Core chat is unblocked even when learning/Mem0/Hermes are 404.
