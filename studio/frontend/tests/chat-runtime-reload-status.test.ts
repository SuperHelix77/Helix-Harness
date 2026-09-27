// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { readIncompleteInfo, restoredAssistantStatus } = await import(
  "../src/features/chat/utils/continuation.ts"
);
const {
  generationChunkCountsTowardTiming,
  generationChunkHasSubstantiveDelta,
  generationIsSettled,
  generationNeedsRecoveredFinalization,
  generationNeedsRecovery,
  loadGenerationOverlaySnapshot,
  projectAdaptiveCheckpointEvent,
  recoverAdaptiveCheckpointEvent,
  recoverCompletedTurnSelfTraining,
  recoveredAdaptiveCheckpointAlreadyCompleted,
  recoveredAdaptiveCheckpointMetadata,
  recoveredAdaptiveResumeRound,
  recoveredGenerationAssistantText,
  recoveredGenerationUserText,
  recoveredReasoningSummaryMetadata,
  recoveredGenerationFinalMetadata,
  generationRecoveryMetadata,
  shouldPreserveGenerationMetadata,
  subscribeGenerationRecoveryTriggers,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

const ADAPTIVE_CONTROL = {
  reason: "context_ratio" as const,
  ratio: 0.8,
  context_length: 10_000,
  trigger_tokens: 8_000,
  prompt_tokens: 6_000,
  completion_tokens: 2_000,
  occupancy_tokens: 8_000,
  segment_max_tokens: 2_000,
};

test("backend lifecycle frames stay pending until an identity-bound receipt", () => {
  const boundary = projectAdaptiveCheckpointEvent({
    current: {},
    event: {
      seq: 40,
      type: "checkpoint.boundary",
      payload: {
        event_id: "event-boundary",
        checkpoint_id: "checkpoint-42",
        phase: "boundary",
        run_id: "run-42",
        thread_id: "thread-42",
        resume_round: 4,
        control: ADAPTIVE_CONTROL,
        control_id: "control-42",
        control_digest: "digest-42",
      },
    },
  });
  assert.equal(boundary.accepted, true);
  assert.equal(boundary.state?.appliedCursor, 39);
  assert.equal(
    (boundary.metadata.adaptiveCheckpoint as { checkpointStatus?: string })
      .checkpointStatus,
    "pending",
  );

  const claimed = projectAdaptiveCheckpointEvent({
    current: boundary.metadata,
    event: {
      seq: 41,
      type: "checkpoint.claimed",
      payload: {
        event_id: "event-claimed",
        checkpoint_id: "checkpoint-42",
        phase: "claimed",
        run_id: "run-42",
        thread_id: "thread-42",
        resume_round: 4,
        control: ADAPTIVE_CONTROL,
        control_id: "control-42",
        control_digest: "digest-42",
      },
    },
  });
  assert.equal(claimed.accepted, true);
  assert.equal(claimed.state?.appliedCursor, 40);

  const missingReceipt = projectAdaptiveCheckpointEvent({
    current: claimed.metadata,
    event: {
      seq: 42,
      type: "checkpoint.completed",
      payload: {
        event_id: "event-completed",
        checkpoint_id: "checkpoint-42",
        phase: "completed",
        run_id: "run-42",
        thread_id: "thread-42",
        resume_round: 4,
        control: ADAPTIVE_CONTROL,
        control_id: "control-42",
        control_digest: "digest-42",
      },
    },
  });
  assert.equal(missingReceipt.accepted, false);
  assert.equal(missingReceipt.metadata, claimed.metadata);

  const completed = projectAdaptiveCheckpointEvent({
    current: claimed.metadata,
    event: {
      seq: 42,
      type: "checkpoint.completed",
      payload: {
        event_id: "event-completed",
        checkpoint_id: "checkpoint-42",
        phase: "completed",
        run_id: "run-42",
        thread_id: "thread-42",
        resume_round: 4,
        control: ADAPTIVE_CONTROL,
        control_id: "control-42",
        control_digest: "digest-42",
        receipt: {
          run_id: "run-42",
          thread_id: "thread-42",
          event_id: "event-completed",
          checkpoint_id: "checkpoint-42",
          resume_round: 4,
          boundary_seq: 40,
          phase: "completed",
          available: true,
          training_deferred: true,
          control_id: "control-42",
          control_digest: "digest-42",
        },
      },
    },
  });
  assert.equal(completed.accepted, true);
  assert.equal(completed.state?.appliedCursor, 42);
  assert.equal(
    (completed.metadata.adaptiveCheckpoint as { checkpointStatus?: string })
      .checkpointStatus,
    "completed",
  );
});

test("durable recovery binds preflight/finalization text to the stored request", () => {
  assert.equal(
    recoveredGenerationUserText({
      messages: [
        { role: "system", content: "sys" },
        {
          role: "user",
          content: [
            { type: "text", text: "inspect this" },
            { type: "image_url", image_url: { url: "data:image/png;base64,x" } },
            { type: "text", text: "carefully" },
          ],
        },
        { role: "assistant", content: "partial" },
      ],
    }),
    "inspect this\ncarefully",
  );
  assert.equal(
    recoveredGenerationAssistantText([
      { type: "reasoning", text: "private" },
      { type: "text", text: "final " },
      { type: "tool-call", result: "hidden" },
      { type: "text", text: "answer" },
    ]),
    "final answer",
  );
});

test("adaptive recovery never resets a continued turn's durable resume budget", () => {
  const base = {
    model: "local/model",
    messages: [],
    stream: true,
    max_tokens: 100,
  };
  assert.equal(recoveredAdaptiveResumeRound(base), 0);
  assert.equal(
    recoveredAdaptiveResumeRound({
      ...base,
      continue_final_message: true,
      adaptive_resume_round: 7,
    }),
    7,
  );
  assert.equal(
    recoveredAdaptiveResumeRound({ ...base, continue_final_message: true }),
    null,
  );
  assert.equal(
    recoveredAdaptiveResumeRound(
      { ...base, continue_final_message: true, adaptive_resume_round: 7 },
      { ...ADAPTIVE_CONTROL, resume_round: 8 },
    ),
    8,
  );
});

test("adaptive recovery stamps the durable run/event identity and dedupes only completed replay", () => {
  const pending = recoveredAdaptiveCheckpointMetadata({}, ADAPTIVE_CONTROL, {
    runId: "run-7",
    eventSeq: 12,
    resumeRound: 3,
    status: "pending",
  });
  assert.equal(
    recoveredAdaptiveCheckpointAlreadyCompleted(pending, "run-7", 12),
    false,
  );
  const completed = recoveredAdaptiveCheckpointMetadata(pending, ADAPTIVE_CONTROL, {
    runId: "run-7",
    eventSeq: 12,
    resumeRound: 3,
    status: "completed",
    available: true,
    trainingDeferred: true,
  });
  assert.deepEqual(completed.adaptiveCheckpoint, {
    ...ADAPTIVE_CONTROL,
    resumeRound: 3,
    checkpointStatus: "completed",
    checkpointEventSeq: 12,
    checkpointRunId: "run-7",
    checkpointAvailable: true,
    trainingDeferred: true,
  });
  assert.equal(
    recoveredAdaptiveCheckpointAlreadyCompleted(completed, "run-7", 12),
    true,
  );
  assert.equal(
    recoveredAdaptiveCheckpointAlreadyCompleted(completed, "run-7", 13),
    false,
  );
});

test("adaptive recovery retries a lost response with one durable event identity", async () => {
  let attempts = 0;
  let sideEffects = 0;
  let committed = false;
  const pending: Record<string, unknown>[] = [];
  const outcome = await recoverAdaptiveCheckpointEvent({
    current: { generationSeq: 11 },
    control: ADAPTIVE_CONTROL,
    runId: "run-lost",
    eventSeq: 12,
    resumeRound: 2,
    persistPending: async (metadata) => {
      pending.push(metadata);
    },
    execute: async () => {
      attempts += 1;
      if (!committed) {
        committed = true;
        sideEffects += 1;
        throw new TypeError("response lost after commit");
      }
      return {
        available: true,
        checkpoint_id: "acp-event-12",
        resume_round: 2,
        training_deferred: true,
      };
    },
  });
  assert.equal(attempts, 2);
  assert.equal(sideEffects, 1, "backend idempotency owns duplicate side-effect suppression");
  assert.equal(pending.length, 1);
  assert.equal(pending[0]?.generationSeq, 11, "pending must not consume the durable event cursor");
  assert.equal(outcome.status, "completed");
  assert.deepEqual(
    (outcome.metadata.adaptiveCheckpoint as Record<string, unknown>).checkpointId,
    "acp-event-12",
  );
});

test("duplicate adaptive event replay returns the persisted receipt without executing again", async () => {
  const completed = recoveredAdaptiveCheckpointMetadata({}, ADAPTIVE_CONTROL, {
    runId: "run-dup",
    eventSeq: 9,
    resumeRound: 1,
    status: "completed",
    checkpointId: "acp-event-9",
    available: true,
    trainingDeferred: true,
  });
  let pendingWrites = 0;
  let executions = 0;
  const outcome = await recoverAdaptiveCheckpointEvent({
    current: completed,
    control: ADAPTIVE_CONTROL,
    runId: "run-dup",
    eventSeq: 9,
    resumeRound: 1,
    persistPending: async () => {
      pendingWrites += 1;
    },
    execute: async () => {
      executions += 1;
      throw new Error("must not execute");
    },
  });
  assert.equal(outcome.status, "completed");
  assert.equal(outcome.metadata, completed);
  assert.equal(pendingWrites, 0);
  assert.equal(executions, 0);
});

test("Stop aborts checkpoint replay and the cancelled identity stays side-effect silent", async () => {
  const controller = new AbortController();
  let executions = 0;
  const first = await recoverAdaptiveCheckpointEvent({
    current: {},
    control: ADAPTIVE_CONTROL,
    runId: "run-stop",
    eventSeq: 4,
    resumeRound: 0,
    signal: controller.signal,
    persistPending: async () => undefined,
    execute: async () => {
      executions += 1;
      controller.abort(new DOMException("Stopped", "AbortError"));
      throw new DOMException("Stopped", "AbortError");
    },
  });
  assert.equal(first.status, "aborted");
  assert.equal(executions, 1);
  assert.match(
    String((first.metadata.adaptiveCheckpoint as Record<string, unknown>).checkpointError),
    /checkpoint_cancelled/,
  );

  const second = await recoverAdaptiveCheckpointEvent({
    current: first.metadata,
    control: ADAPTIVE_CONTROL,
    runId: "run-stop",
    eventSeq: 4,
    resumeRound: 0,
    persistPending: async () => {
      throw new Error("cancelled replay must not rewrite pending");
    },
    execute: async () => {
      executions += 1;
      throw new Error("cancelled replay must not execute");
    },
  });
  assert.equal(second.status, "aborted");
  assert.equal(executions, 1);
});

test("completed-turn replay survives a lost response with one logical key", async () => {
  let attempts = 0;
  let sideEffects = 0;
  let committed = false;
  const first = await recoverCompletedTurnSelfTraining({
    current: {},
    idempotencyKey: "run-final-1",
    execute: async () => {
      attempts += 1;
      if (!committed) {
        committed = true;
        sideEffects += 1;
        throw new TypeError("response lost after completed-turn commit");
      }
      return {
        recorded: true,
        idempotent: true,
        reason: "idempotency_key_replay",
        exampleId: "example-1",
      };
    },
  });
  assert.equal(attempts, 2);
  assert.equal(sideEffects, 1);
  assert.deepEqual(first.recoveredSelfTrainingFinalization, {
    idempotencyKey: "run-final-1",
    status: "completed",
    recorded: true,
    reason: "idempotency_key_replay",
    idempotent: true,
    exampleId: "example-1",
  });

  const second = await recoverCompletedTurnSelfTraining({
    current: first,
    idempotencyKey: "run-final-1",
    execute: async () => {
      attempts += 1;
      throw new Error("persisted finalization must not execute again");
    },
  });
  assert.equal(second, first);
  assert.equal(attempts, 2);
});

test("settled durable turns recover finalization without reviving generation", () => {
  const legacyBase = {
    generationRunId: "run-final",
    generationStatus: "completed",
    generationSettled: true,
  };
  assert.equal(generationNeedsRecovery(legacyBase), false);
  assert.equal(
    generationNeedsRecoveredFinalization(legacyBase),
    false,
    "old durable turns without the persisted keyed contract must not invent replay side effects",
  );
  const base = {
    ...legacyBase,
    finalizationIdempotencyKey: "run-final",
  };
  assert.equal(generationNeedsRecoveredFinalization(base), true);
  assert.equal(
    generationNeedsRecoveredFinalization({
      ...base,
      incomplete: { reason: "length" },
    }),
    false,
  );
  assert.equal(
    generationNeedsRecoveredFinalization({
      ...base,
      recoveryStopRequested: true,
    }),
    false,
  );
  assert.equal(
    generationNeedsRecoveredFinalization({
      ...base,
      recoveredMemoryFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
      recoveredSelfTrainingFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
      recoveredHelixFinalization: {
        idempotencyKey: "run-final",
        status: "skipped",
      },
    }),
    false,
  );
  assert.equal(
    generationNeedsRecoveredFinalization({
      ...base,
      recoveredMemoryFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
      recoveredSelfTrainingFinalization: {
        idempotencyKey: "run-final",
        status: "failed",
      },
      recoveredHelixFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
    }),
    true,
  );
  assert.equal(
    generationNeedsRecoveredFinalization({
      ...base,
      recoveredMemoryFinalization: {
        idempotencyKey: "stale-run",
        status: "completed",
      },
      recoveredSelfTrainingFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
      recoveredHelixFinalization: {
        idempotencyKey: "run-final",
        status: "completed",
      },
    }),
    true,
    "a stale receipt from another logical run cannot suppress replay",
  );
});

test("durable foreground finalization is backend-owned while old-run recovery stays keyed", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const runtime = readSrc("features/chat/runtime-provider.tsx");
  const selfTrainingStart = adapter.indexOf(
    "const selfTrainingWrite = recordSelfTrainingExample",
  );
  const foreground = adapter.slice(
    selfTrainingStart,
    adapter.indexOf("let helixContextTelemetry", selfTrainingStart),
  );
  assert.ok(selfTrainingStart >= 0);
  assert.doesNotMatch(foreground, /idempotencyKey:/);
  const foregroundGuard = adapter.slice(
    Math.max(0, selfTrainingStart - 700),
    selfTrainingStart,
  );
  assert.match(
    foregroundGuard,
    /finalIncompleteReason === null[\s\S]*!isExternalRequest[\s\S]*!generationRunId/,
    "durable runs must not let the browser coordinate self-training/final ingest",
  );
  assert.match(
    runtime,
    /const idempotencyKey = run\.requestPayload\.finalization_idempotency_key;[\s\S]*idempotencyKey !== run\.id/,
  );
  assert.match(
    runtime,
    /recordSelfTrainingExample\(\{[\s\S]*idempotencyKey,[\s\S]*\}\)/,
  );
  assert.match(adapter, /finalization_idempotency_key:\s*cancelId/);
  assert.match(adapter, /runId:\s*cancelId/);
  assert.doesNotMatch(
    foreground,
    /scheduleSelfReflect\([\s\S]*idempotencyKey:\s*generationRunId/,
  );
  assert.match(
    runtime,
    /const finalizationOnly = \(async \(\) => \{[\s\S]*recoverPostAnswerFinalization\(\{[\s\S]*content:\s*storedMessage\.content/,
  );
  assert.match(
    runtime,
    /run\.status === "completed" && !lengthLimited[\s\S]*recoverPostAnswerFinalization\(\{/,
  );
  assert.doesNotMatch(runtime, /const finalizeRecoveredSelfTraining = async/);
});

test("durable Hermes reviews skip frontend learning writes while legacy reviews retain them", async () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const proposalStart = adapter.indexOf("const learningProposal =");
  const proposalDefinitionEnd = adapter.indexOf(
    "const onTheFlySkillDraft",
    proposalStart,
  );
  const proposalEffectStart = adapter.indexOf(
    "const stagedLearningProposal = learningProposal;",
    proposalDefinitionEnd,
  );
  const proposalEffectEnd = adapter.indexOf(
    "// An on-the-fly skill",
    proposalEffectStart,
  );
  const proposalFlow = [
    adapter.slice(proposalStart, proposalDefinitionEnd),
    adapter.slice(proposalEffectStart, proposalEffectEnd),
  ].join("\n");
  assert.ok(proposalStart >= 0);
  assert.ok(proposalDefinitionEnd > proposalStart);
  assert.ok(proposalEffectStart > proposalDefinitionEnd);
  assert.ok(proposalEffectEnd > proposalEffectStart);
  assert.match(
    proposalFlow,
    /finalIncompleteReason === null &&\s*!generationRunId &&\s*isHermesLearningReviewRequest\(latestUserText\)/,
    "only a complete legacy Hermes review may stage a frontend proposal",
  );
  assert.match(
    proposalFlow,
    /if \(stagedLearningProposal\)[\s\S]*createLearningProposal\([\s\S]*setSelfTrainingRecommendation\(/,
    "both writes remain reachable only through the gated proposal value",
  );

  const runProposalFlow = new Function(
    "context",
    `const {
      finalIncompleteReason,
      generationRunId,
      latestUserText,
      finalAssistantText,
      resolvedThreadId,
      isHermesLearningReviewRequest,
      parseHermesLearningProposal,
      createLearningProposal,
      setSelfTrainingRecommendation,
    } = context;
    return (async () => {
      ${proposalFlow}
    })();`,
  ) as (context: Record<string, unknown>) => Promise<void>;
  const calls: string[] = [];
  const dependencies: Record<string, unknown> = {
    finalIncompleteReason: null,
    latestUserText: "review this learning proposal",
    finalAssistantText: "proposal payload",
    resolvedThreadId: "thread-1",
    isHermesLearningReviewRequest: () => true,
    parseHermesLearningProposal: () => ({
      recommendationAction: "train",
      recommendationReason: "repeated behavior",
    }),
    createLearningProposal: () => {
      calls.push("createLearningProposal");
    },
    setSelfTrainingRecommendation: () => {
      calls.push("setSelfTrainingRecommendation");
    },
  };

  await runProposalFlow({ ...dependencies, generationRunId: "durable-run" });
  assert.deepEqual(calls, [], "durable Hermes turns make no frontend learning writes");

  await runProposalFlow({ ...dependencies, generationRunId: null });
  assert.deepEqual(calls, [
    "createLearningProposal",
    "setSelfTrainingRecommendation",
  ]);
});

test("durable recovery projects the backend checkpoint without a learning request", () => {
  const runtime = readSrc("features/chat/runtime-provider.tsx");
  const learning = readSrc("features/chat/api/learning-api.ts");
  assert.doesNotMatch(runtime, /\brunAdaptiveCheckpoint\b/);
  assert.doesNotMatch(runtime, /\brecoverAdaptiveCheckpointEvent\b/);
  assert.match(
    runtime,
    /recoveredAdaptiveCheckpointMetadata\([\s\S]*runId,[\s\S]*eventSeq: update\.event\.seq[\s\S]*resumeRound: durableRound[\s\S]*status: "completed"/,
  );
  assert.match(runtime, /checkpoint_cancelled/);
  assert.match(
    learning,
    /checkpoint_event_seq:\s*input\.checkpointEventSeq[\s\S]*run_id:\s*input\.runId[\s\S]*resume_round:\s*input\.resumeRound/,
  );
  assert.match(
    readSrc("features/chat/utils/chat-generation-recovery.ts"),
    /return \{ \.\.\.current, adaptiveCheckpoint: receipt \};/,
  );
});

test("first-token recovery ignores role and control chunks", () => {
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { role: "assistant" } }],
    }),
    false,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [],
      usage: { completion_tokens: 1 },
    }),
    false,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { content: "token" } }],
    }),
    true,
  );
  assert.deepEqual(
    [
      { choices: [{ delta: { role: "assistant" } }] },
      { context_truncated: { checkpoint: true } },
      { _adaptiveCheckpoint: ADAPTIVE_CONTROL },
      { choices: [], usage: { completion_tokens: 1 } },
      { choices: [{ delta: { content: "token" } }] },
    ].map(generationChunkCountsTowardTiming),
    [true, false, false, false, true],
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { reasoning_content: "thought" } }],
    }),
    true,
  );
  assert.equal(
    generationChunkHasSubstantiveDelta({
      choices: [{ delta: { reasoning_details: [{ text: "thought" }] } }],
    }),
    true,
  );
});

test("reload recovery preserves server reasoning durations", () => {
  const metadata = recoveredReasoningSummaryMetadata(
    {
      reasoningDuration: 1,
      reasoningDurations: [1],
    },
    3200,
  );
  assert.equal(metadata.reasoningDuration, 3);
  assert.deepEqual(metadata.reasoningDurations, [1, 3]);
  assert.equal(recoveredReasoningSummaryMetadata(metadata, -1), metadata);
});

test("stored assistant status remains truthful after reload", () => {
  const interrupted = { custom: { incomplete: { reason: "interrupted" } } };
  assert.deepEqual(restoredAssistantStatus(interrupted), {
    type: "incomplete",
    reason: "error",
  });
  assert.deepEqual(readIncompleteInfo(interrupted), { reason: "interrupted" });

  // Every reason keeps its own identity, so the Continue bar and the error box
  // cannot disagree about what happened.
  const length = { custom: { incomplete: { reason: "length" } } };
  assert.deepEqual(restoredAssistantStatus(length), {
    type: "incomplete",
    reason: "length",
  });
  assert.deepEqual(readIncompleteInfo(length), { reason: "length" });

  assert.deepEqual(
    restoredAssistantStatus({
      custom: { incomplete: { reason: "cancelled" } },
    }),
    { type: "incomplete", reason: "cancelled" },
  );

  assert.deepEqual(restoredAssistantStatus({ custom: {} }), {
    type: "complete",
    reason: "unknown",
  });
  assert.deepEqual(restoredAssistantStatus(undefined), {
    type: "complete",
    reason: "unknown",
  });
});

test("terminal status settles only after the replay cursor catches up", () => {
  assert.equal(generationIsSettled("completed", 3, 5), false);
  assert.equal(generationIsSettled("completed", 5, 5), true);
  assert.equal(generationIsSettled("running", 5, 5), false);
});

test("active runs are read before messages so a concurrent create is visible", async () => {
  const calls: string[] = [];
  const stored = [{ id: "user-1" }];
  const snapshot = await loadGenerationOverlaySnapshot(
    "thread-1",
    () => {
      calls.push("runs");
      stored.push({ id: "assistant-1" });
      return Promise.resolve([{ assistantMessageId: "assistant-1" }]);
    },
    () => {
      calls.push("messages");
      return Promise.resolve([...stored]);
    },
  );
  assert.deepEqual(calls, ["runs", "messages"]);
  assert.equal(
    snapshot.messages.some(
      (message) => message.id === snapshot.activeRuns[0]?.assistantMessageId,
    ),
    true,
  );
});

test("terminal recovery restores final local usage and timing metadata", () => {
  const metadata = recoveredGenerationFinalMetadata({
    current: { generationSettled: true },
    run: {
      id: "run-1",
      requestPayload: { model: "local/model" },
      createdAt: 100,
      startedAt: 120,
      completedAt: 1120,
    },
    usage: {
      prompt_tokens: 8,
      completion_tokens: 12,
      total_tokens: 20,
      prompt_tokens_details: { cached_tokens: 3 },
    },
    timings: { predicted_per_second: 12, prompt_ms: 25 },
    firstChunkAt: 220,
    totalChunks: 4,
  });
  assert.deepEqual(metadata.contextUsage, {
    promptTokens: 8,
    completionTokens: 12,
    totalTokens: 20,
    cachedTokens: 3,
    cacheWriteTokens: 0,
    modelId: "local/model",
  });
  assert.deepEqual(metadata.serverTimings, {
    predicted_per_second: 12,
    prompt_ms: 25,
  });
  assert.deepEqual(metadata.timing, {
    streamStartTime: 120,
    firstTokenTime: 100,
    totalStreamTime: 1000,
    tokenCount: 12,
    tokensPerSecond: 12,
    totalChunks: 4,
    toolCallCount: 0,
  });
  assert.deepEqual(metadata.responseDetails, {
    modelId: "local/model",
    modelLabel: "local/model",
    responseModelId: "local/model",
    providerName: "Local model",
    providerType: "local",
    startedAt: 120,
    finishedAt: 1120,
    durationMs: 1000,
    cancelId: "run-1",
    toolCalls: [],
  });
});

test("reload, wake, and stale-tab recovery stays monotonic and truthful", () => {
  const apply = (
    status: "running" | "completed" | "failed" | "cancelled",
    cursor: number,
    lengthLimited = false,
  ) =>
    generationRecoveryMetadata({
      current: { generationRunId: "run-1" },
      runId: "run-1",
      status,
      cursor,
      lastEventSeq: 4,
      lengthLimited,
    });
  assert.deepEqual(
    [
      ["running", 2],
      ["completed", 2],
      ["completed", 2, true],
      ["completed", 4, true],
      ["failed", 4],
      ["cancelled", 4],
    ].map(([status, cursor, limited]) => {
      const metadata = apply(
        status as "running" | "completed" | "failed" | "cancelled",
        cursor as number,
        Boolean(limited),
      );
      return [generationNeedsRecovery(metadata), metadata.incomplete];
    }),
    [
      [true, { reason: "cancelled" }],
      [true, undefined],
      [true, { reason: "length" }],
      [false, { reason: "length" }],
      [false, { reason: "interrupted" }],
      [false, { reason: "cancelled" }],
    ],
  );

  const windowTarget = new EventTarget();
  const documentTarget = Object.assign(new EventTarget(), {
    visibilityState: "hidden",
  });
  let recoveries = 0;
  const unsubscribe = subscribeGenerationRecoveryTriggers(
    windowTarget,
    documentTarget,
    () => {
      recoveries += 1;
    },
  );
  windowTarget.dispatchEvent(new Event("online"));
  windowTarget.dispatchEvent(new Event("pageshow"));
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  documentTarget.visibilityState = "visible";
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  unsubscribe();
  windowTarget.dispatchEvent(new Event("online"));
  assert.equal(recoveries, 3);

  const existing = {
    generationRunId: "run-1",
    generationSeq: 4,
    generationStatus: "completed",
    generationSettled: true,
    serverManaged: true,
  };
  const incoming = { ...existing };
  assert.deepEqual(
    [
      incoming,
      { ...incoming, generationSeq: 3 },
      { ...incoming, generationRunId: "run-2" },
      { ...incoming, generationStatus: "running" },
      { ...incoming, generationSettled: false },
    ].map((candidate) => shouldPreserveGenerationMetadata(existing, candidate)),
    [false, true, true, true, true],
  );
});

test("recovery persists recovered usage alongside the cursor it advanced", () => {
  // The usage chunk arrives before the terminal event. A cursor published past it and then
  // reloaded would resume after it, so the counts have to travel with the cursor.
  const usage = { prompt_tokens: 8, completion_tokens: 12, total_tokens: 20 };
  const timings = { predicted_per_second: 12 };
  const midStream = generationRecoveryMetadata({
    current: { generationRunId: "run-1" },
    runId: "run-1",
    status: "running",
    cursor: 3,
    lastEventSeq: 5,
    lengthLimited: false,
    usage,
    timings,
  });
  assert.equal(midStream.generationSettled, false);
  assert.deepEqual(midStream.generationRecoveryUsage, usage);
  assert.deepEqual(midStream.generationRecoveryTimings, timings);

  // A recovery that never saw one must not invent or erase it.
  const withoutUsage = generationRecoveryMetadata({
    current: { generationRunId: "run-1" },
    runId: "run-1",
    status: "running",
    cursor: 1,
    lastEventSeq: 5,
    lengthLimited: false,
  });
  assert.equal("generationRecoveryUsage" in withoutUsage, false);

  // Reloading picks the stored counts back up, so settlement still reports them.
  const settled = recoveredGenerationFinalMetadata({
    current: { generationSettled: true },
    run: {
      id: "run-1",
      requestPayload: { model: "local/model" },
      createdAt: 100,
      startedAt: 120,
      completedAt: 1120,
    },
    usage: midStream.generationRecoveryUsage as typeof usage,
    timings: midStream.generationRecoveryTimings as typeof timings,
    firstChunkAt: 220,
    totalChunks: 4,
  });
  assert.deepEqual(settled.contextUsage, {
    promptTokens: 8,
    completionTokens: 12,
    totalTokens: 20,
    cachedTokens: 0,
    cacheWriteTokens: 0,
    modelId: "local/model",
  });
  assert.deepEqual(settled.serverTimings, timings);
});

test("recovery persists the replay prefix statistics it has applied", () => {
  const metadata = generationRecoveryMetadata({
    current: {
      generationRunId: "run-1",
      generationChunkCount: 4,
    },
    runId: "run-1",
    status: "running",
    cursor: 7,
    lastEventSeq: 9,
    lengthLimited: false,
    firstChunkAt: 220,
    totalChunks: 6,
  });
  assert.equal(metadata.generationFirstChunkAt, 220);
  assert.equal(metadata.generationChunkCount, 6);
});
