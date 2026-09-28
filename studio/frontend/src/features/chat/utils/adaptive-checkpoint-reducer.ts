// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AdaptiveCheckpointControl } from "../types/api";

export const ADAPTIVE_CHECKPOINT_PHASES = [
  "boundary",
  "claimed",
  "running",
  "completed",
  "recovered",
  "cancelled",
] as const;

export type AdaptiveCheckpointPhase =
  (typeof ADAPTIVE_CHECKPOINT_PHASES)[number];

export type AdaptiveCheckpointTerminal =
  | "active"
  | "success"
  | "cancelled";

export type AdaptiveCheckpointTerminalReceipt = Readonly<
  Record<string, unknown>
> & {
  readonly runId: string;
  readonly threadId?: string;
  readonly eventId?: string;
  readonly checkpointId: string;
  readonly resumeRound: number;
  readonly boundarySeq: number;
  readonly phase: "completed" | "recovered" | "cancelled";
  readonly available: boolean;
  readonly trainingDeferred: boolean;
  readonly controlId?: string;
  readonly controlDigest?: string;
};

export type AdaptiveCheckpointLifecycleEvent = {
  readonly type: "adaptive_checkpoint";
  readonly eventSeq: number;
  readonly eventId?: string;
  readonly runId: string;
  readonly threadId?: string;
  readonly checkpointId: string;
  readonly resumeRound: number;
  readonly boundarySeq: number;
  readonly phase: AdaptiveCheckpointPhase;
  readonly control: AdaptiveCheckpointControl;
  readonly controlId?: string;
  readonly controlDigest?: string;
  readonly receipt?: AdaptiveCheckpointTerminalReceipt;
};

export type AdaptiveCheckpointEnvelope = {
  readonly seq: number;
  readonly type: string;
  readonly payload: Readonly<Record<string, unknown>>;
};

export type AdaptiveCheckpointState = {
  readonly eventId?: string;
  readonly runId: string;
  readonly threadId?: string;
  readonly checkpointId: string;
  readonly resumeRound: number;
  readonly boundarySeq: number;
  readonly phase: AdaptiveCheckpointPhase;
  readonly terminal: AdaptiveCheckpointTerminal;
  readonly control: Readonly<AdaptiveCheckpointControl>;
  readonly controlId?: string;
  readonly controlDigest?: string;
  readonly receipt: AdaptiveCheckpointTerminalReceipt | null;
  readonly stopLatched: boolean;
  readonly continuationAllowed: boolean;
  readonly appliedCursor: number;
  readonly lastEventSeq: number;
};

export type AdaptiveCheckpointReductionErrorCode =
  | "invalid_event"
  | "missing_control"
  | "invalid_control"
  | "missing_receipt"
  | "invalid_receipt"
  | "identity_conflict"
  | "control_conflict"
  | "event_regression"
  | "invalid_transition";

export type AdaptiveCheckpointReduction =
  | {
      readonly accepted: true;
      readonly state: AdaptiveCheckpointState;
      readonly duplicate: boolean;
      readonly advanceAppliedCursor: boolean;
    }
  | {
      readonly accepted: false;
      readonly state: AdaptiveCheckpointState | null;
      readonly error: {
        readonly code: AdaptiveCheckpointReductionErrorCode;
        readonly message: string;
      };
    };

export type AdaptiveCheckpointReductionOptions = {
  readonly stopLatched?: boolean;
  readonly context?: {
    readonly eventId?: string;
    readonly runId?: string;
    readonly threadId?: string;
    readonly checkpointId?: string;
    readonly resumeRound?: number;
    readonly boundarySeq?: number;
    readonly control?: AdaptiveCheckpointControl;
    readonly controlId?: string;
    readonly controlDigest?: string;
    /** A legacy boundary chunk has no lifecycle event id yet. */
    readonly allowLegacyBoundary?: boolean;
  };
};

type ParsedEvent = {
  readonly event: AdaptiveCheckpointLifecycleEvent;
  readonly control: AdaptiveCheckpointControl;
  readonly controlJson: string;
  readonly receipt: AdaptiveCheckpointTerminalReceipt | null;
  readonly receiptJson: string | null;
};

const PHASES = new Set<string>(ADAPTIVE_CHECKPOINT_PHASES);
const ACTIVE_PHASES = new Set<AdaptiveCheckpointPhase>([
  "boundary",
  "claimed",
  "running",
]);
const SUCCESS_PHASES = new Set<AdaptiveCheckpointPhase>([
  "completed",
  "recovered",
]);
const TERMINAL_PHASES = new Set<AdaptiveCheckpointPhase>([
  "completed",
  "recovered",
  "cancelled",
]);
const CHECKPOINT_EVENT_TYPES = new Map<string, AdaptiveCheckpointPhase>([
  ["checkpoint.boundary", "boundary"],
  ["checkpoint.claimed", "claimed"],
  ["checkpoint.running", "running"],
  ["checkpoint.completed", "completed"],
  ["checkpoint.recovered", "recovered"],
  ["checkpoint.cancelled", "cancelled"],
]);

type ReductionContext = NonNullable<
  AdaptiveCheckpointReductionOptions["context"]
>;

const CONTROL_FIELDS = [
  "context_length",
  "trigger_tokens",
  "prompt_tokens",
  "completion_tokens",
  "occupancy_tokens",
  "segment_max_tokens",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isExactText(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.trim() === value;
}

function firstText(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (isExactText(value)) return value;
  }
  return undefined;
}

function firstInteger(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (isNonNegativeSafeInteger(value)) return value;
  }
  return undefined;
}

function field(value: Record<string, unknown>, camel: string, snake: string): unknown {
  return value[camel] ?? value[snake];
}

function controlFromPayload(
  value: Record<string, unknown>,
): unknown {
  const nested = value.control;
  if (isPlainObject(nested)) return nested;
  const adaptive = value.adaptive_checkpoint;
  if (isPlainObject(adaptive)) return adaptive;
  const control: Record<string, unknown> = {};
  for (const key of ["reason", "ratio", "resume_round", ...CONTROL_FIELDS]) {
    const source = key === "resume_round" ? "resume_round" : key;
    if (source in value) control[key] = value[source];
  }
  return Object.keys(control).length > 0 ? control : undefined;
}

function phaseFromPayload(
  payload: Record<string, unknown>,
  outerType: string | undefined,
): unknown {
  const direct = payload.phase;
  if (typeof direct === "string") return direct;
  if (outerType && CHECKPOINT_EVENT_TYPES.has(outerType)) {
    return CHECKPOINT_EVENT_TYPES.get(outerType);
  }
  if (outerType === "checkpoint.terminal") {
    return payload.terminal_phase ?? payload.outcome ?? payload.receipt_phase;
  }
  return undefined;
}

function normalizeWireEvent(
  value: unknown,
  context: ReductionContext | undefined,
): unknown {
  if (!isPlainObject(value)) return value;
  if (isPositiveSafeInteger(value.eventSeq)) return value;

  const envelope = isPositiveSafeInteger(value.seq) && isPlainObject(value.payload);
  const payload = envelope ? value.payload : value;
  if (!isPlainObject(payload)) return value;
  const outerType = typeof value.type === "string" ? value.type : undefined;
  const phase = phaseFromPayload(payload, outerType);
  const eventSeq = envelope ? value.seq : field(payload, "eventSeq", "event_seq");
  // The transport envelope's `seq` is authoritative. A producer may have
  // stamped the old in-band control value before the durable event row was
  // assigned; that stale field is not an identity authority.

  const runId = firstText(
    field(payload, "runId", "run_id"),
    context?.runId,
  );
  const threadId = firstText(
    field(payload, "threadId", "thread_id"),
    context?.threadId,
  );
  const checkpointId = firstText(
    field(payload, "checkpointId", "checkpoint_id"),
    context?.checkpointId,
  );
  const eventId = firstText(
    field(payload, "eventId", "event_id"),
    context?.eventId,
  );
  const resumeRound = firstInteger(
    field(payload, "resumeRound", "resume_round"),
    context?.resumeRound,
  );
  const boundarySeq = firstInteger(
    field(payload, "boundarySeq", "boundary_seq"),
    context?.boundarySeq,
  );
  const control = controlFromPayload(payload) ?? context?.control;
  const contextControlId = firstText(
    field(payload, "controlId", "control_id"),
    context?.controlId,
  );
  const contextControlDigest = firstText(
    field(payload, "controlDigest", "control_digest"),
    context?.controlDigest,
  );
  const receipt =
    payload.receipt ??
    payload.terminal_receipt ??
    payload.checkpoint_receipt;
  return {
    type: "adaptive_checkpoint",
    eventSeq,
    eventId,
    runId,
    threadId,
    checkpointId,
    resumeRound,
    boundarySeq: boundarySeq ?? (phase === "boundary" ? eventSeq : undefined),
    phase,
    control,
    ...(contextControlId ? { controlId: contextControlId } : {}),
    ...(contextControlDigest ? { controlDigest: contextControlDigest } : {}),
    ...(receipt === undefined ? {} : { receipt }),
  };
}

function canonicalJson(value: unknown): string | null {
  if (value === null) return "null";
  if (typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number") return Number.isFinite(value) ? JSON.stringify(value) : null;
  if (Array.isArray(value)) {
    const encoded = value.map((item) => canonicalJson(item));
    return encoded.some((item) => item === null) ? null : `[${encoded.join(",")}]`;
  }
  if (isPlainObject(value)) {
    const entries = Object.keys(value)
      .sort()
      .map((key) => [key, canonicalJson(value[key])] as const);
    if (entries.some(([, item]) => item === null)) return null;
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${item}`).join(",")}}`;
  }
  return null;
}

function parseControl(
  value: unknown,
  resumeRound: number,
):
  | { readonly ok: true; readonly control: AdaptiveCheckpointControl; readonly json: string }
  | { readonly ok: false; readonly code: "missing_control" | "invalid_control"; readonly message: string } {
  if (value === undefined || value === null) {
    return { ok: false, code: "missing_control", message: "checkpoint control is required" };
  }
  if (!isPlainObject(value)) {
    return { ok: false, code: "invalid_control", message: "checkpoint control must be an object" };
  }
  if (
    value.reason !== "context_ratio" ||
    typeof value.ratio !== "number" ||
    !Number.isFinite(value.ratio) ||
    value.ratio <= 0 ||
    value.ratio >= 1
  ) {
    return { ok: false, code: "invalid_control", message: "checkpoint control ratio is invalid" };
  }
  for (const field of CONTROL_FIELDS) {
    if (!isNonNegativeSafeInteger(value[field])) {
      return { ok: false, code: "invalid_control", message: `checkpoint control ${field} is invalid` };
    }
  }
  if (value.context_length === 0) {
    return { ok: false, code: "invalid_control", message: "checkpoint context_length must be positive" };
  }
  if (
    value.resume_round !== undefined &&
    (!isNonNegativeSafeInteger(value.resume_round) || value.resume_round !== resumeRound)
  ) {
    return { ok: false, code: "invalid_control", message: "checkpoint resume_round does not match identity" };
  }
  const json = canonicalJson(value);
  if (json === null) {
    return { ok: false, code: "invalid_control", message: "checkpoint control is not finite JSON" };
  }
  return { ok: true, control: JSON.parse(json) as AdaptiveCheckpointControl, json };
}

function parseReceipt(
  value: unknown,
  identity: {
    readonly runId: string;
    readonly threadId?: string;
    readonly eventId?: string;
    readonly checkpointId: string;
    readonly resumeRound: number;
    readonly boundarySeq: number;
    readonly phase: AdaptiveCheckpointTerminalReceipt["phase"];
    readonly controlId?: string;
    readonly controlDigest?: string;
  },
):
  | { readonly ok: true; readonly receipt: AdaptiveCheckpointTerminalReceipt | null; readonly json: string | null }
  | { readonly ok: false; readonly code: "missing_receipt" | "invalid_receipt"; readonly message: string } {
  if (value === undefined) {
    return {
      ok: false,
      code: "missing_receipt",
      message: `${identity.phase} requires a receipt`,
    };
  }
  if (value === null) {
    return { ok: false, code: "invalid_receipt", message: `${identity.phase} receipt is null` };
  }
  if (!isPlainObject(value)) {
    return { ok: false, code: "invalid_receipt", message: "checkpoint receipt must be an object" };
  }
  const receiptRunId = field(value, "runId", "run_id");
  const receiptThreadId = field(value, "threadId", "thread_id");
  const receiptEventId = field(value, "eventId", "event_id");
  const receiptCheckpointId = field(value, "checkpointId", "checkpoint_id");
  const receiptResumeRound = field(value, "resumeRound", "resume_round");
  const receiptBoundarySeq = field(value, "boundarySeq", "boundary_seq");
  const receiptPhase = value.phase ?? value.status;
  const receiptAvailable = value.available;
  const receiptTrainingDeferred = field(
    value,
    "trainingDeferred",
    "training_deferred",
  );
  const receiptControlId = field(value, "controlId", "control_id");
  const receiptControlDigest = field(value, "controlDigest", "control_digest");
  if (
    receiptRunId !== identity.runId ||
    (identity.threadId !== undefined && receiptThreadId !== identity.threadId) ||
    (identity.eventId !== undefined && receiptEventId !== identity.eventId) ||
    receiptCheckpointId !== identity.checkpointId ||
    receiptResumeRound !== identity.resumeRound ||
    receiptBoundarySeq !== identity.boundarySeq ||
    receiptPhase !== identity.phase ||
    (identity.controlId !== undefined && receiptControlId !== identity.controlId) ||
    (identity.controlDigest !== undefined &&
      receiptControlDigest !== identity.controlDigest) ||
    typeof receiptAvailable !== "boolean" ||
    typeof receiptTrainingDeferred !== "boolean"
  ) {
    return { ok: false, code: "invalid_receipt", message: "checkpoint receipt identity does not match" };
  }
  const json = canonicalJson(value);
  if (json === null) {
    return { ok: false, code: "invalid_receipt", message: "checkpoint receipt is not finite JSON" };
  }
  return {
    ok: true,
    receipt: Object.freeze({
      ...(JSON.parse(json) as Record<string, unknown>),
      runId: String(receiptRunId),
      ...(receiptThreadId === undefined ? {} : { threadId: String(receiptThreadId) }),
      ...(receiptEventId === undefined ? {} : { eventId: String(receiptEventId) }),
      checkpointId: String(receiptCheckpointId),
      resumeRound: Number(receiptResumeRound),
      boundarySeq: Number(receiptBoundarySeq),
      phase: receiptPhase as AdaptiveCheckpointTerminalReceipt["phase"],
      available: receiptAvailable,
      trainingDeferred: receiptTrainingDeferred,
      ...(receiptControlId === undefined
        ? {}
        : { controlId: String(receiptControlId) }),
      ...(receiptControlDigest === undefined
        ? {}
        : { controlDigest: String(receiptControlDigest) }),
    }),
    json,
  };
}

function parseEvent(
  value: unknown,
  context?: ReductionContext,
): 
  | { readonly ok: true; readonly parsed: ParsedEvent }
  | { readonly ok: false; readonly code: AdaptiveCheckpointReductionErrorCode; readonly message: string } {
  if (!isPlainObject(value) || value.type !== "adaptive_checkpoint") {
    return { ok: false, code: "invalid_event", message: "event must be adaptive_checkpoint" };
  }
  if (
    !isPositiveSafeInteger(value.eventSeq) ||
    !isExactText(value.runId) ||
    !isExactText(value.checkpointId) ||
    !isNonNegativeSafeInteger(value.resumeRound) ||
    !isPositiveSafeInteger(value.boundarySeq) ||
    typeof value.phase !== "string" ||
    !PHASES.has(value.phase)
  ) {
    return { ok: false, code: "invalid_event", message: "event identity or phase is invalid" };
  }
  if (
    (value.eventId !== undefined && !isExactText(value.eventId)) ||
    (value.threadId !== undefined && !isExactText(value.threadId))
  ) {
    return { ok: false, code: "invalid_event", message: "event identity or phase is invalid" };
  }
  const phase = value.phase as AdaptiveCheckpointPhase;
  if (
    (context?.eventId !== undefined && value.eventId !== context.eventId) ||
    (context?.runId !== undefined && value.runId !== context.runId) ||
    (context?.threadId !== undefined && value.threadId !== context.threadId) ||
    (context?.checkpointId !== undefined &&
      value.checkpointId !== context.checkpointId) ||
    (context?.resumeRound !== undefined &&
      value.resumeRound !== context.resumeRound) ||
    (context?.boundarySeq !== undefined &&
      value.boundarySeq !== context.boundarySeq)
  ) {
    return { ok: false, code: "identity_conflict", message: "event context identity does not match" };
  }
  if (phase === "boundary" ? value.boundarySeq !== value.eventSeq : value.boundarySeq >= value.eventSeq) {
    return { ok: false, code: "invalid_event", message: "event sequence is not coherent with its boundary" };
  }
  const control = parseControl(value.control, value.resumeRound);
  if (!control.ok) return control;
  if (phase === "completed" || phase === "recovered" || phase === "cancelled") {
    const receipt = parseReceipt(value.receipt, {
      runId: value.runId,
      threadId: isExactText(value.threadId) ? value.threadId : undefined,
      eventId: isExactText(value.eventId) ? value.eventId : undefined,
      checkpointId: value.checkpointId,
      resumeRound: value.resumeRound,
      boundarySeq: value.boundarySeq,
      phase,
      controlId: isExactText(value.controlId) ? value.controlId : undefined,
      controlDigest: isExactText(value.controlDigest) ? value.controlDigest : undefined,
    });
    if (!receipt.ok) return receipt;
    return {
      ok: true,
      parsed: {
        event: value as unknown as AdaptiveCheckpointLifecycleEvent,
        control: control.control,
        controlJson: control.json,
        receipt: receipt.receipt,
        receiptJson: receipt.json,
      },
    };
  }
  if (value.receipt !== undefined) {
    return { ok: false, code: "invalid_receipt", message: "pending phases cannot carry a terminal receipt" };
  }
  return {
    ok: true,
    parsed: {
      event: value as unknown as AdaptiveCheckpointLifecycleEvent,
      control: control.control,
      controlJson: control.json,
      receipt: null,
      receiptJson: null,
    },
  };
}

function normalizeInput(
  value: unknown,
  context?: ReductionContext,
): 
  | { readonly ok: true; readonly parsed: ParsedEvent }
  | { readonly ok: false; readonly code: AdaptiveCheckpointReductionErrorCode; readonly message: string } {
  return parseEvent(normalizeWireEvent(value, context), context);
}

function error(
  state: AdaptiveCheckpointState | null,
  code: AdaptiveCheckpointReductionErrorCode,
  message: string,
): AdaptiveCheckpointReduction {
  return { accepted: false, state, error: { code, message } };
}

function terminalFor(phase: AdaptiveCheckpointPhase): AdaptiveCheckpointTerminal {
  if (SUCCESS_PHASES.has(phase)) return "success";
  if (phase === "cancelled") return "cancelled";
  return "active";
}

function canTransition(current: AdaptiveCheckpointPhase, next: AdaptiveCheckpointPhase): boolean {
  if (next === "cancelled") return ACTIVE_PHASES.has(current);
  if (next === "recovered") return current === "running";
  if (next === "claimed") return current === "boundary";
  if (next === "running") return current === "claimed";
  if (next === "completed") return current === "running";
  return false;
}

function sameIdentity(state: AdaptiveCheckpointState, event: AdaptiveCheckpointLifecycleEvent): boolean {
  return (
    state.runId === event.runId &&
    state.threadId === event.threadId &&
    state.checkpointId === event.checkpointId &&
    state.resumeRound === event.resumeRound &&
    state.controlId === event.controlId &&
    state.controlDigest === event.controlDigest
  );
}

function sameCheckpoint(
  state: AdaptiveCheckpointState,
  event: AdaptiveCheckpointLifecycleEvent,
): boolean {
  return sameIdentity(state, event) && state.boundarySeq === event.boundarySeq;
}

function sameReceipt(
  current: AdaptiveCheckpointTerminalReceipt | null,
  incoming: AdaptiveCheckpointTerminalReceipt | null,
): boolean {
  if (current === null || incoming === null) return current === incoming;
  const { phase: _currentPhase, ...currentRest } = current;
  const { phase: _incomingPhase, ...incomingRest } = incoming;
  return canonicalJson(currentRest) === canonicalJson(incomingRest);
}

function makeState(
  prior: AdaptiveCheckpointState | null,
  event: AdaptiveCheckpointLifecycleEvent,
  control: AdaptiveCheckpointControl,
  receipt: AdaptiveCheckpointTerminalReceipt | null,
  stopLatched: boolean,
): AdaptiveCheckpointState {
  const terminal = terminalFor(event.phase);
  const latched = prior?.stopLatched === true || stopLatched || event.phase === "cancelled";
  return Object.freeze({
    ...(event.eventId ? { eventId: event.eventId } : {}),
    runId: event.runId,
    ...(event.threadId ? { threadId: event.threadId } : {}),
    checkpointId: event.checkpointId,
    resumeRound: event.resumeRound,
    boundarySeq: event.boundarySeq,
    phase: event.phase,
    terminal,
    control: Object.freeze({ ...control }),
    ...(event.controlId ? { controlId: event.controlId } : {}),
    ...(event.controlDigest ? { controlDigest: event.controlDigest } : {}),
    receipt: receipt === null ? null : Object.freeze({ ...receipt }),
    stopLatched: latched,
    continuationAllowed: terminal === "success" && !latched,
    appliedCursor: terminalFor(event.phase) === "active"
      ? Math.max(0, event.eventSeq - 1)
      : event.eventSeq,
    lastEventSeq: event.eventSeq,
  });
}

function duplicateResult(
  current: AdaptiveCheckpointState,
  parsed: ParsedEvent,
  stopLatched: boolean,
): AdaptiveCheckpointReduction {
  if (!sameIdentity(current, parsed.event)) {
    return error(current, "identity_conflict", "duplicate phase identity differs");
  }
  if (
    current.eventId !== undefined &&
    parsed.event.eventId !== undefined &&
    current.eventId !== parsed.event.eventId
  ) {
    return error(current, "identity_conflict", "duplicate event identity differs");
  }
  if (current.boundarySeq !== parsed.event.boundarySeq) {
    return error(current, "event_regression", "duplicate boundary sequence differs");
  }
  if (parsed.controlJson !== canonicalJson(current.control)) {
    return error(current, "control_conflict", "duplicate phase control differs");
  }
  if ((parsed.receiptJson ?? "null") !== (canonicalJson(current.receipt) ?? "null")) {
    return error(current, "invalid_receipt", "duplicate phase receipt differs");
  }
  if (parsed.event.eventSeq < current.lastEventSeq) {
    return error(current, "event_regression", "duplicate is behind the observed cursor");
  }
  const latched = current.stopLatched || stopLatched;
  const next =
    latched === current.stopLatched
      ? current
      : Object.freeze({
          ...current,
          stopLatched: true,
          continuationAllowed: false,
        });
  return {
    accepted: true,
    state: next,
    duplicate: true,
    advanceAppliedCursor: false,
  };
}

function terminalSuccessReplay(
  current: AdaptiveCheckpointState,
  parsed: ParsedEvent,
  stopLatched: boolean,
): AdaptiveCheckpointReduction {
  if (!sameIdentity(current, parsed.event)) return error(current, "identity_conflict", "terminal replay identity differs");
  if (current.boundarySeq !== parsed.event.boundarySeq) return error(current, "event_regression", "terminal replay boundary sequence differs");
  if (parsed.controlJson !== canonicalJson(current.control)) return error(current, "control_conflict", "terminal replay control differs");
  if (!sameReceipt(current.receipt, parsed.receipt)) return error(current, "invalid_receipt", "terminal replay receipt differs");
  if (parsed.event.eventSeq < current.lastEventSeq) return error(current, "event_regression", "terminal replay is behind the observed cursor");
  const latched = current.stopLatched || stopLatched;
  if (parsed.event.eventSeq === current.lastEventSeq && latched === current.stopLatched) {
    return { accepted: true, state: current, duplicate: true, advanceAppliedCursor: false };
  }
  return {
    accepted: true,
    state: Object.freeze({
      ...current,
      stopLatched: latched,
      continuationAllowed: !latched,
      appliedCursor: parsed.event.eventSeq,
      lastEventSeq: parsed.event.eventSeq,
    }),
    duplicate: true,
    advanceAppliedCursor: parsed.event.eventSeq > current.appliedCursor,
  };
}

export function reduceAdaptiveCheckpointEvent(
  current: AdaptiveCheckpointState | null,
  value: unknown,
  options: AdaptiveCheckpointReductionOptions = {},
): AdaptiveCheckpointReduction {
  const parsedResult = normalizeInput(value, options.context);
  if (!parsedResult.ok) return error(current, parsedResult.code, parsedResult.message);
  const parsed = parsedResult.parsed;
  const event = parsed.event;
  const stopLatched = options.stopLatched === true;

  if (current === null) {
    if (event.phase !== "boundary") return error(null, "invalid_transition", "first lifecycle event must be boundary");
    return {
      accepted: true,
      state: makeState(null, event, parsed.control, parsed.receipt, stopLatched),
      duplicate: false,
      // Hold the normal replay cursor at the event before the boundary. A
      // reconnect must see the same non-terminal frame until a terminal receipt
      // proves that the checkpoint lifecycle is settled.
      advanceAppliedCursor: false,
    };
  }
  if (
    event.eventId !== current.eventId ||
    event.runId !== current.runId ||
    event.threadId !== current.threadId
  ) {
    return error(current, "identity_conflict", "event identity does not match the current run");
  }
  if (event.phase === current.phase) return duplicateResult(current, parsed, stopLatched);
  if (event.phase === "cancelled" && !TERMINAL_PHASES.has(current.phase)) {
    if (!sameIdentity(current, event)) return error(current, "identity_conflict", "non-boundary identity differs");
    if (current.boundarySeq !== event.boundarySeq) return error(current, "event_regression", "non-boundary sequence differs");
    if (event.eventSeq <= current.lastEventSeq) return error(current, "event_regression", "event is behind the observed cursor");
    if (parsed.controlJson !== canonicalJson(current.control)) return error(current, "control_conflict", "checkpoint control changed");
  }
  if (event.phase === "boundary") {
    if (event.checkpointId === current.checkpointId) return error(current, "identity_conflict", "replacement needs a new checkpoint id");
    if (event.resumeRound < current.resumeRound) return error(current, "identity_conflict", "replacement resume round regressed");
    if (event.boundarySeq <= current.boundarySeq || event.eventSeq <= current.lastEventSeq) {
      return error(current, "event_regression", "replacement boundary is not later");
    }
    return {
      accepted: true,
      state: makeState(current, event, parsed.control, parsed.receipt, stopLatched),
      duplicate: false,
      advanceAppliedCursor: false,
    };
  }
  if (current.phase === "cancelled") return error(current, "invalid_transition", "cancelled checkpoint cannot regress");
  if (SUCCESS_PHASES.has(current.phase) && SUCCESS_PHASES.has(event.phase)) {
    return terminalSuccessReplay(current, parsed, stopLatched);
  }
  if (TERMINAL_PHASES.has(current.phase)) return error(current, "invalid_transition", "terminal checkpoint cannot regress");
  if (!sameCheckpoint(current, event)) return error(current, "identity_conflict", "non-boundary identity differs");
  if (event.eventSeq <= current.lastEventSeq) return error(current, "event_regression", "event is behind the observed cursor");
  if (parsed.controlJson !== canonicalJson(current.control)) return error(current, "control_conflict", "checkpoint control changed");
  if (!canTransition(current.phase, event.phase)) return error(current, "invalid_transition", "checkpoint phase regressed or skipped");
  return {
    accepted: true,
    state: makeState(current, event, parsed.control, parsed.receipt, stopLatched),
    duplicate: false,
    advanceAppliedCursor: TERMINAL_PHASES.has(event.phase),
  };
}

export function adaptiveCheckpointCursor(state: AdaptiveCheckpointState | null | undefined): number {
  return state?.appliedCursor ?? 0;
}

export function canContinueAdaptiveCheckpoint(state: AdaptiveCheckpointState | null | undefined): boolean {
  return Boolean(state && state.terminal === "success" && !state.stopLatched);
}

export const reduceAdaptiveCheckpoint = reduceAdaptiveCheckpointEvent;
export const canContinueCheckpoint = canContinueAdaptiveCheckpoint;
