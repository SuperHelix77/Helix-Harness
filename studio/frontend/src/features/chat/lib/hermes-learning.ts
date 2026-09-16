// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LearningKind,
  LearningTarget,
  LearningRecommendationAction,
} from "../api/learning-api";

export const HERMES_LEARNING_REVIEW_PREFIX = "[UNSLOTH_HERMES_LEARNING_REVIEW]";
export const HELIX_SELF_CRITIC_PREFIX = "[HELIX_SELF_CRITIC]";
export const HERMES_LEARNING_TAG = "unsloth-learning";
export const HELIX_SELF_CRITIC_TAG = "helix-self-critic";
export const ON_THE_FLY_SKILL_TAG = "unsloth-skill-draft";
export const HERMES_LEARNING_CHANGED_EVENT = "unsloth-hermes-learning-changed";
export const HERMES_LEARNING_OPEN_EVENT = "unsloth-open-hermes-learning";
export const SELF_QLORA_OPEN_EVENT = "unsloth-open-self-qlora";

export type HelixSelfCritic = {
  finished: "full" | "partial" | "none";
  rightTools: boolean;
  rightSkills: boolean;
  tooManyTools: boolean;
  toolCount?: number;
  usedSkills?: string[];
  notes?: string;
  recommendation: LearningRecommendationAction;
  skillTitle?: string;
  skillContent?: string;
  reason?: string;
};

export function openHermesLearningManager(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(HERMES_LEARNING_OPEN_EVENT));
  }
}

export function openSelfQloraManager(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(SELF_QLORA_OPEN_EVENT));
  }
}

export type ParsedLearningProposal = {
  kind: LearningKind;
  title: string;
  content: string;
  reason?: string;
  name?: string;
  target?: LearningTarget;
  recommendationAction?: LearningRecommendationAction;
  recommendationReason?: string;
};

export function parseOnTheFlySkillDraft(text: string): ParsedLearningProposal | null {
  const match = text.match(
    new RegExp(`<${ON_THE_FLY_SKILL_TAG}>\\s*([\\s\\S]*?)\\s*</${ON_THE_FLY_SKILL_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  try {
    const value = JSON.parse(match[1]) as Record<string, unknown>;
    const name = typeof value.name === "string" ? value.name.trim() : "";
    const title = typeof value.title === "string" ? value.title.trim() : "";
    const content = typeof value.content === "string" ? value.content.trim() : "";
    if (!name || !title || !content || !/^[a-z0-9][a-z0-9._-]{0,63}$/i.test(name)) return null;
    return {
      kind: "skill",
      name: name.toLowerCase(),
      title: title.slice(0, 240),
      content: content.slice(0, 100_000),
      target: "codex",
      recommendationAction: "skill",
      ...(typeof value.reason === "string" ? { reason: value.reason.slice(0, 2_000) } : {}),
      recommendationReason: "A repeatable procedure was identified; stage it as a skill before considering QLoRA.",
    };
  } catch {
    return null;
  }
}

export function stripOnTheFlySkillDraft(text: string): string {
  return text
    .replace(
      new RegExp(`<${ON_THE_FLY_SKILL_TAG}>\\s*[\\s\\S]*?\\s*</${ON_THE_FLY_SKILL_TAG}>`, "gi"),
      "",
    )
    .trim();
}

/** The model proposes; the app stages. It cannot silently edit a memory file or install code. */
export function hermesLearningReviewPrompt(focus: string): string {
  return [
    HERMES_LEARNING_REVIEW_PREFIX,
    "Review the current conversation as a skeptical Hermes-style learning reviewer.",
    "Find at most one durable, reusable lesson or procedure that would improve future work.",
    "Before choosing, ask yourself in this order: (1) Do I already have a skill I should use? (2) Do we need a new skill I can reuse? (3) Do I fix the runtime? (4) Do I need to QLoRA-train myself on this task? QLoRA is expensive; prefer a skill or memory when the gap is procedural or preference-based. Only recommend QLoRA for a repeated, measurable behavior gap that a skill cannot fix. This is a recommendation, never a score or approval.",
    "Prefer a verified engineering fact, user preference, or repeatable plain-text procedure; do not save secrets, transient details, or unsupported guesses.",
    "Return the normal concise explanation first, then exactly one tag with a JSON object, or kind=none if there is no durable lesson:",
    `<${HERMES_LEARNING_TAG}>{\"kind\":\"memory|user|skill|none\",\"title\":\"short title\",\"content\":\"compact lesson or SKILL.md instructions\",\"reason\":\"why it is reusable\",\"name\":\"skill-name when kind is skill\",\"target\":\"codex\"}</${HERMES_LEARNING_TAG}>`,
    "You may additionally include recommendationAction=skill, qlora, runtime-fix, or none and recommendationReason in that JSON. This never authorizes training or promotion.",
    "Do not claim that anything was saved. The app will place a valid proposal in a review inbox for approval.",
    focus.trim() ? `Review focus: ${focus.trim()}` : "Review focus: the current task and its verified outcome.",
  ].join("\n");
}

export function isHermesLearningReviewRequest(text: string): boolean {
  return text.includes(HERMES_LEARNING_REVIEW_PREFIX);
}

export function helixSelfCriticPrompt(focus: string): string {
  return [
    HELIX_SELF_CRITIC_PREFIX,
    "Ask yourself about the turn you just completed. Answer only with the tag below.",
    "1. Did I finish the user's work? full, partial, or none.",
    "2. Did I use the right tools and skills? Prefer an existing skill over inventing a workflow.",
    "3. Did I use too many tools (rereads, duplicate searches, extra calls)?",
    "Then recommend skill, qlora, runtime-fix, or none. QLoRA is never authorized by this answer; Helix Engine decides.",
    `Return exactly: <${HELIX_SELF_CRITIC_TAG}>{"finished":"full|partial|none","rightTools":true,"rightSkills":true,"tooManyTools":false,"toolCount":0,"usedSkills":[],"notes":"short","recommendation":"skill|qlora|runtime-fix|none","skillTitle":"","skillContent":"","reason":""}</${HELIX_SELF_CRITIC_TAG}>`,
    focus.trim() ? `Turn: ${focus.trim()}` : "Turn: the current completed task.",
  ].join("\n");
}

export function isHelixSelfCriticRequest(text: string): boolean {
  return text.includes(HELIX_SELF_CRITIC_PREFIX);
}

function asBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lowered = value.trim().toLowerCase();
    if (lowered === "true" || lowered === "yes") return true;
    if (lowered === "false" || lowered === "no") return false;
  }
  return fallback;
}

export function parseHelixSelfCritic(text: string): HelixSelfCritic | null {
  const match = text.match(
    new RegExp(`<${HELIX_SELF_CRITIC_TAG}>\\s*([\\s\\S]*?)\\s*</${HELIX_SELF_CRITIC_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  let value: unknown;
  try {
    value = JSON.parse(match[1]);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const finishedRaw = String(raw.finished ?? "partial").toLowerCase();
  const finished =
    finishedRaw === "full" || finishedRaw === "partial" || finishedRaw === "none"
      ? finishedRaw
      : "partial";
  const rec = raw.recommendation;
  const recommendation: LearningRecommendationAction =
    rec === "skill" || rec === "qlora" || rec === "runtime-fix" || rec === "none"
      ? rec
      : "none";
  const used = Array.isArray(raw.usedSkills)
    ? raw.usedSkills.filter((item): item is string => typeof item === "string").slice(0, 16)
    : [];
  const toolCount =
    typeof raw.toolCount === "number" && Number.isFinite(raw.toolCount)
      ? Math.max(0, Math.round(raw.toolCount))
      : undefined;
  return {
    finished,
    rightTools: asBool(raw.rightTools, true),
    rightSkills: asBool(raw.rightSkills, true),
    tooManyTools: asBool(raw.tooManyTools, false),
    ...(toolCount !== undefined ? { toolCount } : {}),
    ...(used.length ? { usedSkills: used } : {}),
    ...(typeof raw.notes === "string" ? { notes: raw.notes.slice(0, 2_000) } : {}),
    recommendation,
    ...(typeof raw.skillTitle === "string" ? { skillTitle: raw.skillTitle.slice(0, 240) } : {}),
    ...(typeof raw.skillContent === "string"
      ? { skillContent: raw.skillContent.slice(0, 8_000) }
      : {}),
    ...(typeof raw.reason === "string" ? { reason: raw.reason.slice(0, 2_000) } : {}),
  };
}

function isKind(value: unknown): value is LearningKind {
  return value === "memory" || value === "user" || value === "skill";
}

function isTarget(value: unknown): value is LearningTarget {
  return value === "codex" || value === "claude" || value === "both";
}

export function parseHermesLearningProposal(text: string): ParsedLearningProposal | null {
  const match = text.match(
    new RegExp(`<${HERMES_LEARNING_TAG}>\\s*([\\s\\S]*?)\\s*</${HERMES_LEARNING_TAG}>`, "i"),
  );
  if (!match?.[1]) return null;
  let value: unknown;
  try {
    value = JSON.parse(match[1]);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object") return null;
  const candidate = value as Record<string, unknown>;
  if (!isKind(candidate.kind)) return null;
  const title = typeof candidate.title === "string" ? candidate.title.trim() : "";
  const content = typeof candidate.content === "string" ? candidate.content.trim() : "";
  if (!title || !content) return null;
  return {
    kind: candidate.kind,
    title: title.slice(0, 240),
    content: content.slice(0, 100_000),
    ...(typeof candidate.reason === "string" ? { reason: candidate.reason.slice(0, 2_000) } : {}),
    ...(typeof candidate.name === "string" ? { name: candidate.name.slice(0, 64) } : {}),
    ...(isTarget(candidate.target) ? { target: candidate.target } : {}),
    ...(candidate.recommendationAction === "skill" ||
    candidate.recommendationAction === "qlora" ||
    candidate.recommendationAction === "runtime-fix" ||
    candidate.recommendationAction === "none"
      ? { recommendationAction: candidate.recommendationAction }
      : {}),
    ...(typeof candidate.recommendationReason === "string"
      ? { recommendationReason: candidate.recommendationReason.slice(0, 2_000) }
      : {}),
  };
}

export function stripHermesLearningProposal(text: string): string {
  return text
    .replace(
      new RegExp(`<${HERMES_LEARNING_TAG}>\\s*[\\s\\S]*?\\s*</${HERMES_LEARNING_TAG}>`, "gi"),
      "",
    )
    .trim();
}
