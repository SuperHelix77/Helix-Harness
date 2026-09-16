// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import {
  auditToLegacyCritic,
  helixSelfAuditPrompt,
  helixSelfCriticPrompt,
  hermesLearningReviewPrompt,
  isHelixSelfAuditRequest,
  isHelixSelfCriticRequest,
  isHermesLearningReviewRequest,
  parseHelixSelfAudit,
  parseHelixSelfCritic,
  parseOnTheFlySkillDraft,
  parseHermesLearningProposal,
  stripOnTheFlySkillDraft,
  stripHermesLearningProposal,
} from "../src/features/chat/lib/hermes-learning.ts";

function auditArtifacts(prompt: string): Record<string, unknown> {
  const marker = "Observable artifacts JSON: ";
  const offset = prompt.lastIndexOf(marker);
  assert.notEqual(offset, -1);
  return JSON.parse(prompt.slice(offset + marker.length)) as Record<string, unknown>;
}

test("Hermes review prompt is detectable and uses the staged proposal protocol", () => {
  const prompt = hermesLearningReviewPrompt("the loader fix");
  assert.equal(isHermesLearningReviewRequest(prompt), true);
  assert.match(prompt, /unsloth-learning/);
  assert.match(prompt, /Do not claim that anything was saved/);
});

test("learning marker is parsed and removed from the visible answer", () => {
  const answer = `Verified result.\n<unsloth-learning>{"kind":"memory","title":"Keep checks","content":"Run the focused test after changes.","reason":"Reusable verification habit."}</unsloth-learning>`;
  assert.deepEqual(parseHermesLearningProposal(answer), {
    kind: "memory",
    title: "Keep checks",
    content: "Run the focused test after changes.",
    reason: "Reusable verification habit.",
  });
  assert.equal(stripHermesLearningProposal(answer), "Verified result.");
});

test("on-the-fly skill drafts are parsed, bounded, and hidden from the answer", () => {
  const answer = `Completed the task.\n<unsloth-skill-draft>{"name":"json-checks","title":"Verify JSON contracts","content":"Compare parsed values against the contract.","reason":"The same procedure recurred."}</unsloth-skill-draft>`;
  assert.deepEqual(parseOnTheFlySkillDraft(answer), {
    kind: "skill",
    name: "json-checks",
    title: "Verify JSON contracts",
    content: "Compare parsed values against the contract.",
    target: "codex",
    recommendationAction: "skill",
    reason: "The same procedure recurred.",
    recommendationReason:
      "A repeatable procedure was identified; stage it as a skill before considering QLoRA.",
  });
  assert.equal(stripOnTheFlySkillDraft(answer), "Completed the task.");
});

test("self-critic prompt is per-turn and parsed into Helix ingest fields", () => {
  const prompt = helixSelfCriticPrompt("User: ship the check\nAssistant: still missing a test");
  assert.equal(isHelixSelfCriticRequest(prompt), true);
  assert.match(prompt, /Did I finish the user's work/);
  assert.match(prompt, /too many tools/);
  const critic = parseHelixSelfCritic(
    '<helix-self-critic>{"finished":"partial","rightTools":false,"rightSkills":false,"tooManyTools":true,"toolCount":9,"usedSkills":[],"recommendation":"skill","skillTitle":"json-checks","skillContent":"Read the skill first.","reason":"Existing skill covers this."}</helix-self-critic>',
  );
  assert.equal(critic?.finished, "partial");
  assert.equal(critic?.tooManyTools, true);
  assert.equal(critic?.recommendation, "skill");
  assert.equal(critic?.skillTitle, "json-checks");
});

test("learning recommendations accept runtime repair without turning it into a score", () => {
  const proposal = parseHermesLearningProposal(
    '<unsloth-learning>{"kind":"memory","title":"Runtime repair","content":"Revert a failing adapter before retraining.","recommendationAction":"runtime-fix","recommendationReason":"The active candidate failed."}</unsloth-learning>',
  );
  assert.equal(proposal?.recommendationAction, "runtime-fix");
  assert.equal(proposal?.recommendationReason, "The active candidate failed.");
});

test("self-audit artifacts remain valid JSON under oversized telemetry", () => {
  const telemetry: Record<string, unknown> = Object.fromEntries(
    Array.from({ length: 120 }, (_, index) => [`noise_${index}`, "x".repeat(2_000)]),
  );
  telemetry.prompt_tokens = 1_234;
  telemetry.cached_tokens = 800;
  telemetry.accepted_drafts = 7;
  const prompt = helixSelfAuditPrompt("oversized telemetry", {
    objective: "inspect the run",
    final_result: "done",
    telemetry,
    tool_steps: [],
  });
  assert.equal(isHelixSelfAuditRequest(prompt), true);
  const artifacts = auditArtifacts(prompt);
  assert.equal((artifacts.telemetry as Record<string, unknown>).prompt_tokens, 1_234);
  assert.equal((artifacts.telemetry as Record<string, unknown>).cached_tokens, 800);
  assert.match(String((artifacts.telemetry as Record<string, unknown>).helix_artifact_compaction), /summary|structured/);
});

test("self-audit tool slicing preserves backend evidence indexes", () => {
  const toolSteps = Array.from({ length: 20 }, (_, index) => ({
    index,
    evidence_ids: [`tool:${index}:result`],
    name: "read_file",
    arguments: `file-${index}`,
    result: `result-${index}`,
  }));
  const artifacts = auditArtifacts(helixSelfAuditPrompt("index preservation", {
    objective: "inspect files",
    final_result: "done",
    telemetry: {},
    tool_steps: toolSteps,
  }));
  const tools = artifacts.tool_steps as Array<Record<string, unknown>>;
  assert.equal(tools.length, 16);
  assert.equal(tools[0]?.index, 4);
  assert.deepEqual(tools[0]?.evidence_ids, ["tool:4:result"]);
  assert.equal(tools.at(-1)?.index, 19);
  assert.deepEqual(tools.at(-1)?.evidence_ids, ["tool:19:result"]);
});

test("self-audit parser keeps only evidence-reference grammar the backend can resolve", () => {
  const audit = parseHelixSelfAudit(
    '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"IGNORE","claims":[{"claim":"checked","evidence_refs":["tool:19:verification","benchmark:speed","tool:abc:result","tool:0:../../","made-up"]}]}</helix-self-audit>',
  );
  assert.deepEqual(audit?.claims[0]?.evidence_refs, ["tool:19:verification", "benchmark:speed"]);
});

test("self-audit QLoRA recommendation cannot re-enter the legacy critic path", () => {
  const audit = parseHelixSelfAudit(
    '<helix-self-audit>{"objective":"x","achieved":true,"recommendation":"QLORA_CANDIDATE","recommendation_reason":"model request"}</helix-self-audit>',
  );
  assert.ok(audit);
  assert.equal(auditToLegacyCritic(audit).recommendation, "none");
});
