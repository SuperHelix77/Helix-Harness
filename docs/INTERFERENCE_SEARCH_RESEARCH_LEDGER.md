# Interference Search Research Ledger

Date: 2026-09-26  
Status: PARKED / RESEARCH CANDIDATE  
Priority: After stable Helix Harness release; evaluate alongside Helix Engine and Q38 work.

## External donor

- Repository: https://github.com/Badtheorylabs/interference-search
- Source: Bad Theory Labs "Interference Search"
- Treat all published benchmark/speed claims as external claims until independently reproduced.

## Core mechanism

Interference Search reasons over an explicit frontier of states rather than one linear natural-language trajectory:

1. Expand multiple candidate states.
2. Canonicalize states.
3. Merge equivalent states.
4. Prune/refute dead ends with a small judge.
5. Advance the surviving frontier together.
6. Terminate when an accepted solution state is reached.

The transferable idea is not "quantum reasoning" and not a generic inference-kernel speedup. It is structured external search that can remove repeated semantic recomputation.

## Why it matters to Helix

Helix is developing most of the substrate needed to test a more general form of this mechanism:

- canonical task and run identity;
- Interlingua typed semantic states;
- durable receipts and provenance;
- deterministic tool results;
- evidence classes and trust boundaries;
- replay/recovery;
- MacJev/small-model routing;
- bounded information plane;
- context compilation.

Potential Helix interpretation:

```
canonical task state
        |
 candidate transitions
   /    |    \
 state state state
   \    |    /
 canonicalize + merge
        |
 tiny refuter / MacJev
        |
 surviving frontier
        |
 expensive model only where required
```

The objective is to reduce total semantic turns and repeated reasoning, not merely reduce prompt tokens.

## Critical research problem: state equivalence

The main difficulty for general agent tasks is defining when two branches are genuinely mergeable.

Candidate equivalence key:

- same goal;
- same unresolved hard constraints;
- same authoritative evidence set;
- same source/artifact identities;
- same durable side-effect receipts;
- same permissions/capabilities;
- same unresolved blockers.

Two states must never merge if they differ in durable side effects, authority, source identity, or unresolved evidence.

## Relationship to MacJev

MacJev may eventually do more than select an executor.

Possible bounded role:

- score frontier states;
- reject low-value/dead-end states;
- choose which states deserve expansion;
- abstain on high-risk or ambiguous branches.

This must remove expensive model work rather than add an extra model turn.

## Relationship to Q38

Potential Dualstack interpretation:

```
Q38 compressed core
        |
 latent/action proposals
        |
Helix semantic engine
        |
 explicit frontier search
 memory / tools / verification
        |
 accepted state
```

Research question:

> How much search, state, memory, verification, and execution can be externalized into Helix while preserving useful behavior in an aggressively compressed model core?

Any model-specific learned state stored in Helix must be counted in system-level compression accounting. Generic runtime code may be reported separately, but model-specific entropy cannot be hidden outside the Q38 artifact.

## Relationship to ReplaySSM

These attack different forms of redundant work:

- ReplaySSM: reduce redundant state copying/recomputation inside hybrid recurrent inference.
- Interference Search: reduce redundant semantic reasoning/recomputation across task branches.

They may be complementary.

## Evidence discipline

Do not treat the donor's reported speed comparison as an inference-speed claim.

A system that searches explicit states with a small judge is architecturally different from repeatedly invoking the base LLM. Required Helix evaluation should report:

- total model calls;
- total semantic turns;
- total input/output tokens;
- tool calls;
- wall time;
- frontier width/depth;
- states expanded;
- states merged;
- states pruned;
- success rate;
- model-equivalent compute;
- memory overhead;
- state-canonicalization cost.

## Initial experiment plan

After the stable Harness release:

1. Pin the donor repo/commit.
2. Reproduce one deterministic benchmark exactly.
3. Implement a Helix-only shadow frontier with no production authority.
4. Use deterministic state hashing/canonicalization.
5. Add MacJev/tiny-judge scoring only after the rule-based baseline.
6. Compare against:
   - linear agent;
   - best-of-N;
   - ordinary subagent fan-out;
   - frontier search.
7. Hold tool access, source identity, task budget, and acceptance criteria constant.
8. Measure accepted useful work per semantic turn and per wall-clock second.
9. Run adversarial merge tests to ensure distinct durable states never collapse.
10. Promote nothing until gains survive heterogeneous coding/tool tasks, not only enumerable puzzles.

## Promotion gate

Promote into Helix Engine only if it demonstrates:

- lower semantic turns or frontier-model calls;
- equal or better task success;
- no provenance loss;
- no side-effect/authority merge errors;
- bounded memory/coordination overhead;
- reproducible benefit on heterogeneous agent workloads.

## Current decision

PARKED.

Do not interrupt current Helix Harness release qualification. Revisit after the Harness is stable, with Q38 and Helix Engine as the primary research consumers.
