# Protocol: status, handoffs, iteration lifecycle

All coordination is **file-based**. Agents are stateless subagents spawned by the orchestrator;
their only memory is what they read from / write to these files. Keep files terse and structured.

## Iteration id

`iter-NNN` (zero-padded, e.g. `iter-001`). Every artifact for an iteration is tagged with it.

## `status.md` (one per agent)

```markdown
# <agent> status
iteration: iter-007
state: idle | working | done | blocked
updated: 2026-05-25T14:00Z
summary: <one line of what you're doing / just did>
blockers: <none | description>
```

State machine: `idle` → (orchestrator dispatches) → `working` → `done` (handoff written) →
`idle` (orchestrator picks up). `blocked` means the agent needs orchestrator/human input.

## `handoff.md` (one per agent — the contract to the NEXT agent)

Every handoff begins with this header, then agent-specific sections:

```markdown
# Handoff: <from-agent> → <to-agent>
iteration: iter-007
status: READY | NEEDS-FIX | REJECT | ADOPT
produced: <list of files written this step, repo-relative paths>
```

### Research → Implementation
```markdown
## Hypothesis
<one falsifiable sentence: "X will reduce maxDD by ≥15% without losing >0.2 Sharpe because …">
## Rationale
<data-driven evidence from the last eval report + any literature/SOTA, with links>
## Spec
<precise description of the change: inputs, transform, where it plugs into the stack>
## Pre-registered success criteria
<the exact metrics + thresholds that count as success — MUST align with evaluation_contract.md>
## Risks / things to watch
<look-ahead traps, expected failure modes>
```

### Implementation → Review
```markdown
## What was built
<script path(s), entrypoint, what it does>
## How to run
<exact command, expected runtime, outputs written>
## PIT/look-ahead self-check
<how each feature/label respects point-in-time; shifts, trailing windows, label purge>
## Deviations from spec
<none | what changed and why>
```

### Review → Evaluation (or back to Implementation)
```markdown
## Verdict: PASS | FIX-NEEDED
## Look-ahead audit
<line-referenced findings: shift/window/purge correctness, train/test leakage>
## Correctness
<bugs found, edge cases, numerical issues>
## Required fixes (if FIX-NEEDED)
<numbered, actionable>
```

### Evaluation → Research (closes the loop)
```markdown
## Verdict: ADOPT | REJECT
## Metrics vs baseline & current_best
<table: Sharpe, maxDD, Calmar, folds_positive, nested-OOS Sharpe, placebo percentile,
 paired-CI, per-universe — see evaluation_contract.md>
## Contract gate results
<each gate: PASS/FAIL with the number>
## Why adopt/reject
<plain-language conclusion>
## Insights for next research cycle
<what this result implies; promising / dead directions>
```

## Orchestrator responsibilities each step
1. Read the upstream agent's `handoff.md` + `status.md`.
2. **Validate** the handoff against the schema above and against `evaluation_contract.md`
   (e.g. reject a Research proposal whose success criteria don't include the mandatory gates;
   reject an Evaluation ADOPT that skipped a gate).
3. Update `pipeline_state.json` (phase, active agent).
4. Dispatch the next agent (spawn subagent with its `AGENT.md` + the relevant handoff).
5. On iteration close, append to `orchestrator/iteration_log.md` and update
   `metrics_registry.json`; if ADOPT, update `current_best.md` and `strategy_state.md`.
