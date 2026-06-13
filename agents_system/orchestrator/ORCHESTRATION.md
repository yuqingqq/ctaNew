# Orchestration (the human-in-the-loop driver = me, Claude, the main thread)

I own the loop. I spawn each agent as a subagent (Agent tool) with its `AGENT.md` + the upstream
handoff, validate every handoff, maintain shared state, and decide continue/stop.

## One iteration

```
for each iteration iter-NNN:
  1. RESEARCH:  spawn research agent → reads current_best.md + last evaluation report
                → writes research/handoff.md (hypothesis + spec + pre-registered criteria)
     VALIDATE:  criteria reference the mandatory gates? hypothesis falsifiable? not a re-tread
                of a known-dead direction (baseline.md hard truths)? else send back.
  2. IMPLEMENT: spawn implementation agent → reads research/handoff.md → writes script + handoff
     VALIDATE:  script exists, runs, self-check present.
  3. REVIEW LOOP (repeat until clean):
       a. spawn review agent → reads impl handoff + code → review/handoff.md (PASS | FIX-NEEDED)
       b. if FIX-NEEDED → spawn implementation agent with the review's numbered fixes →
          it edits the SAME script, updates implementation/handoff.md → go back to (a).
       c. if PASS → proceed to EVALUATE.
     The loop continues until Review returns PASS (no remaining issues) — Implementation fixes,
     Review re-reviews, every round. Safety valve: if it fails to converge after 4 rounds (e.g.
     the two disagree or the spec is unbuildable), escalate to the human rather than loop forever.
     Each round increments a fix-round counter logged in review/reviews/iter-NNN.md.
  4. EVALUATE:  spawn evaluation agent → runs backtest, applies evaluation_contract → handoff
                (ADOPT|REJECT + metrics + gate results + insights)
     VALIDATE:  every applicable gate reported with a number; ADOPT only if contract satisfied.
  5. CLOSE:     append iteration_log.md; update metrics_registry.json;
                if ADOPT → update current_best.md + strategy_state.md.
     GATE (if mode=gated): pause, surface summary to human, await go-ahead.
```

## Spawning agents
Use the Agent tool (`general-purpose` or `claude`). Give the subagent: (a) its `AGENT.md` path to
read, (b) `shared/` paths, (c) the specific upstream `handoff.md`. The research agent needs
WebSearch/WebFetch for SOTA/literature. Run independent work concurrently only when truly independent
(usually the loop is sequential). Keep each subagent focused on ONE iteration's task.

## Validation duties (reject bad handoffs early)
- Research proposal must: be ONE change, be falsifiable, pre-register against the gates, and not
  repropose a known-dead idea unchanged.
- Implementation must: be PIT-correct by self-check, runnable, not touch baseline scripts/preds.
- Review must: actually audit look-ahead with line references, not rubber-stamp.
- Evaluation must: report ALL applicable gates with numbers; ADOPT only if contract holds. I
  independently sanity-check the headline numbers before accepting an ADOPT.

## Stop conditions (autonomous mode)
- Reached the iteration budget (default 8), OR
- No ADOPT in the last 4 iterations AND research is recycling ideas (diminishing returns), OR
- An agent is `blocked` needing human/data/paid-feed decisions, OR
- A safety/scope issue (would need live-trading or unapproved paid data).

## Health checks each iteration
- Is research proposing genuinely new directions or recycling? (track in iteration_log)
- Are rejects clustering on the same gate? (signals a systematic issue to surface to human)
- Is wall-clock per iteration blowing up? (cache, shrink scope)
- Did current_best actually move, or are we plateauing? (if plateau, raise to human)

## My posture
Skeptical reviewer, not cheerleader. A REJECT that kills a bad idea cheaply is a *good* iteration.
The goal is a real, honestly-validated Calmar improvement — not a high in-sample number.
