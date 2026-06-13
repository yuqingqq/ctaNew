# Recursive Strategy-Optimization Agents System

A closed-loop, multi-agent system that recursively optimizes the crypto cross-sectional
alpha strategy with the explicit goal of **lowering drawdown and improving Sharpe** —
while refusing changes that only *look* good in-sample.

## The loop

```
            ┌──────────────────────────────────────────────────────────────┐
            │                      ORCHESTRATOR (you)                        │
            │  tracks pipeline_state.json, validates handoffs, runs the loop │
            └──────────────────────────────────────────────────────────────┘
                 │            │              │              │
                 ▼            ▼              ▼              ▼
          ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐
          │ RESEARCH │→ │ IMPLEMENT │→ │  REVIEW  │→ │ EVALUATION │┐
          └──────────┘  └───────────┘  └──────────┘  └────────────┘│
                ▲                            │ (fix)                │
                │                            └──────────────────────┘
                └──── evaluation report (ADOPT/REJECT + insights) ◄──┘
```

1. **Research** reads `shared/current_best.md` + the latest evaluation report → proposes ONE
   concrete, testable optimization (data-driven + literature) with a falsifiable hypothesis
   and pre-registered success criteria. Writes `research/handoff.md`.
2. **Implementation** turns the proposal into efficient, PIT-correct code (a new script under
   `research/convexity_portable_2026-05-20/scripts/`). Writes `implementation/handoff.md`.
3. **Review** audits the code for look-ahead, bugs, and faithfulness to the spec. PASS → eval;
   FIX → back to implementation. Writes `review/handoff.md`.
4. **Evaluation** runs the backtest, applies the **mandatory evaluation contract**
   (`shared/evaluation_contract.md`), and returns ADOPT/REJECT + metrics + an insights note
   for the next research cycle. Writes `evaluation/handoff.md`, updates `metrics_registry.json`,
   and (if ADOPT) `current_best.md`.

The orchestrator drives one **iteration** = one full pass, logs it to
`orchestrator/iteration_log.md`, and decides whether to continue.

## Why the evaluation contract is the heart of this

Prior research on this strategy repeatedly produced changes that beat the baseline in-sample
but **failed honest out-of-sample validation** (cost-margin swap, decay-weighted sleeves,
the sign-flip layer). The contract (`shared/evaluation_contract.md`) makes ADOPT conditional
on nested-OOS, matched placebo, per-fold and per-universe robustness, and paired CIs. **No
change is adopted on raw in-sample Sharpe alone.**

## Operating modes

- **Gated** (default): orchestrator pauses after each Evaluation handoff for human review.
- **Autonomous**: orchestrator runs N iterations back-to-back, stopping on the stop conditions
  in `orchestrator/ORCHESTRATION.md`.

## Folder map

| folder | owner | key files |
|---|---|---|
| `orchestrator/` | you | `pipeline_state.json`, `iteration_log.md`, `ORCHESTRATION.md` |
| `shared/` | all (read); orchestrator/eval (write) | `baseline.md`, `current_best.md`, `evaluation_contract.md`, `strategy_state.md`, `conventions.md`, `metrics_registry.json` |
| `research/` | research agent | `AGENT.md`, `status.md`, `handoff.md`, `insights/` |
| `implementation/` | implementation agent | `AGENT.md`, `status.md`, `handoff.md` |
| `review/` | review agent | `AGENT.md`, `status.md`, `handoff.md`, `reviews/` |
| `evaluation/` | evaluation agent | `AGENT.md`, `status.md`, `handoff.md`, `reports/` |

See `PROTOCOL.md` for the exact handoff/status schemas.
