# Front-pull result: implications for the recovery-model experiment

Read-only integration of the separate front-pull study. Its saved report is as
of **2026-11-13T03:54:02Z**, across the nine consumed dates Oct23/24/25/26/27/31
and Nov1/2/3, covering 5,184 window-sides in the first minute. It reports
591,868 aligned prediction rows and 278,542 undefined-probability steps out of
6,220,800 steps. Undefined predictions keep the order and do not block entry.
These are that study's populations, not the recovery experiment's action rows.

Sources inspected directly:

- [Saved report](../reviews/front_pull_20261113/front_pull.json), fields
  `as_of`, `days`, `readings`, `stat`, `pooled` and `intervals`.
- [Study description and results](20261113_front_pull_results.md).
- Main-worktree `research/pf_front_pull.py`, specifically `fp_fit`, `fp_report`
  and its declared setup/limitations. Its bytes match commit `df320ba3`.

The data and review copies of `front_pull.json` have identical SHA-256
`a2da211be744a50901e0229c59c8f424227e5ed1c30ecc37e4cd6887f187b38d`.
The inspected source SHA is
`42d832d8922b6eb5c7af4363d075b12c39851b29eb0b9f732db6d720ff0d76d6`;
the prose report SHA is
`5d160f23cb3b5fffab6916dbffc7970f5c60657f523918b4cc7feedd347b49e4`.
This review reads their existing results; it does not independently rebuild
their simulations or recompute their statistical tests.

The saved decision readings are negative for policy improvement. `R380` is
`PULL-NOT-SHOWN`: FPQ beats the random-pull p95 on all nine dates under both
reported primary clocks, but beats the guard trigger on zero dates. `R381`
records zero of six comparisons better than J-v3, and `R383` zero of three
positive per-share readings. `R382` reports a limited benefit from tolerating
more predicted risk at the front of the queue. These readings do not establish
a replacement for the guard or for our current frozen candidate.

One prose typo matters: the report says FPQ “loses to the guard trigger on
0 of 9 days.” The machine field is **zero days better**, not zero days worse;
the reported negative paired differences agree with that interpretation.
The other owner's files were left unchanged.

Three distinctions limit what carries into this goal:

| Distinction | What the source actually does | Implication |
|---|---|---|
| Forecast target and population | Fits whether the best level worsens in the delayed future interval, on regular MOVE rows filtered to levels still present at t + 24 ms | This is a future-survival-conditioned level event, not unconditional value of a decision-time order or the benefit of cancelling it. |
| Chronology and inputs | Fits each held-out date using the other eight dates; the input `v` comes from guard models that included the held-out day | Some training dates are later than the evaluated date, and an inherited input has target-date exposure. The reported discrimination is exploratory, not an earlier-only or forward validation result. |
| Execution and comparator | Uses S4's rest lane with one order per side, J-v3 as reference, and queue rules M/U | This differs from our full frozen engine and its inventory/path constraints. The reported sign changes under the front-of-queue assumption do not identify real queue placement or a live profit level. |

For this goal, the result supports keeping **decision value** as the target:
avoidable adverse fills, forgone beneficial fills, retained queue value,
inventory consequences and the re-entry path all matter. A good ranking of
future level moves can still induce poor withdrawal decisions. This is an
interpretation of the target/policy mismatch, not a measured causal explanation
of the front-pull losses.

Our running recovery experiment already keeps direct and component value
comparators, explicit recovery, chronological fits, whole-path value/risk
checks and participation gates. Its own performance is still unproven, and its
neutral-order recovery proxy is not identified fresh-entry value. No front-pull
score, threshold or queue rule is imported; no new variant is selected and no
frozen criterion changes. Live queue/fill/cancel calibration remains unresolved.
