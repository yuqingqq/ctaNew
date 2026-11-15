# Entry information: measured contribution to wait/post targets

Completed **2026-11-15 03:52:15 UTC**. This study inventories available decision
information and measures predictive contribution. It changes no entry or keeper
policy. The model still chooses only wait/post; cancel/keep is separate.

Evaluation: **9,871 eligible action probes, 4,990 bid and 4,881 ask, across 1,434
windows on five consumed dates: Oct 27/31 and Nov 1/2/3**. Training uses only
earlier label days. Seven label days contain 13,911 probes / 2,010 windows.
No untouched-day validation, portfolio replay, policy promotion or causal claim.

The inventory contains 325 named columns: the original 284 plus 41 predeclared
transforms. Aliases and deterministic transforms are not independent sources.
The full field meanings and limitations are in
`20261115_entry_information_inventory.md`; the complete feature and family
scorecards are under `reviews/entry_information_20261115/`.

## Primary target: POST now minus WAIT

Each family is added separately to the same STATE baseline, which includes
candidate/ladder state, current book, fair-value gap, freshness, best-level age
and contract context. Separate bid/ask models; same pinned learner and weights.
Positive numbers below mean a percentage reduction in weighted squared error
relative to STATE. They are **not R2 percentage-point changes**.

| Added family | Bid error reduction | Bid days better | Ask error reduction | Ask days better |
|---|---:|---:|---:|---:|
| Binance return history | +0.71% | 3/5 | -0.98% | 1/5 |
| Binance pressure/activity | -0.90% | 3/5 | -0.35% | 3/5 |
| Predict midprice history/pending flow | +1.25% | 3/5 | -0.88% | 1/5 |
| Recovered upstream dynamics | -0.26% | 2/5 | -0.82% | 0/5 |
| Book price spacing/tick depth | -0.05% | 2/5 | -0.29% | 1/5 |
| Best-price and candidate-level dynamics | +0.73% | 4/5 | -0.16% | 2/5 |
| Implied fair-value dynamics | +0.37% | 4/5 | -0.83% | 0/5 |
| All information together | -0.26% | 3/5 | -0.44% | 2/5 |

No family improves aggregate ask timing error over STATE. Some modest bid
improvements exist, but every family model still has negative R2 against the
prior training mean. STATE itself has timing R2 -0.09255 bid / -0.02331 ask;
ALL has -0.09534 / -0.02780. The four omitted upstream fields were real omissions,
but restoring them is not shown to fix the timing problem. Combining everything
does not improve timing, despite adding information.

## Individual feature screen

Every feature has train/test missingness, weighted rank and Pearson correlation,
train-defined quintile profiles, and prediction error from training-bin means.
Results are kept for every fold, side and target. Missingness gets its own bin;
an apparent benefit may partly reflect availability or associated state.

Examples from the descriptive screen, selected after inspecting all results:

- **Bid: opposite-side gap to the third displayed level** has positive test-day
  rank correlation on all five days, ranging +0.050 to +0.071, mean +0.061.
  Its simple binned predictor has R2 +0.00134 versus the prior mean. The full
  geometry family adds no aggregate timing value beyond STATE. This illustrates
  a stable marginal relationship that does not establish incremental skill.
- **Ask: Binance 100 ms return** has mean test-day rank correlation -0.078,
  ranging -0.155 to +0.072; its binned predictor has R2 +0.00417. Direction is
  unstable across days, and the return family worsens the richer model's timing
  error. This is a research lead, not a confirmed signal.
- **Candidate-price shrink/growth history** offers small univariate improvements
  in some cells, but signs vary and roughly one fifth of actions lack a complete
  visible 250 ms candidate-price segment. Unknown depth was not filled with zero.

The examples are not validated selections. There are 1,300 feature/side/target
screens, including aliases; no marginal p-values or causal claims. Predictive
relationships are weak, often shared with state variables, and learner-dependent.

## Absolute POST value and selection checks

For absolute POST, geometry reduces bid error by 1.87% relative to STATE;
level dynamics reduces ask error by 1.58%. Both improve four of five days in
their respective cells. Every arm/side still loses to its prior-mean baseline
on pooled absolute-POST error. Better absolute-value error is not automatically
better wait/post timing.

**0/18 local economic checks pass**, across nine diagnostic arms and two sides.
Each has 500 controls matched on day, side, hour, rung, initial/later generation,
action count and inverse-sampling mass. Monte Carlo p-values are Bonferroni
adjusted across all 18 tests. Every selected absolute-POST mean is negative.
Some ask selections have positive POST-minus-WAIT value, but that does not make
POST itself profitable or establish skill over the matched controls.

These labels retain the fixed neutral-KEEP continuation. They measure standalone
entry behavior and do not credit the separate cancellation model's benefits.
Raw losses, no-fill outcomes and all eligible actions are retained.

## Interpretation and remaining limits

Market histories contain some measurable relationships, particularly weak bid
timing contributions, but this study does not show a strong entry signal. The
poor model is not explained by the four omitted fields alone. Additional inputs
can improve one target or side while harming another; a single global feature
ranking would hide this. Adding the entire catalog is not supported by the
current comparison.

Any next feature selection or simpler model is a new development candidate.
The modest bid contributions from venue history and level dynamics merit focused
follow-up, while ask timing needs separate investigation. Do not select a winner
from these dates and call them validation. The limited action-state coverage,
long-horizon continuation target and assumed feed/queue clocks remain limits on
what these measurements establish.

## Verification and artifacts

- Six synthetic checks pass, including planted signal/anti-signal, missing-value
  behavior, future invariance, exact timestamps, geometry and bad-null refusal.
- 2,010 historical future-data checks pass; original market inputs reconcile to
  frozen action rows. No eligible action was dropped during the new feature join.
- 180 fitted model files, all source/label/raw-input bindings and output receipts
  verified. Peak worker RSS remained below the 2 GiB guard.
- Source `3109972a`, missing-value checker correction `c6621973`; authoritative
  freeze `1acaf302` / `freeze_v2.json`. Candidate definitions and targets were
  unchanged by the checker correction, made before any historical build or fit.
- The first launch used a Python without NumPy; the second was stopped while
  queued for that checker correction. Both attempts are preserved and contain
  no historical builds/fits. The completed third attempt exited 0. The older
  size supervisor resumed, and the other research session was untouched.

Current receipt: `reviews/entry_information_20261115/current.json`.
Human-readable inventory: `handoffs/20261115_entry_information_inventory.md`.
All feature scores: `reviews/entry_information_20261115/feature_summary.csv`.
All group scores: `reviews/entry_information_20261115/group_summary.csv`.
Full per-day profiles and predictions (large, intentionally outside git):
`data/predict_fun/research/entry_information_20261115_v3/fit_*/associations.json`
and `report/predictions.parquet`, bound by the committed receipts.
