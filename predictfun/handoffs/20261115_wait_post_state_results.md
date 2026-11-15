# Repaired wait/post state, targets and cleanup — completed results

User: add missing inputs, review the model and improve it; then diagnose poor
first-day prediction and whether noise filtering is needed.

Completed asof2026-11-15T02:59:00Z. Five consumed evaluation days Oct27/31 and
Nov1/2/3,9871 eligible action probes (4990bid/4881ask),1434 market windows.
Seven rebuilt label days Oct25/26/27/31,Nov1/2/3:2010 windows,13911 eligible
probes. Statuses:81584 not sampled,344 probe not reached,251 generations in the
six sealed-lookback windows excluded. Per-day source untimed-trade exclusions
are preserved in main/result.json; no silent row removal. Gross local simulated
value under neutral KEEP, not a full strategy replay or forward validation.

## Changes and verification

Client-known per-rung prices, remaining quantities, received fills, age, pending
placement/cancellation, empty age/generation and causal public queue estimates
now reach both labels and runtime inference. Snapshot exposure reconciles.
Public market liquidity scaling and market-level age are separate from own-order
age. Unknown queue levels remain missing. First-order and remaining-portfolio
contributions sum exactly to the original economic objective; no-fill actions
remain in training. Bid/ask heads are separate; no fixed retry timer.

Review repaired an unavailable start-price feature in candidates AND the frozen
continuation. Rebuilding changed994of13911 eligible labels. Six Oct31 windows
with sealed-day volatility lookback are explicitly excluded. This supersedes
the old entry-input/label contract; old results remain as provenance, not directly
comparable acceptance evidence.

19 distinct synthetic checks passed, including positive controls and known-bad
refusals;2010 historical window-level future-feature checks passed.120 fitted
artifacts and all row/fit receipts were verified. All three supervised jobs
exited0; the earlier size queue resumed. Source99173056/freeze42f61bf1;
noise20ba3773/068a4382; cleanup70217846/c9fc924e. The cleanup freeze's copied
multiplicity field is superseded by clean_freeze_v2.json; source/input bindings
and the two named candidates are unchanged, correction before fits.

## Performance

Weighted pooled R2 versus each fold's earlier-training mean. Negative means
worse error than that baseline. Same raw labels and inverse inclusion weights.

| Candidate | Absolute POST R2 | POST-minus-WAIT R2 |
|---|---:|---:|
| BASE | -0.0739 | -0.0450 |
| LADDER | -0.0669 | -0.0528 |
| COMPONENTS | -0.0693 | -0.0596 |
| CLEAN_TREE | -0.0830 | -0.0512 |
| CLEAN_RIDGE | -0.1112 | -0.0835 |

No full local economic gate passes:0/10 side/candidate tests across5 candidates,
500 draws each matched on day,side,hour,rung,generation group,action count and
weighted mass. Every selected absolute-POST mean is negative. The cleanup tree
and ridge variants do not improve aggregate prediction error. LADDER modestly
reduces absolute-value error relative to BASE, while its timing error is worse.
This does not establish useful entry timing or portfolio profitability.

## What the noise investigation says

The separate first-fold diagnosis (training4040 probes onOct25/26; test1992 on
Oct27; asof02:35:02Z) finds sparse/heavy-tailed timing targets and overfitting.
Training external POST variance is3.48x/6.00x the first-order variance (bid/ask).
Failures persist on ordinary-outcome and fresher-data subsets and after oracle
intercept correction. These are descriptive partitions, not deployable filters,
and do not prove an irreducible noise floor or rule out all robust estimators.

We tested train-only economic normalization, public queue scaling, constant/
duplicate removal and fixed ridge regularization, preserving all losses.160 raw
compact fields expand to318-332 columns with missing/extreme flags; this was not
a smaller encoded feature matrix than the284-column raw ladder representation.
Both cleanup candidates fail. Do not label real adverse outcomes as disposable
noise or retain only filled orders.

The main unresolved gap is the target/continuation: entry still predicts a
terminal portfolio effect under neutral KEEP, while the intended strategy uses
a selective keeper. Decomposition narrows attribution, not the valuation
horizon. Relabel under the actual frozen keeper, learn near-term order influence
separately, and expand later-generation/joint-waiting coverage. Reweighting or
more columns cannot create missing examples. The current evidence does not
justify another claim that preprocessing alone makes this model good.

No model promoted. No combined keeper replay, no portfolio acceptance and no
untouched-day validation. Full evidence: reviews/wait_post_state_20261115/current.json.
