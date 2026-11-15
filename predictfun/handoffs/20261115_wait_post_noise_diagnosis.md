# First-fold noise diagnosis — 2026-11-15

User: first day still poor; why, is there too much noise, should we filter it?
Post hoc instrument20ba3773, freeze068a4382. No candidate was changed or added.
Completed asof2026-11-15T02:35:02Z. Training Oct25/26:4040 eligible probes
(2065bid/1975ask); evaluation Oct27:1992 (999bid/993ask). All dates consumed.
Source statuses are retained in the raw receipt. Three diagnostic tests with
positive controls and known-bad refusals passed; saved predictions reconstructed
exactly from the saved model files. Full three-arm comparison resumed afterward.

## What the numbers distinguish

| Observation | Interpretation |
|---|---|
| LADDER POST training R2 +.481 bid / +.482 ask; test -.070 / -.104 versus prior weighted means | Overfitting / failed transfer. Added information is not automatically usable signal. |
| Training timing labels zero in72.1%bid/67.6%ask of raw actions. Largest1% of actions by absolute timing target contributes35.1%/42.2% of weighted squared-target magnitude | Sparse, tail-heavy targets. This is not residual-loss concentration, and the tail may contain essential real risk. |
| Training absolute-POST variance: external component3.48x/6.00x local-order variance; test2.73x/8.24x | Much of what is being predicted concerns later replacements and the rest of the portfolio. Splitting heads did not shorten the60.2s valuation horizon. |
| Weight effective training n566bid/523ask across only2days;284 candidate input columns | Limited representative information. Weight effective n describes weight concentration, not independent observations; actions also share market windows. |
|17/11 constant-or-all-missing columns and40/15 exact duplicate pairs (bid/ask training matrices; counts overlap) | Clean up redundant representation, but these counts do not prove redundancy caused the losses. LightGBM already ignores constant columns. |
| Oracle removal of test mean prediction error still leaves LADDER POST R2 -.020/-.056 and timing -.113/-.011 | A simple calibration intercept is insufficient. This uses test outcomes and is not a deployable fix. |
| Excluding outcomes beyond the training absolute-target99th percentile still gives negative R2 on both heads/sides | Extreme outcomes alone do not explain failure. This is an outcome-based diagnostic, not an admissible training filter. |
| Within training95th-percentile age bounds, LADDER POST R2 -.073bid/-.110ask and timing -.117/-.013 | The failure persists in fresher observations. Stale-row removal alone is not supported as a remedy. |

Feature relationships are also weak and sometimes change sign between days.
For example, basis_bps has high tree gain but unstable marginal relationships.
This is diagnostic evidence, not proof of conditional concept drift or a license
to select features using the evaluation day.

## Answer and next action

There is a noise/variance problem, but a blanket filter is not the fix shown by
this evidence. Do not remove large adverse outcomes or retain only filled orders.
Use train-only normalization, remove duplicate/constant representation, and test
compact regularized features on all action outcomes. Earlier compact-feature
experiments also failed their economic gates, so preprocessing alone should not
be presented as sufficient.

The larger unresolved target issue remains: the labels value an entry under
neutral KEEP through the terminal mark, while the intended strategy uses a
selective keeper. Relabel under the actual frozen keeper once available and
separate near-term order influence from longer-horizon portfolio interactions.
Expand representative later-generation/joint-waiting data; weighting cannot
create examples. Preserve the portfolio criterion so displaced fills and queue
opportunity costs are not silently dropped.

The fixed full comparison continues. One evaluation day cannot establish an
irreducible noise floor or validate any proposed noise filter. No intervals,
portfolio acceptance, model promotion or forward claim.

Evidence: reviews/wait_post_state_20261115/noise/summary.json and result.json.
