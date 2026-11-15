# Entry representation: preserve economically different conditions

User: check the data and make the representation sensible so different
conditions are separable. This follows the completed normalization audit.
WAIT/POST only; cancel/keep stays separate. The component/target-decomposition
study remains unstarted. Do not alter targets to obtain a favorable score.

Three predeclared arms, two new development candidates:

- RAW: the saved 325-input ALL model, with exact reproduction of saved
  predictions. No baseline refit or hyperparameter change.
- RELATIVE: replace public size coordinates with ratios to the existing causal
  joint liquidity scale. Convert shares to lots where required. Preserve both
  absolute log-liquidity fields, absolute top-size logs, own order quantities,
  inventory, exposure, all price/time inputs and all other columns. Each replaced
  quantity is recoverable from its ratio, recorded divisor and preserved scale.
  No empirical duplicate deletion, clipping, imputation or fitted scaling.
- CONDITION: RELATIVE plus actual support of the existing liquidity history
  (span, available duration, frame count, largest inter-frame gap, full-history
  status, last-frame age); side-aligned returns in known-volatility units;
  candidate fair edge in ticks/spreads; recent Binance return translated to a
  local fair-price response using the existing sensitivity; edge plus that
  response; candidate-history availability and shrink relative to currently
  shown size. Zero shown size has its own flag; undefined ratios stay missing.
  Fair transforms stay missing until the start reference is known. This is a
  predeclared bundle, not an individual attribution experiment.

The fair response is an assumption-based transform of recent information, not
an observed future movement or a second fitted forecast. Absolute scale is
retained precisely because more similar marginal distributions are not the
same as more useful conditional information. The nominal 30-second liquidity
estimate stays frame-weighted and unchanged; this study exposes its support
rather than replacing it or pretending short histories are complete.

Data: existing 13,911 eligible probes on Oct 25/26/27/31, Nov 1/2/3. Five consumed
chronological evaluation days Oct 27/31, Nov 1/2/3. Build support at the same
exact action times from frozen received-book inputs. Require exact equality
of every original column/label and full action-key coverage. Preserve source
exclusions and statuses. No outcome filtering, new action sampling or targets.
Every window gets a future-perturbation check and a planted scale-mismatch
refusal. The scale itself must reconcile to each frozen action row.

Evaluation: separate bid/ask; pinned original LightGBM parameters/rounds; inverse
inclusion weights 2/32 in fitting and scoring. Forty new fits: 2 arms x 2 sides x
2 targets x 5 folds. Twenty saved RAW artifacts are controls. Primary target
POST-minus-WAIT (`delta`), secondary POST-minus-NEVER (`qpost`). Report weighted
SSE/R2 versus RAW, prior training mean and price/rung baseline, every fold and
pooled. CONDITION also compared with RELATIVE. Retain all arm results.

Conditional diagnostics: initial/later generation, full/partial history,
known/unknown reference and candidate history, low/high liquidity, volatility
and fair distance. Low/high cuts are training-only weighted medians; test
missingness is its own status. Report n, effective n, raw means and prediction
errors for every cell; these groups overlap and are not independent tests.
No outcome-conditioned bins, significance claim or intervals on these cells.

Economic diagnostic declared before results: 500 controls per arm/side matched
on day, side, hour, rung, initial/later generation, selected action count and
sampling weight mass. Positive post prediction and nonnegative timing
prediction select an action. Six arm/side comparisons; Bonferroni across six.
A local pass also requires positive observed absolute POST and timing value.
These diagnostics cannot qualify a portfolio policy or forward validation.

Seven synthetic checks: lossless scaling with retained liquidity/risk regimes,
side/fair-response/volatility sensitivity, missing versus quiet/zero states,
actual history/future invariance and bad clocks, weighted training-only regime
cuts, economic selection positive/anti-signal/insufficient-draw refusal, and a
planted conditional signal the pinned learner must distinguish. Freeze source,
row/raw inputs, saved baseline models and predictions before historical work.

All dates are already consumed development. No winner is promoted; no claim of
new market information or causality. Terminal neutral-KEEP continuation and
scarce later-generation/joint-waiting examples remain known limits. Jobs run
serially in the Codex light slot with 2 GiB RSS self-guard; the older size
supervisor yields after its current worker and resumes in the runner's finally.
