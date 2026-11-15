# Entry data and model audit — superseding qualification

The entry data is not cleared. A concrete shared fair-price defect was discovered
by the stop-model session and corrected in that model's builder, but the completed
entry study still uses the affected step-5 inputs. This audit supersedes the
interpretation and next-stage priority of the earlier completion receipt; it does
not rewrite its frozen results or claim to measure the correction's entry impact.

## Confirmed data issues

**Hourly fair-price bias.** In `pf_step5_features_rl.py`, each Binance hourly pass
prepends its carry and selects relay observations from the combined interval.
Relay prints in overlapping intervals are appended twice to the basis sample.
This biases the trailing median around the hourly boundary. It is a biased
statistic, not evidence of look-ahead. `pf_stop_model.fair_fix` corrects the basis
and fair price within that model's rows builder. It does not rewrite shared
`rows_rl_d2_pk.parquet` or the frozen entry artifacts.

The entry loader consumes `basis_bps` and `gap` directly. Entry information and
representation code then derives fair changes, sensitivity and candidate edge.
The bootstrap continuation also consumes the original market inputs through
`safe_inputs`; that function masks an unknown start price but does not correct
the hourly basis. Consequently labels may change when corrected inputs alter
future entry choices. The stop model's unchanged labels do not prove entry
labels would be unchanged. No entry-specific effect size has been measured.

**Insufficient pre-window book history.** The frozen entry loader keeps only five
seconds before each market window; liquidity normalization requests thirty.
Availability flags expose this partial support but do not restore observations.
The root loader now has an opt-in pre-roll parameter whose default remains five
seconds, and the stop model opts into thirty. The frozen entry build does neither.
A correction needs an action/engine reconciliation, because extra history can
also affect state initialized from early frames; it is not automatically only a
feature rescaling.

All frozen entry source hashes were checked against `freeze_v2.json` and match.
That confirms precisely which pipeline ran. Exact reproduction of old targets
and inputs establishes preservation, not upstream correctness. The stop model's
raw-input recomputation receipt was inspected separately; its pass applies to
its corrected inputs, not to entry. Its effect sizes are not transferred here.

## Target and model issues that remain

- Entry targets still use neutral KEEP rather than the intended selective keeper.
  Terminal value also includes future entry replacements and inventory/cap effects
  from other slots. This is a task-design limitation, not a new algebraic sign bug.
- The saved development study establishes no economic entry skill on its existing
  pipeline. Poor generalization is observed, but signal weakness, overfitting and
  data defects have not been separated by a corrected controlled experiment.
- Queue position is unobserved; fill timing and clock calibration retain their
  declared assumptions. Larger-size sensitivities do not validate the simulator.
- Denser sampling repaired later-action coverage within the same consumed dates
  and neutral path. New regimes, jointly waiting states and untouched validation
  remain absent. This study did not isolate learner capacity or regularization.

The completed loader boundary filtering, inclusion weights, chronological folds
and action reconciliation still stand within their declared scope. They do not
close the newly identified upstream input defect.

## Revised order of work

Freeze a separate correction, propagate the fair-price and history repairs to
entry, and recompute dependent inputs consistently. Include explicit hourly
boundary cases, ordinary-window controls and planted failures in raw-input
validation. Reconcile engine states and continuation decisions, then rebuild any
affected labels. Preserve the prior pipeline and declare whether bootstrap
weights stay fixed or are rebuilt; do not silently change both data and policy.
Only then revisit the same performance comparison, keeper-aligned targets and
controlled model changes. No rebuild or new training has been launched by this
audit, and no correction is claimed to explain all of the poor performance.

Machine-readable qualification: `reviews/entry_repair_20261115/qualification_v3.json`.
The mutable `current.json` points to this qualification; frozen completion and
result artifacts remain intact.
