# ASK-RESERVATION-GUARD-v1 — combined review

2026-11-21. Pinned target **84a4f999eabdfbe1e756b3f59213a641dc343a4e**.
Range: f8c5e579f15be9cfbf3e525567de55cbfa13780d..84a4f999eabdfbe1e756b3f59213a641dc343a4e.
Reviewed ee632bc2's freeze and 84a4f999's results in detached worktree
`/tmp/pf-lg-review-84a4f999`. Earlier frozen sources/receipts are unchanged;
the results commit changes no research source after this freeze.

**Accept the paired experiment as a consumed-day diagnostic. No blocking
finding requires a rerun. Do not promote the reservation guard.** It improves
the aggregate result, but both arms lose heavily, one day worsens and the
declared local selection screen does not pass. The three prior findings are
closed for the new outputs. Bundle the remaining issues below with the next
experiment; no separate fix-only cycle is warranted.

## Economic result and its meaning

Independently reconciled all 19 pairs, 38 arm ledgers and reported markouts
to their stored fills/mids. All 96 original statuses remain: 19 complete,
54 missing endpoint, 23 stale future book. Complete counts are 7/5/5/2 on
consumed Oct 24–27; this is not untouched validation or four complete days.

| Measure | Baseline | Guard | Change |
|---|---:|---:|---:|
| END cash + marked inventory, c | -209,557.709 | -176,114.061 | +33,443.648 |
| Filled shares | 96,991.071 | 89,771.013 | -7,220.058 |
| +1s markout, common time cutoff, c | -203,083.922 | -168,609.206 | +34,474.716 |
| +1s per valid filled share, c/share | -2.102 | -1.886 | +0.216 |
| Both-side active time | 40.70% | 31.68% | -9.02 pp |

Paired END changes reproduce: +23,889.443 / +10,662.519 / -3,493.829 /
+2,385.515 c, with 15 markets better and 4 worse. The +1s per-share measure
improves on all four days, and on asks each day; bids worsen on three days.
The gain therefore cannot be described simply as “the same fills, fewer
shares,” but the data do not establish incremental predictive selection.

The independently reproduced local control selects 517 of 2,413 uncensored
ask live states, retaining six censored statuses (three selected). There are
104 movable strata / 345 movable selected states. Selected mean is -58.026 c,
null mean -51.440 c, p05 -58.714 c and percentile 7.4%, from the declared 500
draws. The frozen mean-per-state criterion fails. Its `NO_SELECTION_BEYOND_EXPOSURE`
name means **not demonstrated by this screen**, not proof of no signal. Do not
switch after seeing results to the marginally different per-share tail test.
This local control is not a null distribution for the feedback policy's P&L.

The worst regression, Oct 26 market **3838531**, explains why smaller inventory
does not by itself solve the problem. Filled quantity rises 2,224.162 shares;
mean absolute inventory falls 16.671 shares and peak falls 23.283 shares, yet
END deteriorates **10,779.044 c**. The exact accounting decomposition is:

| Contribution to guard minus baseline END | c |
|---|---:|
| +1s markouts on eligible fills | -3,869.287 |
| Movement from each eligible fill's +1s mark to END | -7,212.705 |
| Late / missing-horizon fills valued at END | +302.948 |
| Total | -10,779.044 |

This is an ex-post accounting identity, not a causal attribution to a single
inventory command or a new predictive target. Both sides trade more on this
path. The largest gain (market 3697902) also has a large later-mark component;
END changes must not be read as pure short-horizon adverse-selection skill.
Across all 19 markets the +1s improvement is +34,474.716 c, offset by -424.880 c
after +1s and -606.187 c in the late-fill term. The residual loss is still
dominated in aggregate by adverse movement already visible at +1s.

## Consolidated findings

All are **NONBLOCKING for the next bounded experiment** under the independently
verified evidence here. No current economic number is rejected by this review.

1. **P2 — The checker still does not independently reconcile the new markout
   and exposure metrics it uses for conclusions.**
   `reviews/ladder_guard_20261121/guard_checks.py:134-147` checks arm value,
   fill count and total quantity, then calculates the per-share reading from
   stored `arms.csv` fields. It does not derive those markout fields from the
   ledger's fills/mids. `F_selection` at lines 148–150 compares some numerical
   terms and two stored answers, rather than independently deriving every
   answer/interpretation from pinned draw settings. The supplement exercises
   individual verdict corruptions but not this boundary.
   **Reproduction:** changing a guard `all_cut_mo1s` by +10,000 and updating its
   downstream reading still passes all 21 checks (`evidence.json`,
   `coordinated_markout_corruption_accepted`). No real output was changed.
   **Verified workaround:** this review rederived every reported side's edge,
   +50/+100/+1000ms sums, cutoff quantities/censors, good/bad contributions,
   exact uptime and current input reservations, and reproduced the full
   selection result and predicates. All agree. In the next checker, add one
   ledger-based reconciliation pass and this coordinated known-bad case;
   derive the draw count/seed/answer from the frozen manifest. Do not spend
   another round merely increasing the number of superficial falsifiers.

2. **P2 — Queue-cost attribution remains unmeasured in the guard arm.**
   `pf_ladder_guard.py:115-120` records `Observer` arrays, final orders,
   fill mids and guard logs; unlike `pf_ladder_rest.run_ladder:403-419`, it
   does not save initial ahead or intermediate queue observations. Final
   `ahead_final` and lifetime alone cannot tell how much earned queue priority
   was surrendered at a guard pull or recovered on re-entry. This does not
   invalidate net P&L, which already includes the simulator's queue feedback,
   but it prevents the planned explanation of that cost. Record queue state
   at submission/first live review/guard pull and re-entry for the next
   candidate, with receipt-known estimates and venue diagnostics explicitly
   separated. Keep the same engine queue model. No rerun solely to fill this
   diagnostic gap is required.

3. **P3 — The selftest header overstates the operational guard coverage.**
   `pf_ladder_guard_selftest.py:13-14` names projection, changed-rule and
   write-once refusals, but group 7 at lines 142–153 tests pre-freeze refusal
   and conditionally skips the selection case once committed. It does not
   exercise the other named cases. Actual pilot/projection/freeze records
   reconcile, so this is not a failed current launch gate. Correct the claimed
   coverage or add isolated positive/negative fixtures when touching the next
   runner; defer otherwise.

Two small interpretation details need no repair-only cycle. Inventory averages
at `pf_ladder_guard.py:271` are 50ms grid estimates, unlike exact uptime; exact
integration changes any arm's mean |I| by at most 0.0692 shares and time above
100 by at most 0.0792 percentage points, without changing the reading.
Five of 37,989 guard-withhold callbacks occur where a new order would arrive
after END. Continue calling these **withheld slot proposals**, not distinct
orders or filled opportunities prevented. Their count does not affect replay.

## Requested boundaries and prior closure

- **Signal/timing:** `pf_ladder_guard.py:78-100` wraps `LadderInventory`, checks
  float32 B and overrides only ask KEEP/POST proposals. Inventory restrictions
  keep precedence. The frozen client at `pf_branch_ladder.py:123-128,162-220`
  delivers notices before decisions, includes unclosed cancel-pending
  reservations until notice, and refreshes the view after earlier commands.
  Its alternating side order is preserved. Synthetic checks cover both side
  orders, hidden fill versus delivered notice, immediate signal clearance and
  one pull per generation. Independently reconstructed B and known position
  from submission/fill/closure clocks for every actual intervention, including
  same-tick bid submissions; matched all 588 pulls and 37,989 withholds to
  command arrays and checked that ask KEEP/POST commands never escape an on
  signal. No intervention has positive client-known inventory, as reported.
- **Disabled parity:** all 19 fingerprints, identities, tapes, fills, orders,
  requests and values agree with the original baseline. Also independently
  compared decision/command arrays, client/venue inventory, reservations,
  cash and mids exactly. This is stronger than trusting `control_all` flags.
- **Selection:** `pf_ladder_guard.py:194-231` implements the declared
  right-open remaining/age bands, day/market and inventory-sign strata,
  without-replacement draws, no-fill controls and censor exclusions. The
  `[75, infinity)` final band contains the same valid <=100-share population
  as the review's final size band. All saved statistics and the exact seeded
  selection answer reproduce. No realized fill/value enters matching.
- **Measurement/readings:** lines 246–321 retain late fills in END, use
  `t <= END-1000ms` for +1s, and use valid quantity for that ratio. All eligible
  +1s fills in this run have marks. Paired values use each arm's actual ledger;
  nothing deletes bad baseline fills. The declared interpretation computes
  `mixed` correctly. Gross zero-cost/pre-expiry-mark assumptions persist.
- **Prior per-share denominator:** corrected selected quantities are
  6,592.842 valid, 6,716.500 total and 123.658 excluded shares; reproduction and
  the censored-high-quantity control pass.
- **Prior checker means/ranking:** G rederives the actual audit means, censor
  counts, 78 flags/contrasts and selected candidate. Prior corruptions are now
  refused by the named checks. Current saved numbers independently agree.
- **Prior censor count:** the new summary correctly reports 13. The supplement's
  `X_frozen_censor_field` preserves provenance of the old -3,195 formula; it
  does not rehabilitate that old value. The new corrected count is checked
  separately. The post-run supplement is correctly disclosed as instrument
  testing, not a new candidate or changed economic reading.

## Bounded verification and limitations

`reproduce.py` used the pinned checkout via **PF_SLOT=codex** and the absolute
`run_light.sh`: the selftest, 21 saved checks, seven frozen falsifiers, receipt
copy/hash checks, manifest/bin identity, all 38 arm ledger/markout/control
reconciliations, every intervention's actual known state, the complete
selection statistics and predicates, exact inventory integrals, and the
checker-gap reproduction. Passed in **66.71 user CPU seconds**, peak **458 MiB**.
`summarize.py` then aggregates that independent accounting (under the same slot).
Scripts, logs, `evidence.json`, `independent_arms.csv` and `totals.json` are
beside this report. No tape load/replay, fitting, new label build or new day.

The 16-mutant harness and 22-case supplement were inspected with their saved
receipts; they were not rerun exhaustively. Supplement cases detect their named
corruptions; that is not comprehensive proof of every numerical field. The
development round's saved resource records sum to 0.25948 core-hours, peak
1,136 MiB. No current budget or identity breach was found. Passing synthetic
tests does not establish market realism or profitability; execution remains
conditional on unchanged external flow and the M50 simulator assumptions.

## ONE next priority — BINANCE-PRESSURE-GUARD-v1

Test **one external-market signal**, against the original default-post ladder,
on the same 19 complete markets. Do not tune or stack the reservation guard.
The largest remaining economic gap is adverse value per filled share, and
reservations have not demonstrated selection beyond matched exposure. The
earlier audit's `adv_bn_r250` was the largest pooled market-state separation
but **failed** its all-eight-cells lead rule; this is a declared new hypothesis,
not an accepted signal. Count it as the second guard candidate in this ladder
line. All four days remain consumed development data.

1. **Freeze one rule and one pair of existing edges.** Use the unchanged causal
   feature builder's float32 `adv_bn_r250` at the current receipt-clock grid
   decision. ON for side s iff the value is finite, positive and in q5 under
   the original frozen `live|s|adv_bn_r250` bins. The existing last edges are
   **0.4666730940341952 for bids** and **0.285456120967865 for asks**, in the
   existing feature representation. Bind the original edge-file hash
   `ca688917cd7ddc7c40e94031bee646f4fc1339511bb7c48fbe5270797a38bf80`;
   do not recompute quantiles or manually reconstruct the feature with another
   clock/sign/normalization. Missing inputs cause no additional guard action
   and get explicit counts/ages. This rule responds to movement adverse to
   the particular side, not a volatility spike on both sides.
2. **Use the same controller contract.** On the affected side, PULL a would-be
   KEEP live order and withhold a would-be POST while ON. Inventory and
   structural restrictions retain precedence; leave post-pending generations
   to their first live review, as in the current comparison. Re-enter under
   normal rules immediately after signal clearance; no timer. The other side
   retains its normal policy unless its own signal is ON. Keep two levels,
   nominal 100 shares, 200/100/50 inventory settings, sizes, grid, latencies,
   queue model, full path and END mark fixed. Log reducing-side suppressions
   rather than adding an outcome-driven exception.
3. **Verify the new integration boundary in the same round.** Disabled guard
   must reproduce the accepted baseline. Build/cache market inputs once per
   supported market with the unchanged `pf_pipeline_export.real_features`
   path; avoid calling the full builder separately per order per tick. Compare
   cached values to the accepted saved snapshots at matching keys. On 16
   hash-selected decisions per date, test future-only perturbation leaves the
   signal/action unchanged and a delivered adverse-move control can change it.
   Verify side sign and threshold boundary. Log source receipt time/age,
   feature value, request/effect/ack, current order/remaining size and inventory
   so that pre-effect fills and late signals remain distinguishable.
4. **Keep the selection control separate from feedback value.** On existing
   sampled live states of both sides, use the frozen pressure mask and 500
   fixed-seed draws (seed 20261121), matching action count within day/market,
   side, the existing remaining/age/inventory-sign bands, and the existing
   `float32(B)>=200` indicator. This conditions on the reservation exposure
   already studied. No matching on outcome/fill/venue-only state; retain
   no-fill and censor statuses. Use the current minimum support rules and mean
   +1s-per-state criterion; report per-share quantities separately. Do not
   relax strata or thresholds when support is thin. This diagnostic does not
   replace the actual two-arm replay.
5. **Evaluate full economic feedback, with complete evidence.** Reuse the
   paired per-market/day END and common-cutoff +1s metrics, now independently
   reconciled from fills. Include side/rung, valid denominators at every
   horizon, lost/added positive and negative contributions, exact uptime,
   guard/structural/inventory removals, reposts, first-review pull frequency,
   lifetimes and actual inventory. Add the queue observations from finding 2.
   Carry the exact identity `END = eligible +1s markout + subsequent movement
   to END + late/missing-horizon END contribution` per arm, per market and day.
   It explains differences; it is not a fitted target or an avoided-loss
   estimate. Compute the same consistency/participation/selection readings.
   Better aggregate P&L alone does not establish market prediction; a thin or
   failed control remains so. No promotion or interval claim from these days.
6. **Bound the round.** At most **0.75 additional core-hours**, still within
   the original 2-core-hour ladder allocation at the recorded cumulative use.
   Existing slots and memory limits; the same fixed first-two-complete-per-day
   pilot (8) and all 19 only if its projection fits. Stop with explicit status
   if not; no silent population shrink, alternate threshold or new day.
   Bundle the checker/queue improvements and any cheap operational cleanup
   with this one experiment and submit one combined receipt.

**No additional USER permission is needed for that bounded consumed-day
experiment.** More days, untouched validation, resource expansion or a broader
candidate search do require a separate USER decision. The largest evidenced
limitation is that the current guard changes inventory and participation
without demonstrating robust incremental adverse-fill selection; it still
loses about 1.89 c per valid filled share at +1s. Its 19/96 coverage further
limits generalization. The next test isolates an external market signal while
keeping the baseline strategy and accounting fixed.
