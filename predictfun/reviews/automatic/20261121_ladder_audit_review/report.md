# RECEIPT-LEAD-TIME-AUDIT-v1 — combined review

2026-11-21. Pinned target **6b14fbf951a96baebd885f96a93ecd10a7744bd8**.
Range: 0065191f7b83732aff6994f5b5bef5aad5b0e6bf..6b14fbf951a96baebd885f96a93ecd10a7744bd8.
Reviewed b1167651's freeze, b8642e6d / 349d2f32's checker supersessions and
6b14fbf9's results in detached worktree `/tmp/pf-la-review-6b14fbf9`.

**Accept this audit as descriptive evidence on the 19 complete markets. No
blocking finding requires a baseline or audit rerun.** All eight prior findings
are addressed or explicitly superseded. Three remaining issues are bundled
below; the independently verified workarounds permit the next experiment now.
This is neither a profitable strategy nor demonstrated prediction skill.

## What the evidence establishes

- All **14,968** snapshot identities and their dictionary inputs agree with the
  saved feature rows. Independently reconciled every entry / first / live
  window, its effective time, quantities, censor status, pre-effect fills and
  cancel-pending fills against the saved ledgers. Kinds remain separate:
  5,288 entries, 4,840 first reviews, 4,840 sampled live states.
- First-review future fills at or after `decision + 37ms` carry
  **-185,438.2 c** of valid +1s window markout, versus **-4,130.1 c** in fills
  between that decision and its cancel effect. The first number includes
  subsequent fills over the generation's remaining baseline lifetime; it is
  not a fixed one-second decision horizon. The +1s is measured after each fill.
  Neither number is a counterfactual strategy P&L or an oracle savings claim.
  Inventory already held and cancel-pending fills are separately reported.
- Recomputed all **78** kind/input tail contrasts directly from states, without
  trusting the stored means or selected candidate. The one declared LEAD is
  `live / ctx_bid_reserved_shares`: pooled q5 minus q1 **-23.1380 c/state**,
  negative in all eight day/side cells. Frozen bin edges are unchanged.
- Correcting the per-share denominator in finding 1 preserves the post-hoc
  sign result: **4/4 ask days, 1/4 bid days**. Correct ask-side q5-minus-q1
  differences are -4.7475, -2.8636, -3.7235 and -1.6835 c/valid-window share.
  This is an exposure-weighted descriptive ratio, not a matched causal effect.
- The selected ask-side q5 begins at **200 client-known bid-reserved shares**
  (the last frozen edge of `live|1|ctx_bid_reserved_shares`). That is opposite-side
  outstanding exposure, not an external market-pressure forecast. On bids the
  same feature includes the focal side's quantity; its per-state separation
  mostly fails the per-share check. Do not deploy a symmetric both-side guard
  from the eight-cell sign result.
- All 96 selected-market statuses remain: 19 complete, 54 missing endpoint,
  23 stale future book. Complete markets per day are 7/5/5/2. These are consumed
  Oct 24–27 development data; they are not four complete validation days.

## Consolidated actionable findings

All findings are **NONBLOCKING for the next bounded experiment**, using the
verified workarounds below. Include cheap corrections in its combined receipt;
do not create another fix-only review round or modify the frozen artifacts.

1. **P2 — The post-hoc per-share table mixes valid outcomes with censored
   shares.** `pf_ladder_audit.py:264-267` restricts `mo1s_sum` to uncensored
   windows but sets `win_q` from all windows. The report writer divides those
   fields at `reviews/ladder_audit_20261121/tools/fill_audit.py:94`.
   For example, Oct 26 ask q5 has 1,619.939 valid-window shares, but its
   denominator is 1,704.796 shares. Oct 25 bid q5 similarly includes 100 shares
   whose whole-window markout is censored. This understates adverse cost per
   valid share. **Workaround verified:** recompute numerator and quantity from
   the same uncensored states; all eight sign conclusions are unchanged
   (`evidence.json: lead_per_share_common_population`). Next receipt should
   publish `valid_window_q`, total quantity and excluded quantity separately,
   and test that a censored high-quantity row changes neither valid ratio term.

2. **P2 — The independent checker does not verify all fields used by its lead
   conclusion.** `audit_checks_v3.py:161` checks bin counts and sums, not stored
   means; lines 163–177 then trust those means and the saved pooled contrast,
   comparing the selected key without reconciling its sign/magnitude or every
   flag. In `checker_probe.py`, changing the selected feature's eight q5 means
   to q1 mean +1 reverses every sign, yet all 23 checks still pass. An arbitrary
   censor-counter corruption also passes. These are in-memory falsifiers of
   check coverage, not changes to real artifacts. **Workaround verified:** this
   review independently recomputed all 78 contrasts/flags and the actual lead
   from saved state rows; current results agree. Next checker should derive
   means, signs, all flags, pooled ranking and candidate fields from the same
   counted rows and refuse these corruptions. Existing seven falsifiers still
   pass, but do not cover these cases.

3. **P3 — The generation censor counter is negative.**
   `pf_ladder_audit.py:365` subtracts no-fill generations from the number of
   missing markouts, although `vals([])` already returns zero. The resulting
   `audit.json.cross.generation_tails.mo1s_censored` is **-3,195**, rather than
   **13** (3,208 generations have no fill). The state-level censor flags and
   lead computation are correct; this field is not used by their reading.
   Use the independently counted 13 until a superseding summary fixes it, with
   an explicit `0 <= censored <= generations` check and a no-fill control.

## Prior findings and requested code checks

| Prior item | Disposition and evidence at the pinned source |
|---|---|
| F1 pending/live/actual boundaries | Closed for this audit. `pf_ladder_audit.py:85-95,115-137,233-249` uses the engine's pending max rule and strict removal exclusion. The synthetic request at 50ms lands at 95ms and prevents the 100ms fill. The 235-row, 93-generation interval and its +32,481.2 c END value reconcile. |
| F2 sweep event identity | Closed. Lines 143–155 group by side/public-trade index. 388 multi-generation event fill rows versus 2,132 timestamp-grouped rows; values are confined to event tranches. |
| F3 future-window population | Closed. Lines 222–254 use the same future fill list for quantity, fill indicator and outcome; missing +1s labels remain censored. All 14,968 rows independently checked. Per-share reporting issue is separately qualified above. |
| F4 churn / recovery | Closed. Lines 158–189 distinguish 4 policy first-review pulls, 143 early structural requests, 252 recoveries, 23 direct reversals and 12 censored episodes. |
| F5 unsafe resume | Closed for the named holes by the new API. `pf_branch_ladder_safe.py:27-64` binds executable fields, object tape/mark, envelope identity and next tick; the uninterrupted/resumed positive control and all four prior falsifiers pass. The old API remains explicitly unusable. No checkpoint was used to generate the accepted baseline. |
| F6 environment-dependent selftest | Closed. `pf_ladder_rest_selftest_v2.py:18-53` supplies explicit negative and allowed guard conditions. Independently reran v2 successfully; old fixtures and failures remain provenance. |
| F7 economic attribution | Addressed. Net-market concentration, common-fill markouts and the flat-round-trip example prevent interpreting opposite END legs as independently avoidable losses. Lead windows are still baseline-path descriptions, not neutral no-cancel labels or policy EV. |
| F8 exact uptime / formatting | Addressed. Lines 192–219 integrate active intervals: 40.6981% both-side uptime, versus the old 40.3055% grid estimate. This receipt uses explicit formats. No need to change the frozen baseline writer. |

The causal audit at `pf_ladder_audit.py:374-419` has the declared future receipt
frame / unnotified-fill perturbation and delivered controls. Its saved 64
members match the deterministic hash sample exactly; counts reconcile to rows,
and source hashes/committed receipt copies agree. All 64 market and order
checks pass; market controls fire 64/64 and eligible order controls 59/59.
The check is of reconstructed market/known-order builders, not an end-to-end
perturbation of all 107 persisted inputs and the ladder controller's context.
No raw tape was reloaded in this review. Prior full snapshot reconstruction
and this review's saved-input reconciliation support the descriptive audit;
a new guard must additionally prove its own current-state signal causality.

The v1→v2 checker diff fixes the list/Series membership expression and compares
the two already-declared censor conventions separately. It does not silently
equate generation-censored and per-fill-valid markout sums. v2→v3 adds the day
string dtype; remaining changes are provenance/output names. Both failed logs
are retained, and the source changes are committed before their executions.
The current launcher equals the frozen blob. The reported reverted edit adds
no accepted output; this review cannot independently reconstruct an unrecorded
historical working-tree edit from current files.

## Bounded verification

Both runs used the pinned checkout through the required absolute `run_light.sh`
with **PF_SLOT=codex**, no other slot, no fitting, no label build or tape replay:

- `reproduce.py`: audit selftest, baseline selftest v2, all 23 saved checks and
  seven falsifiers; all output hashes and compressed receipt copies; every
  saved state identity/dictionary input/window; 78 contrasts; causal sample
  identities/counters; common-population per-share correction. **Passed**,
  48.25 user CPU seconds, peak 427 MiB.
- `checker_probe.py`: clean control passes and two previously untested
  corruptions reproduce the checker gap above. **Reproduction passed**,
  13.32 user CPU seconds, peak 410 MiB.

The 16/16 audit, 6/6 resume, 24/24 client and 20/20 runner mutation receipts
were inspected with their harness/source references, not rerun exhaustively.
Synthetic passes establish implementation behavior, not economic performance.
Review evidence and logs are beside this report; baseline and audit sources,
their tests and development artifacts were not edited.

## ONE next priority — ASK-RESERVATION-GUARD-v1

Run **one declared candidate against the frozen default-post ladder**, on the
same 19 complete markets, with all 96 statuses preserved. This directly tests
whether the surviving ask-side association helps after accounting for lost
fills, queue position and inventory feedback. It is an adaptive consumed-day
mechanism test, counted as one new candidate; no bid-side mirror, threshold
sweep, new model or new data day in this round.

1. **Freeze the action rule before replay.** Read current client-known bid
   reservations from the branch after notices, using the same float32
   representation and frozen `live|1|ctx_bid_reserved_shares` bin edges.
   The signal is q5, equivalently represented bid reservations >=200 shares.
   On asks only: request PULL of each currently live ask when the signal is on;
   withhold new ask POSTs while it is on. Existing post-pending orders become
   eligible at their first live review. Once the signal clears, ordinary
   inventory/structural/default-post rules apply immediately; **no timer**.
   Recheck on the existing 50ms grid with the existing 37ms cancel latency.
   This combines cancellation and entry suppression deliberately to avoid an
   unchanged-state cancel/repost loop. Log the two effects separately.
2. **Keep the experiment's other choices fixed.** Two levels, nominal 100
   shares per level, cap/soft-band/recovery 200/100/50, sizes, structural rules,
   reservation accounting, tick/notice order, full supported paths and marks
   remain the baseline's. Use the candidate's own state, not baseline future
   reservations. Bids retain their baseline policy. Do not special-case
   inventory-reducing asks after seeing results: suppressing them can be
   harmful, and that is part of the trade-off this candidate must expose.
3. **Control and validity.** In the same runner, a disabled guard must reproduce
   the accepted baseline ledgers, decisions and values. A synthetic hidden
   fill must not change the signal until its notice; a delivered notice that
   changes represented B across 200 must change it. Test B on both sides of
   the boundary, simultaneous side callbacks, no repeated request on a
   cancel-pending generation, signal clearance and preserved unaffected gids.
   Hash-bind rule, frozen edges, sources, input identity and outputs. Use safe
   resume v2 if needed; a fresh run is simpler at this size.
4. **Predeclare a cheap selection control, kept distinct from replay P&L.** On
   the existing one-hash-live-state-per-ask-generation population, compare the
   q5-selected future-window +1s value with **500 fixed-seed matched draws**.
   Within day/market, match selected action counts and receipt-known remaining
   quantity bands (0–25,25–50,50–75,75–100 shares), age bands (<0.5,0.5–2,2–10,
   >=10 seconds), and inventory sign. State exact interval boundary conventions
   and seed in the freeze; never match on realized fills. Keep no-fill rows and
   censor statuses. Report movable strata/rows and mark thin or degenerate
   controls INCONCLUSIVE, without coarsening after results. This tests the
   local association beyond obvious exposure differences; it is **not** a null
   distribution for the full feedback policy's P&L.
5. **Measure actual economic consequences.** Paired per-market and per-day
   END cash+inventory value is the main replay comparison. Also report gross
   execution edge and +50/+100/+1000ms markouts, total and per filled share,
   with valid quantities and censor counts. Compare the +1s measure on the
   shared fill-time eligibility cutoff END−1000ms; retain late fills in END
   accounting. Break out bid/ask, rung, good and bad fill contributions,
   guard pulls/withholds, structural/inventory removals, reposts and lifetimes,
   actual inventory/time at limits/terminal inventory, exact both-side uptime,
   filled quantity, and reducing-side opportunities suppressed. Never estimate
   policy improvement by deleting bad baseline fills. Do not add overlapping
   snapshot windows or infer inventory value from gross generation END tails.
6. **Interpretation and stop.** Compute whether paired END value and +1s
   per-share value improve, their day/side consistency, and the changes in
   exposure, participation and inventory. Distinguish better selection from
   merely quoting less; mixed results remain mixed. No profitability,
   significance or validation claim from these 19 markets and four consumed
   dates. No automatic promotion or second candidate in this round.
7. **Budget and batching.** At most **0.75 additional core-hours**, within the
   original ladder experiment's unused 2-core-hour allocation; existing slots
   and memory limits. Pilot on the first two complete markets per day in the
   fixed manifest order (8 total); freeze this selection and project cost
   before the full 19. If over budget, retain the pilot/statuses and stop rather
   than changing the population silently. Include the three small summary/
   checker corrections above or retain the verified workaround, then submit
   one combined code/result receipt. No separate approval between repairs.

**No further USER permission is needed for this bounded consumed-day test.**
More days, untouched validation or additional resources remain separate USER
decisions. The largest evidenced limitation is that the audit has found a
reservation/exposure association, not demonstrated incremental market
prediction or an economically useful cancellation rule. Its 19/96 coverage
also prevents generalization to the excluded markets. The next replay is
intended to resolve the economic usefulness of that one association.
