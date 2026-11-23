# COMPETING-TRANSITION-AND-CONTINUATION v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the ONE next priority of review a91f6bb52 (`reviews/automatic/20261123_touch_response_review/report.md`, findings 1–4). It serves the USER goal: action chains and their logic (post / pull / reprice / take).

**The question:** after a visible one-side retreat, which already-declared market conditions precede same-side restoration vs a competing transition? And what happens economically along the candidate order chains?

**Budget:** ≤ 0.030 core-hours ALL-IN, including ≤ 0.010 for the review. The ledger stood at 1.8947 of 2.00 (USER standing CPU authorization).

**Data:** saved evidence only. No tape reload, no new reconstruction, no new days.

**Stop rule (declared now):** a thin or inconsistent answer ENDS this measurement line. No further floor, feature or variant tuning follows.

**Tools** (each with a selftest; frozen v1 tools never edited):

| Tool | Sections | Output (write-once) |
|---|---|---|
| `tools/competing_transition.py` | §1–§4 | `v2/chains/competing_transition/` |
| `tools/continuation_econ.py` | §5 | `v2/chains/continuation_econ/` |

## §1 Population and outcome

**Population:** the same 3,796 episodes of `v2/chains/touch_transition/episodes.parquet`, bound to its receipt sha256. All rows are kept.

**Outcome:** the next detected transition, mutually exclusive, from saved columns (`end_type` / `end_reason` / `end_class` / `resp_type`):

| Group | Outcomes |
|---|---|
| same side | `same_restoring` / `same_partial` / `same_overshoot` |
| other side | `other_improve` / `other_retreat` |
| two sides | `two_side_both_up` / `two_side_both_down` / `two_side_narrow` / `two_side_widen` |
| further retreat | `same_retreat_further` |
| horizon | `horizon_no_change` |
| censored | gap / unknown / empty side / window or book end |

- The two-side sub-label comes from a saved row (e.g. a `touch_transition/reconcile.csv` row of the same end frame pair). Otherwise it is `two_side_sub_unknown`.
- A competing move is an outcome, never a discarded non-response.

## §2 State, contrasts, statistics

**State:** the five frozen contrasts, with their bins and D0 / LAG definitions exactly as saved in the episodes. All five are reported; none is selected.
- btc_with vs flat;
- btc_against vs flat;
- fair_gap > 1 vs ≤ 0;
- own depth ≤ 50 vs > 200;
- time-left ≤ 30 s vs 60–240 s.

**Wording:** fair_gap combines the earlier BTC reference with the **post-retreat best**. It is known at the observed episode start, but it is not a wholly pre-retreat feature.

**Statistics per contrast and clock:**
- state-conditional outcome fractions (every outcome class) per group;
- the matched same-side-response **risk difference** within strata market × side × Δ bin × time-left bin (time-left omitted only for its own contrast). It is pooled over common-support strata with fixed harmonic weights n1·n0 / (n1 + n0), and the raw all-population counts are shown beside it;
- day signs (per-day RD sign), side signs, and contribution concentration (the largest single market's share of the weighted RD).

**Censoring:** same-side response probability bounds:
- lower = responses / all, with censored rows as non-responses;
- upper = (responses + censored) / all.

**No significance:** no p-values and no permutation verdicts. The independent-label shuffles of the previous round are not reused as confirmation.

## §3 Count-based support (frozen now; a reporting-adequacy floor, not power)

A contrast is `adequate` only if, within its mixed strata:
- each group has ≥ 30 episodes;
- each group has ≥ 10 same-side responses AND ≥ 10 competing non-response transitions;
- ≥ 5 markets have ≥ 5 episodes in each group.

Otherwise it is `thin`. Every count is shown, including excluded and unsupported ones. Four related consumed days give no confidence interval and no validated rule.

**Reading per contrast** (computed):
- `adequate_direction_consistent`: D0 and LAG risk differences have the same sign, and the RD sign is the same on ≥ 3 of 4 days;
- `adequate_inconsistent`;
- `thin`.

## §4 Timing (interval bounds, never a latency estimate)

For each response:
- lower bound = max(0, response_lo − retreat_hi);
- upper bound = response_hi − retreat_lo;
- where the retreat interval is (t_prev_ms, t_start_ms] and the response interval is (resp_lo_ms, resp_hi_ms].

Also reported: the next-frame fraction and the observer-clock endpoint difference, each separately. Medians of bounds are summaries of bounds, not an identified median latency.

## §5 Candidate continuation economics (continuation_econ)

**Links:** the 89 candidate response links (`touch_transition/continuations.csv` + identity) and the six chains (chains_v4), with their owner / grade and unresolved paths carried.

**Per candidate order** (fill and close keys de-duplicated across repeated links):
- exact-leg FIFO completed-close net dollars and shares, reusing chains_v3's full-leg-key close attribution and the clean cohort (status ok, non-self);
- closes inside 10 s / 60 s / END kept apart from later closes and from the resolution residual;
- fees included;
- the market END bound enforced;
- take direction relative to known inventory (take_logic_v2). A subsequent take is not automatically an exit.
- Account-level P&L is never summed per candidate. An unavailable exact link is labelled `unavailable`.

**Guards (finding 4):**
- the END upper bound on takes;
- the clean cohort for value views;
- the response-context join requires the creation tick == `resp_tick` ("this cell restored the touch"); otherwise the context is contextual only.

## Acceptance (one combined review)

**Reconciliation:** episode totals, mutual exclusion of outcomes, class-direction identities, common-support denominators, censor bounds and exact leg uniqueness.

**Known-bad cases in the selftests**, each must fail its check:
- swapped outcome labels;
- dropped competitors;
- a take crossing END;
- a duplicated close.

**Receipt:** `v2/report_logic_chains_v6.md` also corrects findings 1–3 of a91f6bb52 in-band:
- "both aggregate best prices changed within the same observed interval", with the 6,723 / 11,804 creation denominator and 4,897 two-side vs 8,298 one-side changed pairs;
- "a subsequent touch change is usually detected within a few book frames", with the endpoint convention;
- exposure 7.2–27.5 s per market.

## Cost

| Builder | Budget (CPU s) |
|---|---|
| competing_transition | ≤ 12 (≤ 2 selftests + one real `run --dry`) |
| continuation_econ | ≤ 15 (≤ 2 selftests + one real `run --dry`) |

- Full runs ≤ 20 s total.
