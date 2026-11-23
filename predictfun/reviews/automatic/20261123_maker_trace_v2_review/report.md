# MAKER-TRACE v2 / completed-close review — 2026-11-23

**Pinned target:** `0e21193f2d403d08bb5172aaf727a6a93110e8a9`

**Reviewed range:** `567f741d9..0e21193f2d403d08bb5172aaf727a6a93110e8a9`
**Prior review:** [567f741d9 report](../20261123_maker_trace_review/report.md). Read the repository instructions, reliability rules and current handoff/state. Review used a detached checkout at the target; no implementation, model, simulator or development test was edited.

**Disposition: accept the descriptive economics and the completed-close packet. No outcome-invalidating accounting defect or reason for a full rerun was found.** The decision-path tables remain conditional, assumption-sensitive descriptions. They do not identify a posting/cancellation policy. Bundle the items below with the next useful measurement; do not start another cleanup-only cycle.

## Consolidated actionable checklist

1. **P2 — NONBLOCKING for these economic results and the next saved-data audit; blocks using the exit-role table as a policy prescription.** The assertion that taker exits lose *because* they pay spread/fees goes beyond the computed attribution. [report_closes.md:32–35](../../maker_trace_20261122/v2/report_closes.md) reports a selected set of completed closes, grouped by their realized closing role. [closes.py:362–426](../../maker_trace_20261122/tools/closes.py) matches fills and prorates fees; it does not value a passive alternative to each aggressive exit. Inventory that never closes is outside that table and appears in residuals. For top-ten maker→taker closes, gross is **−2.6461 c/share**, fees **0.5309 c/share**, net **−3.1770**. Fees are identified; the remaining loss is not separated into entry selection, movement while holding and exit execution cost. A take could still reduce a larger subsequent loss. Conversely, profitable passive completions do not show that waiting would have closed the losing inventory profitably. **Action:** supersede the causal sentence with “FIFO closes ending with a maker fill were profitable in this sample; those ending with a taker fill were negative; the value of replacing either action is unmeasured.” Keep the positive passive-turnover finding. No rerun required.

2. **P2 — NONBLOCKING for P&L; the absolute-inventory-reduction claim is incorrect and must not be used as a model label.** [pf_mt_decision.py:1608–1626](../../../research/pf_mt_decision.py) defines reduction as `side * position_before < 0`. That identifies the opposing side, not `abs(position_after) < abs(position_before)`: a 30-share sell against +10 shares increases absolute exposure to 20. The synthetic reproduction confirms the instrument calls it reducing. On the exact published action population, the independent action sums reproduce every published denominator and opposing-side count:

   | Rank / role | Nonflat actions | Published “reduce” | Actually reduces absolute position |
   |---|---:|---:|---:|
   | 1 / maker | 152,352 | 77,753 | 74,550 |
   | 1 / taker | 156,412 | 92,142 | 90,256 |
   | 2 / maker | 112,615 | 55,750 | 54,380 |
   | 2 / taker | 16,436 | 9,730 | 9,593 |
   | 3 / maker | 104,788 | 57,847 | 55,746 |
   | 3 / taker | 8,581 | 5,225 | 5,095 |

   Rank 1's maker majority changes sides of 50%; the three taker majorities survive. [make_claims:1720–1725](../../../research/pf_mt_decision.py) propagates the original count into an absolute-reduction statement. **Action:** retain the existing statistic as *opposing-side action*, add actual absolute reduction and overshoot separately, and supersede the affected claim. Preserve the declared strict-earlier-settle convention and identify simultaneous actions separately if interpreting actual execution order. This does not alter fill cash flows, FIFO P&L, position sums, or the close results.

3. **P3 — NONBLOCKING presentation corrections, bundle only.** The identifiability control does not show an unobserved additional execution: [decision_identifiability_control.py:12–17, 58–67](../../maker_trace_20261122/tools/decision_identifiability_control.py) compares an originally 5-share order exhausted by the same observed 5-share fill with a 50-share order whose remaining 45 shares cancel unseen. Change [report_closes.md:60](../../maker_trace_20261122/v2/report_closes.md) to “exhaustion of the original size versus partial fill followed by an unseen cancellation”; the observed executed amount is identified in this fixture. Also make clear that `good_turnover_lost_to_residual` means positive turnover **and a negative residual contribution**, not necessarily negative total profit: [closes.py:663–676](../../maker_trace_20261122/tools/closes.py) does not require the residual to erase all turnover. In the presentation's reconciliation, retain residual fees and fee-token drift so close net plus gross residual is not mistaken for an exact net identity. The implementation already includes them at lines 523–529 and 658–660. Neither issue needs a data run.

4. **Prior latent timing defect — still open, NONBLOCKING for this saved population.** [pf_mt_lifecycle.py:274–277](../../../research/pf_mt_lifecycle.py) still chooses `hi=min(UUID, receipt)` but sets `rcpt` only when UUID is absent and assigns a bounded lower end whenever UUID exists. Thus a receipt earlier than an existing UUID is not treated as the receipt-hi fallback. The prior review established no such input in these saved rows; the v2 receipt's `fill_hi_rule` is true. The promised repair was not made when lifecycle changed. Carry it as an explicit reuse limitation, bundle a refusal/fix before admitting that input pattern, and do not rerun this population solely for it.

There is **no BLOCKING finding for retaining these results or proceeding with the next bounded descriptive audit**. Items 1–2 restrict the behavioral conclusions one may draw; they are not reasons to discard the P&L.

## Ruling on the v2 lifecycle change and prior failures

**Accept the same-settle-ms amendment as a declared change to FIFO inference, not as proof the v1 conservative exclusion was a bug.** In [pf_mt_lifecycle.py:548–586](../../../research/pf_mt_lifecycle.py), “later” is a *strictly greater* settle millisecond. Receipt-hi orders remain outside sequence tests. A tied fill alone is not a later reappearance. If one worst-price member demonstrably survives into a later millisecond, FIFO can place the other member's same-ms fills before its exhausting fill; retaining the group is legitimate under that scenario. Multiple exhaustion claims at the last millisecond produce `fifo_tie_conflict`, not a forced size claim. The corresponding size/removal decision is at lines 616–628.

The 199-case pinned lifecycle selftest passes, including the tied-survivor and incompatible-exhaustion cases at selftest lines 419–467. T2 **5/16,433** replaces **3/15,900** with a changed denominator. It remains a low-contradiction screen conditional on the settle-order/FIFO assumptions; it is not an exchange sequencing guarantee or an explanation of maker profitability. The window-certain tests remain distinct. No new evidence makes R1 causal or expands R2 beyond the 86 recorded own filled orders.

The seven old failed families have a coherent v2 disposition:

| Prior failed family | Review disposition |
|---|---|
| posting capacity / raw capacity reparse | The checker now shares the declared display tolerance and violated-cell treatment. The lifecycle explicitly records the per-order tolerance and the single-order bound; not an exact-size observation. |
| priority exhaustion / ahead lower bound / T1–T2 | Accepted scenario amendment above; v1 remains intact. |
| scenario class consistency | The no-priority queue bound `[0, NaN]` is trivial, not informative; the v2 comparison represents that consistently. |
| H2 presence | Repeated, consistent-identity fills can prove persistence even when a book/priority-specific flag excludes other inference. Identity-conflict/malformed identities remain excluded. |
| top of book from levels | Missing side has zero displayed size and a missing price, not a missing size. |

The committed v2 independent-check receipt records **39/39**. I inspected its changed predicates and reran the bounded lifecycle controls; I did **not** rerun its full real-data checker or the 23,668-claim simulated-venue experiment. The formal R3 reconciliation can be accepted with these scenario qualifications; the amount of actual posting/cancel history identified remains limited.

An exact streaming comparison of **8,101,346** v1/v2 `fills_valued` rows found changes only in lifecycle annotations: `life_rep` 164, `life_exh` 519, `life_front` 164 and `life_status` 164. All economic, identity, timing and other columns are exact. `maker_market.parquet` and `identities.json` are byte-identical. Thus **R4 aggregate economics are unchanged**, while “the whole fills file is unchanged” would be false and lifecycle-conditioned group membership may differ.

## Completed-close and decision-path conclusions

The independent review uses a chronological **deque FIFO**, not the production merged-cumulative-boundary implementation. It checks **1,055,691** valid non-self top-ten fill rows on Oct 24–27, reproducing the rank × day × role-pair share, gross and fee aggregates within **2.51e−9** absolute units. All six close-output hashes and all three input hashes match the pinned receipt. The float-boundary correction in `e2201c258` prevents a sliver from taking its partner from the next group; it does not change the FIFO definition. The synthetic close suite passes **239/239**.

The following findings stand as recorded-fill, consumed-day descriptions:

- Top-ten completed closes earn **$109,145.73 net at fill-price fees**, about **1.1913 c per matched share**. Gross is **$130,789.28**. Maker→maker **+1.9332**, taker→maker **+3.9525**, maker→taker **−3.1770**, taker→taker **−1.5905 c/share**.
- Top-three pooled completed turnover is positive, but residual resolution contributions are negative. The reported rank 1–3 totals, recorded position excursions, day classifications and differing fee conventions are consistent with the saved outputs. This is accounting on recorded fills, not proof of flat inventory or a replicable spread rule. FIFO entry/close roles and holding times use the declared settlement-ID ordering, not an independently observed submission sequence.
- Completed turnover and residual exposure are separate. A profitable selected completed-close cohort cannot be evaluated while omitting the inventory that did not complete. A maker→maker close can also contain directional price movement; its entire P&L is not mechanically the quoted spread.

**Relabeling the decision results is sufficient for this descriptive deliverable.** The prior review explicitly allowed a conditional activity-span estimand instead of an unconditional grid. `report_interpretation_v2.md:53–64` now states that the first/last-fill span is outcome-selected, `match bound − L` may be after the real decision, Binance event time does not establish receipt time, and posting-hi can be the first fill. The at-fill centring, inventory skew, take clustering and possible-removal-window associations may be retained with those labels. They cannot yet supply causal training inputs, an unconditional entry hazard or an imitated executable policy. Saved `claims_decision.json` identifiers should be read with the supersession, not promoted by their older “quotes” wording.

The same-public-tape control passes **8/8**, with **78/78** frozen decision controls and the visible-change falsifier. It demonstrates multiple compatible hidden command histories; it does not recover which history actually happened. The removal-window null is descriptive under its declared random-placement assumptions; correlated orders on four consumed days do not provide a new independent-day validation result.

## Checks, provenance and limits

- [Saved-data check](check_saved.py), [evidence](saved_evidence.json), [log](saved_check.log): exact input hashes, independent FIFO and inventory-direction audit. **36.00 s wall / 35.57 s user CPU, 401 MiB**.
- [Pinned controls](check_controls.py), [log](controls_check.log): lifecycle 199, closes 239, identifiability 8 and decision 78 all pass. The wrapper then **failed** an overstrict review assertion that the entire lifecycle-enriched fills file must be byte-identical. This failed check is retained, not silently discarded. **22.82 s wall / 14.11 s user CPU, 689 MiB**.
- [Follow-up identity check](check_identity.py), [evidence](identity_evidence.json), [log](identity_check.log): streaming all-column equality localizes the difference to the four lifecycle annotations; verifies the unchanged economic columns and close output hashes. **16.79 s wall / 15.81 s user CPU, 599 MiB**. No development source change was needed.

All execution was through `PF_SLOT=codex run_light.sh` with absolute script paths. No fit, new labels, real-tape replay, new day, untouched validation or live action. Charge this review **at most 0.025 core-hours including finalization**; do not count the earlier review's 0.021 again. Starting from the submitted 1.68 h task ledger gives **at most 1.705 / 1.85 h**, subject to any concurrent development use. Synthetic controls establish implementation behavior, not profitability. Full decision-grid calculations were not rerun.

## Largest limitation and ONE next priority

**Largest evidenced limitation:** we now have credible turnover accounting and useful associations, but not the maker's executable quote/pull policy or a causal queue advantage. The whole-market top-ten figures also are not yet an aligned comparison to our 19-market ladder runs. Inferring the strategy gap from different markets, start times, inventory bases and terminal horizons would repeat the earlier horizon mistake.

**ONE next priority: `MATCHED-MARKET-TURNOVER-GAP-v1`, a saved-data comparison of the accepted CURRENT_ASYNC_FAST10 ledger with the ex-ante top three and top-ten benchmark on the same supported BOOK markets and intervals.** Bundle the checklist above; no new queue, latency or fitted-rule variant.

Freeze a small manifest and reading rule before computation:

1. Use the existing 19 BOOK markets on Oct 24–27. For each, pin the accepted simulated start/end and both boundary marks. Keep all missing/unsupported intersections as statuses. Do not expand to the makers' more favorable markets or choose winners after reading P&L.
2. Reconcile each book on that **same interval and END mark**, with the same declared fee convention. For the real makers, reconstruct recorded inventory entering the interval and value carried lots at the opening mark; retain actual acquisition time. Separate carried-in contribution from within-window trades. Show a zero-start incremental-flow view separately if useful; do not silently reset actual recorded inventory. Do not compare our END mark with their resolution P&L.
3. Apply one FIFO/accounting implementation to both: completed-turnover net per matched share, matched/unclosed quantities, carry and residual mark contributions, total net, holding times and inventory excursions. Report per market and per day, with pooled values secondary. Attribute each fill/share once and preserve the terminal identity. The opposing-side/absolute-reduction distinction belongs here if reporting inventory actions.
4. The frozen answer should locate the **observed gap**: does our completed turnover already lose while matched makers' turnover earns; is the larger difference residual/carry exposure; are both adverse; or is the answer mixed/unsupported? Keep denominator and market-mix effects explicit. This is not an avoided-loss estimate or an imitation claim.

**USER decision:** no additional permission is needed for this saved-data continuation of the authorized maker/strategy diagnosis **if its entire frozen batch, including the next review, fits ≤0.10 core-hours and the existing 1.85 h task / 5.25 h line caps**. Declare a bounded projection first and account for concurrent work; no new run is launched by this review. If it will not fit, request a resource amendment before running. New days, an executable policy trial, untouched validation or live use require a separate scope/resource decision. A mixed answer should not trigger another tuned variant automatically.
