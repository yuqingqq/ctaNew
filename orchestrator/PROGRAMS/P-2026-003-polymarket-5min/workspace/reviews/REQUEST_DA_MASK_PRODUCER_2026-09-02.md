# REVIEW REQUEST — DA round 6: blackout mask producer + complement block (R-409) — pinned tip `181b4fa`

Filed by the coordinator 2026-09-02T10:07Z. One round, one filing (rule 5 / R-377). Queued behind the BE mask-seam round; dispatch to the reviewer only after that filing lands.

## What to review (execute, never accept from the report)

Commit `181b4fa` (Q-DA-205). Files: `live/pm_research/da_blackout_mask.py` (new), `da_forward_day_verify.py`, `da_cross_venue_forensics.py`, `da_content_liveness_v2_check.py`, `v5_deploy_gates.py`; artifact `data/pm_5min/derived/da_blackout_mask_20260901.json`.

1. **Export, not redefinition.** The mask recomputes v1's window list and REFUSES unless the per-coin count equals the frozen detector's own `n_invisible_thin`. Verify the equality is against the frozen module's number, not against a copy inside the mask builder; verify a one-window disagreement refuses (DA's red-first claim).
2. **The complement.** PRESENT windows minus masked (DA found and fixed its own `range(288) - masked` defect on the open 09-02: 248 unmasked out of 119 present). Check the open-day control fails on the calendar form. Check the denominator PAIRS for P1/P2/P3 are what the filing says (btc 09-02: 93.01 per unmasked hour vs 25.51 per calendar-24h) and that P3's calendar-wide rolling hour excluding masked loss is a defensible reading of the frozen bar rather than a new statistic.
3. **Rule 9.** `L1_over_complement_is_TAUTOLOGICAL: true` — confirm nothing downstream reads the complement's L1 as evidence.
4. **The pins.** `content_thin_vetoes_HEALTHY: false` now `ruled_by: R-409`; `race_accrual_eligible` keeps its four-conjunct definition with the mask entering nowhere — recompute the conjunction from the emitted verdict. The v2 seam is `INERT_PENDING_USER_FREEZE` and refuses by name — confirm it cannot return a mask.
5. **`day_closed_calendar` on the mask** — both-direction control; a partial mask must be refusable by the consumer (BE's adapter does not consume it yet — note whether the schema makes that easy).
6. **Launcher path.** `./da_midnight_verify.sh` into an isolated outdir: 09-01's governing fields byte-identical pre/post batch; only `content_liveness_rule` and `blackout_mask_and_complement` added.
7. **RR7-1 / RR7-2 closure** — each venue's shipped regex matches a REAL line of its own log and rejects another venue's; the v2 comparison is computed (`verdict_changed`, 08-26: 7 string diffs, 1 verdict change).
8. **The escalated constant** (`ESCALATION_no_minimum_complement_size`) — is it the only one, or does the block quietly depend on another (e.g. the 288 calendar denominator on a day with fewer present windows)?

Run in `~/ctaNew-wt-rev` (`git -C ~/ctaNew-wt-rev checkout --detach 181b4fa`), heavy runs inside the research slice. File `REVIEW_DA_MASK_PRODUCER_2026-09-02.md`, commit, push, ONE notification.
