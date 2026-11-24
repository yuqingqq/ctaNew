# AVAILABILITY-AND-ACTION-CHAINS-v1 — completed receipt

2026-11-24. USER: “Yes can you proceed to get the logic/action chains.” The readable deliverable is [logic_chains.md](logic_chains.md), including the population condition/action map and every time-ordered successor, not a chosen profitable example.

Primary source freeze: **46f83815e98568bcf2ec599c9385ffd246bdc17a**. Parent accepted quantity packet: **5f6fa5a98641f014bf9e73c8d16d68d9fc5b3126**. The market-relative extension was frozen separately in **376fc99d52a9bcbde34a7919dab2c4af6aaca54b**, after the primary counts and before its measurement. Contracts, source/input hashes, failed preflight and all logs are retained.

## What is now connected

All 88 fixed signed hashes / 18 markets on consumed Oct24–27; primary as-of2026-11-24T04:18:56Z. Original immutable budgets and observed fills connect to conditional posting/ending bounds, existing exact-maker successor hashes, market-state ranges and later takes. 25 orders have multiple recorded fills; original budget is consumed rather than renewed by each partial fill. Endings remain 41 exhausted / 40 conditional-unavailable / 7 unresolved. This sample was drawn from a prior terminal, fill-selected cohort; its percentages are not unconditional maker action rates.

- Seven successors can be ordered after the old ending under the saved bounds: three exhausted→same-price refill, two exhausted→reprice and two conditional ending→reprice. Thirty-five successor windows overlap, seven are unusable and39 have no recorded successor. Never equate later first fill with new submission.
- Two conditional ending upper bounds tighten by8,480ms and2,250ms. No posting bound improves; no private cancel is identified. Every exhausted order retains the exhaustion control.
- 29,944 competing-record/order combinations are assessed with exact side/actor identity, hard execution-time windows and bracketing pending-quantity envelopes. Counts are repeated evidence within orders, not independent events. The between-fill test has **zero timing-separated opportunities**, so its zero interruptions is uninformative about continuous queue priority.
- 528 interval/clock context rows use the full saved one-second as-of state ranges. Later observed takes produce81 order references to68 unique take records. They are not automatically exits of those orders or reductions of client-known inventory.
- All49 existing successor links are printed apart from their sequence grade. Conditional endings have22 less-aggressive absolute-price successors out of26 links; exhausted orders also have13 of20. Therefore the seven ordered examples cannot establish a clean general depletion-vs-retreat policy.

## The additional market-relative distinction

The original `toward_spread`/`away_from_spread` enums mean only higher/lower own-side price. Their user-facing labels are corrected here; frozen primary values stay unchanged. The descriptive extension reuses the already bound `take_logic_v2` book panel (`kok` is book validity; `bok` is BTC validity), checks probability-to-cent conversion and carries all window combinations, without assuming ambiguous successors followed an ending. Signed quote change equals signed best-price change plus relative-offset change exactly.

42 usable pairs: at D0,5 are always less aggressive relative to touch,3 more aggressive and34 timing ambiguous. With1s lag these are6/5/31. Conditional endings alone are3/0/20 at D0 across23 usable pairs. Seven refused pairs and39 absent links remain statuses. We therefore cannot infer that the observed lower bids/higher asks generally constitute deliberate retreat rather than market following. Nor do ambiguous windows prove market following. The one-second proxy can miss intrasecond changes.

## Verified repeated logic versus remaining gaps

The prior neutral-population map remains the source for repeated condition/action associations:17,100 maker-seconds,11,400 market-side seconds and3,796 retreat episodes. Wider spreads associate with fresh levels; thin depth has different placement and retreat-response associations; passive-fill bursts precede more opposing takes in D0 but the tighter matched effect weakens with1s lag; large inventory associates more clearly with restrained same-direction taking than forced liquidation. No robust BTC-reversal→specific-maker-cancel trigger is identified. Quote creation is mostly anonymous. None of these associations is promoted to a profitable controller.

The actionable working chain is opportunity/quote → same signed budget filled in pieces → exhaustion or conditional ending → refill/reprice/overlap alternative → reconstructed inventory and observed later actions. Exact command time, original size of unsigned successors, hidden reservations, uninterrupted queue tenure and intent remain unidentified. This packet closes joins and preserves alternatives; it does not invent those links.

## Verification and limitations, bundled

Primary selftest23 groups; independent checker re-derives every competing evidence row, every88 chain, all528 context rows, every take link and summary. Ten real-copy corruptions are refused, including false pull, false repost, future context and omitted identity. Extension selftest6 groups; separate scalar checker matches all176 rows and rejects five corruptions. Positive unmodified controls pass. No model fitting, simulator change, new days, intervals or profitability conclusion.

NONBLOCKING interpretation limitations: the signed sample is small and fill/terminal-selected; most private timestamps are intervals; some inference needs full-history/full-display/priority/no-hidden-reservation assumptions; per-record pending envelopes do not identify per-order reservation allocation. Same-hash repeats alone cannot certify uninterrupted rest. The original direction enums are handled by the verified market-relative extension, not renamed inside frozen data. No outcome-invalidating issue remains open for the stated conditional result.

Preflight attempt1 refused the arbitrary150MB source-byte cap; before outcomes it was raised to512MiB while decoding only selected markets/columns. No sample or outcome changed. All failed and successful jobs are charged. Independent checks passed on their first runs. `archive.json` binds two <1MiB compressed parts of the29,944-row evidence; their concatenated CSV (one header) recompresses byte-identically to the original gzip. The original >1MiB file is local/ignored, not committed. Every committed evidence file is bound in SHA256SUMS.

Resource: packet cap .006core-hours (21.6CPU seconds), unchanged task2.00/line5.30 with .05reserve. `resource_final.json` records measured all-attempt usage and a finalization reserve. Only additive owned state is staged, preserving other-session work. No push or external messaging.

## ONE next priority

Coverage-gated expansion of signed order chains to test observed depletion → same-side refill/reprice against partial-fill controls: census existing unused top-maker hashes and complete event membership first; freeze outcome-blind membership only if enough new successor windows can be ordered. Stop if the timing/identity gate fails. Preparation can use current saved data; full computation needs a bounded freeze within the remaining resource allowance, or a USER-approved top-up. New acquisition/days/live use are outside this packet.
