# TAKE-EXIT VENUE-MATCHED COMPARISON v1 — BASE vs FLOOR under venue rules (declared before any run)

USER (2026-11-25): "Can you also check the review commit and run the comparison tests". Review 4e67e0856
([report](../automatic/20261125_take_floor/report.md)) ranks four findings and names one next priority: a venue-matched BASE / FLOOR
comparison on consumed markets.

## Review check (done before this declaration)

- The reviewer's recount ([verify.py](../automatic/20261125_take_floor/verify.py), [evidence.log](../automatic/20261125_take_floor/evidence.log)) reproduces
  from the saved ledgers: 8 included markets have equal start and end prices and were paid as Up (venue: 50 / 50); 57 of 7,941 FLOOR takes
  and 664 of 7,602 exits are below 1 USDT under a buy-everything mapping; the fee-in-shares rule on fixed fills costs FLOOR $69.44 and BASE
  $77.62; 21 FLOOR exit fills increase |position|; 408 of 1,440 scheduled markets entered (selected on later book availability).
- Venue documentation (dev.predict.fun, read 2026-11-25): MARKET orders cancel their unfilled part at once; a buy is worth ≥ 1 USDT, a
  limit sell ≥ 1 share, a market sell ≥ 0.01 share; a buy pays its fee in shares (fee / p of them), a sell in USDT, makers 0; Up / Down
  pays 50 / 50 on equal prices. Speed bump: "Up/down markets ... have a `delayMs` and lock removal"; every order except a post-only
  LIMIT waits `delayMs` and cannot match or be removed meanwhile; "Removals aren't delayed, so makers can update their quotes before a
  waiting order reaches them."
- Value now in force: every market object in the latest complete hour of 2026-11-25 (824 objects, `/v1/categories/` as logged by the
  recorder) carries `speedBump = {delayMs: 150, isRemovalLocked: true, liveDelayMs: null, liveFrom: null}`. This read is venue metadata
  of a day outside the approved Nov 14–24 block and every declared race. On the v3 days the archived market objects have no
  `speedBump` key (reviewer), so the historical regime is unknown; the recorder's `/v1/markets/<id>` rows are order books.

## Days, policies, arms

- **Days:** the five v3 days 20261023, 20261031, 20261101, 20261102, 20261103 (consumed). Nothing else.
- **Policies unchanged:** BASE = the frozen take + maker exit; FLOOR = TAKE_EXIT_FLOOR_v1. Both run on `floor_replay.FloorBranch` (BASE
  with the floor 1,000 ¢ out of reach — exact parity with the v3 ledgers proven on 11 markets).

## Stage 1 — the speed bump alone (`S208`)

The FLOOR v1 harness unchanged except the take reaches the venue 208 ms after the decision (58 ms measured order-live + 150 ms
delayMs; the strict two-frame rule at that arrival). Complete markets (the 408). Settlement with canonical ties. Readings: totals,
by day, paired FLOOR − BASE, against the 58 ms result.

## Stage 2 — venue-matched (`VM`)

- **Population:** every scheduled market of the five days. Statuses: no usable entry (no quotable start book, tick unverified, …),
  no resolution. Included = usable entry and a resolution. The path runs from the frozen start to min(298 s, data end), data end =
  the last frame before the first frame gap > 1 s (or the last frame) + 1 s; no end mark is required. On complete markets the path
  equals the frozen `market_path` exactly (check).
- **Takes:** arrival 208 ms (central) and 258 ms (slow bound); strict rule unchanged. Native orders: the part of a take that opens or
  adds is a BUY of YES or NO, sent only if worth ≥ 1 USDT; the part that reduces is a SELL of held tokens (≥ 0.01 share); a flip
  whose buy remainder is below 1 USDT is cut to the reducing part (status `min_capped`); a take with no reducing part below 1 USDT is
  not sent (`below_min`).
- **Exits:** post-only limit SELLs of held tokens (whole shares ≥ 1), sized on tokens held; measured timing post 58 / cancel 37 /
  cancel-before-live 95 ms (the frozen engine's BASELINE values), desired-levels slot model.
- **Queue:** L — no credit for cancels ahead (only trades at our price move us up; a sweep through our price fills). Trade clock M50
  (primary) and M250 (checklist C2).
- **Fees and tokens:** a buy receives q − fee / p_native tokens and pays q × p_native; a sell pays its fee in USDT; makers 0.
- **Settlement:** YES pays 100 / 0 / 50 ¢ (Up / Down / equal), NO the complement. Resting orders and pending takes at the path end
  are unresolved: bounded by [no further fill, filled in full at their price].
- **Configs (both arms, all included markets):** `VM208-M50` (primary), `VM258-M50`, `VM208-M250`.

## Readings (predicates evaluated by the code)

Per config and arm: settlement net per scheduled market (÷ 1,440) and pooled, by day, days positive; completed turnover (round trips
after fees, by closing kind); held to settlement; inventory left at settlement (shares, loss if all settles against us); the unresolved
bound; C1 — the total without fills on exit orders younger than 100 ms (fixed fills); C3 — the cancel-pending share of exit fills and its $.
Paired FLOOR − BASE by day.

- `survives_venue_match` = FLOOR `VM208-M50` pooled total at its unresolved lower bound > 0 AND day totals > 0 on ≥ 4 of 5 days.
- `floor_beats_base_venue` = paired FLOOR − BASE (`VM208-M50`) > 0 on ≥ 4 of 5 days.
- `holds_at_258`, `holds_at_m250`: `survives_venue_match`'s rule for those configs.
- Stage 1: `speed_bump_cost` = S208 − (58 ms) per arm, pooled and by day (descriptive).

## Checks (each checker with a falsifier)

K1 path parity on complete markets; K2 the VM code with every venue feature off reproduces the v1 BASE / FLOOR ledgers exactly (every
40th complete market); K3 token / cash / settlement identities (selftests with known-bad inputs); K4 no sent take with a buy part below
1 USDT (the v1 ledgers must be flagged); K5 queue L (a synthetic shown-size drop must not move us up; M must); K6 ties valued at 50;
K7 lower ≤ point ≤ upper per market; K8 configured take delays and 58 ms exit activation; K9 one status per scheduled market.

## Budget

Stage 1 ≤ 1,400 CPU s; stage 2 ≤ 9,000 CPU s with a projection after the first day (cost × 5 × 1.1, or stop); RSS self-guard
2,500 MiB; outputs write-once. Development evidence on consumed days; no validation claim.

## Next (unchanged USER decision)

The Nov 14–24 forward test of TAKE_EXIT_FLOOR_v1 waits for this result. It will use the stage-2 harness through a superseding receipt
(v2) declared before any of those days is read, and its input chain needs declared wrappers (the loader's forward boundary, the
imputation-offset day list, padding hours), each proven on a consumed day into a separate output root first.

## Resource-only amendment R1 (before any stage-2 run; nothing else changes)

The stage-2 rehearsal (three consumed markets, nothing written; K1 path parity and K2 OFF parity both exact) measured 0.65–1.6 CPU s per
market-run. Six runs (3 configs × 2 arms) on ~1,439 included markets project to ~8,500 CPU s plus loading, at the 9,000 s cap. New caps:
stage 2 ≤ 11,000 CPU s (projection after 20261023 alone: cost × 5 × 1.1 ≤ 11,000, or stop) and ≤ 3,000 CPU s per day process. After the
projection the other four days run in parallel research slots (claude / claude2 / claude3 / claude4; the launcher's 5-process cap holds).
Merge, checks and the report live in a separate file (vm_report.py), so vm_replay.py stays byte-identical across the day runs.
