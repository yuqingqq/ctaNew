# TAKE-EXIT-REPLAY v2 — more markets (declared before the census and before any v2 outcome)

USER (2026-11-25): "can we test take + maker exit on more markets". v1 used the 19 complete markets that already had replay worlds.
v2 builds worlds for every market of consumed Oct 24–27 with the same loader and path builder, and runs the frozen v1 arms on every
market whose path is complete. Strategy code unchanged: `exit_replay.py` at ee0f16ee6, entry = SIGNAL-TAKE, exit = rung-0 closing
order, guard on the exit side. No new days, tuning or live use.

## Population (outcome-blind)

- **Candidates:** every registry market of 20261024–20261027.
- **Loader:** the unchanged `pf_stop_ev_minutes_inputs.load_day` via `pf_keep_pull.Tape`, one opening hour per batch. Its own day rule
  reads only parts inside the day and assigns statuses to sealed / forward / no-outcome windows.
- **Path status:** each market is run through the unchanged `market_path` (supported end 298 s). Every status is counted:
  complete / missing_endpoint / stale_future_book / no_quotable_book / …
- **Scope of the result:** only complete paths are replayed, so the result covers markets whose book stays two-sided and fresh to
  298 s. This is a selected population (mostly still-contested markets) and is reported as such.
- **Census:** stage 1 records the statuses before any strategy run (census.json, write-once).

## Inputs built for each complete market

- **Signal cache:** `bn_r250` from the raw-tape change points (take_signal_20261125/bn_changes.parquet) on the path grid t_start + 50 k,
  using the builder's exact convention. That convention is Binance event time + 4 ms, inclusive (equivalently a strict + 3 ms for whole-ms times); it was
  established from the saved caches with 0 bit mismatches on all 19 v1 markets. x_ask = float32(r250), x_bid = float32(−r250),
  status ok where finite. The cache's age field is set to NaN (unknown). The saved age counts from the latest bookTicker update of
  any kind, including size-only ones (verified: no mid-change offset reproduces it), and the guard only logs it; ON / OFF uses x and
  status alone (PressureGuard.signal).
- **Identity:** day, instrument_id, start_abs_ms = t_start, end_abs_ms, end_rel, opening_ms = ws, mark source, tape_id; the
  pressure runner's construction.
- **Fee:** the registry fee_bps (must be 200).

## Arms and budget

- **Arms:** PASSIVE_EXIT (primary) and HOLD (control), the v1 definitions unchanged. NOGUARD is not rerun.
- **Budget:** ≤ 900 CPU s (0.25 core-h) for the run stage. If the census-based projection (measured per-arm cost × complete markets
  × 1.2, plus loading) exceeds 900 s, run the largest prefix of the complete markets in sha256('txr_v2|day|mid') order, taken
  round-robin across days, that fits. The count and rule are recorded before any arm runs.

## Gates (any failure stops the run)

- The generated cache equals the saved cache on all 19 v1 markets: x float32 bits on both sides, and status. The guard's ON state is
  therefore identical.
- The v2 PASSIVE_EXIT and HOLD ledgers equal the v1 ledgers exactly on those 19 markets: fills, orders, requests, value, cash,
  position, commands, grid, signal log, IOC results, and the guard log with its logged-age field excluded (the only difference
  allowed).

## Readings (computed; v1's, on the larger population)

- PASSIVE_EXIT pooled END net > 0; days positive (≥ 3 of 4 = robust); markets positive.
- PASSIVE_EXIT − HOLD.
- Net without the value of exit fills on orders < 100 ms old.
- Per-day and per-hour-of-day distribution, and top-10-market concentration.
- v1's 19 markets versus the new markets separately: is v1 representative?

Development evidence on consumed days; no validation or promotion.
