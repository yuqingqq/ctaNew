# TAKE-EXIT-REPLAY v3 — the frozen rule on other days (declared before any v3 data is read)

USER (2026-11-25): "rerun the frozen rule on other days as well. check the result".

## Rule and simulator settings (frozen; no tuning)

The strategy is unchanged: SIGNAL-TAKE entry and the PASSIVE_EXIT exit (exit_replay.py, ee0f16ee6). It runs under the two simulator
settings that bound the result (take_exit_review_20261125):

- **LENIENT40:** the frozen v2 simulator. IOC against the last venue frame before its effect, 40 ms.
- **STRICT58:** the review's conservative bound. The strict two-frame IOC availability rule (strict_rerun.StrictIOCEngine) and the
  measured 58 ms order-live delay.

HOLD and NOGUARD are not rerun.

## Days

The new days are 20261023, 20261031, 20261101, 20261102 and 20261103. These are all the project days that are both:

- permitted by the loader's own day rule (sealed block 2026-10-28 00:00 → 10-31 00:00, forward boundary 2026-11-04 00:00);
- not used to develop this strategy (developed on 10-24..10-27).

They are consumed project days (identity day, earlier studies), so this is out-of-sample for this rule, not untouched-day
validation. Days from 2026-11-04 on are reserved for the pending forward races and are NOT read here.

## Binance input (raw bookTicker, the bn_events.py code unchanged, new allow-list)

- **Stems:** each new day 00–23Z; 20261024_00 only for Oct 23's last seconds of cross-hour flush. Nothing from 10-22, the sealed block
  or 11-04 is opened.
- **Segments:** two contiguous segments, A = Oct 23 and B = Oct 31–Nov 3. The signal for a market uses change points of its own
  segment only, so there is no cross-gap reference.
- **Boundary:** the 20261103 23:55 market is boundary-censored, because its last flush seconds sit in the forward 11-04 file.
- **Builder:** the bn_events loop is reproduced in other_days.build_changes with two additions.
  - Rows before a segment's start are dropped right after parsing. The collector's cross-hour flush puts the last ~5 s of sealed
    Oct 30 into the Oct 31 00Z file, and none of those rows is kept.
  - The carry is discarded at a segment start.
  - Validation gate before any v3 data: on the consumed Oct 24 stems it must reproduce the original bn_changes exactly inside
    Oct 24 00:00:30–23:59:30.
- **Cache convention:** as v2 (event + 4 ms inclusive; age NaN).

## Population and runs

- **Population:** every registry market of the new days through the unchanged loader and path builder, with every status counted.
- **Runs:** both settings on every complete path, in one pass per opening-hour batch.
- **Budget:** ≤ 1,100 CPU s for the run, with a projection after the first day (stop if the projection exceeds the cap). The Binance
  build is ≤ 150 s.

## Readings (computed)

- **Per setting:** pooled END net, per-market net, days positive (of 5), markets positive, net without young exit value,
  and top-10-market share.
- **Comparison with in-sample:** v2 LENIENT40 +$14.00 / market, S2 STRICT58 +$5.76 / market.
- **Robustness:**
  - "Holds out of sample" = STRICT58 pooled > 0 and positive on ≥ 4 of 5 days.
  - "Similar size" = per-market STRICT58 within 50–200 % of in-sample.
- **Checks:**
  - the stems read equal the allow-list;
  - delays 40 / 58 ms per setting;
  - the strict cap respected;
  - exit orders on the closing side at rung 0;
  - IOC fills re-derived (limit, fee, ledger), ledger identities;
  - triggers recomputed from the v3 cache;
  - a no-trigger falsifier (takes disabled → no orders).

Development evidence on consumed days; no validation or promotion claim.

## Resource-only amendment R1 (after the first-day projection refusal, before any further run)

- **Refusal:** the run refused at its declared gate. 20261023 took 202 CPU s, and the 5-day projection with a ×1.2 margin was
  1,211 s against the 1,100 s cap (run_light 06:53:17Z, rc = 1; v3_projection.json). The gate worked as declared.
- **New cap:** total ≤ 1,300 CPU s (≈ 0.36 core-h, under 30 min wall).
- **Oct 23 reuse:** the 20261023 ledgers already written by the frozen code are reused. A fresh 20261023 census (paths only, no
  strategy run) must show exactly the same complete-market set as the saved ledgers, with both settings present for each market.
  Their sha256 are recorded at resume time.
- **Resume:** runs 20261031..20261103 with the same code. Projection after 20261031 (cost × 3 remaining days × 1.1 plus spent
  ≤ 1,300) or stop.
- **Unchanged:** no rule, setting, day or check changes.

**R1 note (first resume attempt):**
- **What happened:** it stopped at the 2,500 MiB RSS self-guard (peak 2,662 MiB, under the 4 GiB hard cap) after writing 2 Oct 31
  markets × 2 settings. Cause: the resume kept the full 20261023 ledger records in memory.
- **Fix:** keep only metrics and sha256.
- **Existing ledgers:** any ledger already on disk is recomputed and must be byte-identical, a determinism check.
- **CPU:** spent CPU now counts every txr3_run / txr3_resume attempt from run_light.log against the 1,300 s cap.
