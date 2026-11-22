# ASYNC-EXECUTION-SENSITIVITY-v1 — declaration

2026-11-22, session aea5bfb9. This is review `b8ab077a`'s ONE next priority: the two ASYNC_FAST10 cases of the USER's concurrency amendment `c3fe1d4a`, which supersedes the FAST10 notice changes of `9f7fe703`.

**Approval.** The USER approved it on 2026-11-22: ≤ 0.25 additional core-hours for the whole batch and a ladder-line ceiling of 2.50 including review.

**The question:** would the same pressure ladder rescue itself under a faster, concurrent execution contract, at the current queue and at the hypothetical front queue?

This is a combined execution-profile sensitivity:
- timing and concurrency change together, so it is not wire latency;
- not an executable contract, an upper bound, a validation or a profitability claim;
- on the 19 complete consumed Oct 24–27 markets;
- no new data, threshold, size, cap or parameter search;
- POSITIVE-PRESSURE-GUARD-v1 stays stopped.

The specification is the header of `predictfun/research/pf_ladder_async_fast.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any replay)

| Item | File |
|---|---|
| Runner (ProfileEngine / ProfileBranch, arms, parts, projection, measurement, readings) | `pf_ladder_async_fast.py` |
| Selftest (groups 1–7: the amendment's preflight) | `pf_ladder_async_fast_selftest.py` |
| Mutation harness | `pf_ladder_async_fast_mutants.py`: 24 / 24 caught (`logs/mutants_try1`) |
| Identities, reused-record sha256s, parts, caps, ledger lists | `manifest.json` (write-once) |
| Independent checks | `fa_checks.py`: A / B / K / S / E / T / F / M / P / I and an 11-case falsifier |
| Contract-check rehearsal, counts only | `tools/rehearse_contract.py`: 0 violations on the 38 reused real baseline ledgers and on 6 synthetic runs; planted corruptions caught |
| Launcher | `launch_async_fast.sh` |

## Profiles

| | BASELINE | ASYNC_FAST10 |
|---|---:|---:|
| Submission → activation | 58 ms | 10 ms |
| Live cancel request → effect | 37 ms | 10 ms |
| Cancel of a still-pending post | max(req + 37, submit + 95) | max(req + 10, submit + 20), a per-order race rule |
| Fill notice / cancel acknowledgement after effect / post notice | 100 / 100 / 0 ms | 100 / 100 / 0 ms (unchanged) |
| Slot model | frozen: a retiring order holds its slot until acknowledged | desired levels: a cancel request frees the order's desired slot |

**What stays the same in both profiles:**
- A retiring order stays unclosed until its acknowledgement. Its known remaining stays reserved against the 200-share cap and the desired-exposure sizing, with no early release and no overlap allowance.
- Its price stays locked, and own-cross protection still sees it.
- The 50 ms grid, the received Predict / Binance clocks and the pressure cache are unchanged, so no event is exposed sooner.
- The frozen PressureGuard policy, trades-before-commands ties, inclusive activation and strict removal.
- The near-END check uses each profile's activation.
- ASYNC settings record the effective timing and slot model. BASELINE keeps the frozen settings, for parity.

## Arms

| Arm | Content |
|---|---|
| PARITY_BASELINE | BASELINE profile, current queue, through the wrapper; pilot markets only. It must equal the accepted pressure arm on every pilot market. |
| CURRENT_ASYNC_FAST10 | ASYNC_FAST10, the engine's current queue |
| FRONT_ASYNC_FAST10 | ASYNC_FAST10 with `pf_ladder_queue_front`'s one-time zero external queue at activation |
| CURRENT_BASELINE (reused) | QUEUE-PRIORITY-SENSITIVITY-v1's CURRENT_QUEUE records, bound by sha256 in the manifest |
| FRONT_BASELINE (reused) | Its FRONT_AT_ACTIVATION records, bound the same way |

## Parts and gate

- **Pilot:** the first two complete markets of each date (8 markets, 3 arms each), reused in the result.
- **Rest:** the other 11 markets (2 arms each). They run only if the write-once projection fits:
  - this batch's spent core-hours + the projected rest + a 60 s tail ≤ 0.25;
  - the line + those + a 0.03 review reserve ≤ 2.50.
- **Otherwise:** stop.

## Measurement and readings (declared before any replay)

**Measurement:**
- per arm, the frozen `arm_row`. Its `withholds_near_end` label uses the frozen 58 ms window for every arm;
- the class table;
- the treatment census;
- blocks (from each arm's saved orders and closures):
  - desired-quote occupancy;
  - retiring order-ticks;
  - cap / near_end / own_cross / unavailable_price constraints;
  - cap refusals made while that side had a retiring generation, which separates idle-by-risk from idle-by-slot.

**Contrasts:**

| Contrast | Definition |
|---|---|
| EXEC_AT_CURRENT | CURRENT_ASYNC − CURRENT_BASELINE |
| EXEC_AT_FRONT | FRONT_ASYNC − FRONT_BASELINE |
| QUEUE_AT_FAST | FRONT_ASYNC − CURRENT_ASYNC |
| INTERACTION | FRONT_ASYNC − CURRENT_ASYNC − FRONT_BASELINE + CURRENT_BASELINE |

For each contrast:
- END and eligible +1 s, each per day: POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED;
- economic: IMPROVES / WORSENS / MIXED.

Also every arm's absolute END, per day and pooled.

**reading** = `nonnegative_development_result` if any arm's pooled END ≥ 0, else `assumptions_do_not_rescue`.

How each outcome is used:
- A mixed or negative result starts no tuned speed / queue variant.
- A nonnegative one would motivate timing feasibility and separately approved untouched validation, not promotion.
- Four days: point estimates only.

## Checks (`fa_checks.py`, its own code)

| Family | Checks |
|---|---|
| A | Identity: frozen files, parent and reused records with their bytes, this run's records, the measurement hashes |
| B | Membership |
| K | Parity on the 8 pilot markets, compared here |
| S | ASYNC settings and profile records |
| E | The execution contract recomputed per arm from the saved orders, fills and closures: activation, removal rule, and at every POST the desired / unclosed slot count, the price lock, the reservation room under the 200 cap, own-cross and near-END |
| T | The front treatment |
| F | Fill classes and volume conservation |
| M | END / +1 s / shares, the class and block tables |
| P | The contrasts and the reading |
| I | Batch ≤ 0.25, peak < 1,536 MiB, line ≤ 2.50 |

The falsifier refuses eleven corruptions while the clean evidence passes.

## Resources

At freeze the batch has used about 0.03 core-hours: selftests, one 24-mutant run and the contract rehearsal. The line is about 2.20. The projection gates the rest. Jobs go through `run_light.sh`, PF_SLOT claude..claude4, at most 4 at once, under the 1,536 MiB checkpoint.
