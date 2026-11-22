# ASYNC-EXECUTION-SENSITIVITY-v1 — receipt

2026-11-22, session aea5bfb9.
- **Specified by:** review `b8ab077a`'s ONE next priority, with the USER's amendment `c3fe1d4a`.
- **Approval:** the USER approved ≤ 0.25 additional core-hours and a line ceiling of 2.50.
- **Frozen:** `09418aff`, before any replay.
- **Run:** 4 pilot dates in parallel → write-once projection (run_rest true) → 3 rest dates → measurement → checks.
- **Scope:** 19 complete markets, consumed Oct 24–27. **A combined execution-profile sensitivity** (timing and concurrency together), not wire latency, an executable contract, an upper bound, a validation or a profitability claim. Four days: point estimates only.

## Integrity

| Item | Result |
|---|---|
| Parity | PARITY_BASELINE, the wrapper in BASELINE mode, = the accepted pressure arm on **8 / 8** pilot markets: fingerprint, fills, orders, requests, value, guard log, command array, queue records |
| Settings | Every ASYNC record: activation 10 / live cancel 10 / pending 20 ms, desired_levels, notices 100 / 100 / 0 |
| Treatment | FRONT_ASYNC treated every accepted generation |
| Independent checks | `fa_checks.py` A / B / K / S / E / T / F / M / P / I all pass. **E** recomputed the execution contract for all four measured arms on all 19 markets from the saved orders, fills and closures, with **0 violations**: activation, removal rule, and at every POST the slot count, price lock, reservation room under the 200 cap, own-cross and near-END |
| Falsifier | 12 / 12 corruptions refused |
| Mutation | 24 / 24 caught |
| Resources | Batch **0.201 core-hours** (cap 0.25; the projection allowed 0.223). Line **2.375 of 2.50**. Peak 1,187 MiB |

## Result

| Arm | END (c) | Eligible +1 s (c) | +1 s c / share | Filled shares |
|---|---:|---:|---:|---:|
| BASELINE (no guard; context) | −209,558 | −203,084 | −2.10 | 96,991 |
| CURRENT_BASELINE (accepted pressure ladder) | −86,110 | −85,164 | −1.08 | 79,399 |
| FRONT_BASELINE | −65,937 | −61,812 | −0.53 | 117,506 |
| CURRENT_ASYNC_FAST10 | −68,320 | −78,486 | −0.98 | 80,828 |
| FRONT_ASYNC_FAST10 | −86,961 | −61,052 | −0.48 | 126,504 |

**Computed reading: `assumptions_do_not_rescue`.** Every arm is negative pooled and on every day.

Contrasts (days Oct 24 / 25 / 26 / 27):

| Contrast | END total (by day) | +1 s total (by day) | economic |
|---|---|---|---|
| EXEC_AT_CURRENT (CURRENT_ASYNC − CURRENT_BASELINE) | +17,790 (+10.6k / +6.5k / −6.7k / +7.4k); 11 of 19 markets better | +6,678 (+1.3k / +5.4k / +6.6k / −6.6k) | MIXED |
| EXEC_AT_FRONT (FRONT_ASYNC − FRONT_BASELINE) | −21,024 (−14.4k / −1.5k / +0.8k / −6.0k) | +761 (−3.5k / −1.7k / +5.2k / +0.7k) | MIXED |
| QUEUE_AT_FAST (FRONT_ASYNC − CURRENT_ASYNC) | −18,641 (−1.1k / −6.8k / +4.5k / −15.3k) | **+17,434, better every day** | MIXED |
| INTERACTION | −38,813 | −5,918 | MIXED |

**Read the END contrasts with care.** The movement from +1 s to END is −0.9k / −4.1k / +10.2k / −25.9k c for CURRENT_BASELINE / FRONT_BASELINE / CURRENT_ASYNC / FRONT_ASYNC. That is directional inventory exposure, and it is as large as the contrasts themselves. The +1 s contrasts are the closer measure of execution.

**Sweeps still carry the loss.** Through-price fills at eligible +1 s:

| Arm | Through-price shares | c / share | +1 s value (c) |
|---|---:|---:|---:|
| CURRENT_BASELINE | 41,632 | −2.09 | −86,932 |
| CURRENT_ASYNC | 40,446 | −2.06 | −83,265 |
| FRONT_BASELINE | 31,905 | −2.59 | −82,694 |
| FRONT_ASYNC | 33,306 | −2.41 | −80,105 |

The faster contract barely changes sweep exposure at the current queue. Same-price fills stay small and positive at +1 s: +0.12 c/share for CURRENT_ASYNC, +0.21 for FRONT_ASYNC.

**What the faster contract does change** (ASYNC vs BASELINE, at current / front queue):
- desired-quote occupancy: 0.611 → 0.649 and 0.559 → 0.624;
- both-sides quoting share: 0.402 → 0.436 and 0.356 → 0.428;
- generations: +21 % / +21 %;
- pressure pulls: +34 % / +4 %;
- withholds: +51 % / +21 %;
- own-cross blocks: 1,835 → 3,697 and 1,646 → 3,115;
- mean |inventory|: 83.9 → 77.8 and 84.7 → 79.0 shares.

**Cap refusals are 0 in every arm**, so the blocks table cannot show idle-by-risk as cap refusals. The frozen inventory sizing never proposes more than the reservation room; risk-limited idleness appears inside the policy's own WAITs and sizes. Retiring order-ticks rise from 11.9k to 14.0k at the current queue and from 11.8k to 13.7k at the front queue: more cancels are in flight.

## What this shows and does not show

**Under these favourable execution assumptions** — 10 ms placement and cancel, concurrent sends, and even perfect front priority — **the fixed pressure ladder still loses on every day.**
- The best +1 s outcome is FRONT_ASYNC at −0.48 c/share, −61,052 c.
- The faster contract mainly raises quoting activity and occupancy. It barely reduces the through-price sweep loss.
- So the residual short-horizon loss is not removed by acting 27–48 ms sooner on the same decision-time information. That is consistent with the earlier residual audit (74 % of the loss was never flagged in time).

It does not show:
- whether better or earlier information could avoid sweeps;
- whether any achievable latency is near 10 ms;
- whether a different quoting stance could break even.

Per the declared reading, no tuned speed or queue variant follows.

Limits:
- Unchanged external flow.
- Through-price fills ignore queue in the engine.
- END includes directional exposure.
- The withholds_near_end label uses the frozen 58 ms window for every arm.
- Four consumed days.

**Largest evidenced limitation:** the policy's decision-time information does not anticipate the sweeps that carry the loss. Faster execution and better priority leave about −2.1 to −2.4 c/share on through-price fills.

Files:
- `outputs/` holds the measurement copies (`arms`, `parity`, `classes`, `treatment`, `blocks` .csv.gz and `audit.json`);
- `checks.json` and `checks_falsifier.json`;
- `logs/`, including `projection.json` and the contract-check rehearsal.
