# QUEUE-PRIORITY-SENSITIVITY-v1 — declaration

2026-11-22, session aea5bfb9. This implements the USER's plan `425e9fbc` (`reviews/automatic/20261122_queue_sweep_direction/plan.md`).

**Approval.** The USER approved it on 2026-11-22 with:
- the plan's caps: ≤ 0.25 additional core-hours for the whole development batch, and the ladder line ≤ 2.25 including review;
- the plan's pilot: the first two complete markets of each date.

**The question:** would the SAME accepted pressure ladder do better if every new order started at the front of its price level?

This is one paired execution sensitivity:
- on the 19 complete markets of the consumed Oct 24–27 (all 96 statuses stay in the parent receipts);
- no new day, fit, ladder, size, threshold, inventory parameter or trained guard;
- POSITIVE-PRESSURE-GUARD-v1 stays stopped.

**FRONT_AT_ACTIVATION is hypothetical.** It is not an executable policy, a monetary upper bound or exchange-truth priority. The simulator's unchanged-external-flow assumption is strongest here.

The specification is the header of `predictfun/research/pf_ladder_queue_front.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before any replay)

| Item | File |
|---|---|
| Runner (treatment engine, arms, fill classes, parts, projection, measurement, readings) | `pf_ladder_queue_front.py` |
| Selftest (groups 1–7: the plan's fixtures) | `pf_ladder_queue_front_selftest.py` |
| Mutation harness | `pf_ladder_queue_front_mutants.py`: 26 / 26 caught (`logs/mutants_try1`) |
| Identities, parts, caps, line ledger lists | `manifest.json` (write-once) |
| Independent checks | `fq_checks.py`: A / B / K / T / F / M / P / I and a 10-case falsifier |
| Launcher | `launch_queue_front.sh` |
| Scope read (counts only, before approval) | `tools/scope_ahead0.py` and its log |

## Arms and treatment

| Arm | Content |
|---|---|
| CURRENT_QUEUE | The frozen PressureGuard (the accepted FULL_11 / Q5 pressure ladder), run on `FrontLadderEngine(front=False)`: the override is engaged but never triggered. It must equal the accepted pressure arm on every market: fingerprint, fills, orders, requests, value, guard log, command array and queue records. |
| FRONT_AT_ACTIVATION | The identical policy on `FrontLadderEngine(front=True)`. The frozen `add9` runs unchanged; then every accepted new generation (new, remaining > 0, not refused) gets its external queue ahead set to 0, once, at activation. No outcome selects the generations. |

What the treatment keeps:
- The engine's running minimum never raises the queue, so the order stays first at its level.
- Same-price prints fill it before the external queue; through-price trades fill it as before.
- Own priority, partial remainders, the per-trade volume check, post-only refusal and strict removal are unchanged.
- The frozen engine file is not edited.

Both arms run the complete feedback loop. Each arm's own fills feed its notices, reservations, inventory and later decisions.

## Fill classes and parts

**Fill classes.** Each fill is classed from the engine's own trade row: `same_price` (trade price = order price) or `through` (better for the aggressor). Any other relation refuses. The classes are explanatory, never an input.

**Parts:**
- `pilot`: the first two complete markets of each date, reused in the full result.
- `rest`: the others (Oct 24: 5, Oct 25: 3, Oct 26: 3, Oct 27: none).

The rest runs only if the write-once projection fits. The projection measures the pilot's mean load, its per-market wall time and CPU per wall-second, and requires both:
- this batch's spent core-hours + the projected rest + a 60 s tail ≤ 0.25;
- the line + those + a 0.03 review reserve ≤ 2.25.

Otherwise the round stops: no outcome-based continuation and no repeated pilot.

## Measurement and readings (declared before any replay)

**Measurement:**
- per arm: the frozen `arm_row` (END, eligible +1 s, per share, participation, inventory, pulls / withholds, first-review churn, generations; BASELINE from the pressure receipt's disabled arm as context);
- lost / added fills;
- the class table per arm × market × class (fills, shares, END, eligible +1 s value and shares, positive / negative parts, fills sharing a public trade, 5 / 30 s marks with their censored counts);
- the treatment census.

**Readings:**
- each arm's absolute END sign, per day and pooled;
- FRONT − CURRENT for END and for eligible +1 s, each per day as POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED;
- **economic**: IMPROVES iff both are POSITIVE_EVERY_DAY, WORSENS iff both NEGATIVE_EVERY_DAY, else MIXED;
- adds_fills (more filled shares, pooled);
- recovered = END change / |CURRENT END|;
- the +1 s mix split into a volume term (share changes at CURRENT's per-share, by class) and a value term.

**The reading map** (the plan's interpretation):

| Reading | Condition | Implication |
|---|---|---|
| `queue_helps_policy_nonnegative` | IMPROVES and FRONT END ≥ 0 | Supports work on acquiring and preserving priority; it does not prove a method. |
| `queue_helps_but_insufficient` | IMPROVES and FRONT END < 0 | Queue helps but is insufficient. |
| `front_priority_does_not_rescue` | More fills and a pooled END change ≤ 0 | Stop this hypothesis; no intermediate queue advantage is tuned. |
| `no_added_fills` | No added fills | — |
| `mixed` | Otherwise | — |

Four consumed days: point estimates only. No automatic variant, fit, classifier or data request follows any reading.

## Checks (`fq_checks.py`, its own code)

| Family | Checks |
|---|---|
| A | Frozen files committed. Parent records = the manifest's = the committed copies, with their gzip / npz bytes. This run's records, bytes and treatment identity. The measurement hashes. |
| B | The 19 markets once each; the pilot / rest split as declared. |
| K | CURRENT = the accepted pressure arm on every market, compared here (not read from the module's flags). |
| T | FRONT's every accepted generation at ahead 0, with front_set = accepted_new = the accepted count. CURRENT untreated. |
| F | Classes recomputed; own fills ≤ each public trade's size. |
| M | END from the fills, eligible +1 s and shares against arms.csv. classes.csv, lost_added.csv and treatment.csv recomputed. |
| P | The readings recomputed. |
| I | Batch ≤ 0.25, peak < 1,536 MiB, line ≤ 2.25. |

The falsifier refuses ten corruptions while the clean evidence passes.

## Resources and limits

**Resources.** As of this freeze, the batch has used the scope read, the selftests and one 26-mutant run, about 0.037 core-hours. The line is about 2.013, including this batch.

The estimate from measured replays is about 0.19 core-hours for the whole batch. Jobs go through `run_light.sh` with PF_SLOT=claude..claude4, at most 4 at once, absolute paths, under the 1,536 MiB checkpoint.

**Limits:**
- The simulator does not model how real makers would respond to our earlier fills.
- Through-price trades ignore queue in the engine.
- 8.4 % of accepted generations already start at 0 ahead (the scope read). Their FRONT changes arise only through feedback.
