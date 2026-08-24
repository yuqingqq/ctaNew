# Measurement-layer design — DA-State, EV-Markout, EV-Calibration

**Revision 13** — iterations **1-7** of `MEASUREMENT_PLAN_REVIEW_LOOP.md` applied.
*(This banner read "iteration 1 … ~25 MUST-FIX" through Revisions 2-7: a STALE
BANNER, not a provenance statement. The revision number was bumped five times and
the sentence beside it never was — the plan's own named failure mode, in its
masthead. Iteration 1 found ~25 MUST-FIX across conformance, omission and
buildability — a look-ahead clause inside the admissibility rule, two dead
authority pointers, and two figures contradicted by the Tier-2 output committed
against this plan. Iterations 2-5 added: an inverted FLB gate, three unratified
rule ids, a retracted weighting-arm near-identity, a generation-blind
double-count, and a frozen rule whose declared fields never reach its
evaluator.)*

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


P-2026-003. Design planner's output, 2026-08-20. **Plan only — no implementation.**
Refines `PRELIMINARY_PLANS.md` §1–3. Canonical contracts are in
`contracts/contracts_measurement_delta.yaml` (a proposed v12→v13 delta;
`contracts.yaml` is not edited here). Where this document and the YAML disagree,
the YAML wins — prose explains, it does not define. **CORRECTED 2026-08-23: that
pointer is DEAD and must not be followed.** `contracts_measurement_delta.yaml`
opens *"STATUS: SUPERSEDED HISTORICAL INPUT … **Do not merge this v12->v13
file** … Canonical v17 plus `PM_MEASUREMENT_PREREG.md` are authoritative."*
Canonical `contracts.yaml` is now **v22** and has no `MarkoutEstimate`,
`InferenceSpec`, `MarkoutComponent`, `AckBracket` or `ClaimLadder`; EV-Markout
produces `TerminalMarkoutObservation` / `MarkoutSummary` under
`R-GROSS`/`R-DUAL`. **The type authority is `contracts.yaml` v23** (in force per R-68; this read v22 until iteration 4), plus
`PM_MEASUREMENT_PREREG.md` for frozen infrastructure.** Concretely: §1.3's
`Coverage` field list (`admissible`, `rule`, `provenance`) is the v13 shape;
v22's `Coverage` has neither `admissible` nor `rule` and adds `strike_readable`,
because `R-ADMISS` now separates the coverage FACT from the admissibility
DECISION — so obeying the dead pointer rebuilds the exact conflation R-ADMISS
exists to prevent.

Every number tagged **[measured]** was computed during this planning pass from
`data/pm_5min/` and `data/mm_hf/`. Six of them refute or sharpen something the
corpus currently asserts; those are marked **[correction]**.

---

## 0. Why these three, and why first

The architecture's build order is `Known[V]` → EV-Markout/EV-Calibration →
mechanism experiments → any order. That ordering is not aesthetic. At `r = 2 s`
to expiry, the settlement diffusion being modelled is ~0.007 bps while a 1.7 s
head start on the settlement stream is worth ~0.08 bps — **inside the final
seconds the look-ahead is worth ten times the risk**, so a replay without a
knowledge-time layer does not produce a slightly optimistic number, it produces
free money. Meanwhile the strategy identity is already decided against us: the
book beats our forecast at every horizon by a stable 2.5–3.2 Brier points, so
this is spread capture, and **spread capture is a markout question, not a
forecasting question**. Neither harness exists. Nothing else should be built.

Status on disk: `DA-Normalize / DA-State / DA-Settlement` — none built.
`EV-Markout / EV-Calibration` — "methodology only". Adding module records for
them is open MUST-FIX M11-5/M12-5, not scope creep.

---

## 1. DA-State — the knowledge-time layer

### 1.1 What is actually on disk (measured, not assumed)

Three streams, three completely different lag regimes. The corpus discusses the
1.7 s figure as though it were global; it is not.

| stream | file | `recv_ns − payload_ts` p50 | p90 | p99 | negatives |
|---|---|---|---|---|---|
| PM CLOB market WS | `raw/<day>/<slug>.jsonl*` | **48 ms** (`price_change`), 52 (`book`), 62 (`last_trade_price`) | 593–810 ms | 3.7–5.6 s | **0** |
| PM prices WS (Chainlink TWAP) | `prices/crypto_prices_twap_sixty/` | **1,713 ms** | 2,189 | 2,605 | **0** |
| Binance bookTicker | `mm_hf/raw/bookTicker/` | ~80 ms | — | — | 0 |

**[measured]** and the TWAP stream decomposes exactly as the corpus says, on
145,054 messages:

```
t_event   = payload.timestamp     Chainlink observation
t_publish = envelope.timestamp    venue publish   t_publish − t_event  p50 1,456 ms
t_known   = recv_ns               ours            recv_ns − t_publish  p50   257 ms
                                                  total                p50 1,713 ms
```

Zero negative lags on any stream, on every symbol, at every quantile. No clock
faults. This decomposition is the reason imputation can be credible at all:
`D_publish` is a **venue** property and portable across deployments;
`D_transport` is **ours** and must be re-measured per host and region.

**[correction] The 1.7 s knowledge-time problem is specific to the settlement
stream.** The CLOB market data we would actually quote into is knowledge-clean
to ~50 ms. This matters: it means the peek risk is concentrated in exactly one
place (the settlement read at the end of a window), and every markout number is
essentially unaffected by it. That is a narrower and more defensible claim than
the corpus currently makes, and it should replace the global framing.

### 1.2 The imputation policy

The trap, restated precisely: `t_known` is `recv_ns`, an observation that exists
on **our** wire and nowhere else. Historical archives, Gamma/CLOB REST, and
on-chain records carry only the venue's timestamp. The natural implementation
`t_known := t_event` makes `t_known ≤ now` hold **by construction**, passes every
assertion `EV-Replay` will ever make, and returns the peek type-laundered. That
is strictly worse than no guard, because a legible bug gets fixed.

**Three source classes, and the rule for each.**

**Class A — OBSERVED.** A stream captured on our own wire, `recv_ns` present.
`t_known := recv_ns`; `t_known_err := SourceProfile.clock_err` (the local NTP
bound — small, *not* zero); `prov = OBSERVED`. Covers all three streams in §1.1.

Note the subtlety for **polled** sources (`markets.jsonl`, `resolutions.jsonl`):
`recv_ns` is the poll time, and the fact became true earlier. That makes
`recv_ns` an **upper** bound on knowledge time, which is the conservative
direction, so polled sources are legitimately OBSERVED. They are only wrong in
the safe direction.

**Class B — IMPUTED.** An event time exists, no observation on our wire, but the
same venue leg has been observed with `recv_ns` over an overlapping period.
```
t_known     := t_event + D̂(source)        D̂ from SP-Params.at(t_event)
t_known_err := q95(D) − q05(D)            from the same calibration
prov         = IMPUTED,  imputation_rule = <named rule id>
```
The calibration must come from the *same leg*. The corpus already burned this
once: a 471 ms latency measured on the Binance **spot mirror** was applied to the
TWAP stream, which is 3.6× slower — a keying failure, not an arithmetic one.

**Class C — ASSUMED.** No overlap, no measurement. `D_assumed` is a pessimistic
bound (the max observed on any comparable leg), `t_known_err` the full range,
`bias_direction = PESSIMISTIC`. R-PROV already bars ASSUMED values from gating a
decision. They may appear in a *reported* number, flagged.

**Never `t_known := t_event`.** Enforcement is structural, not disciplinary:

1. `Known` is constructible only through a factory bound to a `SourceProfile`. A
   profile with `has_recv_ns = false` **cannot emit `OBSERVED`.** This turns the
   failure from a code-review finding into a construction error.
2. `R-IMPUTE` asserts `t_known > t_event` **strictly** for every non-OBSERVED
   value. A zero delay is physically impossible; it is the laundering signature.
   This is testable, and it holds on every stream we have: zero negatives, zero
   zeros.
3. `t_known_err > 0` whenever `prov ≠ OBSERVED`, and a named `ImputationRule`.
4. `EventTime` and `KnowledgeTime` are distinct nominal types with **no shared
   comparison operator**. `t_event <= now` is a type error. Sorting a collection
   by `t_event` stays legal (a TWAP integral requires it) and stays safe, because
   the collection was truncated at the view boundary and no reordering of a
   truncated set can produce a future value.

**How replay refuses inside `t_known_err`.** Admission is on the upper bound, not
the point estimate:
```
t_known_hi = t_known + refuse_k · t_known_err          refuse_k ∈ SP-Params, default 1.0
admit(k, now)  ⟺  t_known_hi ≤ now
otherwise      →  Unavailable(reason = WITHIN_TKNOWN_ERR)
```
Composition generalises the `MAX(inputs)` rule so it survives nesting:
```
t_known      = MAX_i t_known_i
t_known_hi   = MAX_i (t_known_i + t_known_err_i)
t_known_err  = t_known_hi − t_known
t_known_prov = WORST_i prov_i           OBSERVED < IMPUTED < ASSUMED

  UNSPECIFIED, and an implementer must invent an answer a contract check then
  grades: composition assigns NO value to `t_event`, `imputation_rule` or
  `source`. Two consequences. (a) R-IMPUTE's `rule_named` check fires on any
  composite whose worst provenance is IMPUTED or ASSUMED -- reachable from a
  single IMPUTED input -- and composition names no `ImputationRuleId` for it, so
  the composite cannot be built conformantly. (b) `t_event` MAX-vs-MIN is not
  cosmetic: `EventTimeView` truncates on `t_event`, so the choice changes the
  LEAK-CANARY DELTA, and a delta of 0.0 is the `INVALID_UNBOUND_GUARD` verdict.
  A gate-bearing number is left to implementer discretion.

  AND THE SHIPPED WRITER DOES NOT USE THIS LATTICE. `da_state.py` implements it
  correctly (`KnownProvenance(IntEnum)`, composite worst-provenance asserted),
  but `evaluation_pipeline.py` writes
  `provenance = "OBSERVED" if provenances == {"OBSERVED"} else "MIXED"`.
  `"MIXED"` is a member of no ruled provenance enum -- the same defect §3.3
  corrects for `provenance = fitted`. All 2,031,865 committed rows currently read
  OBSERVED so nothing is numerically wrong YET; the moment one non-OBSERVED input
  appears, the receipt says MIXED, no consumer can tell IMPUTED from ASSUMED, and
  **R-PROV's ASSUMED-cannot-gate rule becomes unenforceable from the published
  artifact, silently and with no error.**
seq          = MAX_i seq_i
```
Taking the error of the argmax input instead is wrong: an input that is *earlier*
but much less certain can dominate the upper bound. This form is associative, so
a composite of composites does not silently lose its error bar.

Consequence, stated so nobody is surprised by it later: at `r = 2 s` with
`t_known_err` on the order of the 1,440 ms publish leg, **the entire endgame study
is refused on backfilled data.** That is the correct answer, and it is the answer
nobody reaches voluntarily once the numbers look good.

**The leak canary (`R-CANARY`).** A guard never observed to bite is
indistinguishable from a guard that is not wired. Every replay/evaluation run
executes twice — once through `StateView`, once through the deliberately leaky
`EventTimeView` — and reports the delta. EXP-M6 already produced one reading:
**99.3% at knowledge time vs 99.8% at event time**, a 0.5 pp bite. That is the
reference calibration. A later run reporting 0.0 pp has broken the harness, not
fixed the leak, and is marked `INVALID_UNBOUND_GUARD`. **RE-CORRECTED AT ITERATION 5 — ITERATION 4'S "CORRECTION" WAS WRONG ON FOUR
COUNTS AND POINTED AT RE-OPENING A GRANTED RULING.** It claimed the decision was
open, that both committed days were partial, that `reference_delta_pp` carried no
value, and that the deltas were Brier-scale and so incommensurable with 0.5 pp.
Verified against primary files: **R-7 is `AMENDMENT GRANTED, with the guard kept
live`** (`COORDINATION.md:6784`) — option 1, four conditions, applied and
verified; **zero of 100 canary receipts carry `partial: true`**;
`replay_canary.py:55` sets **`REFERENCE_DELTA_PP = 0.5`**, stamped on every
receipt; and the deltas carry
`metric: "winner_reproduction:S60(T)>=S60(t0)"` — **the same
settlement-agreement quantity as EXP-M6's 99.3 / 99.8**, not Brier. *Root cause,
recorded because it is the reusable part:* iteration 4 cited DA's own earlier
diagnosis without checking whether a ruling had superseded it — the stale-premise
class that R-40 clause 3 exists to prevent, applied to register rows and never to
plan edits.

**THE REAL DEFECT, WHICH THE FALSE CORRECTION HID.** `delta` is a **fraction**
and `reference_delta_pp` is in **percentage points**, they sit side by side in
one receipt, and `classify()` **never compares them**. btc 08-20's `0.0139` reads
36× *below* a reference of 0.5 at face value and 2.8× *above* it once aligned.
The fix is a ×100 unit alignment plus wiring a comparison that does not exist —
not deleting the reference. And the measured bite runs **0.000–1.389 pp with 13
of 21 receipts ABOVE the 0.5 pp reference**, mean 0.678 pp, so **§8 step 4's gate
is already satisfied by the committed receipts**
**— POPULATION: `canary_version == leak_canary_v2_r7` ONLY, twins deduped by content address, 21 coin-days over 2026-08-20/21/22, as of 2026-08-24.**
~~23 of 36~~ **WAS WRONG AND WAS STRUCK HERE AT ITERATION 8.** It pooled the 8 `leak_canary_v1`
receipts with the 35 `v2_r7` ones and counted content-addressed twins twice — the
exact double-count §4 warns about, in a paragraph citing §4. The correction was
recorded in §4 at iteration 5 and **the corrected statistic was left standing at its own
site for three iterations**: a vacated claim surviving in the text that uses it, which is
the class R-94 names for code. **AND THE CORRECTION ITSELF AGED**: it read *9 of 14 on 2
day-clusters*, which was exactly right when written and went stale when 2026-08-22 landed.
That is why the population and as-of are now stated inline — a bare count over a growing
corpus goes stale silently and reads as current. and the drift is a live R-7
condition-4 question. **R-7 IS VACATED (R-89) — NOTHING BELOW MAY CITE IT AS AUTHORITY.** OPS ruled it DIES: the licence and the challenge were computed on the **same 14 coin-days**, so R-7's own basis does not support it. **THE AMENDMENT SURVIVES — RE-FOUNDED ON ORDERING (R-94, 2026-08-24), NOT ON A DISTRIBUTION.**
BE's re-founding succeeded, so `BOUND_ZERO_SCORE_DELTA` and the removal of the
`disagreements == 0` fatal branch are **LICENSED AGAIN, on a different basis than the one
they were proposed with**: pre-amendment, zero disagreements with zero harm was INVALID
while five disagreements with the *same* zero harm was fine — the strictly safer
observation was punished more harshly, so the rule was **non-monotone in its own
evidence**. No distribution enters that argument, nothing about it can drift, and **G=2
cannot break it**. R-7's licence stays dead and nothing may cite it. Anything that was
inheriting PROVISIONAL / UNLICENSED status from this amendment no longer does. The text below is retained as the record of what was granted and vacated: `DA_PIPELINE_OUTAGE_DIAGNOSIS_2026-08-23.md` puts three options with their costs and states *"that is an argument, not a decision, and **the decision is not DA's**"*; this plan stated option 3 — *"leave it fatal"* — as design. **Measured cost:** a zero delta is a property of the DAY'S DATA, not the harness — the leaky twin read past the boundary **568 times on doge** and the knowledge view refused every one, so wiring was never in question; what was zero is `decision_disagreements`, i.e. the look-ahead never flipped a winner. **About seven days in ten abort on this guard by chance** (P ≈ 0.686 for a 7-coin day; 2 of 8 coin-days observed), and since `run_from_partitions` raises and aborts every remaining coin, **~~this is why both committed Tier-2 days are partial.~~ **FALSE, struck at iteration 6:** zero of 100 canary receipts carry `partial: true`, and both days are COMPLETE. **BUILD THIS:** nothing — the abort path was removed by R-7 and `run_from_partitions` raises only on `INVALID_UNBOUND_GUARD`.** v23 carries a third status for exactly this case — `BOUND_ZERO_SCORE_DELTA` — which the binary rule above does not use. *Second defect, same paragraph:* ~~the on-disk deltas are 0.0035–0.0139 on a Brier scale~~ **STRUCK AT ITERATION 6 — this is one of the very claims the block above lists as WRONG, still asserted verbatim later in the same paragraph.** Iteration 5 corrected the head and left the tail: the identical shape it was correcting. **BUILD THIS:** the deltas carry `metric="winner_reproduction:S60(T)>=S60(t0)"` — the same settlement-agreement quantity as EXP-M6's 99.3/99.8 — so multiply by 100 and compare against `reference_delta_pp`. *Superseded:* while "0.5 pp" is a settlement-agreement percentage, so the stated reference calibration is not the quantity the shipped canary compares against; ~~`LeakCanary.reference_delta_pp` is typed and carries no value anywhere.~~ **FALSE, struck at iteration 6:** `replay_canary.py:55` sets `REFERENCE_DELTA_PP = 0.5` and `:397` stamps it on every receipt. **BUILD THIS:** compare `100 * delta` against `reference_delta_pp` — a comparison `classify()` does not currently make.

`Known.seq` is **restored** (it was in the reviewed ITER2 design and lost in a
later rewrite). The Environment seam contracts deterministic tie-breaking and
warm-restart parity; neither is expressible without a total order within source.

### 1.3 Coverage

The shipped `Coverage{segments, complete_frac}` keeps non-contiguity and loses
the thing that matters. `complete_frac` is **span**-fraction; the estimand is
**weighted**. Extend (never replace):

```
Coverage{ field, target, covered, gaps[Gap], weight_missing, tail_deficit,
          observed_n, expected_n, duplicate_n, max_gap, complete_frac,
          admissible, rule, provenance }
Gap{ from, to, cause, evidence }
cause ∈ STREAM_GAP | COLLECTOR_RESTART | SLOW_CONSUMER | VENUE_OUTAGE
      | NOT_YET_SUBSCRIBED | POST_UNSUBSCRIBE | UNKNOWN
```

**Why weight and not span.** **[measured]** The TWAP stream has **77 gaps > 5 s
per symbol with a maximum of 44 s**, and the gap set is *identical* across all
eight symbols (77/77/77/…, max 44 s or 43 s) — these are collector-level,
common-mode outages, not per-feed events. The settlement statistic averages over
`w = 60 s`. **[ARITHMETIC CORRECTED 2026-08-23: the scenario as stated does NOT pass —
1 − 44/310 = 0.858, and 0.858 ≥ 0.90 is false, so `complete_frac` already
rejects it. The extension is still justified — **but NOT by `max_gap`, and iteration 2's
correction named the wrong clause (corrected at iteration 3).** Measured on the
same 4,032 committed coverage rows this document's §1.4 correction uses,
`max_gap ≤ 30 s` excludes **ZERO** rows that `complete_frac ≥ 0.90` does not
already exclude: MAX_GAP-fail-and-COMPLETE_FRAC-pass **0**, both-fail 203,
COMPLETE_FRAC-fail-only 414. **`max_gap` has no marginal screening power on this
corpus**, and it is a span-style scalar besides, so naming it argues against the
thesis this subsection exists to make. The clauses that do the work are
`PROTECTED_SPAN` and `MAX_WEIGHT_MISSING`, each failing **93.6 %** alone (§1.4) —
both weight-style, which is the argument. **This matters forward:** §3.1 says
*"an `A-CALIB-1` rule shaped like A-TWAP-1 is owed"*, and an author shaping it
from this subsection would carry across a clause of zero measured power. The two
prior statements of this justification were both wrong: the original arithmetic
(0.858 ≥ 0.90 is false, so the 44 s scenario never passed) and iteration 2's
substitute; the stated reason was arithmetically wrong and
a reviewer checking it concludes the opposite of what is intended. The same error
sits in `contracts_measurement_delta.yaml` with different numbers again.]**
A 44 s hole can consume 73% of the decisive averaging window while
`complete_frac` over `[t0−5s, T+5s]` still reports 0.86 and the window passes.**
`weight_missing` reports it as ~0.73 — **and this is the definition the SHIPPED
rule uses** (`coverage_ledger.py`: `weight_missing = protected_missing /
protected_expected`, the missing fraction of the 60 protected slots), confirmed
by `PM_MEASUREMENT_PREREG.md`. The superseded delta YAML defines it as *"the
fraction of the ESTIMAND's weight absent … ~1.0"*, a near-binary rather than a
graded 0..1, which gives a **different admissible set** under the frozen
`max_weight_missing: 0.0` gate. §1's dead precedence pointer sent an implementer
to the wrong one; the shipped semantics govern.
`tail_deficit` reports the right-edge hole, `tail_deficit` reports the right-edge hole,
which is structurally at least the 1.7 s knowledge lag at every settlement read
and larger whenever the stream was also down.

**Why `cause` is load-bearing.** **[measured]** From `collector.log`: 47 WS
disconnects, of which **27 are `1013 slow consumer: send buffer full` — our own
client failing to keep up — and 32 of 47 are BTC alone.** This is not a venue
outage. It is data loss that is *activity-correlated by construction*, on the
symbol carrying 82–85% of notional, missing precisely in the busiest intervals a
markout is trying to measure. `NOT_YET_SUBSCRIBED` may be pooled over;
`SLOW_CONSUMER` may not. Conflating them into one "gap" count is the difference
between missing-at-random and missing-not-at-random.

### 1.4 The admissibility rule

Pre-registered, frozen with a `spec_hash`, and **never applied exclude-only**.

```
A-TWAP-1   (settlement / calibration)
  field    twap60[symbol]
  target   [t0 − 5s, T + 5s]
  require  complete_frac        ≥ 0.90            (expected_n = span_s at 1 Hz)
       AND weight_missing       = 0.0             <- WAS MISSING FROM THIS PLAN
  [!] AT ITERATION 5 THIS CLAUSE AND THE PROTECTED-SPAN CLAUSE WERE DEMONSTRATED
      TO BE ONE PREDICATE, AND THE FROZEN RULE'S FIELDS DO NOT REACH THE
      EVALUATOR. `coverage_ledger.py` hardcodes `protected_start_ms = end_ms -
      60_000` and `protected_expected = 60_000 // cadence_ms`; fail-closed
      probes show `protected_span`, `pre_boundary_span` and
      `require_known_at_boundary` are ABSENT from the evaluator, though
      `config/a_twap_1.json` freezes them inside `spec_hash ab098a55…` and v23
      declares `protected_span_expr` and `require_known_at_boundary` as
      `AdmissibilityRule` fields. On the committed corpus `PROTECTED_SPAN` and
      `MAX_WEIGHT_MISSING` each fail 3,774 rows with **symmetric difference 0** —
      one predicate wearing two names. **Demonstrated, not argued:** a hash-valid
      FROZEN A-TWAP-2 declaring `protected_span: T-10s..T` was accepted, and a
      window whose only missing slot sat at T−45 s — OUTSIDE its declared span —
      was still excluded, reporting `weight_missing = 1/60`. **Counterfactual:**
      relaxing `max_weight_missing` to 0.05 / 0.5 / 0.9 leaves admissible at
      **213 / 213 / 213**; wired to the rule it would give 1,948 / 3,373 / 3,373.
      Observed `weight_missing` IS graded (p50 0.05, max 0.617), so §1.3's
      gradedness argument is real and the shipped gate never uses it.
      **CONSEQUENCE FOR §10.4's FALSIFIER:** the rule's owner has exactly two
      levers in the frozen JSON and BOTH ARE INERT — they would edit, re-run, see
      213 unchanged against a changed `spec_hash`, and conclude "the stream is
      unfit", retiring A-TWAP-1 under 94.7 % of both committed Tier-2 days
      **having never run the test they think they ran.** A `spec_hash` that
      certifies text the code does not consult is the frozen-formula-with-free-
      inputs shape, inverted. Filed as Q-DA-38; the code fix is not DA's alone.
       AND at least one sample in (t0 − 5s, t0] with t_known + clock_err ≤ t0
                                                  (the strike was KNOWN at t0)
       AND max_gap              ≤ 30 s            <- NEW
       AND no gap intersecting [T − 60s, T]       <- NEW, protected span
  on_fail  EXCLUDE_UNIT, with the INDICATOR arm mandatory
```

The first two clauses are the existing E0 rule. **[correction]** The existing
rule has neither `max_gap` nor a protected span, and with a measured max gap of
44 s against `w = 60 s` that is not a theoretical hole: a window can pass a
90%-density test while the entire settlement average is missing.

**Current yield — CORRECTED 2026-08-23, and the correction is large.** The
superseded figures (*1,642 final resolutions, 1,340 admissible / 301 excluded =
18.3%*) are the yield of the **first two clauses only**, measured before the two
`<- NEW` clauses and `weight_missing` existed. Measured on the committed coverage
partitions under the rule as actually frozen and shipped: **4,032 rows, 213
admissible, 94.72% EXCLUDED.** The old clauses still pass at 84.7%
(`complete_frac`) and 95.0% (`max_gap`); the new ones do all the work —
`PROTECTED_SPAN` and `MAX_WEIGHT_MISSING` each fail **93.6%** alone, because the
protected span is 60 zero-tolerance one-second slots against a stream whose
*median* max gap is 2 s. **Consequences:** §7.11's selection-risk calibration is
computed on the wrong exclusion rate; the surviving sample is **213
coin-window-days, not ~1,340**; and `route_a_admissible` in both committed Tier-2
days carries the shipped rule, not this figure. Whether a zero-tolerance 60-slot
span is the intended bar is for the rule's owner — DA does not move a frozen
threshold. (The superseded triple never reconciled either: 1,642 − 1,340 = 302,
not 301.) The per-coin split looks alarming (btc/eth/sol/xrp lose 48–49
each, bnb/doge/hype 35–36) and the corpus reads it as uneven-by-coin. It is not:
btc/eth/sol/xrp were *discovered* 13 windows earlier than the other three
(244 vs 231), and 49 − 36 = 13. **[correction] The differential exclusion is
entirely a collector-startup artifact, not a coin-dependent stream property.**
Resolved/discovered is 0.98 uniform across all seven coins.

**Exclusion is never the only arm.** Dropping gap-containing intervals *selects
on volatility* — outages cluster with bursts, and here they are literally caused
by them. `InferenceSpec.gap_arm = BOTH` is the default: report the excluded
estimate and the all-data estimate carrying a gap indicator, and treat a sign
difference between the two as a finding, not a nuisance.

A parallel rule guards the markout reference:
```
A-BOOK-1   (markout)
  field    top_of_book_up[market]
  target   [t_parent − 250ms, t_parent + max(τ)]
  require  book two-sided throughout; no gap intersecting the target
  on_fail  EXCLUDE_EVENT, counted on the face of the estimate
```
**[measured]** 2.8% of BTC `book` snapshots are one-sided; those trades have no
mid, and excluding them is a selection on liquidity state that must be reported,
not buried.

### 1.5 SelfState: three clocks, one of which does not exist

| fact | knowable at | status |
|---|---|---|
| we submitted | our submit clock | **OBSERVED**, exact to `clock_err` |
| the venue accepted | submit + `US_TO_ACK` | **never measured** — **DO NOT BUILD FROM THESE (corrected iteration 4):** this read *"assumed 75 ms US / 3 ms EU"* while `OP-LatencyBudget`, the owner named below, publishes **`ack` = 100 ms optimistic / 200 ms central, ASSUMED and flagged BIASED OPTIMISTIC** (the order path is not the market-data path; venue-side and rate-limit queuing ignored). 3 ms is 25–67× faster than the owner's live assumption, and in `D_max` on an EU host it makes the bracket `t_acked := t_submitted` — **the exact defect this section opens by naming.** Use OPS's values with the bias flag. OPS's τ verdict is ack-insensitive (*"set ack to zero and it still fails"*), so R-48 does not move; the bracket does |
| the order was live in the book | *earlier than the ack*, unobservable | not representable |
| we were filled | fill-message `recv_ns` | OBSERVED |
| queue position `Q_ahead` | never (no L3) | bracket exists for this |

The v12 `OrderRecord` collapses these into two bare optionals. That is the
SelfState analogue of `t_known := t_event`: **imputing `t_acked := t_submitted`
makes the order live at submit, over-states queue priority, and lets a replay
fabricate fills that could not have happened — a look-ahead the `t_known ≤ now`
assertion cannot catch, because the timestamp is genuinely ours.**

So `t_acked` is `Unavailable`, never imputed, and the consumer receives a
bracket:
```
AckBracket{ t_live_earliest = t_submitted
            t_live_latest   = t_submitted + D_max
            t_first_consequence : Known[Timestamp] | Unavailable
            confirmed_by : OpenOrderReconciliation | InferredFromFill | None }
```

**The conservative assumption is a direction, not a value, and the three
consumers face opposite losses.** One bracket is published; each consumer picks
its edge by contract, not convention:

| consumer | question | edge taken | why it is the conservative one |
|---|---|---|---|
| risk / exposure | can I be filled? | `t_live_earliest` | assume live instantly ⇒ maximum exposure |
| cancellation | is my quote gone? | `t_live_latest` | never assume a cancel landed (R-HALT already says this for cancel-all) |
| P&L / markout | did I earn this spread? | `t_live_latest` | shortest queue credit, fewest claimable fills |

`ExposureEnvelope` gains `lo = acked + filled` and `hi = lo + all in-flight
assumed filled`, with the routing rule as a contract: **DE-Constraints evaluates
on `hi`; BE-FlowAndFills evaluates on `acked`/`fills`; no consumer may read a
collapsed single number.**

`D_min`/`D_max` come from `OP-LatencyBudget`, whose ack leg is unobserved and
therefore ASSUMED — so under R-PROV it cannot gate a decision until measured.
That is the intended forcing function: **the programme must measure round-trip
ack latency before it may size on it.**

A real partial identification is available and should be taken: the first
observable consequence of an order (our size appearing at our level in a
`price_change` level total, or a fill) **upper-bounds** the ack. That gives
`Known[Timestamp]` with `prov = IMPUTED` and `t_known_err = t_first_consequence −
t_submitted` — honest, and it shrinks with activity.

The write path goes through the **world**, not the code: DE-Actuator's outbox is
part of the Environment seam and DA-Feeds reads it as a feed. If the Actuator
wrote SelfState directly, DE→DA would be an upward edge *and* live would diverge
from replay (live through the venue, replay through a back-channel).

### 1.6 StateView surface

```
DA-State.view(now: KnowledgeTime) -> StateView          # the ONLY entry point
StateView.get(field)        -> Known[V] | Unavailable
StateView.history(field, span) -> list[Known[V]]        # also truncated
StateView.coverage(field, span) -> Coverage
  # UNSATISFIABLE AS SPECIFIED: `span` is a Duration measured back from a
  # `KnowledgeTime`, `Coverage.target` is a pair of `EventTime`, and §1.2
  # clause 4 says the two have NO shared comparison -- `t_event <= now` is a
  # type error. Filling `target` requires exactly the comparison the plan's
  # central rule forbids, and `StateView` has no event-time `now` to use
  # instead; giving it one would BE the leak. This is the one place where the
  # accessor surface and the type discipline cannot both hold, and it sits in
  # the point-in-time layer. Needs a third option, not an implementer's guess.
StateView.phase()           -> PRE | IN_WINDOW | POST_T_PRE_RESOLUTION | RESOLVED
StateView.refusals()        -> RefusalLedger
```
There is deliberately **no** accessor returning a tape, a gap registry or a raw
buffer: if one existed the truncation would be bypassable and the guarantee void.
BE/DE modules receive a `StateView` and never a `DA-State` handle, so reaching
for the raw feed is a wiring error rather than a code-review finding.

`phase()` lands S2-C5, which never landed: `DE-Constraints` cannot express "no
quoting after T" without it, and it is derivable from **`SP-Instrument.T`** and (**corrected iteration 4: Rev 5 said `SP-Instrument.horizon`, which DOES NOT EXIST** — v23's `SP-Instrument` is `settlement · T · payoff · complement · incentive_contract`. `SP_PLANE_PLAN` §4 flagged this against this plan by name: *"'horizon' is overloaded three ways and needs one name … a builder resolving 'no quoting after T' against a 5 s value halts every market 5 s in."* The registered names that DO resolve are `primary_horizon` = **5 s**, Class D, and `ScopeKey.horizon`, the decision horizon — neither is the 300 s market window. `T` exists and is the correct source.)
`now`, so it introduces no new data. §2.5 shows why it is also required for
markout.

**EV modules do not read through `StateView`.** They hold `read_all`, because a
markout at horizon τ legitimately needs data *after* the trade. `EV-Replay` is
the exception that proves the rule: it *simulates decisions* and must therefore
read through `StateView`, never `read_all`.

---

## 2. EV-Markout

### 2.1 Estimands

```
es      = q(p − m)/m                 effective half-spread
Λ(τ)    = q(m_{t+τ} − m)/m           adverse selection
rs(τ)   = es − Λ(τ)                  maker gross revenue
```
`q = +1` for a taker buy; the maker is the counterparty. Cash forms
`es_cents = q(p − m)·100` etc. are computed alongside and are the P&L-additive
ones.

**τ grid, frozen: {1, 5, 15, 30, 60, 300 s, TO_RESOLUTION}, primary =
TO_RESOLUTION.** **CORRECTED iteration 4: the corpus DOES have a PM-side grid, and it censors.**
`BE_FLOWANDFILLS_PLAN` §2.3 carries `h ∈ {0.25, 0.5, 1, 2, 5, 15, 30, 60, 120 s}
∪ {resolution}`, **each censored at `T − τ` and reported with the censoring
flag**, because *"pooling censored and uncensored h is a real bias in a 300 s
instrument."* Nine finite rungs against six; only `{1,5,15,30,60}` overlap; **this
plan states no censoring convention at all**, and `τ = 300 s` in a 300 s
instrument is censored for every fill after the open — a pooled-censored
statistic under BE's own rule. **Consequence for §2.7:** that section asserts the
partition holds *"exactly"* and defers the formulas to BE, which defines
`permanent_AS + transient_AS = AS(h)` at **h**, censored, with a flag. An
identity stated at τ and defined at h shares no rung outside `{1,5,15,30,60}`.
The premise below is retained as superseded: The corpus has no PM-side τ grid at all — only the sibling
programme's `{1,5,15,30,60,300}` and an abstract `markout(τ)`. This fixes it. The
primary is settlement because the venue doctrine is never to unwind: a binary
held to resolution pays 0 or 1, and the finite-τ markouts are Λ-decay
diagnostics, not the P&L. ~~The finite grid matches the sibling so decay profiles are comparable.~~ **STRUCK AT ITERATION 5 — REFUTED BY THE CORRECTION ABOVE.** Five of nine finite rungs overlap; 0.25/0.5/2/120 s have no counterpart here and 300 s none there, so profiles are **not** comparable off `{1,5,15,30,60}`. This was the only surviving rationale for the six-rung freeze, and iteration 4 corrected the premise while leaving the conclusion standing on it. *Superseded text:* the finite grid matches the sibling so decay profiles
are comparable.

### 2.2 Building the parent event

**Dedup**, then **collapse**, then sign.

Dedup key is `(transaction_hash, asset_id, payload.timestamp)`. Line-level dedup
does **not** work: `recv_ns` differs per collector process, which is exactly what
the 30 s duplicate-collector overlap produced.

Collapse merges prints sharing `(t_event, q_up)` within a market.

**[correction] Both steps are near no-ops on this venue today, and the corpus
expects otherwise.** [measured] on 15,860 prints across 7 coins:

| step | result |
|---|---|
| dedup on `transaction_hash` | **0 duplicates** — 15,860/15,860 hashes unique |
| collapse on `(t_event, q_up)` | multiplicity {1: 15,768, 2: 43, 3: 2} → **47 prints merged, 0.30%** |

The corpus states "trades are probably double-reported (we subscribe to both
tokens; one on-chain fill touches both) — dedup on `transaction_hash`, otherwise
every intensity estimate is 2× too high." **That is not happening.** Polymarket
emits `last_trade_price` **once per fill**, on one `asset_id`, and the sibling
token gets no print. This is corroborated by the independent observation that
668/668 transactions in a BTC window touched exactly one token.

Keep both guards anyway, for two reasons: the SE-inflation they prevent is real
at higher print rates, and **a guard that starts firing is the signal that the
venue changed.** The measured rates (0 duplicates, 0.30% merge) become asserted
data-quality regression bounds reported on the face of every estimate.

### 2.3 Aggressor sign — resolved by measurement

`side` alone is not aggressor direction (BUY 87% / SELL 13% is structural: in a
two-token market you express "down" by *buying* Down). Direction is
`(side × asset_id)`. In UP-space:
```
q_up = +1  if (asset == UP and side == BUY) or (asset == DOWN and side == SELL)
q_up = −1  otherwise
```

This is the largest open sign risk in the corpus: X-11 flags that **if the WS
`side` were the *maker's*, the headline maker gross flips from +95 bps to

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and the `+95 bps` maker gross are the SAME book-derived number and both fall together: `book` snapshots are p90 6.2 s stale. Rebuilt with no book at all the figure is **+0.17 ¢/share**, and it is **NOT DISTINGUISHABLE FROM ZERO** — window-clustered bootstrap gives **+0.173 [-0.251, +0.596]**, with all seven per-coin CIs spanning zero. **The maker-edge sign is UNDETERMINED at two days.** Also settled: `side` IS the taker's (G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0]), so the `+95 → −95` flip scenario is closed. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

−95 bps and the programme is dead on arrival**, and the existing circumstantial
check (63.7% of BUY prints at the best ask vs 15.1% at the best bid) is weakened
by stale-book contamination.

**[measured] Removing the contamination resolves it.** Classifying 15,262 BTC
prints against the prevailing UP-space quote taken **200 ms before** the print:

| quote lookback | consistent with taker reading | inconsistent | inside spread |
|---|---|---|---|
| 0 ms | 75.6% | 20.7% | 3.7% |
| 50 ms | 80.9% | 15.0% | 4.0% |
| **200 ms** | **90.8%** | **6.6%** | 2.6% |
| 1000 ms | 74.7% | 23.5% | 1.7% |

Under the maker reading every one of those 90.8% would require a resting maker to
have lifted its own offer or hit its own bid. The likelihood ratio is ~14:1 per
print over 15k prints. **`side` is the taker side.** Provenance on `side_signed`
may be recorded as MEASURED rather than ASSUMED, and the consistency rate becomes
a standing regression check. The definitive confirmation remains cheap and should
still be run: on-chain `OrderFilled` receipts carry maker and taker addresses.

### 2.4 The reference mid — the decisive convention

This is where the v12 markout decomposition is currently invalid, and the fix is
one frozen number.

**The problem.** The venue emits the `price_change` carrying `best_bid/best_ask`
for a match **before** it emits the `last_trade_price` for the same match. A mid
read at the print has therefore already moved toward the print. The symptom is on
record: measured capture of +0.11…+0.50 c against observed spreads of 1.1–1.8 c
is far below a half-spread, which is impossible for a maker and is the signature
of exactly that contamination. The **net** markout never uses `m` and is immune;
the **partition** is not.

**The fix.** `ReferenceMid.lag_used` is **frozen at 250 ms**. The table in §2.3 is
the same experiment viewed as a lag diagnostic: consistency is a plateau with an
interior optimum, low at 0 ms (own-impact contamination) and low at 1000 ms
(staleness). 250 ms sits inside the plateau. **It is chosen over the
sample-optimal 200 ms *because* it was pre-specified independently in
PM_DEEP_REVIEW before this grid was run** — adopting the argmax of a grid fitted
on the evaluation sample would turn a measurement into a tuned parameter. The
grid `{0, 50, 100, 250, 500, 1000}` ms is reported as a sensitivity band beside
every headline.

**Mid vs micro-price — decided.** `mid = (bid_up + ask_up)/2` is primary;
micro-price is a reported secondary arm only. Reason: micro-price moves with our
own resting size, so a maker's own quote would contaminate its own markout, and
`rs = es − Λ` requires the *same* `m` in both terms or `rs` stops meaning maker
revenue. The corpus lists "mid vs microprice" as open; this closes it for the
markout reference specifically (it stays open as a *belief* input, which is a
different question).

**UP-space, always.** The book is unified across the pair (`neg_risk = false`,
plain binary CTF; an Up bid at 0.60 crosses a Down bid at 0.40 via mint, atomic
at match):
```
bid_up = max(bid_UP, 1 − ask_DOWN)
ask_up = min(ask_UP, 1 − bid_DOWN)
```
This can be **tighter** than either token alone. If it crosses, that is an
arbitrage or a data fault — never a mid.

**[correction] The "2–4 ticks wide" figure is not what the quote spine shows.**
Measured spread from `book` snapshots: p50 = **1 tick (1¢)**, p90 = 11–12¢. The
corpus figure (and the FLB's "3–6¢ against a 2–4¢ spread" comparison) is
therefore probably too wide, which *understates* the FLB's tradeability. But the
measurement carries its own defect and cannot yet replace it: **`book` snapshots
arrive on (re)subscription, not on a timer**, so they are a biased sample of
*time* — clustered at connection events, which cluster at busy moments. Any
time-uniform spread statistic must come from the `price_change`
`best_bid/best_ask` spine instead, which is what Tier-1 (§4) provides. Until then
neither number should be quoted as settled. A further complication: `tick_size`
is absent from most `book` events and `tick_size_change` (0.01 → 0.001) fires
188× per day-sample, so a spread measured in *ticks* is not comparable across the
change; measure in cents.

### 2.5 Aggregation protocol

- **Unit of observation:** the parent event.
- **Weight:** notional, ex-ante, **and** equal-weighted reported beside it.
  **THE FORMULA IS UNSTATED AND THE HEADLINE IS WEIGHTED BY IT.** For a
  0.01-0.99 binary, `size × price`, `size × 1` and `size × min(p, 1−p)` differ by
  up to 100×, and §2.5's own argument is that the weighting choice flips signs.
  `notional` is given a formula nowhere in this plan; `ParentEvent.notional_cents`
  is typed `Cents` with no definition in either YAML. **`risk-notional`, used once
  for "the top 1 % of prints carry 24.2 % of risk-notional", is defined nowhere
  in the tree** — so the tail-risk claim is stated in an undefined unit. Separately
  the live contract disagrees with this bullet: `R-DUAL` requires *per-fill AND
  share-weighted*, and the code follows the contract
  (`"weighting_contract": "PER_FILL_AND_SHARE"`), not the `{notional, equal}` pair
  named here. **RETRACTED 2026-08-23 — THE TWO ARMS ARE NOT NEAR-IDENTICAL, AND THIS SENTENCE WAS THE LICENCE FOR CITING ONE.** Measured over all 1,995,577 committed `markout_events` rows: `POST_CLOSE` is **+0.652 c per-fill against −0.303 c share-weighted — OPPOSITE SIGNS**; `IN_WINDOW` +0.285 vs +0.115 with coin-day sign agreement **4 of 14**; `PRE_OPEN` +0.931 vs +0.612. Overall coin-day sign agreement 27/44; pooled cross-arm Pearson **+0.045** — uninformative, not near-identical. **The disagreeing cell is the exact one §2.5(iii) cites at −0.303** to mark the post-`T` cancel prescription UNSUPPORTED, and which the `CANCEL_POLICY_PROTOCOL` owner reads as the state of evidence. **Provenance of the retracted claim, recorded because it matters more than the claim:** "91.66 % vs 91.79 %" came from a review note where it measured **share of notional in-window** — a different quantity from agreement between weighting arms — and was repeated here without checking what it measured. **Consequence: every c/share figure in §2.5(iii) and §8 step 5 is stated in ONE arm while the live contract `R-DUAL` requires both; they must be dual-reported.**
  but the notional arm's whole justification is the tail, and shares vs
  `shares × price` differ by exactly the price factor that dominates it —
  never one alone. On the sibling programme (ADA) equal weighting gave
  `rs = +2.443` bps, positive on 31/31 days, and the cell screened as a PASS;
  notional weighting gave `−0.322`, positive on 7/31, and killed it. Small prints
  capture the half-tick; the *dollars* are adversely selected.
  **[correction] the shorthand "equal-weighting flipped +2.44 to −0.32" inverts
  the causal direction** — equal weighting *manufactured* the false positive;
  notional weighting produced the kill. Cite it the right way round.
  **[measured]** the same asymmetry is present here: the top 1% of prints carry
  **24.2%** of risk-notional and the top 10% carry **63.0%**; BTC alone is 82–85%
  of notional, giving effective breadth ≈ 1.4 coins.
- **But not for identity tests.** Notional weighting on a mechanism/identity
  question makes every such claim a BTC claim: a settlement convention
  reproducing BTC at 99.9% and XRP at 95% passes a notional-weighted pooled bar
  at 99.1% while being wrong about four feeds. EXP-M6 is already correctly
  notional-blind. `R-WEIGHT` now makes that an explicit function of
  `estimand_kind ∈ {ECONOMIC, IDENTITY}` rather than an exception.
- **Units:** headline in **cents per share**. Mid-relative bps `q(p−m)/m` is
  reported but is *not* the headline: on a 0.01–0.99 grid it diverges as `m → 0`
  (a 1¢ edge at `p = 0.03` is 33% of mid) and it is not P&L-additive.
  `WealthLedger` wants cash, so cash it is.
- **Inference:** *(SCHEME COLLISION, flagged 2026-08-23: §5 names this estimate's
  scheme three different ways — "stationary block bootstrap" here, "wild cluster
  bootstrap (Rademacher)" and "day-block bootstrap" in §5's ladder — and
  `InferenceSpec.scheme` makes those DISTINCT enum members, so "they are the same
  thing" is unavailable. No G→scheme mapping exists anywhere; `ClaimLadder` maps
  G→`ClaimClass` only. One scheme per estimand must be named.)*
  day-clustered, stationary block bootstrap over UTC days,
  B = 2000; Holm across each experiment's own family; fails-final,
  passes-provisional. See §5 for what that actually permits today.
- **Phase stratification is mandatory.** **THREE CORRECTIONS, 2026-08-23 —
  read them before citing any figure in this bullet.**
  **(i) `StateView.phase()` DOES NOT EXIST.** `da_state.py` contains zero
  occurrences of `phase`, and v22 lists `StateView` as `get / history / coverage
  / refusals`. What shipped is a private four-value enum in the EV layer
  (`evaluation_pipeline.py:284-301`): `PRE_OPEN | IN_WINDOW | POST_CLOSE |
  AMBIGUOUS_BOUNDARY`. So the mandatory stratification axis is unimplemented at
  its named location and re-implemented in the wrong plane.
  **(ii) FOUR phases, not three** — `phase()` as specified distinguishes
  `POST_T_PRE_RESOLUTION` from `RESOLVED`, and the shipped enum **collapses
  them into one `POST_CLOSE`**. That is precisely the split the remedy below
  turns on, so it is not measurable from the committed output. `R-STRATA`
  freezes strata by hash before the estimate, so a three-vs-four disagreement
  changes the hash and two builds cannot be reconciled afterwards.
  **(iii) THE COMMITTED RECEIPTS CONTRADICT THE FIGURES BELOW.** Measured over
  both committed Tier-2 days: `PRE_OPEN` **+0.612** c/share on 2,649,539 shares
  (6.14% of notional); `POST_CLOSE` **−0.303** c/share on 941,365 shares — and
  `POST_CLOSE` is **POSITIVE on 12 of 14 coin-days, including BOTH btc days**
  (+0.488, +0.444), which are 83% of the book. The pooled negative is carried
  entirely by two coin-days (sol 08-20 at −33.2 c/share, hype 08-21 at −8.1).
  A sign carried by two tiny in-sample-selected cells is the shape R-50 ruled
  licenses nothing — the same logic applies to a negative one used to justify a
  prescription. **So the remedy below is UNSUPPORTED, not refuted**: everything
  here is `DESCRIPTIVE` at G=2 with no CI, and the populations are not identical
  because of (ii). §2.3 and §7.12 already carry `WITHDRAWN 2026-08-21` blocks for
  sibling book-derived figures; this bullet was left standing and should not have
  been. *Superseded text:* the three phases are economically different
  instruments: pre-window `t < t0` is 7.4% of notional at +0.70 c/share to the
  maker; post-`T` is 0.9% of notional at −1.46 c/share (−806 bps) — 350,929
  shares traded after the outcome is already determined, ~8% of total maker gross
  destroyed in 1.4 days. Pooling hides both the largest per-share loss on the
  venue and its trivial remedy (cancel at `T − ε`, do not re-quote before
  resolution).
- **Model-free strata only** (`R-STRATA`): `r`, `|book_mid − 0.5|`, coin, phase.
  Never `|d|` or anything else derived from a forecast — that lets the model
  grade its own exam across a ~210-cell space.

### 2.6 Every number is an upper bound

The tape's fill is the **average** resting maker's. A back-of-queue entrant is
filled later and more often on the wrong side. The measured haircut is severe and
moneyness-dependent: an unconditional +1.8 c/share at `t = 290 s`,
`mid ∈ [0.95, 1.00)` realises **+0.72 c** on matched fills (~60% haircut); an
unconditional −9.4 c at `t = 30 s`, `mid ∈ [0.15, 0.30)` realises **+0.25 c**
**— and this cell REFUTES the section header rather than illustrating it (corrected 2026-08-23).** `1 − 0.25/(−9.4) = +103%`; the stated "~97%" is `1 − |0.25|/|−9.4|`, a **second, sign-blind formula** written down nowhere. The realised value is ABOVE the unconditional, so the unconditional is **not** an upper bound here and selection did not destroy edge — **it flipped the sign**, which this same section calls *"a failure, not an average"*. The sign-blind formula launders that flip into a magnitude. The defensible claim is that **|number| upper-bounds |realised|**, weaker than the field the contract carries; a DE consumer haircutting `−9.4` downward would refuse a cell whose matched-fill value is `+0.25`. *Superseded:* Selection destroys most of the available edge at every moneyness.

`MarkoutEstimate` therefore carries `upper_bound_disclaimer` as a field, and the
queue bracket (pessimistic = back-of-queue, decremented by trades only, all
cancellations assumed ahead; optimistic = front-of-queue) is reported as a
bracket. **A sign flip across the bracket is a failure, not an average.**

### 2.7 Partition

`spread + transient_AS + permanent_AS + snipe + own_impact = markout(τ)`, exactly.
**NOT BUILDABLE AS WRITTEN (flagged 2026-08-23).** Those four component terms
appear exactly once in this document — on this line — with no estimator, no
inputs and no owner; formulas for four of them exist only in
`BE_FLOWANDFILLS_PLAN` §§3-6, which this plan never cites. And `own_impact` is
separately **ruled non-identifiable** (`FLOW_UNCERTAINTY_LOOP.md:611`), with
`PM_STRUCT_ITER2_B.md:835` naming `own_impact` returning `0.0` as *"an `assumed`
value entering the objective silently, which is the FATAL-1 shape"* requiring
*"a **declared pin** with a bias direction, not a float."* An identity asserted
as *exact* and closed with an unpinned zero on a non-identifiable term is this
programme's named fatal pattern. Until a pin with a bias direction exists, the
partition is **DEFERRED, not available** — and §8's build order already omits it,
which was the only prior hint.
Each component becomes a `MarkoutComponent{value, unit, conditioning_measure,
sign_convention, coverage}` — the architecture prose has mandated those four keys
all along while the YAML carried five bare floats, and under the standing rule
that a fix living in prose is not a fix, the prose version did not exist.

`conditioning_measure` is **pair-scoped**, not instrument-scoped: some fills are
minted against a counterparty who wanted the *complement*, whose information
content differs from a same-token taker's, so `E[markout | fill]` otherwise
conditions on the wrong selection event.

Report **net unconditionally**; report the **split only with the frozen lag and
the sensitivity band attached**.

### 2.8 The blocker

`fee` is one of EV-Markout's four required spec facts (with `tick`, `w_declared`,
and the 2026-08-20 era boundary) and it is unreconciled: the WS `fee_rate_bps` is
**0 on every print measured**, the docs say crypto taker feeRate 0.07, the CLOB
market record says `maker/taker_base_fee 1000`, and `PRELIMINARY_PLANS` records
"maker 0 + ~70 bps rebate/fill; taker ~3.5% ATM". **Never read the fee from the WS
field** — it is simply unpopulated; the real fee appears in Polygon
`OrderFilled`/`FeeCharged` logs. ~~A *net* markout number cannot be signed until~~ **DISCHARGED 2026-08-24 (R-84), with the three conditions in §8 step 5.** The taker leg is measured (`0.07·p(1−p)`, 600/600 charged) and the maker leg is zero on **744 of 754** legs; what remains is not the fee but the unmeasured maker **rebate**, the `Disputed` rewards band. *Superseded:* a net markout number cannot be signed until
one source is established, because the sign of net maker revenue depends on it.
Gross markout can be reported now.

---

## 3. EV-Calibration

### 3.1 Panel and scoring

One row per `(window, r)`; `r` grid frozen at
`{270, 240, 180, 120, 60, 30, 10, 5, 2}` s. **NO STALENESS BOUND IS SPECIFIED ON THIS PANEL — AND MEASURED AT ITERATION 5,
THE SHIPPED PANEL IS ~20x STALER THAN THE FIGURE THIS PARAGRAPH CALIBRATES A
READER TO.** `evaluation_pipeline.py` takes the max-by-event-time over every
quote admitted at knowledge time, with **no lower bound and no maximum age**;
`staleness_ns` is recorded and never gated. On the committed `calib_panel`
(36,288 rows): `quote_status == "AVAILABLE"` on **36,288 of 36,288** —
`NO_ADMITTED_QUOTE` never fires — and at `r = 2 s` staleness is **p50 57.8 s, p90
119.3 s, max 627.0 s on a 300 s window** (a quote from 327 s before the window
opened, stamped AVAILABLE). **The short rungs are not independent reads:**
`r=10` and `r=2` share the same quote event in **96.1 %** of windows, `r=5` and
`r=2` in **98.5 %**. Part of this may be genuine book quiescence — `price_change`
is a delta stream — but 793 of the 1,937 windows whose last quote precedes `T` by
>60 s carried **6,765 trades in that final minute**, and the finding does not
depend on the mechanism. **Two consequences:** an owner setting `A-CALIB-1`'s
bound from BE's 6.2 s figure refuses essentially the whole `r ≤ 30` panel on both
committed days; and §0's cross-horizon stability — *"the book beats our forecast
at every horizon by a stable 2.5–3.2 Brier points"* — reads as corroboration
while §3.1 already retires that reading on the `y` side (*"six rows are one
observation"*). The predictor side has the same defect at the short end, and
whoever ratifies the spread-capture identity needs that on the page. The sibling
plane's measured cost: No
maximum quote age, no as-of convention, no refusal on a stale quote, and no
statement of whether the mid comes from `book` snapshots or the `price_change`
spine. `BE_BELIEF_PLAN` measures this exact series: **book-snapshot inter-arrival
p90 = 6,240 ms**, and the fit moves `b̂ = 1.145` dense versus **1.182**
snapshot-only. §2.4 and §7.5 quarantine book-derived statistics carefully — *"a
biased sample of time"* — and that doubt is never applied to the book-derived mid
that **constitutes** this panel. The asymmetry is inside one document. An
`A-CALIB-1` rule **IS NOW WRITTEN — `config/a_calib_1.json`, DRAFT pending freeze (R-85, R-91).**
**The bound is ADOPTED FROM MEASUREMENT, not chosen: `max_quote_age = r`, the rung's own value.**
A read labelled *r seconds before T* may not be older than `r`, or the label names a window position
the quote predates — **no free parameter, so nothing in it can be tuned after a result is visible.**
*Measured yield* (36,288 committed rows, generation pinned): **22,318 admitted / 13,970 demoted =
61.5 %**, per-rung **1.0 % at r=2 · 3.6 % at r=5 · 9.8 % at r=10 · 51.2 % at r=30 · 88.6 % at r=60 ·
99.8 % at r≥120**. `on_fail: DEMOTE_R_TO_NOMINAL` — per R-85 the committed days keep demoted labels
rather than retro-refusal, so a row survives as a stale-quote observation and only the *r-indexed
freshness* reading is refused. **What the measurement says:** staleness is monotone in `r` and
collapses at `r ≥ 120` — the book goes quiet in the final minute, so every short rung reads the same
pre-quiet quote. That is a property of the venue's quote process, not of the rule. **R-91
consequence:** a coordinator-set rule now exists, so `admissible` on these rows evaluates a rule
instead of expressing a worker's judgement — which is what converts it back into a fact.
*Superseded:* an `A-CALIB-1` rule shaped like A-TWAP-1 is owed: source pinned to the spine, a max
quote age tightening with `r`, refuse rather than carry forward, and EXP-BLEND
re-scored with it on. `p_book(r)` is the UP-space mid at
**knowledge time** `T − r`. Scoring is **paired** — model and book on the same
`(window, r)` — because an unpaired comparison invites the
difference-in-significance fallacy, which is this repo's own historical killer.

**Brier primary, log-loss secondary.** A Brier/log-loss disagreement means the
difference lives in the tails and must be reported as such.

**[correction] The corpus specifies "a frozen clip" and never gives a number.**
This fixes it:
```
clip(state) = max( tick(state)/2 , 5e-4 )      applied to BOTH forecasts
```
The tick term is the load-bearing one and it is **state-dependent**:
`tick_size_change` (0.01 → 0.001) fires 328× across 130 windows, so a hard-coded
`[0.01, 0.99]` both forbids the only levels that exist near the boundary and
leaves the exploit open. Without the clamp, late in a window the book mid is
pinned to the grid while a continuous model can "win" on pure tick-quantization
with zero economic content. The 5e-4 floor is numerical only.

**Three arms, not two.** The model must beat `BOOK_ISOTONIC` — the walk-forward
isotonic recalibration of the book mid — not merely `BOOK_RAW`. If it beats raw
mid but not recalibrated mid, the "edge" is public recalibration available to
every competitor on the same terms, not information. This is the self-defeat
check in operational form: *a belief that tracks the book cannot profit from
disagreeing with it, so the recalibration IS the edge or there is none.*

**Stratify by `r`; do not pool across it for the headline.** Base rate and
variance both move by an order of magnitude (Brier 0.237 at `r = 270 s` vs 0.017
at `r = 30 s`), so a pooled number is dominated by the long horizons.

**One row per window (`R-ONEROW`).** There is exactly one `y` per window.
Stacking a window's ~300 per-second decision rows as independent inflates
apparent information by up to 300×, and the late rows carry ~0 information
anyway. **The six horizon rows of the current EXP-BLEND table are the same
windows re-scored** — near-perfectly dependent. Six rows are one observation, and
their striking stability across horizons is not corroboration.

### 3.2 Separating the +4.5 pp drift from the FLB

**[measured]** pooled up-rate **0.5436** (naive +3.54σ, and that σ is itself an
upper bound on significance because windows share paths); consistent across every
coin (0.515–0.567) and both days (0.550 / 0.538). This is a two-day rally, not a
property of the venue.

The two effects are separately identified because they have different **shapes**:
```
P(Up | p_book) = g( a + b·g⁻¹(p_book) + φ(d) )
```
- **FLB is the slope. CORRECTED 2026-08-23 — THE INEQUALITY WAS INVERTED, AND IT
WOULD HAVE INVERTED A GATE.** Under the parameterisation above, outcomes MORE
extreme than quoted means **`b > 1`**, not `b < 1`. This document's own two
bucket gaps prove it: −0.043 at `[0.1,0.2)` and +0.059 at `[0.6,0.7)` fit to
**`b = 1.280`**, intercept `a = +0.098` — matching the `a > 0` drift stated here,
so the arithmetic was right and only the comparison's direction was wrong.
`BE_BELIEF_PLAN` states the correct convention — *"`b > 1` ⇔ underconfident"* —
and measures `b̂ = 1.145` / `1.037`. **Had this stood:** EV-Gates wires G2 as
`assert b_hat < 1` and the gate returns **FAIL on data that CONFIRMS the
effect**; and §3.3's `edge_vs_cost` rides on a recalibration pointing the wrong
way, since BE's inverts and `p_recal − p_book` flips sign in every window. The
programme's recurring class in its purest form: **the rule's text reads correctly
and what it evaluates is the opposite.** *Superseded text:* `b < 1` ⇒ realised outcomes more extreme than quoted ⇒
  the book is underconfident, longshots overpriced and favourites underpriced.
  A slope is invariant to any level shift.
- **Drift is a level, i.e. a hump.** A pure up-drift enters as a constant `a > 0`
  in link space, which in probability space is maximal near `p = 0.5` and
  vanishes at both ends. It cannot generate a monotone low-under/high-over
  pattern — but it *does* inflate the high side.

Estimate `b` with **day fixed effects absorbing `a`**, which de-drifts by
construction. `g` is a `LinkFunction` port — never hardcode the Gaussian Φ; the
Φ-link misspecification is documented to bias the fit −3.5…−15% and to
under-price the far tail by ~28× at `d = 3`, in exactly the direction that
matters for a longshot-selling book.

**The bucket table is not the estimator.** Gaps of −0.043 at `[0.1,0.2)` and
+0.059 at `[0.6,0.7)` are *consistent with* **`b > 1`** — they fit `b = 1.280`; this line read `b < 1` before 2026-08-23; the claim to be gated is
**`b̂ > 1`** with a day-clustered interval. *(This was `b̂ < 1` until iteration 3: the iteration-2 repair corrected the DESCRIPTION two sentences above and left the PRESCRIPTION inverted — the author's named failure mode, committed inside the repair for an inverted gate.)*

**Independent drift control.** `drift_control` is the realised underlying return
over the window measured from **Binance bookTicker**, deliberately *not* from
Chainlink: Chainlink is the settlement source, so a Chainlink-based drift control
is endogenous to the outcome it is meant to purge. (HYPE has no Binance leg and
is excluded from the controlled arm; report it separately.)

### 3.3 Recalibration and the path to a decision

The **only** legal route from a fitted calibration into a decision is offline
through the spec plane, and **R-37 fixes all three roles**: **`BE` fits and
publishes** (it is BE's model, and Class C is publish-then-adopt); **the SP
owner — `DA` — writes the `SP-Params` entry on adoption**; **`EV` does
neither** — it scores. `DE` then reads `SP-Params.at(t)` with
`fit_data_through`. A direct DE read of EV would violate "EV is read by
none" *and* destroy the walk-forward guard.

> **Correction — plane-order violation. R-37 (confirming DA's R-33 resolution).**
> This section previously had **EV** writing the `SP-Params` entry. The deciding
> consideration is **direction, not authorship**: an `EV → SP` write is an edge
> out of a plane that is read by no one, and SP is read by DA, BE and DE. It is
> the same violation class as `de_constraints.py`'s `EV → DE` import (Q-DA-13),
> **one plane further up and sitting undetected in a PLAN rather than in code**. EV writing
> into a plane that `DA`, `BE` and `DE` all read is EV reaching a decision by
> proxy — the same shape as the `EV → OP → DE` health-source edge that
> `PM_ARCHITECTURE` v7 removed deliberately. `BE_BELIEF_PLAN` already has BE
> publishing `â`/`b̂` for adoption, and `COORDINATION.md` records BE doing so, so
> the two plans disagreed and the plane rule settles it. **Also corrected:**
> `provenance = fitted` is not a member of any ruled enum — R-3 enumerated
> `MEASURED · DECLARED · CHOSEN · ASSUMED · IMPUTED` — so a fitted value carries
> `MEASURED` with its `artifact_id` and `fit_data_through`. Filed as Q-DA-7;
> DA has resolved its own side and the BE-side wording is BE's to confirm.
`measured_at` is when the fit ran; **`fit_data_through` is the last date of the
fitting data**, and only the second is PIT-relevant.

The FLB is a claim about money only in the form
`edge_vs_cost = |p_recal − p_book| − half_spread − fee`, and that comparison
**IS UNBLOCKED AS OF ITERATION 6 — BOTH PRECONDITIONS ARE DISCHARGED IN THE PAGE THIS PLAN'S MASTHEAD SAYS WINS ON FACTS.** (i) *Fee:* `FLOW_MODEL_STATE.md` §1 records **"Taker pays the fee, maker does not — `0.07·p·(1−p)` $/share, 600/600 taker legs charged, 744/754 maker legs zero"**, measured on-chain, n=600, and v23 ships the `FeeFamily` carrier; under Class-C rules a measured fee is publish-and-adopt, so no ruling was ever owed. (ii) *Spread on the spine:* the same page gives median **0.0100**, p90 0.020 over **2.29 M executable quotes**, read from `price_change.best_bid/best_ask` rather than `book` snapshots — exactly the time-uniformity §2.4 and §7.5 demand. **BUILD THIS:** compute `edge_vs_cost` on btc/eth now at **ATM half-spread 0.50 c** and a maker
fee of **zero WITH ITS EXCEPTION CARRIED — 744 of 754 maker legs zero, so TEN legs (1.3 %) WERE
CHARGED.** "Measured at zero" is the right headline and the wrong absolute; any `edge_vs_cost`
citing it carries the 1.3 % exception or states why it is immaterial at the precision claimed.
*(Corrected 2026-08-24: the quotation above carried 744/754 and this prescription said "maker fee
zero" flat — the caveat in the head, the absolute in the tail, which is this document's own
recurring shape.)* **NAME THE LEG BEFORE COMPUTING THIS (added 2026-08-24).** `FLOW_MODEL_STATE.md:60` pairs the two terms as ONE side: *"Crossing costs ~2.25 c/share ATM | 0.50 c half-spread + 1.75 c fee = 225 bps"*. The prescription above subtracts the **crossing** side's half-spread while using the **resting** side's zero fee — **a resting maker EARNS the spread** — and the legs differ by 1.75 c on a 0.50 c base, **3.5x, in the direction that flatters.** Compute the resting leg (earn half-spread, pay ~0) or the crossing leg (pay half-spread + `0.07·p(1−p)`), and say which. **Then the residual is the MAKER REBATE** (`rebate-unknown`) — the `Disputed` rewards band, which has a named carrier and can only move `edge_vs_cost` FAVOURABLY, so it does not block a signed lower bound. **§8 step 5 CARRIES FROM GROSS TO NET as of 2026-08-24.** Net is signable on btc/eth with the
maker leg at zero-with-a-1.3 %-exception and the taker leg at `0.07·p(1−p)`, both measured on-chain.
**The maker rebate remains unmeasured and can only move net FAVOURABLY**, so what is signable is a
**lower bound on net**, and it must be stated as one. **NET RESTATED AGAINST THE POLICY COMPARISON (R-109, filed as Q-DA-48, 2026-08-24).**
On the maker leg **a fee can only SUBTRACT**, so net ≤ gross and the restatement
cannot flatter these numbers. All **20 arm-coin-day values are negative gross**
(`policy_comparison_v2.json`; **n=141 paired windows per coin — 08-24 is PARTIAL
at 21/30, so "5 days" is NOT five equal days** — as-of 2026-08-24). The fee term
is **0.0232 c/share** at incidence 10/754 and **1.75 c/share** at the absolute
bound (every maker leg at max `0.07·p(1−p)`). **Across that entire range every
value stays negative: the STOP VERDICT IS INVARIANT to the user's unpinned fee
parameter.** The only sign-flipping term is the **unmeasured maker rebate**,
needing **> 0.5164 c/share = 52 bps** to flip the least-negative coin-day and
**> 2.8620 c/share = 286 bps** to flip all twenty. *The verdict is invariant; the
NUMBERS are not, and R-107 ships numbers — so materiality stays the user's call.*

**STOP CONSEQUENCE, FILED AS Q-DA-42:** *fee
treatment* is one of the three unpinned `STOP-MM-VIABLE` inputs named in Q-DA-14, which R-35 reserved
to **the user** — so this restatement moves a STOP input and the STOP dossier does not go to the user
until that is settled either way. *Superseded:* cannot be made until §2.4's spread measurement is redone on the time-uniform spine and §2.8's fee is resolved.

### 3.4 Standing variance audit

`z_i = (X_T,i − E_t[X_T,i]) / √(budget.total_i)`; report `var(z)` with a
day-clustered CI and PIT coverage at 50/80/95 by `(r, |d|)` bin.
`var(z) < 1` ⇒ the budget is too large ⇒ a double count of the declared-disjoint
kind. `var(z) > 1` ⇒ a component is missing. A 32–41% σ inflation appears as
`var(z) ≈ 0.51–0.57`. **The registry catches declaration errors; this audit
catches declaration lies, and only the second is falsifiable.** Wire it into
EV-Gates as a standing gate, not a one-off study.

### 3.5 Ground truth

EV-Calibration owns resolution ingestion, previously unowned — `resolutions.jsonl`
is consumed by no module. Finality filter: `closed == true` **and** `winners`
present. **[measured]** 1,650 rows → 1,642 final, **exactly 8 non-final**, zero
duplicate slugs, winners always exactly one of (Up, Down): 892 Up / 750 Down.
The 8 bad rows carry a **distinct schema signature**
(`{closed, closedTime, outcomePrices, outcomes, recv_ns, slug, umaResolutionStatus,
volumeNum}` with `closed: false`) versus the good schema
(`{closed, conditionId, recv_ns, slug, source, winners}`), so they are dropped by
schema signature, not by heuristic — and a row matching neither schema is a
fail-loud, not a silent skip.

---

## 4. Storage and indexing

**The problem, measured.** Ad-hoc parsing costs **5.3 s for 10 BTC windows**
single-threaded → **~13 min for a full 1,500-window pass**, on 4.9 GB of raw
gzipped JSON. `price_change` dominates: 833k lines carrying 1.58M level-total
entries per 10 windows.

**The observation that makes it tractable.** Only **0.75%** of those level-total
entries change the top of book — 11,954 distinct top-of-book states per 10
windows (~1,200/window). Trades are ~1,320/window. So the entire useful surface
of the corpus is **~3.8M rows**, which is tens of MB of Parquet and loads in
seconds.

**Three tiers.**

**Tier 0 — landing zone, immutable.** The `.jsonl.gz` / `.csv.gz` as written.
Never rewritten. Content-addressed in a manifest.

**Tier 1 — distilled spine, Parquet, Hive-partitioned `day=/coin=`:**

| dataset | rows/window | key columns |
|---|---|---|
| `quotes` | ~1,200 | `t_known_ns, t_event_ms, slug, bid_up, ask_up, bid_sz, ask_sz, seq, shard_id, src_file_id` |
| `trades` | ~1,320 | `t_known_ns, t_event_ms, slug, asset_id, token_side, price, size, side_raw, q_up, tx_hash, fee_rate_bps, parent_id, …` |
| `book_snap` | sparse | full ladders — **flagged subscription-triggered, not time-uniform** |
| `twap` | 8/s | `t_event_ms, t_publish_ms, t_known_ns, symbol, window_s, full_accuracy_value` |
| `binance_bt` | high | `t_known_ns, E, T, updateId, bid, bid_sz, ask, ask_sz` |
| `windows` | 1/window | `coin, t0, T, condition_id, up_asset, down_asset, winner, coverage_flags, admissible_*, era` |

Every row carries **both** `t_event` and `t_known`, so an event-time sort is
available for integrals and a knowledge-time filter for decisions — the type
system, not the storage layer, is what prevents mixing them.

**Tier 2 — derived caches:** `markout_events` (one row per parent event with all
τ precomputed), `calib_panel` (one row per `(slug, r)`).

**Mechanics.**
- Sorted by `t_known_ns` within partition; row groups ~64k rows; zstd
  (pyarrow-bundled) with snappy fallback. `pyarrow 24` and `pandas 3` are
  present; there is no duckdb or polars, so the reader is `pyarrow.dataset`. **BUT A NAIVE `ds.dataset()` OVER A TIER-1 ROOT DOUBLE-COUNTS, AND THE ERROR IS INVISIBLE IN THE RATE (corrected 2026-08-23).** R-10 versioned the partition address so an amended generation is written *alongside* and the superseded one is **kept** — `twap`, `coverage`, `windows` **and `canary`** currently hold two co-resident generations — **`canary` was omitted from this list until 2026-08-24, and that omission produced a bad citation in front of a live ruling.** DA cited *"23 of 36 canary receipts above the 0.5 pp reference"*; `tier1/canary/` holds **8 pre-R-7 `leak_canary_v1` receipts plus 28 `leak_canary_v2_r7` files that are 14 coin-days × 2 content-addressed twins**, so the honest figure is **9 of 14 coin-days on two day-clusters** — and the pooled set carried **two `INVALID_UNBOUND_GUARD` verdicts that exist only in the generation R-7 abolished**, quoted back at the ruling that abolished them. **Any statistic over `canary` must pin `canary_version`, exactly as §1.4's 213/4,032 must pin `distiller`.** (`day=/coin=/part-0.parquet`, `tier1_v3`, and `day=/coin=/distiller=tier1_v4_r12/…`). Executed with the reader this section names, `tier1/coverage` returns **8,064 rows / 426 admissible** against a truth of **4,032 / 213**; `windows` 8,064 for 4,032; `twap` 7,542,428 for 3,771,214. **Because the exclusion RATE is identical at 94.72 %, the doubling is invisible in the one number a reviewer sanity-checks and shows only in N** — exactly what §7.11's selection-risk calibration and §5's `n_eff` ladder consume. Shipped readers escape it only because `daily_pipeline.validate_existing_partition` reconstructs the explicit `distiller=` path, **which this section never states**. Any re-derivation of §1.4's 213/4,032 must pin the generation.
- **Shard merge:** for slug `S`, read `S.jsonl.gz`, `S.jsonl.1.gz`, … in numeric
  order then the bare in-flight `S.jsonl`; sort by `recv_ns`; verify (i)
  `recv_ns` monotone within shard, (ii) shard boundaries do not overlap in
  payload time, (iii) post-concat duplicate rate under the identity key.
  **[measured]** 1,741 window files, 1,669 distinct slugs, **54 multi-shard**.
- **Blank/partial lines** exist (a truncated final line per file is normal);
  they are counted, not silently skipped, and a rate above a frozen bound is a
  `HealthEvent`.
- **Atomicity:** write to a temp partition, then rename. A partial partition is
  never visible. Re-derivation is idempotent; a manifest mismatch is **fatal**.
  This repo has already destroyed 163 of 176 symbol histories once with an
  overwrite-semantics incremental updater — the merge-never-overwrite lesson
  applies verbatim here.
- **Duplicate-collector overlap** (28 duplicate TWAP ticks/symbol, epoch-ms
  1787186761000–1787186791000) is removed by the identity keys, keeping the
  **earliest** `recv_ns` from a declared primary collector — earliest is the
  honest knowledge time for what we *did* know, and naming the primary collector
  prevents a second process silently improving our history.
- **Era stamping:** every partition carries the spec **content hash**, not a
  date. Settlement changed 2026-08-07 (pre-change data is liquidity-anatomy only,
  never pooled) and the rewards band was re-cut 2026-08-20 *inside* the current
  1.5 days; Gamma served a stale band after the re-cut with both values on disk
  simultaneously.

---

## 5. What is claimable, and when

The cluster is the **calendar day**. Not the window (288 windows/day ride one
path; consecutive windows additionally share a 60 s average, giving a mechanical
`ρ ≈ 0.036`). Not coin-day (the seven coins load on one crypto factor; measured
effective breadth under notional weighting is **≈ 1.4 coins**, so
`n_eff ≈ 288 × 1.4 ≈ 400` windows/day, not 2,016).

| horizon | G | claimable | not claimable |
|---|---|---|---|
| **2 days (now)** | **2** | **`DETERMINISTIC` only.** Settlement-rule reproduction (a near-identity, 99.8%), data-integrity counts, the leak-canary delta, dedup/collapse/one-sided rates, sign-consistency rate. | **Anything with a CI.** With G = 2 a day-clustered variance has ~1 df; no interval exists. Every ΔBrier and every bps number is a point estimate labelled `DESCRIPTIVE`. |
| **7 days** | 7 | `PROVISIONAL`, via wild cluster bootstrap (Rademacher) — the standard small-G fix. Directional claims only: sign of the FLB slope **`b̂ > 1`** (corrected iteration 3; see §3.2), sign of gross `rs` at short τ, sign of the post-`T` phase loss (large and unambiguous). | Net-of-fee profitability, the τ-curve shape, per-coin differences, anything conditioned on a rare state. |
| **30 days** | 30 | `INFERENTIAL`. Day-block bootstrap CIs; paired ΔBrier vs both book arms; `rs(TO_RESOLUTION)` against a cost threshold; frozen model-free strata with Holm/Romano–Wolf. | Rare-regime conditioning (there will be 1–2 instances). Heavy-tailed markout quantities may still need 4–6 weeks. |

**The binding constraint is often not window count.** The day-level term dominates
whenever `σ_day` is comparable to the edge, requiring `G ≥ 4(σ_day/μ)²` **days
regardless of window count** — ratio 2 → 16 days, ratio 3 → 36 days. Reference
window-noise-floor MDE for paired ΔBrier at `σ_ΔB ≈ 0.02`, `n_eff ≈ 400/day`:
G = 14 → 5.5e-4, G = 28 → 3.9e-4, G = 56 → 2.8e-4. The corpus's own read dates
follow: **G2 (calibration contest) readable at ≥ 28 days; markout economics 4–6
weeks**; a two-week read may produce a FAIL-on-sign and never a pass.

Markout has a second, worse tail problem: **49% of BTC maker gross came from 5 of
171 windows**, and excluding the best 10 takes gross from $48.4k to $10.8k. A
quantity with that tail needs tens of day-clusters before a mean means anything.

**Stated as plainly as possible: the current one-test-day results have no
confidence interval.** The EXP-BLEND ΔBrier of +0.0245…+0.0316 is a
`DESCRIPTIVE` point estimate. Its stability across six horizons is *not* six
corroborations — those are the same 1,477 windows re-scored. ****§5's G-BRANCH IS STRUCK (R-86, 2026-08-24).** At G=2, `ci: Unavailable` is **not a limitation — it is the correct answer**, and the contract states it accurately. The claim ladder is refiled as **explicit DEBT with trigger G ≥ 7 day-clusters**, named by DE in `CONTRACTS_BATCH_v24` §3; it is not a defect to be worked around at G=2. The diagnosis below is retained because it remains true of the *rule ids*, which is Q-DA-40's separate ask: `R-CLUSTER` IS NOT A RATIFIED RULE — CORRECTED 2026-08-23.** Fail-closed probe over `contracts.yaml` v22: **`R-CLUSTER`, `R-WEIGHT`, `R-STRATA`, `INSUFFICIENT_CLUSTERS` and `estimand_kind` are ALL ABSENT (n=0)**, while `R-IMPUTE`, `R-ADMISS`, `R-CANARY`, `R-PROV`, `R-DUAL` and `R-ONEROW` resolve. The three inference rules this plan enforces against exist in exactly two files: **`contracts_measurement_delta.yaml`, which §1 declares DEAD and must not be followed**, and this document. They are **PROPOSED, not ratified.** *Live divergence:* v22 pins `MarkoutSummary.ci: Unavailable` and `claim_status: enum:DESCRIPTIVE_POINT_ESTIMATE` **unconditionally, no G threshold** — an implementer following v22 hard-codes "never a CI" and cannot relax at G≥7 without a contract change, while one following this section writes a G-branch keyed on a reason enum in no schema. Incompatible artifacts. Same for weighting: v22 `R-DUAL` requires per-fill AND share-weighted; §2.5 says `{notional, equal}` keyed on `estimand_kind`, **a field that does not exist.** As a PROPOSAL, `R-CLUSTER` makes
this structural: at `G < 5` the `ci` field is
`Unavailable(INSUFFICIENT_CLUSTERS)` and cannot be filled in by choosing a
different estimator.

The MDE is published **from variance components only** at freeze time. A
variance-only peek is pre-registration-safe and must be declared as such.

**CORRECTED 2026-08-23 — half true, and the load-bearing half was false.**
`PM_PREREG.md` under that exact name does not exist; but
**`PM_MEASUREMENT_PREREG.md` DOES, frozen `2026-08-21T03:26:27Z`**, supplying
A-TWAP-1's `spec_hash` and the source-profile registry hash, while
`coverage_ledger.py` refuses to evaluate any rule that is not FROZEN. **A-TWAP-1
IS frozen**, so "nothing above is frozen" was false from 08-21 onward — and that
falsehood is what licensed the §1.4 strike-clause defect corrected above: while
it stood, §1.4's divergent rule body read as a live proposal rather than as a
contradiction of a hash. The prereg freezes *infrastructure only* — not a
markout, calibration, probability, sigma, book or flow/fill estimate — so the
`strata_hash` half below remains true. *Superseded text:* `PM_PREREG.md` does not
exist. Until it does, nothing above is frozen, and the
`Gate` fields `frozen_at`/`strata_hash`/`spec_hash` are unfilled.

---

## 6. What EV-Replay will need from this layer

Designed in now so the interface does not change when replay arrives.

1. **`t_known_prov` + `t_known_err` on every value**, worst-provenance
   propagation, and the associative `t_known_hi` composition (§1.2). Replay
   **refuses** — does not warn — any result whose decision horizon `r` falls
   within `refuse_k · t_known_err` of a non-OBSERVED input, and stamps every
   output with the worst `t_known_prov` among its inputs.
2. **Collection-level truncation on every accessor**, including `history()`, and
   **no raw-handle escape**. `EV-Replay` reads through `StateView`, never
   `read_all` — the opposite of every other EV module.
3. **`Known.seq`**, a total order within source, for deterministic tie-breaking
   and warm-restart parity.
4. **`SelfState` as a replayable feed**, sourced through the Actuator outbox as
   an Environment seam, with `AckBracket` rather than an imputed ack.
5. **`EventTimeView`** for the `R-CANARY` twin run.
6. **`params.at(t) -> Known[value]`** on SP-Params. Without it,
   `params.get(name)` resolves *today's* value inside yesterday's replay — a
   look-ahead in parameter space, and this repo's own pitfall class.
7. **Field-level spec dependency, not boundary-level refusal.** An analysis
   declares which spec fields it depends on; replay refuses only if a
   *depended-on* field changed. The current all-or-nothing rule caps replay at
   ~12 days and, after 2026-08-20, at under one — a fee change must not
   invalidate a settlement study. A `--span-spec` escape must record the boundary
   and both `spec_id`s.
8. **`spec_id` by content hash** stamped on every dataset and run.
9. **`Coverage` on every estimate**, so a replay result can be compared against
   the coverage of the data it consumed rather than assuming a full panel.

---

## 7. Ways this could be wrong

Ordered by how much damage each does.

1. **Missing-not-at-random data loss, correlated with activity and concentrated
   in BTC.** 27 of 47 disconnects are our own send buffer overflowing; 32 of 47
   are BTC, which is 82–85% of notional. Messages we never received are missing
   *precisely* in the intervals where adverse selection is highest, so every
   markout is biased toward calm and toward optimism. `SLOW_CONSUMER` gap
   accounting and the `gap_arm = BOTH` requirement expose it; **they do not fix
   it.** The fix is a bigger receive queue and batched I/O in the collector, and
   the data already lost is not recoverable. *This is the largest risk in the
   plan and it is not a statistical problem.*
2. **The 250 ms reference-mid lag is a frozen guess inside a measured plateau.**
   It was pre-specified, which is the best available defence, but the plateau was
   measured on BTC on one day. If the venue's emission ordering changes, or if
   other coins differ, the partition moves. Mitigation: the sensitivity band is
   mandatory and the sign-consistency rate is a standing regression check —
   but a systematic shift would move `es` and `Λ` in opposite directions while
   leaving `rs` and net markout untouched, so the *net* number stays safe and
   only the *decomposition* is exposed.
3. **`G = 2`.** Everything stochastic in this programme is currently
   uninterpretable, and the temptation to read the stable-looking ΔBrier table as
   evidence is exactly the failure `R-CLUSTER` exists to block.
4. **Mid-relative bps is the contract's identity but the wrong headline.**
   `q(p−m)/m` diverges as `m → 0` and is not P&L-additive; if anyone reports the
   bps number as the headline on a sample containing deep longshots, the
   aggregate is dominated by the `p → 0` tail. The cash headline is the guard.
5. **Book snapshots are subscription-triggered.** Every statistic I computed from
   `book` events — including the 1¢ modal spread and the 2.8% one-sided rate —
   is a biased sample of time. They are indicative, not settled, until recomputed
   on the `price_change` spine.
6. **The imputation calibration is host- and region-specific.** `D̂_transport` is
   ours; only `D̂_publish` is portable. A deployment move silently invalidates
   every IMPUTED `t_known` fitted here, and nothing in the type system notices.
   `ImputationRule.calibrated_from` + `calibration_overlap` make it auditable,
   not automatic.
7. **The dedup/collapse guards are near no-ops today, and I am inferring from
   ~16k prints on one day.** If Polymarket ever does double-report, the guard
   fires correctly. But the opposite error is live: if the true structure is one
   print per parent event, then collapsing on `(t_event, q_up)` **over**-merges
   genuinely distinct trades that happen to share a millisecond, which *deflates*
   standard errors rather than inflating them. At 0.30% it is immaterial; at 10×
   the print rate it would not be.
8. **Chainlink is both the settlement source and a candidate signal.** Using it
   for the drift control would make the control endogenous. Binance bookTicker
   avoids that — but HYPE has no Binance leg, so the controlled arm silently
   drops a coin.
9. **Population-vs-marginal, plus subsidy contamination.** Every number is an
   upper bound on the average maker. Additionally, maker rewards
   (`rewards_registry.jsonl`) may make observed makers' behaviour
   unrepresentative of an unsubsidised entrant, and the subsidy is time-boxed and
   only partly verified.
10. **The canary can be defeated by the distillation itself.** Tier-1
    `windows.parquet` deliberately carries the winner. If a decision path ever
    joins it directly instead of reading through `StateView`, the leak is
    upstream of every guard in this document. The only defence is the port
    discipline: Tier-1 is reachable through `read_all` (EV) or `StateView`
    (replay), and never otherwise.
11. **Admissibility exclusion is itself a selection.** **94.72%** of windows are
    (corrected 2026-08-23; 18.3% was the pre-`<- NEW`-clause rate, §1.4) —
    dropped by A-TWAP-1. I show the differential is a startup artifact, but the
    *level* is not — and gap-containing windows are volatile windows. The
    `INDICATOR` arm is mandatory for this reason and a sign difference between
    arms must be treated as a result.
12. **~~The fee is unresolved and its sign matters.~~ **SUPERSEDED 2026-08-24: the FEE is resolved (taker `0.07·p(1−p)`, maker zero on 744/754); the REBATE is not.** The sign point survives and transfers to the rebate, which can only move net favourably — unlike the 1.3 % charged-maker-leg exception, which moves it the other way.** Gross maker revenue of
    ~+95 bps is not a conclusion until the fee source is settled, and the WS

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and the `+95 bps` maker gross are the SAME book-derived number and both fall together: `book` snapshots are p90 6.2 s stale. Rebuilt with no book at all the figure is **+0.17 ¢/share**, and it is **NOT DISTINGUISHABLE FROM ZERO** — window-clustered bootstrap gives **+0.173 [-0.251, +0.596]**, with all seven per-coin CIs spanning zero. **The maker-edge sign is UNDETERMINED at two days.** Also settled: `side` IS the taker's (G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0]), so the `+95 → −95` flip scenario is closed. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

    field is known to be unpopulated rather than zero.
13. **This plan freezes several parameters (250 ms, the τ grid, the clip, the
    r grid, `refuse_k = 1.0`) that no experiment has yet stressed.** They are
    frozen on purpose — a parameter chosen after seeing the estimate is a knob —
    but "pre-registered" is not "correct", and each should have a written
    revision trigger in `PM_PREREG.md` when that document is created.

---

## 8. Build order within the layer

1. `SourceProfile` registry + `Known` factory + `R-IMPUTE` checks (no consumers
   yet; the invariants are testable against all three streams immediately).
2. Tier-1 distiller with manifests and shard merge. Everything after this is
   seconds instead of minutes.
3. `Coverage` + `AdmissibilityRule` + the coverage ledger. Re-run E0 against them.
4. `StateView` + `EventTimeView` + `RefusalLedger`; re-run EXP-M6 through both and
   confirm the canary reproduces the 0.5 pp bite.
5. `EV-Markout`: parent events → `MarkoutObs` → **net** `rs(TO_RESOLUTION)` **as a LOWER BOUND** (carried from gross 2026-08-24 per R-84; §3.3). **THREE CONDITIONS, none optional:** (a) `R-GROSS` (`contracts.yaml:124-129`) pins `fee_status == GROSS_ONLY_NOT_APPLIED`, a single-member enum **asserted at `evaluation_pipeline.py:1189`** — **net is UNEXECUTABLE until v24 amends it** (Q-DA-42); (b) the maker fee is zero on **744 of 754** legs, so ten (1.3 %) were charged and that term is **UNFAVOURABLE** — a lower bound needs every unmeasured term favourable, and this one is not; (c) §5's ladder puts *net-of-fee profitability* at **G=7** and *`rs` against a cost threshold* at **G=30**, and the corpus is at **G=2**, so the figure is **computable and not yet citable**. *Superseded:* gross `rs(TO_RESOLUTION)` by
   phase and moneyness, `DESCRIPTIVE`.
6. `EV-Calibration`: panel → three-arm paired scores → `FLBFit.b̂` with day fixed
   effects, `DESCRIPTIVE`.
7. `SelfState`/`AckBracket` — needed only when an order exists, but the contract
   is fixed now so `EV-Replay` does not force a rewrite.

Steps 1–6 run entirely on data already on disk. None of them can be *read* as a
gate until G ≥ 7, and that is the point: build the harness now, so that when the
days accumulate the numbers are already honest.

---

## 9. What this plan deliberately does not do

*Added at review iteration 1 (2026-08-23). Until now this document had no scope
boundary and no falsifier section, while every sibling plan has both — so apart
from §7.13's five parameters, **nothing in it was a marked deferral and absence
was indistinguishable from oversight.** That is why the first review returned so
many findings. Several of them are converted to honest deferrals below; the rest
were real.*

- **Does not build the markout partition.** `spread + transient_AS +
  permanent_AS + snipe + own_impact` is deferred, not available — `own_impact`
  is ruled non-identifiable and has no declared pin (§2.7).
- **Does not implement the τ grid.** `{1, 5, 15, 30, 60, 300 s}` is specified and
  **not built**: `MARKOUT_SCHEMA` carries no τ columns, no `es`, no `Λ`, no `rs`,
  no `m`, and the shipped `markout_events` is **TO_RESOLUTION only** —
  `contracts.yaml:1758` has already followed the code. So `Λ(τ)` decay, which §0
  names as how spread capture is told from adverse selection, is **not
  measurable from any committed dataset**, and §4's Tier-2 spec describes a
  dataset that does not exist while two days are committed under its name. This
  is the largest gap between this plan and the shipped layer.
- **Does not build `book_snap` or `binance_bt`.** §4 names six Tier-1 datasets;
  **four** exist (`quotes`, `trades`, `twap`, `windows`) — this bullet's own
  header names **two** missing, so "five" was wrong (corrected iteration 3). Consequently §3.2's `drift_control` — the realised underlying
  return measured from Binance bookTicker — is **unbuildable today**, and §7.5's
  "until recomputed on the `price_change` spine" has no baseline to compare
  against.
- **Does not freeze A-BOOK-1.** It has no computable form: no cadence (so "gap"
  is undefined on an event-driven feed), no source pinned between `book_snap` and
  the `price_change` spine, `max(τ)` includes `TO_RESOLUTION`, and no
  `spec_hash`. `config/` holds only `a_twap_1.json`. Read literally it excludes
  every event, so `MarkoutEstimate.excluded_coverage` is unfillable.
- **Does not acquire history.** Backfill is named once, at §1.3, only to be
  refused — and refused for knowledge-time replay at `r = 2 s`, a case that does
  **not** apply to net markout, which §2.4 states *"never uses `m` and is
  immune"*. `pm_backfill.py` was queued in `PM_MM_PLAN` §7 and never built. So
  §5's read dates and §8's build-then-wait order are decisions taken with one
  option on the table. Recorded as a **choice**, not a conclusion.
- **Does not define a taker estimand.** `es`, `Λ`, `rs` are maker-only, while
  `DE_MODULE_PLAN` §1.1 ships `CROSS(side, size)`. The measurement layer cannot
  score an action the decision layer can already take.
- **Does not sweep the location estimator.** The mean is the only estimator
  named, on a quantity §5 shows is tail-dominated (49% of BTC maker gross from 5
  of 171 windows). The 4–6 week read date is partly an artefact of insisting on
  the mean.
- **Does not decompose the cluster variance.** "The day is the cluster" is applied
  globally, including to *paired* contrasts where the common daily path cancels.
  The day-level variance component **of the difference** is never estimated,
  though §5 licenses exactly that peek as pre-registration-safe.

## 10. What would falsify this design

*Distinguished from §7 deliberately: §7 is a risk register with mitigations —
these are observations that would show the plan's own CHOICES are wrong.*

1. **The 250 ms reference lag.** The measured grid was `{0, 50, 200, 1000}` ms;
   **250 was never measured** and the sample-optimal 200 is not in the mandated
   `{0, 50, 100, 250, 500, 1000}` sensitivity band. If the consistency rate on
   the six-point band has no interior plateau around 250, the frozen value is
   wrong and every markout conditioned on it moves.
2. **`MAX_i` composition.** If a composite `Known` built from one OBSERVED and
   one IMPUTED input cannot be constructed conformantly — R-IMPUTE's `rule_named`
   check fires and composition assigns no `imputation_rule` — then the rule is
   incomplete, not the implementation. Composition is also silent on `t_event`,
   and MAX vs MIN there changes the **leak-canary delta**, which is a
   gate-bearing number.
3. **`StateView.coverage(field, span)`.** `span` runs back from a
   `KnowledgeTime`; `Coverage.target` is a pair of `EventTime`; §1.2 clause 4
   says those have no shared comparison. If the only implementation requires the
   comparison the plan forbids, the accessor surface and the type discipline are
   mutually unsatisfiable and one must give.
4. **The protected span.** If a zero-tolerance 60-slot requirement excludes 94.7%
   of windows on a stream whose median max gap is 2 s — as measured — then either
   the bar is wrong or the stream is unfit for the estimand. Both are findings;
   neither is a threshold DA may move.
5. **Venue emission latency.** Measured 2026-08-23 on 21.6 M PM messages:
   `recv_ns − venue timestamp` has a 43.8 ms floor with a heavy tail — p99
   **3,631 ms**, p99.9 **7,967 ms**, and **3.68% above 750 ms**. Attributed
   venue-side rather than collector-side (94.2% of affected wall-clock seconds hit
   only ONE slug; a collector stall would be common-mode), so `t_known` is honest
   — but if that attribution is wrong, knowledge-time truncation is wrong
   everywhere in this programme, including both committed Tier-2 days.

