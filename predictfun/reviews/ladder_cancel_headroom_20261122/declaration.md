# FEASIBLE-CANCEL-HEADROOM-v1 — declaration (frozen before any table)

2026-11-22, session aea5bfb9.
- **Specified by:** review `3e84ca6d` (ONE next priority after ASYNC-EXECUTION-SENSITIVITY-v1, target `85127b3d`).
- **Amended by:** the USER's amendment `6d200317` (`reviews/automatic/20261122_queue_fill_tradeoff_amendment/plan.md`), folded in here before any output.
- **Builder:** `predictfun/research/pf_ladder_cancel_headroom.py`. Its docstring is the full rule; this file states it for the receipt.

**What this is:** an ORACLE fixed-path deletion accounting diagnostic on saved data. It is not:
- a deployable signal, strategy or speed / queue variant;
- an upper bound on market-making policies;
- a validation or a profitability claim.

**What it does not use:** no tape, replay, model fit, threshold search, new day or post-hoc market exclusion. Consumed Oct 24–27 only, on the same 19 complete markets. POSITIVE-PRESSURE-GUARD-v1 stays stopped.

## Population and inputs

Every submitted generation is included: refused, unfilled and never-reviewed ones stay, as statuses.
- **Arms:** CURRENT_ASYNC_FAST10 (primary) and FRONT_ASYNC_FAST10 (an artificial-priority reference). The two are reported separately and **never pooled**.
- **Inputs:** ASYNC-EXECUTION-SENSITIVITY-v1's day records: orders, fills, fill marks, closures, queue records and command arrays. They are sha256-bound in `manifest.json`, and each record's gzip / npz bytes are checked.
- **Refusal:** the run refuses unless the unchanged path reproduces that receipt's committed per-market eligible +1 s and END (`outputs/arms.csv.gz`, its sha256 = its committed audit). It also refuses unless every fill is counted once.

## Choices and values

**Choices:**
- KEEP the recorded path; or
- replace ONE recorded live KEEP callback *d* of the generation by a permanent PULL.
  - Any reason counts; fallback KEEPs are counted separately.
  - The PULL takes effect at *d* + 10 ms, the profile's live cancel, checked on every record.
  - The deleted suffix S(*d*) = that generation's recorded fills with t ≥ *d* + 10. Earlier fills, the request tick included, have happened.

**What is not allowed:**
- no cut before the first live review;
- no deletion of single fills (same-time fills are indivisible);
- no replacement, re-entry, queue reset or inventory feedback.

Every other order and the external flow stay fixed. No recorded live KEEP → `no_feasible_callback`.

**Values per fill (native eligibility):**
- **v1** = s·q·(mid(t + 1 s) − p), when t ≤ END − 1 s and the mark exists; otherwise the fill is *unvalued*.
- **v5 / v30:** the same with their own marks; None = censored.
- **vEND** = s·q·(mark_END − p).

**On a suffix:**
- G = Σ max(v1, 0): the good value the cut sacrifices.
- L = −Σ min(v1, 0): the bad value it avoids.
- net = G − L: KEEP minus cancel.
- Also: counts and shares of positive, negative, exactly-zero and unvalued fills.

Tolerance TOL = 1e-6 c.

## Suffix classes

Exhaustive, applied in this precedence:

| Class | Definition |
|---|---|
| `incomplete_valuation` | a suffix fill has no v1 (observed components still shown; never called pure or compensated) |
| `pure_observed_loss` | L > TOL, G ≤ TOL |
| `compensated_mixed` | G, L > TOL and G − L > TOL |
| `uncompensated_mixed` | G, L > TOL and L − G > TOL |
| `mixed_tie` | G, L > TOL and \|G − L\| ≤ TOL |
| `good_only_or_no_fill` | the rest (denominators kept; absence of a loss is not a success signal) |

These are descriptive hindsight classes. They are never trading states, predictable states or training labels.

## Anchors

Economic totals reconcile once per generation at each anchor.
- **(a)** The first feasible callback.
- **(b)** The **ORACLE:** the cut maximizing the retained eligible +1 s (unvalued fills count 0).
  - It cuts only if its value exceeds KEEP's by more than TOL.
  - It then cuts at the earliest callback whose value exceeds KEEP's by more than TOL and is within TOL of the best.
  - Classes at (b) add `oracle_keep` and `no_feasible_callback`. A chosen cut can only be pure, uncompensated or incomplete; this is checked.

Without a feasible callback, the prefix is the whole generation and the suffix is empty at both anchors.
- **END:** recomputed for each anchor's cut (not re-optimized), with late / unvalued fills apart.
- **5 s / 30 s / END:** reported at the **same cut**, on native eligibility and on the common cohort (fills valued at 1 s, 5 s and 30 s). There is no reselection by horizon.

## Sequences

Per generation, computed on distinct fill times, where a time's sign comes from its fills' G / L. The flags **overlap**; they are never summed as disjoint populations.

| Flag | Definition |
|---|---|
| `bad_then_good` | a good time after the first bad time. Reported with the latest feasible callback whose effect is at or before that bad time, and G / L / class of its suffix. |
| `good_then_bad` | for the first bad time preceded by a good time, and the last good time before it. Reported with the latest callback whose effect lies after the good time and at or before the bad time (or infeasible). |
| `interleaved` | ≥ 2 sign changes |
| `simultaneous_good_bad` | one time carries both a good and a bad value (impossible within one generation here, since all its fills share price and side; counted anyway) |

The whole-generation class is also reported.

## Queue and decision states

Only recorded values are used; unknown stays NaN and is never read as front.
- **Recorded fields:**
  - submitted size;
  - activation venue ahead (`queue_venue.ahead0`);
  - first-review venue ahead / remaining and its time (`queue_venue.first`);
  - the client's first-review known ahead (`queue_known`);
  - a pressure pull's venue record.
- **Per anchor:**
  - order age;
  - client-known remaining (fills noticed at t + 100 ms);
  - venue remaining at the effect (hindsight, never a decision input).
- **Same-price replacement:** a generation on the same side at the same price within 1 s of a recorded request is a NEW generation, with its own activation ahead. Same-gid continuity is never merged with it.
- **Decision panel:** one row per (generation, feasible callback). It is a labeled panel of decision states, not independent samples.
- **Casebook:** per category and arm, the 3 generations with the smallest sha256(arm|day|m|gid). The key is outcome-independent, not hand-picked. Each case carries its exact fill and callback times.

## Reading (declared; computed by the builder and the checker)

**The rule:** per arm and day, if the oracle's retained eligible +1 s total is < −TOL, the day reads `no_local_cancel_rescue`; otherwise `local_cancel_headroom`. The same rule applies pooled.

**What a reading permits:**
- `no_local_cancel_rescue` starts no cancellation-only variant.
- Headroom permits only a later causal-discrimination question, and is reported with its good-fill cost and concentration. It permits no model fit or promotion.

**Wording rules for the report:**
- Saved data can show a trade-off *linked under the recorded keep-versus-permanent-cancel choice*. It cannot show that a bad fill was *necessary to earn front priority*: re-entry is untested, so queue dependence stays **unidentified**.
- A purely bad suffix is a candidate for avoidance, not proof that it was detectable before the fills.
- The bound is optimistic: other orders, external flow and inventory are fixed.

## Outputs

**Data directory** `data/predict_fun/research/ladder_cancel_headroom_v1_20261122/`, write-once:
- `generations.csv` (full) and `generations_core.csv`;
- `summary.csv`, `classes.csv`, `markets.csv`, `sequences.csv`, `panel_summary.csv`, `cases.csv`;
- `panel.csv.gz`;
- `audit.json`, holding every sha256 and the projection's sha256.

**Receipt copies** under `outputs/`, except the full generations table and the panel.

**Post-run readout:** `tools/read_headroom.py`, frozen here before the run. It is descriptive arithmetic on the frozen outputs and adds no reading rule. It writes `outputs/read_headroom.json`:
- **T1** anchor-(a) classes with the Codex split of `good_only_or_no_fill` into NO_SUFFIX_FILLS / GOOD_ONLY / NEAR_ZERO, checked against the Codex counts and nets;
- **T2** the oracle per arm × day;
- **T3** each anchor-(a) class linked to its full recorded prefix (loss / gain only / none) and end cause;
- **T4** horizons at the oracle cut;
- **T5** sequences;
- **T6** the decision panel;
- **T7** recorded queue fields of cut vs kept generations.

**Bundled correction** (review `3e84ca6d` P2): `tools/async_identity.py` writes `outputs/async_identity.{csv,json}` from the ASYNC receipt's committed arms copy:
- the END identity per arm: eligible +1 s, subsequent movement on that cohort, late / unsupported fills at END, END;
- a check against the review's committed `end_decomposition.csv` (0.01 c), with a falsifier.

A superseding interpretation of the ASYNC receipt (`ladder_async_fast_20261122/report_interpretation_v2.md`) cites it; the frozen ASYNC report stays as provenance. The P3 checker-coverage item stays deferred, as the review allowed.

## Verification (before the freeze)

- **Synthetic selftest:** 8 groups, including the parent's list and the amendment's hand cases:
  - bad-then-good with net gain and with net loss;
  - good-then-bad with and without a feasible intervening effect;
  - simultaneous fills;
  - a missing later mark;
  - a new gid at the same price.
- **Mutation:** 38 / 38 planted errors caught by a check (`logs/mutants_try5`).
- **Independent checker:** `ch_checks.py`, families A / B / E / O / Q / H / R / S / K / C / P / I. It recomputes every generation, the panel and the casebook with its own code.
- **Cross-check X:** the Codex review's independent first-anchor classification (`c72f0d37`, its committed generations). Per generation it compares the class (with its finer good-only split), suffix G / L, prefix G / L, whole +1 s / END, the end cause and the anchor.
- **Falsifier:** 19 corruptions. A known-bad classification is among them and must fail `O_anchors`; a disagreeing Codex class must fail X.
- **Rehearsal:** `tools/rehearse_checks.py`, latest `logs/ch_rehearse_try3.log`, ran the builder and every checker family on a synthetic two-day record set. A self-consistent stand-in exercised the X code path. All families passed except `A_committed` (expected before the freeze), and all 19 cases were refused.

## Resources

- **Caps:** this round ≤ 0.05 core-hours; review reserve 0.02; cumulative line ≤ 2.50, including the reserve.
- **Ledger:** this round's run_light logs, plus a declared 30 s allowance for work with no surviving log:
  - ~12 s of population-only structure inspection run outside run_light before the design was fixed;
  - two first-draft mutant batches, 0.1 CPU min each by their surviving summaries, whose per-mutant logs were deleted on 2026-11-22 before the rewrite.
- **Codex classification CPU:** its 3.01 CPU s count inside this round, as its declaration requires. They are copied verbatim into `logs/counted_codex_queue_fill_classes.log`; its directory is not in `line_reviews`, so there is no double count.
- **Gate:** the write-once projection, which must be true before the run:
  - spent + 7 × the timing job + 15 s ≤ 0.05;
  - line + that + 0.02 ≤ 2.50;
  - peak < 1,536 MiB.
- **If the projection does not fit:** stop and ask the USER; do not extend.

## Later directions read before this freeze

**`2a8cde50` / `c72f0d37` — Codex's independent first-anchor classification.** Reproduced per generation by X and per arm by T1. The oracle analysis is kept separate.

**`90e82f8e` — USER correction: early losses may pay for later use of retained priority.**
- The suffix classes start at the first feasible cancel. They do NOT count or test that whole-life mechanism.
- T3 links each class to its full prefix and end cause, and the report states the anchor distinction.
- The matched retain vs avoid / re-enter comparison needs a replay design (below).

**`c590b083` / `60768c83` — USER direction: longer-continuation, two-sided economic value.**
- This diagnostic is not the action objective. The +1 s suffix value locates adverse selection; it does not decide KEEP / PULL.
- The same combined deliverable adds a design-only contract, written after this run: a paired retain vs avoid / re-enter continuation to a common END, with full-ledger two-sided labels, its controls and a pilot cost projection.
- No replay, fit, day or budget is launched or extended here.

## Multiplicity

There is no candidate or forward race. It is one diagnostic on two separately reported arms, with two accounting controls: the recorded KEEP and the first-callback cut.
