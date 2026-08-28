# Lane 4 — seven-arm replay parity-stub battery (SPEC ONLY)

**Authorised by** the user's plan `d506a06` — *"common action-value interface
plus seven-arm offline replay"*, **build/preregister only, nothing scored**.
All code remains offline replay/research code.
**Status:** SPEC. **Owner:** DA (battery design). **Consumers:** DE, BE.

## 1. Why stubs first, and why they are typed

The battery is built and proved **before any predictor exists**. Every arm is a
**typed stub** returning declared-shape output with no model behind it. The
point is to establish that the harness itself is neutral: **if the arms differ
while every predictor is inert, the difference is the harness, and any later
result would inherit it invisibly.**

This ordering is not caution for its own sake. The programme's own history is
that path-coupled overlays amplified prediction noise 10–20x and produced large
replay deltas with zero ranking improvement. A battery that cannot first
demonstrate zero difference under zero signal cannot attribute a later
difference to signal.

## 2. THE ANCHOR TEST

> **A disabled predictor must be BIT-IDENTICAL to `QR_SKEW_ONLY`.**

Bit-identical, not "statistically indistinguishable", not "within tolerance":
identical fills, identical cancellations, identical inventory path, identical
per-window totals, byte-equal serialised trajectory. **A tolerance here would
hide exactly the coupling the test exists to find** — and after today's finding
that non-associative summation alone moves totals by ~1e-11 on identical terms,
"close" cannot be distinguished from "differently ordered but wrong".

Corollary anchors from the user's list, each equally bit-exact:
- **Infinite cancel threshold** ≡ `QR_SKEW_ONLY` (nothing ever crosses).
- **Zero repost threshold with permanent hold** ≡ cancel-and-hold.

## 3. The seven arms (the user's list, unchanged)

1. `QR_SKEW_ONLY`
2. `QR_CANCEL_HOLD_X_SKEW`
3. Fill-hazard-only cancel, neutral placement
4. Conditional-value cancel, neutral placement
5. Conditional-value cancel × frozen skew
6. Conditional-value cancel × frozen skew × fair-price residual
7. Random cancel, **matched on action count, side, hour and budget**

Run on the **same neutral opportunities and independent event clocks**. Arm 7 is
the matched control and inherits rule 7: matched on the decision variable,
compared on the decision metric, never on a proxy.

## 4. Lifecycle invariants the battery must enforce

From the user's list, each a behavioural test with a known-bad:
- One generation is cancelled **at most once**.
- Cancelled skewed orders **cannot fill after simulated effectiveness**.
- **Pre-effectiveness fills remain charged as stale** — the latency estimand on
  the replay side.
- Rate limits count **requested, effective and suppressed** cancellations
  separately.
- **No policy-generated trajectory is reused as its own training population.**

That last one is the outcome-selection rule (rule 1) in replay form, and it is
the one a battery is most likely to violate silently: a policy that generates
its own training data conditions on the event it exists to prevent.

## 5. Falsifiers the battery ships with (rule 15)

Each must FIRE on a known-bad, and each must have a **positive control** —
today's lesson: *the same battery that passes a correct harness must refuse a
broken one, or "all arms agree" is evidence of an unrun battery.*

1. **Anchor, both directions:** disabled predictor ≡ `QR_SKEW_ONLY` bit-exact
   (positive), **and** a deliberately perturbed stub — one extra cancel — must
   BREAK parity. If a one-cancel perturbation does not break it, the comparison
   is not bit-exact and the anchor is decorative.
2. **Determinism, cross-process:** two runs under **different `PYTHONHASHSEED`**
   produce byte-identical trajectories. Today's blocker-7 finding was exactly
   this class — a fixed RNG seed over a process-dependent iteration order is an
   independent draw, not a reproduction — so the battery must not inherit it.
3. **Matched control:** arm 7's action count, side and hour distribution equal
   the arm it is matched against, asserted rather than assumed.
4. **Double-cancel:** a stub attempting a second cancel on one generation is
   REFUSED.
5. **Stale charging:** a pre-effectiveness fill appears as stale, not as
   prevented; a post-effectiveness fill does not.
6. **Empty run refuses:** a battery over zero opportunities must NOT report
   seven passing arms. Zero difference under zero data is not parity.

## 6. What this document does NOT authorise

No scoring, no promotion, no forward clock. Any arm later evaluated on data
starts its own ≥5 complete-UTC-day clock on unconsumed days; consumed days stay
consumed. Whether any arm is adopted is a policy decision with its own priced
trade-offs (rule 14) — the battery estimates, it never decides.

---

## AMENDMENT B1 — hardening round 2 (Codex batch 2, item 5)

Five findings against the first build, plus four defects the hardening's own
falsifiers found in the hardening. Instrument: `da_replay_parity_battery.py`,
**62 checks, green**.

### B1.1 The matched control did not match — it could not have failed

`matched_control(opps, cancels)` took `cancels` and **ignored it**: 0, 1, 6 and
99 all produced 12 cancels. Both arms cancelled every generation above a
threshold, so the profiles agreed **by construction**, and every "matched: True"
it ever reported was uninformative.

**The fix is not to honour the argument — it is to delete it.** A matched
control's action count is *determined* by the treated arm; a count the caller
can choose is one that gets chosen after the numbers are visible. Same
reasoning as the date-predicate granularity in `da_forward_day_verify`. The
budget knob now lives one level down as a tested primitive:

`budget_matched_selection(pool, budget, seed)` draws exactly `budget`
generations uniformly without replacement from the cell's eligible pool.
Budgets 0/1/6 return 0/1/6; **99 REFUSES** rather than clamping, because a
control that silently drew fewer actions than the treatment is no longer
matched on the decision variable and the shortfall would be invisible in the
profile it reports.

Two properties the draw must have, both checked red-first:
- **order first, then draw** — the pool is sorted into a total order before the
  RNG touches it, and shuffling the input does not change the selection.
  Reproducibly sampling an unstably-ordered sequence is blocker-7's defect with
  a seed bolted on.
- **`selection_differs`** — the control must select *different* generations than
  the treatment. A control reproducing the treated selection would match
  perfectly and measure nothing. `strict_subset` likewise: matching an arm that
  cancels everything is vacuous, so the treated arm must cancel some but not
  all (6 of 12 on the fixture).

The stub scorer is **sha256-derived, not builtin `hash()`** — `hash()` of a str
is salted by `PYTHONHASHSEED`, so a scorer built on it would select a different
cancel set per process.

### B1.2 The receipt: enumerate, don't derive

`battery()` ran two anchors and returned them. A reader could not tell which
checks existed, which had run, or whether the top-level boolean covered all of
them or the two that happened to be present.

`all_pass` is now the conjunction over an enumerated `REQUIRED_CHECKS`, and **a
required check ABSENT from the receipt makes it False** and is named in
`missing_checks`. Absence is the failure mode that reads as success. The receipt
also carries `fixture_sha256` and `battery_code_sha256`, and the fixture digest
moves with the fixture — a parity result read six weeks later is verifiable at
the artifact it claims (rule 16).

### B1.3 The zero-repost anchor — the arm that wins by not trading

An arm that cancels once and never reposts has no further adverse fills, and no
further fills at all. On harm share, adverse-per-fill, or rho it **wins**. That
degeneracy is invisible to any metric normalised by activity.

`permanent_hold_anchor` measures **exposure, not harm**: withheld share against
the declared `PERMANENT_HOLD_WITHHELD_SHARE = 0.25`. The holder is flagged
(9/12 withheld) and takes 3 fills against the normal arm's 12; the normal arm is
**not** flagged, so the flag is not simply what the anchor always reports.

This requires a representational change: a withheld quote is a **status**
(`PLACE_WITHHELD`), never a silent absence. An arm that just stopped emitting
would be indistinguishable from one that ran out of opportunities (rule 4).

### B1.4 Rate limits: requested ≠ effective

A cancel **requested** is not a cancel **effective**. It binds only after
`CANCEL_EFFECTIVE_LAG_S`, and only if the limiter let it through. A request the
venue suppressed **prevented nothing**, and valuing it as prevented harm
inflates the estimand — while looking, to any counter that reads
`CANCEL_REQUESTED`, exactly like a request that bound.

Evaluated as an identity: `requested = effective + suppressed` (12 = 3 + 9 on a
run where the limiter actually bound, both sides non-zero). The load-bearing
anchor: **under a limit of zero, the fills are bit-identical to the arm that
never cancelled at all.** If they were not, suppression is being credited
somewhere.

Per-window effective counts are exact and enter the verdict, bucketed on the
**request** time — the limiter admits or refuses at request, so bucketing the
effective event would smear counts across the window boundary and report a
limit that was never applied.

### B1.5 The training-reuse guard (rule 11)

`assert_no_training_reuse(train_days, score_days)` refuses on overlap, and
refuses when scoring days are **not strictly later** than every fitting day —
disjoint is not enough; validation is *later untouched* days, not merely other
days.

**It also refuses on two empty sets.** They are disjoint, so a guard testing
only `train & score == ∅` would pass loudest exactly when it had nothing to
check. This programme has already shipped an invariance check that compared two
empty files and printed IDENTICAL.

### B1.6 The external-arm interface — a data contract, not an import

`load_external_trajectory` reads a declared-shape object and rebuilds the event
list under DA's own canon. **It imports nothing from BE** (R-235): the moment it
called BE's serializer, agreement would stop being evidence.

Refusals, each with a red-first test: canon mismatch (trajectories under
different canonical forms cannot be byte-compared); undeclared arm; empty
trajectory (it would satisfy every invariant trivially); **missing field**
(never defaulted — a checker that supplies what the producer failed to produce
is testing its own defaults); **undeclared extra field** (ignoring it would take
the digest over a *projection* of what the producer actually did); undeclared
kind; duplicate `seq`; non-finite stamp.

`external_lifecycle` then checks *behaviour*, because shape validity is not
behavioural validity: a well-formed trajectory that cancels one generation twice
loads cleanly and fails. Duplicate arm submissions are flagged — otherwise one
arm could be scored twice and a missing arm go unnoticed.

### B1.7 Four defects the falsifiers found in the hardening itself

Recorded because they are the same class the battery exists to catch, and
because three of the four were **decorative fields beside a verdict they did not
enter** — a shape this programme has already paid for:

1. `effective_within_limit` multiplied the limit by the number of windows
   spanned (a loose bound) and **was not in `pass` at all**. Now exact and
   load-bearing, with a falsifier proving it can return False on real output
   (unlimited: one window carries 12 > 3).
2. `no_fill_after_effective` checked only `FILL`, so a producer could relabel a
   post-cancel fill as `FILL_STALE` and pass — and `FILL_STALE` is *defined* as
   pre-effectiveness, so that is precisely the mislabel the check exists for.
   Both kinds now checked.
3. `matches_reference` was computed per arm and **left out of `pass`**, so a
   submission could report `pass: True` beside a reference mismatch. It now
   enters the verdict, and with no reference supplied the key is **absent, not
   True** — an unrun comparison must not read as a passed one.
4. The tightened signature check for the deleted `cancels` argument was written
   with a clause that made it unsatisfiable; it **failed red on the first run**
   and was fixed. Its predecessor (a `co_varnames` grep) would have passed while
   proving less — the source-grep-vs-identity shape.

Also fixed: `unrequested` was derived by iterating a set of string tuples, whose
order is `PYTHONHASHSEED`-dependent; and an outcome without its request now
fails, since `requested = effective + suppressed` could otherwise be satisfied
by two compensating errors.

### B1.8 What this still does not do

Stubs only, plus a contract for external submissions. **No BE trajectory has
been checked through it** — the interface is designed and tested against
round-tripped stub output, and its first real use is the first thing that can
falsify the contract. Nothing here is scored, no stub is ever a candidate, and
the battery estimates rather than decides (rule 14).

---

## AMENDMENT B2 — identity is two-dimensional (composition × predictor)

Raised by BE against the external-arm contract (`a3e8382`), which **refused to
guess the arm-name mapping** rather than label a run with the nearest-looking
name. The refusal was correct and the contract was underspecified.

### B2.1 The mismatch

`ARMS` names **policy compositions** — which components are active and whether
they interact. 011's `composed_linear` / `composed_lgbm` are **predictor
candidates** — which estimator produced the scores a composition consumed.
These are orthogonal axes, and one string cannot carry both.

Worse, the axes are not independently nameable: `CONDVALUE_X_SKEW` asserts an
**interaction** that 011's arms do not distinguish. Mapping `composed_linear`
onto it would assert a composition BE did not implement — a claim the checker
would then be structurally unable to falsify.

### B2.2 THE CONSEQUENCE THAT IS NOT ABOUT FIELDS

**A run is identified by the PAIR, so the count of candidates in a forward race
— rule 12 multiplicity, recorded at freeze time — is the number of PAIRS.**
Seven compositions over two predictors is **fourteen** candidates, not seven.
Any multiplicity already recorded on an arm count is wrong by that factor, and
a race is corrected before it starts or not at all.

### B2.3 Predictor identity is trajectory-level and EXCLUDED from the digest

Same argument that excludes the arm name, and it binds exactly as hard: a
per-event predictor string would make `composed_linear` and `composed_lgbm`
differ in **every event**, and the inert anchor — all arms bit-identical with
every predictor disabled — could never pass. A parity anchor that cannot pass
is not an anchor.

So the split is: **canonical bytes = what was DONE; trajectory-level identity =
who DID it.** Two consequences, both checkable *only because* identity is out:

- two submissions that are bit-identical but differ in identity is the
  **interesting** case — two predictors that behaved identically;
- two with the **same** identity that differ is a determinism failure.

### B2.4 A required field, not a manifest

`predictor` and `predictor_active` are **required top-level keys**, exact in
both directions like the event fields: absent refuses, undeclared refuses.

Not a sidecar manifest, and today supplied the argument: a manifest is a second
artifact that can drift from the thing it describes. The nightly log said
`verdict artifact written` beside a file that had since been replaced, and the
fix was to make the artifact **self-describing**. A trajectory that travels
alone must still say what produced it.

### B2.5 The arm name must be checkable, not a label

The producer states the `components` it actually ran and whether they
`interaction`; the loader verifies both against the declared `ARM_SPEC`
decomposition. A mismatch **refuses**.

This is the clause that **forces BE's 011 question into the open rather than
resolving it by resemblance**: a run named `CONDVALUE_X_SKEW` reporting
`interaction: false` is refused, so the mapping gets *decided* — by BE and the
coordinator, on the record — instead of being inherited from a name that looked
close enough. An `X` in the name is an interaction claim.

### B2.6 Two contract clauses this makes enforceable

- **inert agreement**: every submission with `predictor_active=False` over the
  same opportunities must be bit-identical, whatever its arm or predictor. The
  inert anchor, generalised to trajectories this harness did not build — with
  every predictor off, a difference can only be the harness.
- **declared-active-but-inert**: a submission claiming an active predictor that
  is bit-identical to the inert set is **reported**. Not necessarily an error —
  a threshold may never have been crossed — but *"we ran the model"* must not
  read as *"the model acted"*.

### B2.7 A defect this exposed in my own checker

`check_external_arms` keyed its results on `tr.arm`, so **two predictors running
the same composition silently overwrote each other** — a whole candidate could
vanish from the results while the submission count still looked right. Now keyed
on the pair, with duplicates flagged. Taking identity seriously found it; the
one-dimensional contract concealed it.

### B2.8 What this does NOT decide

**Which 011 arm maps to which composition.** That is BE's to state and the
coordinator's to rule. This amendment makes the mapping a **decision that must
be made explicitly** rather than one that can happen by accident — and until it
is made, a real trajectory cannot load. Which is the correct state to be in,
since no trajectory may be scored under the standing hold anyway.

**11 new red-first checks; battery at 73.**
