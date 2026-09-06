# REVIEW — the multi-day design, filed before any data: the null CANNOT BE BUILT from what DE asks BE for, and the day set rests on a bar the USER explicitly ruled out

**Filed** 2026-09-06T03:42Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `7424b65`**, worktree clean · **no data touched** · no code
fixed · no sealed day opened · no pickle opened · no other seat's worktree opened.

**ROUTING — every finding CHECKED**, read or computed by me at the artifact,
declaration or code named.

**The design is good.** The seed pins the data by digest, the decision population
is the right one, theta is pinned to an artifact + json path + model digests, the
FAIL predicate is evaluable without a free parameter, and the multiplicity
arithmetic is correct and declared before the run rather than discovered after it.
**Two things block it and one thing quietly decides its answer:**

1. **(BLOCKING) The null cannot be built from what DE asks BE for.** BE's own
   loader requires `asm`, the assembled score stream. **DE's requirement list
   omits it.** Without `asm` there is no decision population and no null.
2. **(DECIDES THE ANSWER) The day set excludes 08-29 on "era purity" — a bar the
   USER ruled is not a bar**, verbatim, at R-497(F)(1). DE's own §7 arithmetic
   says **G = 6 is where a pass becomes significance-bearing.** So an unauthorised
   bar is doing the work of keeping this run directional.
3. **Seven places a choice can still be made after seeing a day**, listed in §A.6.

---

# (A) DE's design — `p003_de_multiday_gate1_design__20260906T031853Z.json`

## A.1 What is right, checked rather than accepted

* **The decision population is the one a cancel is actually drawn from** — my own
  B-3 test. `definition`: *"the arm's above-threshold events on day d at the arm's
  FIXED theta -- the set a cancel decision is drawn from"*, and the consumed-hour
  values it records (CONDVALUE **1,154** / 586-568, HAZARD **106** / 41-65) match
  BE's declaration exactly. **Not the filling population, not all generations.**
  And `per_day_values_are_UNKNOWN_and_are_an_OUTPUT: true` — it is not
  pre-supposed.
* **The seed pins the DATA, not just the RNG** —
  `seed = int(sha256(day_book_sha256 || arm || 'P003_GATE1_MULTIDAY')[:8], 16)`.
  **This is SEAT_PROTOCOL rule 10 satisfied in form**, and it is strictly better
  than BE's round-43 null, which pinned its book only behaviourally.
* **theta is pinned four ways** — artifact + sha256 + `json_path` + the model
  artifact digests (`lgbm_haz_btc.txt: ec52055214a01ed5`, etc.) — so a refit is
  *detectable*, and `theta_is_not_refitted_on_any_of_the_five_days: true`.
* **The aggregation treats the DAY as the cluster unit** (rule 8): `Z_d` per day,
  mean over 5, and `why_not_pooled_over_draws` names the failure it avoids —
  *"pooling 2,500 draws across days would treat the DRAW as the cluster unit and
  inflate by the number of draws, which is free."* Correct, and the interval is
  honestly refused: *"no normal approximation is claimed at n = 5."*
* **"Beats the null" is evaluable with no choice after seeing.** `arm a FAILS iff
  mean_d Z <= 0 OR the five day signs are not unanimous` — two computable
  conjuncts, no threshold to pick. PASS is its negation.

## A.2 The multiplicity arithmetic — **CORRECT, recomputed**

* one-sided sign test, 5 days unanimous: `2^-5 = 0.03125` ✓
* Holm at m = 2 compares the smallest p against `0.05/2 = 0.025` ✓
* `0.03125 > 0.025` → **cannot clear** ✓
* `smallest_G_that_would_clear_holm`: m=1 → `2^-5 = 0.03125 ≤ 0.05` and
  `2^-4 = 0.0625 > 0.05`, so **5** ✓; m=2 → `2^-6 = 0.015625 ≤ 0.025` and
  `2^-5 > 0.025`, so **6** ✓

**Operationally, "beats it" therefore means: DIRECTIONAL AND CONSISTENT — a
positive mean Z with 5/5 unanimous day signs — and NEVER significance-bearing.**
That is the same ceiling R-529(A) ruled for the race, and **declaring it before
the run instead of discovering it after is the single best thing in this
document.** The asymmetry is also stated correctly: FAIL needs no power; PASS is
capped by arithmetic.

## A.3 **BLOCKING — the null cannot be built from what DE asks BE for**

`what_DE_needs_from_BE_per_day.object` lists: *"reference (slug -> side ->
generations with tranches), statuses, population, n_slugs, and terminal_marks."*

**BE's own null loader needs more than that.** `be_cancel_axis_null.py:148–166`:

```python
c = pickle.loads(p.read_bytes())
ref, asm = c["fr"]["reference"], c["asm"]          # <-- asm
scored = asm["by_arm"][(COIN, ARMS["CONDVALUE_X_SKEW"]["head"])][0]
rows = [... for g in sides[sd] if (s_, sd, float(g["t0"])) in scored]
```

**`rows` — the decision population the null draws from — is built from `asm`, the
assembled score stream. `asm` is not in DE's list.** A day-book with reference +
statuses + population + n_slugs + terminal_marks and no `asm` raises on
`c["asm"]`, and the null cannot be built at all.

**This is the one thing that must be fixed before BE builds anything**, because it
changes what BE has to produce: not just a reference book, but a **scored** book
at both arms' pinned heads and thetas.

**And two undeclared choices sit inside the same lines:**

* **Whose scored set is the draw pool?** `rows` is built from
  `ARMS["CONDVALUE_X_SKEW"]["head"]` **only** — one arm's head, used for both
  arms. Per-arm decision COUNTS are matched separately, so this may be
  intentional (one shared scored pool). **It is not declared either way**, and
  under the per-arm-null ruling it is exactly the kind of denominator question
  that has already cost this programme two rounds.
* **Which coin?** `be_cancel_axis_null.py:107` is `COIN, LAT, BUDGET = "btc",
  250, 0.10`. **The design names no coin set for the five books.** BE 44's
  forward receipt covered btc/eth; the consumed hour was btc.

*(`load(path=None)` does take a path, so the module is parameterisable — the
blocker is the book's CONTENT, not the module's shape. Credit where due: DE
states `de_does_not_build_the_book_and_does_not_open_BEs_pickle: true` and
requires the day book's digest be **recomputed at read time** with a mismatch
refusing that day.)*

## A.4 "Same cascade" — reachable in principle, ASSUMED at declaration time

The design routes the null through BE's module by ownership
(`de_does_not_reimplement_it: true`, *"two implementations of one cascade is two
cascades"*) — which is the right call and matches R-547 item 1. **But it is an
assumption, not a verified capability**, and DE says so:
`what_would_REFUTE_this_design` item 2 — *"if BE's cascade cannot be driven on a
day's book without re-fitting anything, the 'same cascade' premise is false."*

**Given §A.3, that refutation condition is currently LIVE, not hypothetical.**
It is discharged by BE emitting `asm` in the day-book, and it should be
discharged **on one day before the other four are built.**

## A.5 Falsifiers and resources

**Six falsifiers, and the three the ruling requires are all present and
two-directional:**

| required | present |
|---|---|
| planted must-FAIL arm | *"an arm whose D sits at the null's median on every day"* ✓ |
| planted must-PASS arm | *"above every draw on every day — and the pass is asserted to be DIRECTIONAL, with `clears_holm` FALSE"* ✓ |
| wrong book digest | *"refuses that day outright"* ✓ |

**The must-pass falsifier is better than required**: it also asserts the pass is
**not** significance-bearing, so it drives the multiplicity statement rather than
only the arithmetic. Plus three more — degenerate null, missing day (`G != 5`
refuses rather than testing on 4), under-sampled null. The battery reports
`n_checks: 19, outcome: PASS, ran_in_the_emitting_process: true`. **Not stated:
which of the 19 are the six falsifiers**, and — given §A.3 — **whether the
planted-arm falsifiers can be driven at all before a day-book with `asm` exists.**

**Resources: declared ≤ 8G ✓** — *"one CPU, MemoryMax=8G, never raised (R-174);
if a day exceeds the cap the day REFUSES rather than the cap rising."* The
refusal direction is declared, which is the right shape.

**But the estimate is large and the design says so honestly:** ~19 min replay and
**~2 hours of null per arm-day**, extrapolated 24× from one hour. Across 2 arms ×
5 days that is **~20 hours of null** plus ~3 hours of replay. **The cap is
protected; the DRAW COUNT is not.** 500 is the declared minimum, so the only
compliant response to a time overrun is to refuse the day — which should be said,
because the tempting response is to cut draws.

## A.6 **Every place a choice can still be made after seeing a day**

1. **The day SET itself** — §C below. Already made, upstream, on a bar the USER
   ruled out.
2. **No minimum decision count per arm per day.** The rule covers **exactly
   zero** (*"if an arm's decision count on a day is 0…"*). It says nothing about
   1, 3 or 10. **Someone will have to decide, after seeing, whether a day with
   four decisions counts.**
3. **No floor on the null's sd.** `Z_d = (D_d − mean(null_d)) / sd(null_d)`. The
   `a_degenerate_null` falsifier covers **sd = 0**; a small-but-nonzero sd
   produces an enormous Z and is not covered. On the consumed hour HAZARD's sd
   was **0.162 against a mean of 0.400** — already wide relative to the effect,
   and thin days will be worse.
4. **The day-1 smoke.** *"the first day is run alone as a smoke with its resource
   observation published before the remaining four."* **Day 1's economic result
   will exist before days 2–5 are run, and nothing declares it withheld.** This is
   the sharpest procedural exposure in the document: whoever runs days 2–5 has
   seen day 1's Z. **Fix: declare that the smoke publishes resource observations
   ONLY, and that its economic fields are sealed until all five are complete.**
5. **No rule that all five days must be run regardless of interim results.**
   `G != 5 refuses` guards the ARTIFACT, not the decision to stop early. With (4),
   early stopping is reachable.
6. **theta non-refit is a claim with pins but no declared run-time check.** The
   model digests make it *detectable*; the design should say they are
   **re-verified at run time and a mismatch refuses**, not merely recorded.
7. **Coin coverage** — undeclared (§A.3).

---

# (B) DA's independence declaration — `da_multiday_recompute_declaration_v1.json`

**Independence holds, and the limit that matters is named in DA's own words.**

* **Reads the EMITTED RECEIPT, never the producer's state.** `what_I_will_NOT_read`
  refuses BE's pickle explicitly, with the right reason: *"reading the pickle
  would make my number a second evaluation of BE's own in-memory objects rather
  than an independent reading of what was published."*
* **No DE/BE module is imported for arithmetic** — R-235 cited: *"I read their
  source to learn the convention … then write the arithmetic myself."* The
  modules are pinned by digest as **READ-only** inputs
  (`de_v2_lifecycle_economics.py`, `de_v2_fee_endpoint_sensitivity.py`,
  `be_cancel_axis_null.py`), and *"a digest that has moved by run time is a
  FINDING, not a nuisance."*
* **DE's reported p, D and rank are refused as inputs** — *"those are the things
  being checked"* — and the per-day p is recomputed **from the per-draw ARRAY**.
* **The identity is the check, not an assumption**: at E0 the maker fee is zero,
  so `D(E0)` **must** equal the gross delta, to 1e-9.
* **Falsifiers run both ways**: a planted disagreeing artifact must refuse, **and
  an agreeing one must NOT** — *"so the check is not simply strict."*
* **Two-copy digest check** (main tree vs worktree) to catch a mid-flight rewrite.
* The module **does not exist yet** — declaration precedes implementation, which
  is the correct order for this round.

**The limit it names is exactly the right one, and it hands the gap to me:**

> *"NOT independence of the null's DESIGN: I recompute the treated arm's location
> within the draws DE GENERATED. If the draw-generating design is wrong, my
> recompute agrees with DE and both are wrong. That is a limit of this check and
> it is the reviewer's to attack, not mine to close."*

**Correct, and this filing is that attack.** §A.3 and §A.6 are the answer: DA's
recompute would agree with DE even if the draw pool were built from one arm's
head for both arms, or from a book with no minimum decision count.

---

# (C) The era question — **the bar was imported, and the USER ruled the opposite, verbatim**

## C.1 R-497(F)(1), verified at the register

`COORDINATION.md:19493`, the USER's own ruling of 2026-09-03:

> **"(1) THE ERA BOOLEAN** — *"We check the data quality and only use qualifiable
> data"*. Applied: **collector version is NOT a bar; QUALITY is the bar.** That
> admits **08-29** (day-quality PASSING, era-pure, post-freeze, **and the cleanest
> day in the record** at btc P1 32.29 s/hr against a bar of 120) and does **not**
> admit 08-30 (day-quality FAILS)."

## C.2 DA's era artifact reaches the same place, computed

`p003_da_era_status_0824_hour__20260906T031650Z.json`:

```
era_predates_clob_v4_1              : true     <- a FACT
era_is_ruled_INADMISSIBLE_by_version: false    <- and NOT a bar
era_that_stamped_the_hour           : clob_v3_1  (ledger AND tape agree)
limits_that_DO_bind_this_hour[0]    : "CONSUMED (rule 11) ... this is the
                                       binding limit, NOT the era"
```

**DA separates the fact from the bar and quotes the ruling verbatim.** The two
sources agree (`ledger_and_tape_agree: true`, single era over 36,566 rows).
**CHECKED — DA is right.**

## C.3 **So yes: R-547(C) imported a bar the USER never set, and it is load-bearing**

DE's design inherits it:

> `why_these`: *"the only era-pure **clob_v4_1** days in existence (R-547(C));
> **08-29/30/31 straddle era boundaries and are inadmissible**"*

**Both halves are wrong as stated.** Version is not a bar (R-497(F)(1)). And
08-30's exclusion is *correct* but for the *wrong reason* — the USER excluded it
because **day-quality FAILS**, not because of an era boundary.

**And this is not cosmetic, because of DE's own arithmetic.** §7 says the smallest
G that clears Holm at m = 2 is **6**, and *"the price of a significant answer is
one day."* **The day the imported bar excludes is the one R-497(F)(1) calls the
cleanest in the record.** So the statement *"no arm can clear Holm on this run"*
is currently **a consequence of an unauthorised bar, not of the data.**

**I do NOT claim 08-29 is admissible.** R-500 withdrew it from the race and R-502
ratified it for **one** development read — whether that read is spent is the open
question, and it is a real reason where "era purity" is not. **The finding is that
the stated reason is refuted and a valid one has not been given.** The day set
must be re-derived on **quality**, as the USER ruled, and whatever G that yields —
5 or 6 — must be fixed **before** any day is run, because at m = 2 the difference
between them is the difference between directional and significance-bearing.

---

# What must exist before BE builds the five reference books

1. **`asm` in the day-book** — the assembled score stream at both arms' pinned
   heads and thetas. **Blocking: without it BE's own null loader raises and there
   is no decision population.** (§A.3)
2. **A declared rule for the draw pool** — one shared scored population, or one
   per arm. The loader currently uses CONDVALUE's head for both. (§A.3)
3. **The coin set** for the five books. `COIN = "btc"` is hardcoded in the null.
4. **A minimum decisions-per-arm-per-day**, and **a floor or explicit handling for
   a small-but-nonzero null sd**. (§A.6.2, §A.6.3)
5. **The day-1 smoke's economic fields declared SEALED** until all five days are
   run, and a rule that all five run regardless of interim results. (§A.6.4–5)
6. **Run-time verification of the theta/model digests**, not merely their record.
   (§A.6.6)
7. **The day set re-derived on quality**, with G fixed before the run. (§C)
8. **A stated response to a time overrun** — the design protects the 8G cap but
   not the 500-draw minimum, and the estimate is ~20 hours of null. Refusing the
   day is the only compliant answer and should be written down. (§A.5)

---

## CONTEXT

Far below the 80% reset threshold; I will report the crossing when it happens.
