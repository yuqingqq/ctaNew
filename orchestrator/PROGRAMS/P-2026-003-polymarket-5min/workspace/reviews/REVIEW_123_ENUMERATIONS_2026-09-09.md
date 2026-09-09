# REVIEW 123 — the meta-class: which enumerations are samples?

**REV, 2026-09-09T08:46Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**TWO SILENT FINDINGS, AND BOTH ARE IN FIXES TO MY OWN FINDINGS.**

1. **`ECONOMIC_FIELDS` does not name the ruled PRIMARY endpoint.** A sealed artifact may
   carry `D_E_settle`, both arm totals and both legs, and nothing flags it. **The seal
   protects the diagnostic and not the result.**
2. **`SCORING_PATH_MODULES` is a hand-typed 5 of a RECORDED 49**, and is not even a superset
   of the 10-module cascade — so the closure predicate I asked for in REVIEW 111 checks a
   sample of the closure the book already records.

**And one candidate on the list is NOT of this class, and is the model for fixing the
others: the heavy-run lock decides heaviness by MEASUREMENT, not by enumeration.**

---

## THE TEST I APPLIED TO EACH

For each set: **what operation would constitute membership; was the set built that way; and
would a missed member be LOUD or SILENT.** Only the silent ones matter — rule 28 pointed at
the enumerations rather than at the checks.

## 1. `ECONOMIC_FIELDS` — SILENT, and live

**Membership would be:** *a quantity that is sealed until the read.* **Built:** by inspection
when R-599 was written, extended twice by ruling (R-599's `sd_over_abs_mean`, R-659's three
counts). **Never revisited when R-801 made the settlement P&L the primary endpoint.**

Driven, in a SEALED artifact at 1 of 6 days:

```
economic_settlement: {D_E_settle 30045.04, arm_total_cents 88884.09,
                      zero_cancel_baseline_total_cents 81238.30,
                      arm_legs {trades_leg_cents …, residual_leg_cents …}}
  -> assert_no_economic_leak: NO REFUSAL

the same artifact plus ONE named field (n_cancels_issued: 137)
  -> REFUSED at ['per_day_sealed_artifacts[0].n_cancels_issued'] with 1 of 6
```

**The guard fires on what it names, and it does not name the ruled result.** The reasons
guard behaves the same way: a reason quoting `D_E_settle` is not refused, while the same
reason quoting `Z` is. *(My first probe of this looked like a refusal — because my fixture
also carried `Z`. I removed it and re-drove; recording that, because the wrong field
explaining a refusal is how a hole reads as a guard.)*

**Fix by the operation, not by adding a name:** the sealed set should be *the fields the
receipt publishes under `economic` and `economic_settlement`* — derived, so a new endpoint
joins the seal by existing rather than by being remembered.

## 2. `SCORING_PATH_MODULES` — SILENT, and it is the fix for REVIEW 111's site 3

**Membership would be:** *a module whose bytes can change the values in `asm["by_arm"]`.*
**Built:** five typed names. Measured:

```
SCORING_PATH_MODULES (typed) : 5
be_cascade.modules (declared): 10   -- six of them NOT in the predicate, including
                                       phase4_generation_tables.py and de_matched_random_control.py
the book's own import closure: 49   -- of which the predicate checks 5
```

`phase2_arms.py` is in the predicate and not in the cascade; six cascade modules are in the
cascade and not in the predicate. **A change to `phase4_generation_tables.py` — the
generation tables the assembly is built from — moves the scores and
`assert_book_scoring_code` passes.** The closure the book records is the set that would
constitute membership, and it is already in the artifact.

## 3. `fit_manifest.json`'s twelve `fit_code_files` — SILENT, same class, not driven

**Membership would be:** *a file whose bytes can change a fit artifact.* **Built:** by
inspection — BE 113 found the surface by accident. A missed member means the manifest
verifies while the code that produced the fit moved: **silent, and it is the model-identity
equivalent of (2).** I did not drive it — enumerating the fitting closure needs the fit
pipeline, which is heavy — so I name it as the same class rather than assert a defect.

## 4. `_GEN_REQUIRED` — LOUD. Discarded, with the reason

**Membership would be:** *a field the replay reads.* **Built:** typed
(`gen, t0, t1, level, displayed, status, tranches`). **But a missed member is LOUD**:
`validate_reference` raises on absence — *I tripped it myself this session*
(`ReferenceIntegrityError: s1/BUY_UP: generation missing 't1'`) — and a field read but not
required fails at the read with a `KeyError`. **A missing member cannot be silently
absorbed, so it is not of this class.**

## 5. The exclusion-status vocabulary — SILENT BY CONSTRUCTION, and the next sweep worth doing

**Membership would be:** *every reason a row or a window can be dropped.* **Built:** by
inspection, accreting a name per defect found. **A missed member is a drop with no status —
which is precisely what rule 4 exists to prevent, so this set failing IS the rule failing.**
The constructive test exists: every filtering branch in a drop path (`continue`, a
comprehension guard, a `if not …: return`) should map to a named status, and the sweep is a
walk of those branches against the status vocabulary. **That is a real round's work and I did
not do it here** — but it is the candidate I would take next, because unlike (3) it is
sweepable without heavy data.

## 6. The heavy-run lock — ALREADY BUILT BY THE OPERATION, and the model for the rest

`assert_rule20` computes `heavy = (wall_s > HEAVY_WALL_S or peak_rss_mb/1024 > HEAVY_RSS_GB)`
and refuses a run that **was heavy by measurement** without the lock. **There is no
enumeration of heavy operations to be incomplete.** This is the shape the other five should
take: **replace the list with the property**, so membership is decided by what a thing DOES
rather than by whether someone remembered it.

## SCOPE

Closed over: the six sets the round named, each read at the tip and — for (1) and (2) —
driven with controls in both directions. **Not closed over:** every other typed set in the
package. The general question "which other enumerations are samples" is answered here for six
cases and **the method is the transferable part**: for any set, ask what operation
constitutes membership, and if the answer is available as data (a closure, a receipt's own
fields, a measured property), the set should be derived from it rather than typed beside it.

## ROUTED

1. **DE — `ECONOMIC_FIELDS`.** Derive the sealed set from the receipt's own economic blocks;
   today the seal does not cover the ruled primary endpoint. **This is the one I would fix
   first**, because a sealed artifact carrying `D_E_settle` is a pre-read leak of the result.
2. **DE — `SCORING_PATH_MODULES`.** Check the closure the book records, or at minimum the ten
   cascade modules; five typed names is a sample of forty-nine recorded ones.
3. **BE — `fit_code_files`**, same question asked of the fitting closure.
4. **Whoever takes the next untargeted round — the exclusion-status vocabulary (§5)**, which
   is sweepable without heavy data and where a miss is silent by construction.
5. **`_GEN_REQUIRED` needs nothing** (loud), and **rule 20 is the model** (measured).
