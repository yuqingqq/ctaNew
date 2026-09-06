# REVIEW — three batches: BE 43 stands with one dangling pointer, DA 51's 0/4 measures the HARNESS not DE's census, and the rc deferral CAN swallow a real failure — I drove the path

**Filed** 2026-09-06T02:28Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `e67252d`**, worktree clean · no code fixed · no write under
`data/` · no sealed day opened · **no other seat's worktree opened** — where an
artifact names a path inside one I report the name and did not follow it.

**ROUTING — every finding below is CHECKED**, computed or driven by me in my own
worktree. Two are second observations of another seat's claim; the rest I
derived. **Nothing is AGREED.**

**THREE FINDINGS ROUTED FOR ACTION**, in severity order:

1. **(C-1) The rc deferral CAN swallow a real failure on an open day.** I drove
   the production function directly and got `DEFERRED` on a genuine disk-error
   traceback. **The falsifier's own summary line overstates what its ten cases
   establish.**
2. **(B-2) DA 51's `0/4` is a property of the HARNESS, not of DE's census** —
   structurally guaranteed before any mutant was written. The finding was real;
   the predicate's *name* over-attributes it. **DE has already closed the
   underlying gap at `e67252d`.**
3. **(C-2) One live consumer still pins the superseded seam v1** with no
   supersession pointer.

Everything else in all three batches checks out.

---

# (A) BE 43 — the cancel-axis null: EVERY CLAIM VERIFIED, and the declaration discipline is fixed

## A.1 Was the null declared before the draw? YES — and this time the declaration was not touched

`8b930b4` adds **only** the declaration (113 lines, one file). `4c17646` adds the
result, the module and one register line — **and does not touch the declaration.**

**Byte-identical at both ends:** sha256 of
`be_cancel_axis_null_declaration_v1.json` is `aabefbde2f4bbc3c…` at `8b930b4`
**and** at `HEAD`. `git log --all` on that path returns **one commit**.

**Declared design == what ran**, checked field by field at
`be_cancel_axis_null_v1.json`:

| declared | ran |
|---|---|
| `n_draws_per_cell` **500**, 2 cells | `cells.*.n_draws` = **500**, 2 cells |
| `seed` **20260905** | `cells.*.seed` = **20260905** |
| decisions **1154** / **106** | **1154** / **106** |
| by_side {586, 568} / {41, 65} | identical |
| `declared_family_m` **6** | family unchanged |
| `min_draws_enforced` 200 | 500 > 200 |

**This is the round-42 defect fixed.** Last round I found v1 edited in the same
commit as its result and a hardcoded declaration hash that dangled across a
rebase. The receipt now resolves the declaration **by sha256**, and says why:
*"a hardcoded hash dangles across a rebase; round 42's did."*

## A.2 …and the adjacent field dangles anyway — **FINDING A-1**

`declaration.commit_that_last_touched_it` = **`6eaa538a630c30a6…`**. I checked it:

* the object **exists** (it is the pre-rebase declaration commit),
* it is on **no branch** (`git branch -a --contains` returns nothing),
* and `git merge-base --is-ancestor 6eaa538a 4c17646` → **NOT an ancestor**.
* `8b930b4` **is** an ancestor, and is on the branch.

So the receipt asserts *"the declaration commit PRECEDES the run commit — an
ordering, which no rebase disturbs"* while **naming a commit from which that
ordering cannot be checked.** **BE diagnosed the disease and reproduced it in the
field beside the cure.** The digest is load-bearing and resolves, so nothing is
wrong with the result — but the pointer should be superseded in band to
`8b930b4`. **Low severity, exactly reproducible.**

## A.3 "Same stateful cascade" — same code path, not a re-implementation: CHECKED at the code

`be_cancel_axis_null.py` **calls** rather than copies:

* `:191` `HSP.replay_policy(bk["ref"], scores, params_for(theta))` — **the
  treatment's own function**
* `:190` `HSP.validate_scores(scores)`, called explicitly
* `:173` `R.cell_params`, `:192` `R.received_fills`, `:218` `R.cancel_mechanics`

**Seven delegated call sites; no second copy of the cascade.** And it is proved
behaviourally, not asserted: the `reproduction_gate` is **PASS** on three points
through this harness — baseline **0 cancels / 4315 fills**, CONDVALUE
**333 / 1440**, HAZARD **48 / 107** — every one matching the filed arms. **The
null and the arms are on identical machinery, checked.**

Also correct and worth naming: the module **refuses to borrow a real head's
manifest** to get a random stream through `de_score_stream.score_events`, calling
`HSP.validate_scores` directly instead. Borrowing one would have been a false
provenance claim, and the declaration says so before the run.

## A.4 Does the seed pin the DATA, not just the RNG? (SEAT_PROTOCOL rule 10) — **IN SUBSTANCE YES, IN FORM NO**

Rule 10 verbatim (`SEAT_PROTOCOL.md:55–57`): *"the seed must pin the data the RNG
is applied to, not just the RNG."*

* **What is pinned cryptographically:** the declaration (sha256), the seed, the
  decision counts and side splits.
* **What is NOT:** the book. `population.source` is
  `/home/yuqing/ctaNew-wt-be/…/de_section81_cache_12.pkl` — **a path inside BE's
  worktree, a pickle, with no digest.**
* **What binds it anyway:** the three-point reproduction gate. If that pickle
  were a different book, the baseline and both arms could not return
  0/4315, 333/1440 and 48/107.

**So the data is pinned BEHAVIOURALLY and not by digest.** That is materially
stronger than nothing and materially weaker than a hash — and because the path
lies inside a worktree **no other seat may open**, the behavioural binding is the
*only* check available to anyone else. **A sha256 of the pickle costs one line
and would make rule 10 satisfiable from outside BE.** Recommended, not blocking.
*(Same standing note as round 42: an uncommitted pickle in a seat worktree is
rule 12's scratch-builder hazard, so this artifact is not freeze-grade for that
reason alone.)*

## A.5 The four claims — all verified by my own arithmetic

| claim | verified |
|---|---|
| random replays cascade at **0.497** fills/cancel, n=500, sd **0.072**, range **[0.310, 0.818]** | mean **0.496975**, sd **0.072333**, min **0.309829**, max **0.818182**, n **500** ✓ |
| CONDVALUE **4.324 = 8.70×** | 4.324324 / 0.496975 = **8.7013×**, `inside_the_null_range: false` ✓ |
| HAZARD **2.229 = 5.57×** | 2.229167 / 0.400327 = **5.5684×**, outside ✓ |
| `the_cascade_is_machinery_not_selection` = **false** | ✓ — **the cascade is SELECTION** |

**BE's own withdrawal is sound, and the reason is dispersion.** CONDVALUE's
2.8646 ¢/cancel sits in a null of mean **2.1075**, sd **3.8084**, spanning
**[−13.14, +15.29]**, p **0.6228**; HAZARD p **0.4112**; both
`COST_IS_INSIDE_THE_NULL_RANGE: true`. **The null is wider than the effect by a
factor of several, so "1.29× worse than a blind cancel" was never distinguishable
from noise.** Withdrawing it is right.

## A.6 BE's claim against DE's module — CONFIRMED, and it is an exact identity

BE says DE's `fills_per_generation = 1.1175861` is fills per **FILLING**
generation. **4315 / 3861 = 1.1175861175861175**, and the receipt's
`DEs_ASSUMED_blind_rate` is **1.1175861175861175** — equal to sixteen digits. The
population block carries `baseline_fills: 4315` and
`n_generations_with_fills: 3861` beside `generations: 29813`; per *generation* the
rate would be **0.1447**.

And it is **outside both drawn ranges** — CONDVALUE `[0.3098, 0.8182]`, HAZARD
`[0.0417, 1.0426]` — with `DEs_assumed_rate_is_inside_every_drawn_range: false`
while `DEs_assumed_cost_is_inside_every_drawn_range: true`. **The rate assumption
is wrong; the cost assumption is not.** DE has since corrected the baseline at
`e67252d`.

---

# (B) DA 51

## B.1 The attainable ceiling: SOUND, and its positive control is the good kind

* **Reproduction gate PASS on 6 fields**, all matching the filed artifact: net
  **8,598.758849499998**, V_oracle **60,303.760723**, n_fills **4,315**,
  n_neg/n_pos/n_zero **2,072 / 2,234 / 9**.
* **701.3077% → 516.1128% of net**, overstatement **26.407%**, growing with
  budget: **10.750%** (k=107) → 12.627% (216) → 15.445% (432) → **23.928%**
  (1,440) → 26.407% (4,315). Monotone across every reported cell.
* **The positive control is a real two-directional falsifier.** With the cascade
  **off**, attainable must collapse onto the oracle — and it does, at **every**
  budget, to `overstatement_pct` of order **1e-14**. An attainable bound that
  could not return to its oracle would be measuring something else.
* **Independent agreement with BE**: `oracle_curve_agrees_with_BE_at_every_cell:
  true`, and `cross_check_against_BE` shows `agree_to_1e_9: true` on all five
  cells. **Two seats' implementations of the oracle curve agree — that is
  replication, not reproduction.**

**DA's killed monotonicity assertion was correctly killed**, and I checked the
direction rather than taking the note: at dwell **0.25 s** the attainable share
at k=1,440 is **0.3389**; at the declared **2.0 s** it is **0.7607**. Attainable
**rises** with dwell — because the budget counts **fills removed**, so a longer
silence sweeps more losers per cancel. A naive "monotone decreasing in dwell"
assertion would have been false, and `limits[2]` explains exactly why.

## B.2 The four mutants: REAL DEFECT CLASSES, none a strawman

| mutant | class | why it is real |
|---|---|---|
| `STATISTIC_returns_zero` | always-PASS | a checker that can never fire — rule 15's exact concern |
| `STATISTIC_drops_abs` | silent | TVD without the absolute value is *the* classic slip; it returns a plausible small number, not zero |
| `NULL_does_not_permute` | null machinery | a permutation null that does not permute |
| `STATUS_always_nothing_excluded` | **interface** | it emits **the exact string DE's artifact cites**, unconditionally |

The fourth is the strongest because it is the only one aimed at the **token
another seat reads** rather than at internals. And DA's own `limits[0]` says four
mutants on three surfaces is a **sample**, never a proof of correctness.

*(Small instrument note: on mutant 4, `da_population_audit` went red with
`named_failures: []` — it caught the mutant without being able to say why.)*

## B.3 **FINDING B-2 — the `0/4` measures the HARNESS, and it was guaranteed before any mutant was written**

I read `de_section81_mid_census.py` **as it was when DA audited it** — the parent
of `e67252d`, `c476d0f` — rather than at HEAD, because DE has since changed it.

At that version the consumer **did** import the oracle (`import
da_population_audit as PA`) but had only **three** `PA.` sites: one call inside a
function (`:194 PA.compare(...)`) and two exception clauses (`:345`, `:392`).
**Its selftest asserts nothing whatever about `PA.compare`'s output** — every
`ok(...)` tests `census`, `tranche_records`, `duration_tail`,
`denominator_check` and refusals.

**A suite that never asserts on the mutated output cannot detect its mutation.**
So `0/4` was **structurally guaranteed** before a single mutant existed. It is
evidence about the consumer's **selftest coverage**, not about the census it
performs on real data.

**DA disclosed this** — `limits[2]`: *"the consumer is run at its own selftest,
which is not the same as the production census it performs on real data."*
**But the predicate names carry the other reading:**
`every_applied_mutant_caught_by_the_consumer: false` and a
`surviving_mutants` list of four read as four holes in DE's census. **The finding
is real — there was no consumer-side falsifier — but what was measured is
narrower than the naming implies.** Supersede the naming, not the result.

**And it was acted on, which is the best evidence it was not a strawman:**
`e67252d` takes `PA.` call sites from **3 → 12** and adds explicit
consumer-side falsifiers in both directions (*"REFUSED: the imported oracle DID
NOT FLAG a maximally…"*, *"…FLAGGED an exactly balanced…"*). **Closed.**

## B.4 The 22 charged taker legs

`p003_da_onchain_fee_audit__20260905T155346Z.json` carries it. DA's own statement
— **formula NOT established** for the 22, mechanism unknown, 19 submitters each
100% non-conforming — is the correct disposition and I do not dispute it. **It
matters to my own bar**: it is exactly the A4 term in
`REVIEW_SPEC_ATTACKS_AND_STRADDLE`'s §1.2 that loosens the rebate bound upward,
and it remains the reason `D(E0) > 0` would have been a blocking condition. That
`D(E0)` came back **negative** is why it did not bind.

---

# (C) DA 52

## C.1 **FINDING C-1 — the deferral CAN swallow a real failure on an open day. I drove it.**

**In every case the falsifier drives, it cannot.** `classify_mask_failure`
requires **both** conjuncts, ANDed: `closed == "0"` **and** the log matching
`CONTENT_LIVENESS_(UNRESOLVED|UNJUDGEABLE)`. Everything else falls to FAILURE.
The battery is **10 checks, both directions**, and it drives the exact question:
`classify_mask_failure 0 realfail` → **FAILURE** (open day + disk error);
`0 empty` → **FAILURE** (*"silence is not a deferral"* — rule 11);
`'?'` → **FAILURE**; a missing log → **FAILURE**. **I ran it: 10/10 PASSED.** The
unit declares `SuccessExitStatus=2` and **not 4** (checked), and `broke` is
raised and never lowered.

Design credit: the falsifier **awk-extracts the function from the production
script** and **refuses if it is not found**, so it tests the real code, not a copy
that can drift.

**But the deferral is keyed on a PROSE MATCH, and exactly one path is not
driven: a log carrying BOTH the liveness token AND a real failure.** I extracted
the production function the same way the falsifier does and drove it with:

```
Traceback (most recent call last):
RuntimeError: while handling CONTENT_LIVENESS_UNJUDGEABLE the writer died
OSError: [Errno 28] No space left on device
```

**→ `DEFERRED`.**

A genuine disk-full failure, on an open day, classified as an expected status —
and because the unit maps rc 2 to success, **`systemctl` would report success and
the instrument failure would be invisible.**

**And the falsifier's own summary line asserts what its cases do not
establish:** *"rc 2 is reachable ONLY for an open day refusing for want of
windows."* My probe reaches rc 2 by another route. **That is a printed conclusion
beside a passing test set — rule 10's shape, in a shell `echo`.**

**Severity: low probability, high invisibility.** It needs a failure whose output
quotes the detector's token — but that is not exotic, since a wrapper handling a
`MaskRefused` and then dying would produce exactly this. **Cheap fix:** require
the liveness token **and** the absence of a traceback marker, or have the mask
builder emit a structured status line and match that instead of prose. Either way
the mixed-log case belongs in the battery.

## C.2 The seam supersession: COMPLETE at DA's end — **FINDING C-2: one live consumer still pins v1**

**Complete, and correctly shaped.** `p003_da_fee_interval_seam_v2__20260906T021955Z.json`
carries `supersedes` with path **and sha256 `a7b562f0ab467316…`** (the digest DE
also pinned, so the chain is checkable from both ends); `arms_whose_bracket_straddles_zero`
is **`[]`**; every arm reads `admissible_reading_is: "A POINT AT E0, NOT AN
INTERVAL"`; v1 is **not edited**. **And the withdrawn endpoint is RETAINED as a
field** — `WITHDRAWN: true`, `authority: FLOW_MODEL_STATE.md:79`, the row
verbatim, and `would_have_given_strategy_net_cents: −3074.31`. **Withdrawing
without erasing is the right shape**, and v1's `sign_invariant_on_every_arm_probed`
is gone rather than left to be misread.

**But a consumer still reads v1.** Grep over `*.py/*.md/*.json/*.yml/*.sh`
returns **five** referencing files. Four are provenance (DA's own producer, the
register, my two dated review filings). **The fifth is live code:**

> `live/pm_research/de_v2_fee_endpoint_sensitivity.py:85` —
> `DA_SEAM_ARTIFACT = ("p003_da_fee_interval_seam__20260905T155346Z.json", …)`

— the **ruled run's module**, whose `citation_cross_check` block reads v1 by name
and digest.

**The USE is legitimate and I want to be exact about that:** DE cites v1 as *the
artifact being corrected*, pinned by hash, to set `claim_holds: false` against my
withdrawn §1.5.1. Citing a superseded artifact by digest is correct. **What is
missing is the pointer:** the ruled receipt tells its reader nothing about v2, so
under rule 13 the chain is discoverable only from v2's end. **No number is wrong;
a reader of the ruled receipt cannot learn that v1's estimand was withdrawn.**

**Fix, and it is already routed:** add a sixth point to DE's pending
supersession — carry `superseded_by:
p003_da_fee_interval_seam_v2__20260906T021955Z.json` beside the v1 citation.

*(Minor, same family: v2's `protocol` string is still
`P003_DA_FEE_INTERVAL_SEAM_V1`. An automated reader resolving by protocol sees
V1 for both files.)*

---

## CONTEXT

Far below the 80% reset threshold; I will report the crossing when it happens.
