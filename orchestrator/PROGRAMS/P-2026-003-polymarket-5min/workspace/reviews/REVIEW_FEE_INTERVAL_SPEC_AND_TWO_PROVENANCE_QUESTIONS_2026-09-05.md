# REVIEW — the ruled fee-endpoint specification (the upper endpoint is mis-shaped and collapses to zero), what `e3a1f088` licenses, and BE's amendment adjudicated

**Filed** 2026-09-05T16:00Z (clock read before composing; the round opened at
15:51Z and two fact additions landed inside it) · reviewer seat (pm-codex) ·
**tip `641cfb6`** (worktree refreshed from `1404302` at round start, then again
for R-536/537/538), clean · no code fixed · nothing run · no sealed day opened · no write
under `data/` · **no other seat's worktree opened** — where an artifact names a
path inside one, I report the name and did not follow it.

**ROUTING — everything below is CHECKED**, computed or read by me at the
artifact. Two items began as DA claims and are now second observations: I
re-derived them independently rather than reading DA's filing across.

---

# ITEM 1 — SPECIFICATION: the fee endpoints for the ruled Gate-1e re-run (R-537)

**Status.** R-537 makes this the declared bar for a ruled run, not a proposal.
R-536 (venue docs) and R-538 (901 chain receipts) landed while it was in flight
and **both change it**. The endpoints below are the third version of my own
derivation; §1.0 says where I moved and why, because a bar that quietly changed
shape is not a bar.

## 1.0 I have moved twice on the direction. Here is the final derivation and what killed each earlier one

**v1** — `1aa9e4b` §4: *"a maker-leg fee can only subtract, so if the gross
result is already unfavourable the fee cannot rescue it."* **True of an arm's
LEVEL, false of the DECISION METRIC, which is a DELTA.** CHECKED at
`de_v2_lifecycle_economics.py:132, 295–310`: the treatment cancels, so it
receives fewer fills, so it pays less charge, so a CHARGE improves its delta.

**v2** — this file's first draft: fee = 0 as the treatment's worst case, a
0.07·p(1−p) charge as its best. **Killed by R-538.** Two ways, and the second is
worse than the first: the charge endpoint is the wrong SHAPE, and 0.07 is also
from the wrong TABLE — it is the **taker** category rate (700 bps), while the
observed non-zero **maker** signed rates are **1000 and 5000 bps**. So it was
not merely mis-shaped, it was **not even conservative**. A bound that is wrong in
kind and short in magnitude is worse than no bound.

**v3, final.** The maker fee is a **signed order parameter**. We sign it. It is
zero. The only live term is the **rebate** — and rebate forfeiture **penalises**
a cancelling arm. **So the envelope is one-sided and it runs AGAINST the
treatment**, which is the opposite of both earlier versions.

## 1.1 The upper endpoint: I take the first horn — it is ZERO BY CONSTRUCTION

R-538(B), verified at the register and consistent with the artifact I read: each
of the ten charged maker legs pays **exactly its own order's signed rate** on
`C·rate·p(1−p)` — seven at 1000 bps (max deviation 8.2e-5), three at 5000 bps
(2.5e-6) — all at p = 0.9900, across 5 transactions and 6 addresses, and **every
maker leg those six addresses appear on is charged, none at zero.**

**The rate is a property of the ORDER, fixed at signing.** Our replay models
orders **we** would place, so the signed rate is a **policy parameter, not an
environmental unknown**. There is no state of the world in which we sign 0 bps
and are charged. "The venue might charge us 0.07·p(1−p)" is therefore a category
error: **the venue does not select our maker rate.**

So the charge endpoint **collapses into the zero endpoint**, and the economic
interval is **[−rebate, 0]**.

### 1.1.1 The residual is real — and it is an IMPLEMENTATION DEFECT, so it gets a GUARD, not an endpoint

Seven of the ten are at **exactly 1000 bps**, which is precisely the CLOB's
advertised `maker_base_fee = taker_base_fee = 1000` that `PM_SKETCH_REVIEW_ITER1_M.md:42`
called *"a legacy signature cap, not the charged rate."* **That is not a
coincidence — it is the number a client signs if it copies the market's
advertised base fee into the order's fee field instead of writing 0.** The three
at 5000 bps show the space is not bounded by 1000 either.

**This is a signing bug with precedent on this chain, not market risk.** Pricing
it as an economic endpoint would make the interval a statement about our own
possible defects rather than about the economics, and would not bound it anyway.
**Requirement instead:** any future order-signing path asserts
`signed_feeRateBps == 0` at build time, with **1000 as its known-bad** — because
1000 is the value the metadata offers. Filed as a standing build requirement, not
as an endpoint. **No live-trading code is authorised here; this binds whoever
writes it, whenever that is.**

## 1.2 The rebate: DEFINED, and the per-market share CANCELS

**Answering the attack directly: NO, it does not collapse the way §2.3's
owned-order join does, and the distinction is sharp.**

* **The owned-order join requires an EVENT THAT DID NOT OCCUR** — a venue
  acknowledgement of an order never placed. No function of simulated data yields
  it. **Undefined.**
* **The rebate is a FUNCTION OF FILLS, and the replay simulates fills.** Its
  numerator, `fee_equivalent = C·feeRate·p(1−p)` on filled maker orders, is the
  **same expression** as the charge endpoint everyone agreed was computable.
  **If that was computable, this is — they are one formula.**
* **The decisive test:** reject the rebate as counterfactual and you must reject
  the **gross maker P&L**, computed on the same simulated fills. That would
  discard the entire Gate-1e ledger. **The rebate is exactly as admissible as the
  number Gate 1e already computed — no more, no less.**

**And the share does not need to be known, because it cancels.** From R-536(C)
verbatim: `rebate = (your_fee_equivalent / total_fee_equivalent) × rebate_pool`,
`rebate_pool` = 20% (crypto) of the taker fees collected **in that market**, and
`fee_equivalent = C × feeRate × p × (1−p)`.

Every matched fill carries one taker leg and one maker leg at the same `C` and
`p`, and the taker fee uses the same expression. Summing over the market,
`total_fee_equivalent = P`, the pool's own base. Therefore

```
rebate_us = (our_fe / P) × 0.20 × P = 0.20 × our_fe
```

**The per-market share cancels.** It is also **displacement-invariant**: if our
simulated fills displace another maker's, `our_fe` rises and theirs falls by the
same amount while `P` is unchanged — so `0.20 × our_fe` still holds. The one
quantity R-536(F) called *"the one a counterfactual replay cannot know"* **does
not have to be known.**

**Three caveats, each with its DIRECTION stated, because direction is what
decides whether they matter:**

1. **Quantisation TIGHTENS it.** Taker fees are floored to 10 µUSDC (R-538(C)),
   so `P ≤ Σ fe` and `rebate ≤ 0.20 × our_fe`. Safe.
2. **The 22 over-charged taker legs LOOSEN it.** 22 of 901 paid more than the
   formula, ratios 1.03–20.05, mechanism unknown — so `P` may exceed `Σ fe` and
   the rebate may exceed `0.20 × our_fe`. **This widens the interval DOWNWARD,
   further against the treatment.** It therefore cannot turn a negative delta
   positive — only a positive one negative. **Conditional requirement: if the run
   finds `D` at the zero endpoint POSITIVE, the rebate bound must be made
   airtight before any verdict is read from it.**
3. **The $1 daily minimum can only shrink it toward zero**, i.e. toward the
   decision-bearing endpoint. It is a **daily, cross-market** threshold, so a
   one-window replay **cannot settle it** — report as a counted status, never
   resolve it.

**Not a fourth endpoint:** Liquidity Rewards. Separate programme, formula
undisclosed, and eligibility for crypto 5-minute markets **UNSTATED** (R-536(D)).
Carried as a status; assumed in neither direction.

## 1.3 The two endpoints, exactly

Let `fe_arm = Σ over that arm's received fills of 7 · p · (1−p) · shares` cents
(`feeRate = 0.07`, crypto). Anchor: at `p = 0.5, shares = 1` this is **1.75 ¢**.

| endpoint | maker fee supplied | meaning |
|---|---|---|
| **E0** | `0.0` on every fill | the venue's default, our signed rate, **and the estimand V2 declares** |
| **E−R** | `−0.20 × fe_fill` on every fill | the rebate at its bound |

`Δfe := fe_B − fe_T ≥ 0` (the baseline receives more fills). Then

```
D(E0)  = gross_delta
D(E−R) = gross_delta − 0.20 · Δfe          ⇒   D ∈ [gross_delta − 0.20·Δfe,  gross_delta]
```

**One-sided, downward, against the treatment.**

**Reuse, not rework:** the quantity DA already computed and labelled
`endpoint_worst_case.maker_fee_cents` **is numerically `fe_arm`** — the retired
charge endpoint's arithmetic is exactly the rebate's base, scaled by 0.20 and
sign-flipped. Nothing computed is wasted.

## 1.4 Population — CHECKED, and smaller than the programme's other numbers

`p003_v2_gate1_economics_smoke__20260905T052605Z.json`: **202 arms** = 1 baseline
(QR_SKEW_ONLY) + 1 treatment + **200** recorded Gate-1d control phases; **one
consumed BTC five-minute window** (`interval.population` verbatim); **5,869**
source rows → **3,557** canonical actions; cluster unit UTC day, **n = 1**,
`NONE_G0_COMPLETE_UTC_DAYS`, **G = 0**; data **consumed** (rule 11). **This is not
BE's 4,315-fill hour and the two must never be pooled.**

## 1.5 The bar — DECLARED NOW, and honest about what is already run

Report at **both** endpoints, select nothing: `D(E0)`, `D(E−R)`; the treatment's
location among its own 200 controls at each endpoint,
`p = (1 + #{controls ≥ treatment}) / 201`, one-sided; `fe_T`, `fe_B` and every
control's `fe`; and `materiality = 0.20·Δfe / |gross_delta|`.

**INVARIANT** ≡ `sign(D(E0)) == sign(D(E−R))` **and** `|p(E−R) − p(E0)| ≤ 0.05`
**and** both p on the same side of 0.5. **MATERIAL** ≡ `materiality > 0.10`.

**These are thresholds on a DESCRIPTION, not a test.** G = 0, n = 1, consumed
data, and `decision_metric.matched_null` is **hardcoded `None`**
(`de_v2_lifecycle_economics.py:381`) — a complete ledger yields a **point, not a
comparison**. The receipt must say so in its own fields.

**⚠ WHAT IS ALREADY RUN, AND I SAY SO RATHER THAN PRE-REGISTERING IT.** DA drove
both LEVEL endpoints on the real Gate-1e fill identities at 15:53Z
(`p003_da_fee_interval_seam__20260905T155346Z.json`, `0ac9de9`), and **I have
read it.** So this bar is **post-hoc with respect to DA's two-arm LEVEL bracket.**
What remains genuinely unrun and is declared here before it exists: **the DELTA
on all 202 arms, the control p-locations, and the rebate endpoint.** Rule 11
applies to reviewers.

### 1.5.1 A reviewer catch on DA's own framing: the straddle is on a quantity that does not decide

DA reports *"baseline QR_SKEW_ONLY: [−3074.3, +288.4] — THE BRACKET STRADDLES
ZERO."* **That is a LEVEL. Gate 1e's decision metric is a DELTA**, and the two
move in opposite directions: both levels fall as a charge rises, while the delta
*rises*. Sizing it from DA's own per-arm numbers, which I read at the artifact —
baseline `n_fills` 458, `fe` 3362.7278 ¢, gross **+288.4178** ¢; treatment
`n_fills` 313, `fe` 2318.2082 ¢, gross **−3927.4644** ¢:

```
gross_delta = −4215.8822 ¢        Δfe = 1044.5196 ¢        0.20·Δfe = 208.9039 ¢
D ∈ [−4424.7861, −4215.8822] ¢    width/|gross_delta| = 4.955%
```

**Sign-invariant across the whole interval, and below the 10% materiality bar.**

**This is NOT the result and must not be cited as one.** It is arithmetic on
**two** arms; the decision needs all **202** and the 200-control p-locations,
which nobody has computed. I state it only to show the bar is well-posed and to
size the endpoints. **DE must compute it, not inherit it.**

## 1.6 Predicates the run must COMPUTE, not print (rule 10)

* `fe_T ≤ fe_B` and `n_received_fills_T ≤ n_received_fills_B` — **per arm. Do not
  assume the direction I argued in §1.3.**
* At **E0**, `fee_adjusted_strategy_net == gross_after_queue_reset` **exactly**,
  all 202 arms. If that identity fails the ledger is not being applied where it is
  claimed to be and **the run is void**.
* Every supplied fee finite; at E−R every value ≤ 0.
* The endpoint-E−R ledger recomputed independently from `(p, shares)` by the
  verifying seat and compared bit-for-bit.
* **`D(E0)` reported with its sign explicitly**, because §1.2 caveat 2 makes a
  positive value a blocking condition rather than a result.

## 1.7 Falsifiers, both directions (rule 15)

* **Positive control:** an arm avoiding exactly `N` fills at `p = 0.5, shares = s`
  must move the delta by **exactly `−0.35·N·s` cents** (20% of 1.75), hand-computed.
* **Known-bad, keep it:** an unknown `fill_id` refuses (`:494`, and DA re-verified
  the guard unweakened).
* **Known-bad:** a non-finite fee refuses.
* **Known-bad, ready-made:** a ledger built with `0.07·min(p, 1−p)` instead of
  `0.07·p·(1−p)` must be caught by the hand-computed identity. **That wrong form
  is the refuted Q5 reading, 2× too large** (`STATUS.yml:3795–3797`) — it exists,
  it is plausible, and it is the error this run could actually make.
* **Known-bad, new and specific to E−R:** a rebate supplied with the **wrong
  sign** must be caught by the E0 identity.

## 1.8 What this does NOT license — and the trap that must be closed in code

1. **It cannot clear Gate 1.** The three sampler refusals stand and none is about
   fees: iid **1 of 200** matched draws in 4,000 proposals; exact-fiber **ESS
   10.53 < 100**; sequential quota **16 of 1,000** with **16** distinct states
   against a declared minimum of 50.
2. **⚠ THE TRAP, CHECKED AT THE CODE.** `de_v2_lifecycle_economics.py:333` is
   `gate1_green = every_gross_identity and every_fee_complete`, and Gate 1e
   reported every gross identity green across all 202 arms. **So supplying any
   complete ledger flips `gate1_exit.cleared` to `true`, `status` to
   `GATE1_LIFECYCLE_ECONOMICS_COMPLETE` and `decision_metric.status` to
   `AVAILABLE` — and EMPTIES `reasons_not_cleared`, deleting the
   owned-order-causality caveat** appended unconditionally at `:352`.
   **Therefore this run must NOT emit a `gate1_exit` block at all.** It is a
   side-car with its own protocol and its own status —
   `FEE_ENDPOINT_SENSITIVITY_NOT_A_GATE_RESULT` — carrying the three sampler
   refusals by name and the causality limitation as explicit fields. **A wrapper
   that lets the module's own `gate1_exit` reach a receipt is a wrong build,
   however correct its arithmetic.**
3. **Not a validation.** Consumed data, G = 0, n = 1. It cannot enter the race.
4. **It does not settle ack/fill causality**, and it does not make the two
   earliest V2 receipts citable (Item 2).

## 1.9 Which endpoint is DECISION-BEARING, and the answer to "stronger or weaker?"

**E0 is decision-bearing.** Three reasons, in order of force:

1. **It is the term we control.** The maker rate is signed by us and we sign zero
   (§1.1). This is a stronger footing than a venue promise — R-538's phrasing, and
   it is right.
2. **V2's own scope says so.** `FEE_COMPONENT =
   VENUE_MAKER_FEE_EXCLUDING_REBATES_REWARDS` — **the rebate is excluded from
   Gate 1e's estimand by the plan.** E−R is robustness, not the verdict.
3. **And at E0, `D = gross_delta` EXACTLY.** So Gate 1e's decision metric, at the
   venue's own schedule and our own signature, **is identically the gross number
   the receipt was already holding and refused to interpret.**

**Stronger or weaker than the two endpoints I had?** **The question changed,
because one of the three was retired as mis-shaped, and I withdraw the "superset"
claim my draft was going to make.** `[E−R, E0]` and the old `[L, H]` are **not
nested and not comparable**: the old interval's upper half priced an exposure a
maker who signs 0 does not have. **Width is not strength when the axis is wrong.**

What is true: **invariance across `[E−R, E0]` is stronger than E0 alone**, because
it survives the one live term at its bound — and it is a claim about **the world
we are actually in**, which the old upper endpoint was not. That is the honest
comparison, and it is the smaller of the two claims I could have made.

## 1.10 Seat, cap, and a constraint that still binds

**DE builds and runs** (`de_v2_lifecycle_economics.py` is DE's). **DA verifies**
— on recorded standby per R-538(F) — by independent re-run and by recomputing
the E−R ledger in its own code. **I specify and do not run.** DA's seam already
establishes the mechanics: **no module edit is needed, and the caller must build
one dict PER ARM after that arm's fills are known — 202 dicts, not one.**

**Cap:** one CPU, `MemoryMax=3G`, swap disabled, **ten-minute ceiling**. Gate 1e
itself was 22.98 s / 338,556 KiB. Caps are never raised (R-174).

**⚠ CONSTRAINT.** The economics receipt pins `de_phase4_diag_runner.py` at
`5097de1b…`, the blob at **`9b37088`**; the file has since drifted (`21961ab5…`
at `751bbe6`), and the Gate-1d drift guard **refuses** — DA hit this
independently. **Run at `9b37088`, or DE re-pins Gate 1d prospectively and says so
in the receipt. The guard must not be defeated, widened or bypassed** — it is one
of the few instruments here that has fired correctly on a real drift.

## 1.11 A citation of mine that R-538 corrects

My `1aa9e4b` §2 table cited the two decoded receipts at
`PM_SKETCH_REVIEW_ITER1_M.md:20–42`. **R-538(D): both transactions are NOT IN THE
CACHE — that decisive test is not reproducible from this repository as-of
2026-09-05T15:53Z.** The finding is unchanged but its evidence is re-based: it now
rests on the **901-receipt decode (1,046 of 1,056 maker legs at exactly zero)**,
which is reproducible. ITER1_M's *"exact to 6 decimals"* also does not generalise
— it holds for its two worked examples and for **110 of 901** at that tolerance;
the calibrated form is `floor(C·0.07·p(1−p) / 10 µUSDC) × 10 µUSDC`, exact on
**879 of 901**. **Cite the 901, not the two.**

*(Also noted: R-538(E) independently quantifies my truncation finding — 1,800 of
13,048 bytes, **86% of the block outside the check**. My own measure was 1,800 of
13,046 to the first `])`. Same defect, same size, two instruments.)*

---

# ITEM 2 — pricing `e3a1f088`

## 2.1 What I verified myself, by execution

I hashed **every version of `de_phase4_diag_runner.py` in all of git history**
(`git log --all`), 25 commits, 25 distinct content hashes.
**`e3a1f088…` is not among them.** DA's claim is confirmed as a second
observation, not read across. The neighbouring versions are `508e2ce3…` at
`f7ba45f` (09-04 14:05:41Z) and `5097de1b…` at `9b37088` (09-05 11:30:23Z);
`e3a1f088` is a working-tree state that existed between them and was never
committed.

Both early receipts name it, CHECKED in their own `source_identity.file_sha256`:
`p003_v2_gate0_smoke__20260904T160623Z` (8 files pinned, 20 dirty entries) and
`p003_v2_gate1_switch_smoke__20260904T163438Z` (11 files, 24 entries). Both
carry `freeze_status: NOT_FROZEN_UNCOMMITTED_V2_WORK`, `working_tree_clean:
False`, `git_head: 8fe1201`. **They told the truth about being unfrozen. What
they cannot do is say what the missing file contained.**

## 2.2 The direct question: is a bit-identical reproduction at DIFFERENT code stronger or weaker?

**It is a different claim, stronger on the numbers and strictly weaker on
scope — and it does not restore the receipts.**

**Stronger, and genuinely so.** The original bytes would have supported exactly
one proposition: *these numbers came from this code.* A single-implementation
chain answers *what produced it* and says nothing about *whether it is right*.
The reproduction supports a proposition the original could never have supported:
*these numbers arise from two different versions of the producing code.* A
quantity that survives a code change does not depend on the changed code. On the
narrow question "are the numbers an artifact of an unrecoverable build?", the
answer is **no, demonstrably** — which is more than intact provenance would have
given.

**Weaker, in three ways that must travel with any citation.**

1. **Agreement is not independence.** `e3a1f088` and `5097de1b` are two states
   of one file by one author in one lineage. A common-mode error survives both.
   This is *reproduction*, never *replication*.
2. **It cannot tell you what changed.** If `e3a1f088` carried a defect that
   `5097de1b` fixed, and the defect did not touch the compared fields, the
   observation is identical. "The numbers match" ≠ "the code matched" ≠ "the
   code was correct at 16:06."
3. **It covers the emitted fields, not the choice of what to emit.** A recursive
   whole-document diff verifies what the receipt says; it cannot verify what the
   receipt omitted.

**And the asymmetry that decides the question: reproduction transfers the
NUMBERS forward; it cannot transfer PROVENANCE backward.** Under rule 12 those
two receipts are permanently non-freeze-grade. Nothing found later fixes that.

## 2.3 Citable form, precisely

**CITABLE — cite the reproduction, never the receipt.** *"5,869 source rows →
3,557 canonical actions, 458 valued tranches, 200 draws, 6/6 identities —
reproduced bit-for-bit at the landed code `9b37088` (DA, 2026-09-05T15:13Z);
the original producing bytes are unrecoverable."* The as-of and the
unrecoverability are **part of the citation**, not a footnote.

**NOT CITABLE.**

* The two receipts as **provenance** for anything — not as evidence of the
  pipeline's state at 16:06/16:34Z, not as a freeze, not as a record of what ran.
* Any field in them that was **not** re-run and compared.
* **Gate 0 as a result of any kind.** Independent of `e3a1f088`, its own status
  strings say so: `PIPELINE_SMOKE_COMPLETE_NOT_AN_ECONOMIC_RESULT` and, inside,
  `STATIC_SCREEN_COMPLETE_NOT_GATE_CLEARED` — and its own
  `declared_before_run.window_limit` is **1**, `n_slugs` is **1**, with
  `treated_at_or_above_random_p95: false`. **It did not clear its own screen.**

## 2.4 The symmetry worth recording

* **gate0 / switch:** reproduction **good**, provenance **gone**.
* **gate1e economics:** provenance **good** (`5097de1b` is at `9b37088`),
  reproduction **blocked** — by its own drift guard, working correctly.

Neither pair is freeze-grade, for opposite reasons. Both facts belong in the
same sentence whenever the V2 line is described, and the second one is a live
constraint on Item 1 (§1.8).

---

# ITEM 3 — BE's amendment: LEGITIMATE on the tie treatment, NOT legitimate on the second amendment, and the isolation is only PARTIAL

## 3.1 The v1 declaration was not substantively edited — CHECKED by semantic diff

I parsed v1 as declared (`a165d19`) and v1 now, and diffed the **leaf values**,
not the text:

* **0 keys removed. 0 values changed in place. 9 keys added** — all of them the
  `SUPERSEDED_BY` block and `THIS_FILES_OWN_COMMIT` (a rebase-pointer repair).
* The file grew 10,600 → 12,394 B; **the growth is entirely annotation**, and
  the 183-line raw diff is mostly JSON re-indentation.

v2's carry-forward claim also checks out **exactly**: `n_draws` 10000, `seed`
**20260905** (the same seed is in v1's own `N_DRAWS_AND_ITS_HEADROOM`),
`min_draws` 200, family `m = 27`, and all five `k` (216 / 432 / 647 / 1440 /
107) identical.

**And the family stayed at 27 after five cells were withdrawn.** That is the
conservative choice and the correct one — multiplicity you have *looked at* is
not returned when you withdraw a cell. BE did not shop the denominator.

*(One tension, named not charged: appending `SUPERSEDED_BY` into the superseded
file is an edit to it. Rule 13 forbids sidecar annotations because automated
readers resolve fields — so an in-file pointer is the right call, and BE says
"not edited in substance" inside the file. Verified true. Noted so the precedent
is explicit rather than assumed.)*

## 3.2 The SIZE_LARGEST_FIRST withdrawal is LEGITIMATE — and the strongest evidence is that the rule cut against BE

The defect is real and mechanical: `argsort(kind='stable')` on a key with
**3,098 identical values** returns the reference's iteration order. CHECKED in
the artifact: `SIZE_LARGEST_FIRST` has `tie_decided_share_of_k = 1.0` and
`tie_group_size = 3098` **in all five cells** — 3,098 / 4,315 = **71.80%**.
That is not a ranking; it is a tie-break rule wearing a ranking's name.

Four properties make this a correction rather than a selection, all CHECKED:

1. **One rule, all five orderings.** The tie treatment is not applied to the arm
   whose number was disliked; it is applied uniformly, and its effects vary
   because the data vary (`SPREAD_WIDEST_FIRST` tie share runs 0.009 → 0.697
   across cells).
2. **The falsifiers fire in the REAL data, not only synthetically.**
   `EARLIEST_FIRST` and `LATEST_FIRST` report `tie_share 0.0`, `sd 0.0`,
   `identified True` in every cell — the declared positive control ("a
   strictly-ordered key must report a degenerate distribution") is live.
3. **It suppressed the best naive number in the table.**
   `SPREAD_NARROWEST_FIRST` at `MATCHED_HAZARD` has the only positive naive mean
   anywhere (**+0.0030**) and the rule flags it **not identified** (tie share
   0.907). A rule chosen to flatter does not delete the best number it can see.
4. **It left standing the one that hurts.** `SPREAD_NARROWEST_FIRST` at
   `MATCHED_CONDVALUE` is **identified** (tie share 0.0049) with mean
   **−0.00818**, which **beats CONDVALUE's −0.01582** — and the artifact records
   it: `arms_beaten_by_naive.CONDVALUE_X_SKEW.beaten_by_any_identified_naive =
   true`. The correction did not protect the arm.

**Verdict: a construction defect removed by a uniform rule, not selection
wearing rule 13's clothes.**

**With one permanent scar, which BE does not state and should.** The defect was
knowable **before** the run, from the policy's own construction: the quoter
rests a **constant size**. CHECKED in the Gate-0 receipt's own action
population — `resting` is **5.0 on 3,557 of 3,557 actions**. A "rank by size"
ordering over a constant-size quoter is degenerate a priori, and one histogram
of the sort key would have shown it. **The discovery channel was the result.**
So v2's naive numbers are **post-hoc diagnostics, not pre-registered results**:
citable as *"these orderings are not identified"*, never as *"this naive policy
captures X."*

## 3.3 The SECOND amendment is a different animal, and it IS load-bearing

`INFORMATION_AT_DECISION_TIME_added_after_the_run` classifies
`SPREAD_WIDEST_FIRST`, `SPREAD_NARROWEST_FIRST` and `SIZE_LARGEST_FIRST` as
`decision_time_FALSE`. **BE labels it honestly** — *"this was NOT declared in
advance. It is added because the results made it decisive."*

**But look at what it does.** In the artifact, for CONDVALUE:

```
beaten_by_any_identified_naive              = TRUE   (by SPREAD_NARROWEST_FIRST)
beaten_by_a_DECISION_TIME_identified_naive  = FALSE
```

**The entire difference between those two predicates is the post-hoc
classification** — and the second one is the arm-protecting form. So v2's claim
that *"the correction is confined to the naive arm"* is **FALSE AS STATED**: the
naive arm's redefinition propagates into a predicate about the **ARM**.

I am **not** disputing the physics — a cancel is decided before the fill, and
realised spread and size are genuinely post-decision quantities. The argument is
sound. **I am disputing its status.** A correct filter applied after seeing which
arms it exonerates is still applied after seeing. It belongs in the **next**
declaration, before the next run — not as a lens on this one. Until then,
`beaten_by_any_identified_naive = TRUE` is the citable predicate for CONDVALUE
and the decision-time form is not.

## 3.4 Is the random arm's isolation enough? PARTIALLY — and there is a bigger limit than the amendment

**YES, for the matched-null comparison.** CHECKED and untouched by v2:

| arm | cell | capture | null mean | excess | above p95 | p-location |
|---|---|---|---|---|---|---|
| CONDVALUE_X_SKEW | MATCHED_CONDVALUE (k=1440) | −1.5819% | −4.7462% | **+3.164 pp** | **false** | 0.1146 |
| HAZARD_OVER_SKEWED_REF | MATCHED_HAZARD (k=107) | −0.0208% | −0.3459% | **+0.325 pp** | **false** | 0.3581 |

The random null's grid, seed, `n_draws` and both arm locations are byte-identical
between v1 and v2 — I verified the declaration fields myself. **The amendment
cannot have touched these numbers.**

**NO, for the naive comparison** (§3.3). The isolation holds on one axis and
fails on the other.

**And a limit that outranks the whole amendment question.**
`random_capture_is_negative_at_every_cell = true`, values −0.0035 to −0.0475,
**because the book's mean fill P&L is positive (+1.9928 ¢)** — so a decline-only
overlay drawing at random destroys value in expectation. The artifact computes
this and says so itself: *"why_that_is_arithmetic_not_a_finding."*

**Beating that null means only that the arm declines fills worse than average.**
It is close to the least a ranker could do. Against it, both arms clear the mean
and **neither clears p95**, and the artifact's own predicate
`nothing_tested_captures_1pct_of_the_ceiling` is **true** — the best identified
capture anywhere is `LATEST_FIRST` at **0.55%**. **So the honest reading of the
whole table is: the +3.16 pp is real, is measured against a null that is negative
by construction, and does not reach the programme's own bar.** That reading does
not depend on the amendment at all, which is why the amendment question, though
worth answering, is not what decides this result.

## 3.5 One provenance note, filed without opening anything

The result's book is `de_section81_cache_12.pkl` and its reproduction gate reads
`de_section81_arms__20260904T140543Z.json`, **both at paths inside
`~/ctaNew-wt-be`** — another seat's worktree. I did not open them; I report the
paths as the artifact records them. BE mitigates correctly: a **10-field
reproduction gate, status PASS**, binding the cache to the filed artifact. That
is the right mitigation. It does not change the fact that the input of record is
an **uncommitted pickle in a seat worktree** — rule 12's scratch-builder hazard,
the one that voided a freeze before. **`be_ceiling_null_v1.json` is not
freeze-grade for that reason alone**, independent of everything in §3.1–§3.4.

---

## CONTEXT

Well under the 80% reset threshold; I will report the crossing when it happens.
