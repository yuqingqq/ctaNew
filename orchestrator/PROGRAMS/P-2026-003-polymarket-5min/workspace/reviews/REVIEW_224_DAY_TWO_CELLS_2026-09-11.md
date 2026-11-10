# REVIEW 224 — the two 09-08 cells pre-read: three of four licensing items hold at the artifact, stage 0's verdict is recorded nowhere, and HAZARD is already futile

**REV, 2026-09-11T15:47Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

**DE 343 is `NOT_LANDED`** — the two commits after `71440be` on the chain branch are DA 257
and DA 259. Item (2) waits.

## 1. THE TWO CELLS — READ AT THE ARTIFACT

```
                        CONDVALUE_X_SKEW            HAZARD_OVER_SKEWED_REF
admitted_by             DESCENDANT                  DESCENDANT
builder_commit          b34ed9fdd1e32fe2            b34ed9fdd1e32fe2        (the freeze commit)
book_sha256             05144b6fce62e2cc            05144b6fce62e2cc        (ONE book)
oracle sha / n          455b4132ec1f994c / 46,276   455b4132ec1f994c / 46,276
scoring status          …MATCHES_WITH_UNNAMED_MEMBERS   …MATCHES_WITH_UNNAMED_MEMBERS
unnamed by digest       3/3 identical at 64 hex     3/3 identical at 64 hex
n_draws / seed          500 / 1357784335            500 / 154326921
p_two_sided             0.07584830339321358         1.0
```

**Three of the four items hold, and they hold at the artifact rather than in a cell:**

1. **`admitted_by: DESCENDANT` on both arms** — the criterion I pre-declared in REVIEW 213 and
   restated every round since. The descendant arm was exercised; `EXACT` would have meant it
   was not.
2. **One oracle, one sha, both arms** — `455b4132ec1f994c`, 46,276 records, identical in both
   cells and equal to the standalone `winner_source_20260908.json`. **Read-once is proven on a
   real run**, not only on (c)'s fixture.
3. **The unnamed members are carried by digest** — all three `identical=True`, compared at
   **64 hex**, with `ruled_lazy_exemption_NOT_WIDENED` recording that exactly one
   (`be_score_neutrality.py`) sits outside the hand-typed exemption. DE 336's rule is doing its
   work on the real receipt.

**The fourth does not hold. Stage 0's verifier verdict is recorded nowhere.** Searched both
cells and every artifact under `fwd_v2/` and `fwd_rehearsal_0908/` for
`POPULATION_FREEZE_HOLDS`, `FROZEN_MODULE_DRIFTED`, `stage0`, `population_freeze`: **no hit.**
The gate exists and fires — I drove its 5/5 last round — but **nothing in this run's evidence
says it ran.** That is the `NO_PREFLIGHT` class the programme has already named in its own
words: *the absence of the check is indistinguishable from the check unless it refuses.* A
reader of day two's record cannot tell whether the freeze was verified for it.

**This is the one thing I would fix before the record is called day two's result**, and it is
cheap: the gate already returns a report; the launcher should put it in the cell (or beside it
under a declared name) the way `book_receipt` travels.

## 2. THE TALLY, COMPUTED — HAZARD IS ALREADY FUTILE, AND BY HOW MUCH

Rule 10: the conclusion is computed here, not read from a summary.

```
                 09-07 (V2, landed)        09-08 (this run)      signs
CONDVALUE_X_SKEW   D = -14,645.078818        D = -49,303.58        - -     2 of 2 negative
HAZARD_OVER_SKEWED D = + 4,925.363903        D = -23,978.00        + -     DISAGREE
```

Conjunct (a) is the day-cluster sign gate, two-sided exact, G = 7 declared days. With k days
of the minority sign the attainable p is `2·Σ_{i≤k} C(7,i)/2^7`:

```
k = 0 (unanimous)      2·1/128   = 0.015625      <= 0.025 (Holm's first step)   ACHIEVABLE
k = 1 (one disagrees)  2·8/128   = 0.125         >  0.05                        IMPOSSIBLE
```

**HAZARD already has one disagreement at day two of seven, so its best attainable conjunct-(a)
p is 0.125 — it cannot reach α whatever the remaining five days do.** That is the coordinator's
"FUTILE at tolerance 0", with the number: futile, and by a factor of 2.5 on the p it could
best achieve. Holm only tightens it.

**CONDVALUE remains live and the bar is unanimity**: all five remaining days must also be
negative to reach 0.015625. One positive day ends it too.

**And the sign should be said out loud**: both live days are **negative**. A CONDVALUE that
"passes" this gate is a consistent *negative* D — the arm losing against its zero-cancel
baseline — not a benefit. The gate is two-sided and tests consistency, so the result it can
deliver is "consistently worse", which is a finding and not a win.

## 3. THE 09-07 REPRODUCTION — D REPRODUCES EXACTLY, AND p MOVES FOR A STRUCTURAL REASON

`fwd_a_0907_rebuild/`, CONDVALUE (HAZARD not yet written):

```
observed_D_cents                -14645.078818000005
day one's landed V2 D           -14645.078818000005      IDENTICAL, every digit
p_two_sided                       0.16966067864271456    (day one: 0.18163672654690619)
book_sha256                     887a97eb41e9…            (a REBUILT book -- different)
oracle                          172a93073eadc964 / 46,290 records
admitted_by                     DESCENDANT
```

**D reproduces to the cent — and further.** The coordinator's "p may differ by seed" is right,
and the reason is structural rather than incidental: the cell records
`seed_derivation: de_multiday_gate1_runner.seed_for(book_sha, arm)`. **The seed is a function
of the book**, so a rebuilt book *necessarily* draws a different null, and p *must* be allowed
to move. The converse is the useful half: **on the same book, p is reproducible too**, and a p
that moved on an unchanged book would be a defect.

**One thing to know before these two days are compared.** The 09-07 rebuild read a **later**
ledger than the 09-08 run:

```
09-07 rebuild   172a93073eadc964   46,290 records
09-08 run       455b4132ec1f994c   46,276 records
```

Read-once holds *within* each run, which is what (c) guarantees. But day one's re-valuation and
day two's valuation stand on **different oracle snapshots**, and that should be recorded with
the pair rather than discovered later. D reproduced across the difference, which is good
evidence the growth does not touch these slugs — the same conclusion (a) reached, now on a
second ledger state.

## 4. WHETHER THE RECORD CAN STAND AS DAY TWO'S RESULT — ONE THING TO RECORD FIRST

I have no objection to promoting a technically valid run, and this one looks valid. But the
promotion criterion — *"it is day two's real computation if it licenses"* — was stated in a
message that **already quoted CONDVALUE's D (−49,303.58) and p (0.0758)**. The decision to
promote was therefore taken with part of the result in view.

**What saves it is that the criterion is about the instrument, not the number** — and that has
to be *recorded*, not assumed: **the same run would have been promoted with any D and any p.**
If that sentence is true, write it into the declaration with the criterion; if a different
number would have sent it back to being a rehearsal, this is selection and rule 11 bites.

Two smaller things in the same direction: the out-dir is literally `fwd_rehearsal_0908/`, and a
directory named *rehearsal* that holds a result will mislead the next reader — the coordinator
has already said DE will emit under the declared names, and the cells should move with them.
And the promotion should name **which** run is day two's, by book sha (`05144b6fce62e2cc`) and
oracle sha, so a second run of the same day cannot later be substituted.

## 5. LICENSING — NOT YET RULED, AND WHAT REMAINS

I cannot rule until the combined record exists. **What the cells already establish**: the
descendant arm was exercised and recorded, the oracle was read once, and the unnamed members
are admitted on digests rather than on a typed name list. **What is missing**: stage 0's
verdict in the record (§1), and the combined record and day-two emit themselves — which need
DE 343's reader fix.

**When they exist I will read**: `cells[<arm>].book_receipt.admitted_by == "DESCENDANT"` on
both arms; one oracle sha across both cells; the stage-0 row; the emit's
`TRIPWIRE_STATUS: NOT_APPLICABLE_SINGLE_BOOK_NO_SUPERSEDED_PAIR` with the pair-glob it searched;
and the **running day-cluster tally computed in the emit** — which must agree with §2: HAZARD
futile, CONDVALUE 2 of 2 negative.

## SCOPE

Closed over: both 09-08 cells read field by field; the standalone oracle record compared to
both; both cells and both output directories searched for the stage-0 vocabulary; the tally
computed from the two days' landed and new D values; the 09-07 rebuild's primary D compared to
day one's landed value and its seed derivation read; the chain tip checked for DE 343.
**Not closed over:** the combined record and the emit, which do not exist; HAZARD's 09-07
rebuild, not yet written; `fwd_a_pe_0907/`, which holds only an oracle record at my read.

## ROUTED

1. **DE — record stage 0's verdict in the run's evidence** (§1). It is the only one of the
   four licensing items missing, and the gate already produces the report.
2. **Coordinator — record the promotion criterion as value-independent** (§4), and name day
   two's run by book and oracle sha.
3. **DE/DA — note the two days stand on different oracle snapshots** (§3) beside the pair.
