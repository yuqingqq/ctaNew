# REVIEW — design v3: APPROVED for the runner to be built against, with four items that must close before any real day runs — and the shell trap bit me again mid-round

**Filed** 2026-09-06T04:14Z (clock read before composing) · reviewer seat
(pm-codex) · tip `cacc060` · no code fixed · no write under `data/` · nothing
sealed opened · **every `data/` read in this filing re-verified at the absolute
ledger path** for the reason in §D.

**ROUTING — CHECKED**, every claim driven or recomputed by me.

## VERDICT

**v3 is APPROVED for DE 72 to build the runner against, on fixtures.** All eight
must-exist items are present as fields, R4 and the book-digest check are real
two-sided code paths, R7's derivation returns exactly the six days I get from the
ledger, the imported era bar is withdrawn verbatim, and my Set-B refinement is
handled **structurally and better than I asked**.

**Four items must close before any real day runs. Three are on the runner; one is
on the design.**

1. **R2's 0.90 is the only bar in the document with no calibration** — and unlike
   R4's 0.25, it is not checked against the one population already seen. **This is
   the design-level item.**
2. **R5's seal is a requirement, not yet a code path.**
3. **R6 is half a code path** — the book digest refuses; the model/theta digests
   have no verifier.
4. **The root derivation does not refuse.** Driven: a non-ledger root returns an
   **empty day set, silently.**

---

# (A) The eight items

## R1 — asm: CLOSED, and specified beyond what I asked

`what_asm_must_contain_per_day` names `by_arm` keyed by `(coin, head)` **for both
pinned heads**, `[0]` as the scored `(slug, side, t0)` keys, coverage of every
generation the reference carries (*"a generation absent from `scored` is a
scorer-coverage fact and not a silent drop"*), and **scored at the pinned thetas**.
DE cites the loader lines I cited. **And the digest question is answered better
than I framed it:** the seed derives from the whole book's digest which *contains*
`asm`, **and** BE must publish `sha256(asm)` separately *"so a reader can tell a
score-stream change from a reference change."* The one-day discharge is required
before the other four books are built. ✓

## R2 — the shared pool: the CHOICE is right, the THRESHOLD is uncalibrated

**The choice is correct and the reasoning is the one I would have given**: the null
is per arm because the *draw* is matched to that arm; the *pool* is a property of
the book, and a per-arm pool *"would make each arm's random cancel land in a
different universe and the two nulls incomparable — the same category error as a
per-arm shared denominator, in the other direction."* The cost is stated, not
hidden, and BE must publish `|scored(head)|` per head per day plus the overlap.

**FINDING — 0.90 is declared but not calibrated, and R4 shows what calibration
looks like.** R4 states *"HAZARD's null had sd 0.16234 against mean 0.40033, a
ratio of 0.4055 — ABOVE the 0.25 floor, so the bar as declared would have admitted
the consumed hour. That is stated so the floor cannot be read as chosen to exclude
something already seen."* **R2 has no equivalent sentence.** I searched the whole
document for `overlap`: two hits, both the rule and the publish-requirement, **no
number.**

**And the number is computable today** — the 08-24 cache carries `asm["by_arm"]`
for both heads, which is exactly what R1 requires per day. **Until the consumed
hour's overlap is published, 0.90 has no basis: if it is below 0.90 the bar is
inconsistent with the precedent run; if just above, the threshold looks chosen to
clear it.** One number closes this, and it should be in the design, not the runner.

## R3 — btc-only: HOLDS, and I verified all four digests

The reasoning is that the pinned thetas *are* btc thetas and an eth arm would be a
different frozen object. **Checked at the artifacts:**

```
lgbm_thresholds_btc.json  0fa2f1f7a5a4c58f     linear_d_btc.json  18701008c2bd18c6
lgbm_thresholds_eth.json  ce67009e38a07e8c     linear_d_eth.json  fb371f6352214a92
```

**All four match R3's claims exactly.** The eth siblings exist and are **different
artifacts**, and `theta_pins` pins only the btc pair — so running eth would require
a new pin. **"A different frozen object" holds.** The multiplicity consequence is
right too: m 2→4 ⇒ threshold 0.0125 ⇒ smallest clearing G rises 6→7 (2⁻⁷ = 0.0078
≤ 0.0125; 2⁻⁶ = 0.0156 > 0.0125). And the scope limit is stated: a btc run *"says
nothing about eth and must not be reported as a venue-level result."* ✓

## R4 — the degeneracy bars: a REAL code path, and I drove it

`arm_day_admissible()` at `de_multiday_design_declaration.py:228–247`. Driven:

| case | admissible | status |
|---|---|---|
| n=106, HAZARD-like sd (0.379) | **True** | OK |
| n=4, wide sd | **False** | `decisions 4 < declared minimum 30` |
| n=106, tight sd (0.052) | **False** | `null sd 0.0205 < 0.25·\|mean 0.398\| = 0.0995` |
| n=4, tight sd | **False** | **both reasons accumulate** |
| **n=30 exactly** | **True** | boundary admits |
| **n=29** | **False** | boundary refuses |

**Two-sided, boundary-correct, reasons accumulated rather than short-circuited.**
And a refused arm-day is a **status** that does not silently shrink G. ✓

## R5 — the seal: correct as a requirement, NOT YET a code path

The content is exactly right — day 1 publishes resource observations, population
counts and refusal statuses; it does **not** publish `D(E0)`, `D(E−R)`, `Z`, any
per-day location or any null summary; `all_G_days_run_regardless_of_interim_results:
true`; and *"the sealed fields are absent from the artifact, not present-and-ignored."*

**But the per-day emitter does not exist** — DE 72 is building the runner. So the
artifact-level refusal is **a promise the runner must keep.** Until it is a code
path and driven, early stopping remains reachable in practice. **Condition on the
runner, not a defect in the design.**

## R6 — half a code path

* **Book digest: REAL.** `verify_book_digest()` at `:249` raises `DesignRefused`
  on mismatch — *"the whole day, not the offending draw."* ✓
* **Theta / model digests: NO VERIFIER.** The pins sit at `:99` and `:107`; nothing
  consumes them. The field says *"verified at run time not merely recorded"* and
  *"a recorded digest that nobody compares is provenance theatre"* — **which is
  precisely the state it is currently in for the model half.** Condition on the
  runner.

## R7 — the day set: CLOSED, and my Set-B refinement is handled better than I asked

* **`n_verdict_files_read: 12`** and `qualifying_on_quality` =
  `[08-29, 09-01, 09-02, 09-03, 09-04, 09-05]` — **identical to my own independent
  re-run at the ledger.**
* **`version_is_NOT_a_bar`** quotes R-497(F)(1) verbatim and says *"R-547(C)'s 'only
  era-pure clob_v4_1 days' imported a bar the USER never set, and v1 of this
  declaration inherited it. WITHDRAWN HERE."* ✓
* **`withdrawn_sentence`** records the phrase, who withdrew it, what survives, and
  that v1/v2 carried it. ✓
* **`day_read_state`** gives each day a `previously_opened_for` field with its
  authority, and **`set_b_rule`: a day with no field REFUSES rather than defaulting
  into it.** ✓
* **Both sets computed with their Holm arithmetic**: Set A G=6, best p 0.015625,
  clears at m=2; Set B G=3, clears at neither. The parameter is put to the USER
  with the consequence stated.

**My Set-B refinement is in v3 — structurally, and it is a better fix than the one
I proposed.** I said day-quality must be evaluated only on complete days. v3 makes
`day_closed_calendar` one of the four **conjuncts**, so 09-06 (`day_closed_calendar:
false`, `post_freeze_pass: false`) is excluded **before quality is ever consulted** —
the open day never reaches the quality test. And `accrual_schedule` states when it
will: *"closes 00:00Z 09-07 and is verdicted at 00:06Z 09-07 … set A reaches G = 7
on 09-07."* **No v4 needed for R7.**

*(One note on R-550(D): its purist set was named as 09-03/04/05 + 09-06/07/08. v3's
Set B is 09-03/04/05 only, G=3, derived from the field. v3 is right and the register
entry was ahead of the ledger.)*

## R8 — computed

`estimate_source: "resources.totals — COMPUTED from the measured seconds, not
typed"`, `agrees_with_resources_totals: true`, and the numbers are the corrected
ones: **G5 null 9.6967 h, total 11.2633 h; G6 null 11.636 h, total 13.516 h** —
matching my own arithmetic from last round exactly. `never_by: [lowering the draw
count, raising the cap, sampling fewer days]`, and the 500-draw minimum is
protected **by refusing the arm-day**. ✓

## The battery and the chain

`expected_checks_in_the_source: 42`, `n_checks_run: 42`,
`run_count_equals_source_expected: true` — **the v1 receipt-count defect closed
exactly as recommended: both numbers with a computed equality**, and the v1 defect
named. The supersession chain v1 → v2 → v3 **resolves by digest at both hops**
(verified at the ledger; see §D for why my first attempt said otherwise).

---

# (B) The tightened provenance rule — driven, and one half still fails

| path | class | required |
|---|---|---|
| `population_block.source_cache` (round-69 defect) | **`string`** | not swallowed ✓ |
| `provenance.as_of` | **`provenance`** | ✓ |
| `resource_observation.as_of` | **`provenance`** | ✓ |
| `arms.X.as_of` | `string-embedded-provenance` | — |
| **top-level `as_of`** | **`provenance`** | **✗** |

**The round-69 defect is NOT swallowed** — it lands in `string`, which is a
reported class. *(The taxonomy has no "substantive string" class, so `string` is
the correct outcome; the requirement is that it not be classified provenance, and
it is not.)*

**But my A-1b finding is NOT closed, and I re-ran the decisive test:** the actual
BE `cb9bf8a` diff through the **current** classifier still returns

```
counts: {'provenance': 1, ...all others 0}     moved: {'as_of': 'provenance'}
```

— **identical to last round.** The one substantive in-place edit in that commit
would still be reported as moving nothing but provenance. The tightening moved
`as_of` **under a result parent** from `provenance` to
`string-embedded-provenance`, but a **top-level** `as_of` — which is exactly the
demonstrated case — is unchanged. Still open; still cheap (require a provenance
parent, or give timestamps their own reported class).

---

# (C) `worktree_data_shell_trap` — records the root, does NOT refuse

The field is honest and the diagnosis is right: *"a path that resolves in both
trees but means different things in each. A digest cannot catch it, a green suite
cannot catch it, and a count that nobody compares to a hand check cannot either."*
It records `root_read_this_emission` and resolves `<root>/data/data` when present.

**Driven — and it does not refuse:**

```
day_sets_from_the_ledger(<empty temp root>)
  -> RETURNED normally.  n_verdict_files_read = 0.  qualifying_on_quality = []
day_sets_from_the_ledger(/home/yuqing/ctaNew)
  -> n_verdict_files_read = 12, qualifying = the six days
```

**A run pointed at the wrong root reports "no day qualifies" rather than an error.**
That is *worse* than the 3-day set that prompted the field, because zero looks like
a finding. **The field's own lesson applies to itself: nothing compares
`n_verdict_files_read` to an expected minimum.** One line — refuse when it is zero,
or when the resolved root carries no ledger marker.

---

# (D) The trap bit me again, in the round I am filing about it

My first chain check reported **both supersession digests mismatched**. That was
false. `git checkout --detach origin/mm-research` at the start of this round
**replaced my `data` symlink with a materialised directory again**, because the
three newly-tracked design artifacts were not in the `skip-worktree` set, so git
recreated `data/` to write v3 into. My worktree then held **only v3**, and the two
predecessors read as absent.

At the absolute ledger path **both hops MATCH**.

**The operational finding: the `skip-worktree` fix is not durable across a checkout
that introduces newly-tracked `data/` paths — and every DE/BE round adds some.** So
every seat that refreshes will silently get a shell back. **A seat must either
re-run the skip-worktree line after every checkout that touches `data/`, or read
`data/` only at the absolute ledger path.** I am doing the latter from here.

*(Recorded plainly: I caught it only because a digest mismatch on a freshly-emitted
chain was implausible enough to re-check. That is the second false finding this
discipline stopped in two rounds, and both were the same trap.)*

---

# Every remaining place a choice could be made after seeing

1. **R2's 0.90** — declared, uncalibrated, and the calibrating number is computable
   today (§A/R2). **The one design-level item.**
2. **R5's seal** — a promise until the emitter refuses (§A/R5).
3. **R6's model digests** — recorded, unverified (§A/R6).
4. **The root derivation** — silent on a wrong root (§C).
5. **The USER parameter (Set A vs Set B)** — **not a defect**: it is put before the
   run with both sets and both Holm outcomes computed and published.

---

# Verdict, stated for the record

**APPROVED — DE 72 may build the runner against v3, on fixtures.** The design's
substance is sound and three of my four remaining items are properties the runner
must have rather than defects in the declaration.

**NOT YET CLEARED TO RUN A REAL DAY.** Before the first day: publish R2's
consumed-hour overlap (design), and make R5's seal, R6's model-digest check and
the root refusal into driven code paths (runner). **No v4 is required for R7, R8,
R1, R3 or R4** — those are closed.

---

## CONTEXT

Far below the 80% reset threshold. Standing by for BE 46.
