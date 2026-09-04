# REVIEW — the threshold, and the tail that has already crossed it

**Filed** 2026-09-04T13:26Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `e4a0eb6`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· read-only against
`de_section81_arms__20260904T125340Z.json` and the code at this tip · no sealed
forward day opened, no write under `data/`, no other seat's worktree opened.

**ROUTING.** Everything computed here is **CHECKED** — arithmetic on the
artifact's own published totals. Three inputs are **AGREED** and marked where
used: DA's fill-level tail decomposition, its per-generation dispersion, and its
random-cancel null (all R-520, none re-derived by me because none is
recomputable from the published aggregates).

---

## 0. The answer in three lines

**Break-even is r = adverse/spread ≥ 27.60 %. This book reads 18.63 %.**
**But 18.63 % is a ratio of two totals of which one is 43 fills, and excluding
the top 1 % the ratio is ≥ 110.6 % — a hard bound, true for any split of the
tail.** So the specification is not waiting on a different venue: **on 99 % of
this book the threshold is already exceeded by a factor of four**, and the whole
question becomes whether a ranker can decline the body without declining the
tail.

---

## 1. Your item 1 — the threshold, as a specification

Let `s`, `a` be the book's average spread and adverse per fill, `r = a/s`, and
let the policy's removed fills carry `α·a` of adverse at `σ·s` of spread. Then
declining one such fill is worth `α·a − σ·s = s·(α·r − σ)`, so

> **BREAK-EVEN: r ≥ σ/α.**

Both parameters are measured on this artifact, per **lost fill** (R515-R1's
denominator, not `n_cancels`):

| arm | α (adverse) | σ (spread) | **break-even r** | this book | gap |
|---|---:|---:|---:|---:|---:|
| CONDVALUE_X_SKEW | **3.0140×** | **0.8319×** | **27.60 %** | 18.63 % | needs **1.48×** |
| HAZARD_OVER_SKEWED_REF | **3.8137×** | **0.7583×** | **19.88 %** | 18.63 % | needs **1.07×** |

**HAZARD is 6.7 % of relative adverse away from break-even.** That is a far
smaller gap than "the overlay does not pay" conveys, and it is the sharper of
the two rankers on both axes.

**And "enough to matter", as uplift on baseline maker P&L** —
`f·(α·r − σ)/(1 − r)` at the measured removal fraction `f`:

| uplift on baseline P&L | CONDVALUE (f = 33.4 %) | HAZARD (f = 2.5 %) |
|---|---:|---:|
| break-even | **27.60 %** | **19.88 %** |
| +5 % | 31.03 % | 47.59 % |
| +10 % | **34.15 %** | 61.06 % |
| +25 % | 42.01 % | 78.01 % |
| +50 % | 51.64 % | 87.26 % |

**Read the two columns against each other, because the comparison is the
specification.** CONDVALUE reaches +10 % uplift at r = 34 % because it removes a
third of the book; HAZARD needs r = 61 % for the same uplift because it removes
2.5 %. **Breadth buys uplift, sharpness buys a lower break-even.** A book near
the threshold wants HAZARD's selectivity; a book well past it wants CONDVALUE's
breadth.

**The window, stated as the specification's real shape.** Market-making is worth
doing only while `r < 1` (spread exceeds adverse) and the overlay pays only while
`r > σ/α`. **So the target is `0.28 < r < 1`**, and the overlay's contribution
rises steeply toward the top of it.

**The one assumption, stated rather than buried: α and σ are held constant as r
moves.** They are properties of the ranker *on this book*. A book with more
adverse selection may be one where adverse is easier or harder to concentrate.
**Nothing here establishes that α = 3.01 transports**, and the specification
should be written as *"at α ≈ 3, σ ≈ 0.83"* rather than as a property of the
model.

---

## 2. Your item 2 — is it testable on data this repository already holds? **Yes, on ~600 coin-hours, with no replay and no sealed day opened**

**Yes, and cheaply, for a reason R-520(A) established (AGREED, DA's, verified by
the coordinator): at zero cancels the baseline's fills ARE the reference's kept
tranches, so the whole decomposition is recomputable from the reference alone,
without DE's replay.** DA reproduced maker P&L, spread capture, adverse and
`n_fills` to 7.3e-12 that way.

**`r` is a property of the BASELINE, not of any arm.** So measuring
`adverse/spread` across coins and hours needs no policy, no replay, no arm and no
threshold — one pass over the reference tranches per coin-hour.

**The admissible window, and it does not touch a sealed day:**

* sub-second-reliable Binance data begins **2026-08-24T13:48:54Z** (the
  `hf_ws_v2` stamp boundary). **The measured hour, 13:50–14:50Z on 08-24, starts
  66 seconds after it** — it is the first admissible hour, which is why it is the
  one that exists;
* the freeze epoch is **2026-08-28T06:09:00Z**;
* between them lie **≈ 88 hours × 7 coins ≈ 600 coin-hours** of development data
  that is already consumed as development evidence and cannot be consumed again;
* **09-01 / 09-02 are consumed economic reads and 09-03 is accrued but unopened
  — none of them is needed and none should be touched.**

**So the answer is yes, without qualification about sealed days.** The measured
sample is **1 coin-hour of ~600 available**. Whether any of them reaches 27.6 %
is unknown and is exactly what should be measured next — and the result is
decision-bearing either way: a distribution of `r` that never approaches the
threshold closes the overlay programme on evidence rather than on one hour.

**One caution on the estimator, not the availability.** `r` is a ratio of sums,
so per-coin-hour it inherits the same tail sensitivity §3 describes. **Report
the tail-excluded `r` beside the raw one for every cell**, or the survey will
reproduce the very artefact it is being run to escape.

---

## 3. Your item 3 — the counterweight, and it is the largest number in this filing

**No, 18.63 % is not trustworthy as a description of the book, and the corrected
figure is not a small adjustment.**

Using DA's fill-level decomposition (**AGREED**, R-520(C): top 1 % = 43 fills
carry **113 %** of net; the other 99 % sum to **−13 %**) with the artifact's
three totals:

* tail net = 1.13 × 8,598.76 = **+9,716.60 c**; the other 4,272 fills sum to
  **−1,117.84 c**;
* let `x` be the tail's share of **spread**. Then
  `r_ex = (1,968.19 + 9,716.60 − x) / (10,566.95 − x)`;
* the numerator at `x = 0` is **11,684.79 c** against a denominator of
  **10,566.95 c**, and `d/dx > 0` because the numerator exceeds the denominator.
  **So the minimum is at `x = 0`:**

> **r_ex ≥ 110.58 %, for ANY split of the tail's P&L between spread and
> adverse.** No assumption about the tail is required.

It also follows trivially and independently: **the other 99 % of fills sum to
−1,117.84 c, and a negative maker P&L means adverse exceeded spread — r > 1 by
definition.**

**Which version should a specification be written against? Neither alone, and
the reason is the whole finding.**

* **18.63 % understates the opportunity by 6×.** It is a ratio dominated by 43
  fills and it describes the tail, not the book.
* **110.6 % overstates what is capturable**, because *"exclude the top 1 %"* is
  not an available policy. A cancellation overlay does not get to keep the tail
  and decline the body; it declines what its score selects, tail included.

**So the specification is conditional, and the condition is measurable:**

> **The overlay pays on this book if and only if it declines the body without
> declining the tail.** On the body alone `r ≥ 110.6 %` against a break-even of
> 27.60 %, a factor of four in hand. On the whole book `r = 18.63 %` and it
> loses. **The entire result sits in 43 fills.**

**And that makes one unmeasured quantity decisive.** CONDVALUE removed **1,440
fills — 33.4 % of the book**. If it removed even a proportional share of the
tail (≈ 14 of the 43, worth ≈ 3,200 c) the delta would be far more negative than
−953.92 c, so **it evidently did not remove many**. But *"evidently"* is an
inference from an aggregate. **The measurement is one pass: how many of the top
43 fills did each arm decline, and what was their net?** If the answer is zero,
the ranker already does the hard part and the case is much stronger than
anything filed so far. If it is more than a handful, the 3.01× concentration is
partly the ranker walking into the tail.

---

## 4. What reconciles my mechanism with DA's null — the CASCADE, and it separates the two arms (CHECKED)

R-520(D) reports neither arm distinguishable from random cancellation (z = −0.20,
+0.26). R515-R2 reports the ranker selecting correctly on both axes. **Both are
true, and the quantity that reconciles them is unreported.**

The book runs **4,315 fills over 3,861 fill-bearing generations = 1.1176 fills
per generation** (AGREED: the generation count is DA's). A *random* cancel
therefore removes ≈ 1.12 fills. **These cancels removed 4.32 and 2.23.**

> **`cents_per_cancel` = (cascade multiplier) × (per-fill cost ratio) × (mean
> generation P&L)**

| arm | cascade vs the book rate | per-lost-fill cost vs the average fill | product | per cancel | vs a random cancel |
|---|---:|---:|---:|---:|---|
| CONDVALUE | **3.869×** | **0.3324×** | 1.2862 | **2.8645 c** (reported 2.8646) | **1.29× WORSE** |
| HAZARD | **1.995×** | **0.0589×** | 0.1175 | **0.2616 c** (reported 0.2616) | **8.51× BETTER** |

Both reconcile to four decimals against the artifact's own `cents_per_cancel`.

**The reading, and it is new:** CONDVALUE's ranking edge is real (per lost fill
it costs a third of what a random fill costs) **and it is spent almost exactly
over again by the cascade its own cancels cause** — 3.01× of selection against
3.87× of cascade, netting 1.29× worse per decision. **HAZARD does not cascade
(1.99× against a per-fill cost of 0.059×) and is 8.5× better than random per
cancel.** DA's null reports both as indistinguishable, correctly — **but HAZARD's
point estimate is 8.5× better on 48 cancels, and the z-statistic hides that the
two arms' point estimates are a factor of seven apart.**

**So the actionable lever is not a better ranker. It is a cancel that does not
cascade.** Fills-lost-per-cancel is the quantity that separates these two arms,
it is absent from `cancellation_economics`, and it is `n_fills` differencing —
one subtraction that is already in the file.

---

## 5. Your item 4 — dispositions, so they stop being carried

**Re-emission provenance census: RUN. CLOSE the item; the FINDING is open.**
Filed last round as **R514-R7** (`7d439b6`) and re-verified again at this tip —
no new arms artifact has landed, so
`de_section81_arms__20260904T125340Z.json` is still the only one and the census
result stands: `carrying_commit` **`b43a9ce` is not on `mm-research`**, **3 of 7
identity files differ at the commit the artifact names** (including
`de_section81_arms.py`, its own producer), all 7 match at the tip, and
`working_tree_dirty` is **`None`**. **Nothing further is gained by running it a
third time.** What is owed is the fix — stamp the commit *after* the producing
edits land, or record `working_tree_dirty`/`dirty_paths` beside the ref, as
iter011's `producing_code.why` already requires.

**Arms diff: CLOSE as UNRECOVERABLE — and here is the substitute that is not.**
`arms53.py` and `arms53.json` are absent from disk and were **never committed**;
there is no second run to diff. **Carrying it open implies a check that can
still be performed, and none can.**

**But the question the diff existed to answer is still live and is now
answerable another way.** The diff was to establish *"does the committed runner
reproduce the scratch runner's numbers?"* The scratch runner's decisive numbers
survive as committed source comments — `de_section81_arms.py:62` and `:592`
carry the **333-vs-496** figures (the line numbers moved at `0e8f40c`, which added 133 lines; re-checked
at this tip) that R-506(A) rests on. **So the substitute
check is: run the committed runner and ask whether it reproduces 333 realised
treated actions against 496 control.** That is a reproduction, it is executable
today, and it delivers exactly what the diff was for. **Close the diff; open
that.**
