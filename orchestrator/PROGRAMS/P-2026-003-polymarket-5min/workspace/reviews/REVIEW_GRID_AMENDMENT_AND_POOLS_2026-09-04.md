# REVIEW — the grid amendment attacked: legitimate, but not for the stated reason, and the pool it adds is worse than the one it fixes

**Filed** 2026-09-04T13:04Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `e4c9b97`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· read-only against the canonical tree's artifacts · no sealed forward day
opened, no write under `data/`, **no file in another seat's worktree opened**
(a name-only `find` enumerated paths; nothing under `-wt-de`/`-wt-da` was read).

**ROUTING.** Every numbered finding is **CHECKED**. Where I revise my own
previous filing I say so in the same line.

**Headline, and it is against the amendment I was asked to attack: the
PREVENTABLE pool cannot supply the budget at two of the three budgets, is a
point mass at those two, and is anti-conservative at the third. It should not
run as declared.**

---

## 1. Your item 1 — is the grid amendment rule 11 in disguise?

### R514-R1 — the amendment is LEGITIMATE and your stated defence is the weak half. Do not put it in the receipt. (CHECKED)

Your defence has two clauses. The second one does not survive.

> *"nothing was drawn, and your analysis used only quantities the artifact
> already published — so no outcome was seen."*

**The published quantities ARE functions of the evaluation data.**
`n_actions_preventable`, `n_actions`, `m_good`, `m_harm`, `p_pos`, `p_neg` are
all statistics of the eval population, computed at
`D.TARGET_LATENCY_MS = 50` — the same latency `val()` reads. My prediction was
a deduction *from the data being tested*. **I did not need to draw to know the
answer, and "no draw" is therefore a claim about procedure, not about
information.** If a reviewer deducing the outcome to near-certainty from
published aggregates does not count as seeing, then "seeing" has been defined
as "sampling", and rule 11 becomes a rule about RNG calls. It is not.

**So concede the clause. The amendment is still legitimate, on two grounds that
do not depend on it:**

1. **THERE IS NOTHING LEFT TO CONSUME.** Rule 11's teeth are *"seen days are
   consumed"*. 2026-08-25 was consumed before this artifact was written:
   `development_evidence.is_a_validation` **false**, both fitting and evaluation
   populations development, `G_complete_utc_days` **0**,
   `intervals_claimable` **false**. The preventable-restricted arm **cannot be
   consumed by this amendment because the population it runs on was already
   spent.** Validation requires later untouched complete UTC days, and no
   amendment made today can change what that day is worth.
2. **DIRECTION AND COMPLETENESS.** Rule 11 exists to stop data from selecting a
   design that flatters a result. The amendment's declared intent was to make
   the null *harder*, both cells are declared, both run, and **neither is
   discarded**. A grid where every cell is reported cannot be selection —
   selection requires discarding cells.

**And the receipt should record something better than a defence: that the
amendment FAILED at its own purpose.** §2 measures that the preventable pool
makes the null easier, not harder. Recording *"we amended toward a stronger
null and measurement showed the amendment weakens it"* is stronger evidence of
good faith than any claim about what was or was not seen.

### R514-R2 — HIGH — the grid changes the MULTIPLICITY, and at the new denominator the artifact's four surviving results DIE (CHECKED)

`family.holm_denominator` is **24**, `declared_family.n_cells` is **24**, and
`holm_denominator_is_declared_not_evaluated` is **true** — the denominator is a
*declaration*, so enlarging it is a declaration change and rule 12 records
multiplicity at freeze time.

**Running both pools produces 12 Q4 cells where 6 were declared.** If both
enter the family, the denominator is 30. Computed at the artifact's own floor
p = 0.001996007984:

| denominator | Holm-adjusted smallest p | the four distinct surviving results |
|---|---:|---|
| **24 (declared)** | 0.047904 | **SURVIVE** (by 4.2 %) |
| **30 (both pools adjudicated)** | **0.059880** | **DIE** |
| 36 | 0.071856 | die |

**So the grid amendment, left as written, would retrospectively kill `Q1` and
`Q3` — the artifact's only surviving results — through the denominator.** That
is a consequence nobody has ruled on and it must be ruled BEFORE the run, not
discovered after.

**The fix is one line and the programme already has the pattern.** Designate
**UNRESTRICTED as the ADJUDICATING cell** (it is the one that preserves
comparability with the five heads already carrying this null) and mark
**PREVENTABLE as REPORTED-NOT-ADJUDICATED** — exactly what `COIN_SLICE` already
does for eth (*"eth is reported and never adjudicated, so an eth slice can never
carry a verdict"*, R-306). Denominator stays 24, the grid stays honest, and
nothing is selected after seeing.

---

## 2. Your item 2 — the preventable pool, and it is worse than the one it replaces

### R514-R3 — HIGH — the PREVENTABLE pool CANNOT SUPPLY THE BUDGET at 10 % and 15 %, and is a POINT MASS there (CHECKED)

There are **17,604** preventable actions. The budgets are **8,883 / 17,767 /
26,651** actions.

| budget | k | k as % of the preventable pool | feasible? |
|---|---:|---:|---|
| 5 % | 8,883 | 50.5 % | yes — but half the pool every draw |
| **10 %** | 17,767 | **100.9 %** | **NO — 163 short** |
| **15 %** | 26,651 | **151.4 %** | **NO — 9,047 short** |

At 10 % and 15 % the null **cannot draw k distinct preventable actions because
there are not k of them.** Two consequences, both fatal as declared:

1. **The count match breaks silently.** `pick = pool if cnt >= len(pool) else
   rng.sample(pool, cnt)` (`harmful_action_eval.py:180`) hands back the whole
   pool when it is short, so the null cancels **fewer** actions than the
   candidate — rule 7's first half, violated by the arm built to satisfy rule 7.
   BE's new refusal covers a stratum with **no** eligible generation; it does
   **not** cover a stratum with **too few**. Those are different failures and
   only one is guarded.
2. **The null becomes a POINT MASS.** Consuming ~100 % of the pool every draw
   leaves no variance: the "null distribution" is one number, p = 1/(n+1) with
   certainty, and n = 2,000 buys nothing. **This is the `movable == 0`
   degeneracy** `matched_random_null` guards by name
   (`phase2_iter011.py:1080-1082`) — *"a null that CANNOT MOVE is not a null"* —
   reappearing in the pool that was added to make the test meaningful. It is
   also exactly the saturation case I pre-specified at R513-R6 last round;
   I flagged it as unlikely under the unrestricted pool, and the amendment made
   it certain.

### R514-R4 — HIGH — and even at 5 % the restriction makes the null WEAKER, which inverts the amendment's purpose (CHECKED)

The preventable population's mean value is **negative**: −1.6364 c per
preventable row, **−3.1254 c per preventable action** (total −55,019 c over
17,604 actions). So forcing the null to spend its entire budget inside it
guarantees a large loss:

| budget | UNRESTRICTED null mean | **PREVENTABLE null mean** (row-mean … action-mean) | observed |
|---|---:|---:|---:|
| 5 % | −1,440 c | **−14,536 c … −27,763 c** | +7,869.68 c |
| 10 % | −2,881 c | −29,074 c … −55,529 c *(infeasible; point mass)* | +12,333.50 c |
| 15 % | −4,321 c | −43,612 c … −83,294 c *(infeasible; point mass)* | +14,476.99 c |

**At 5 % the gap grows from 9,310 c to 22,406–35,633 c — the "stronger" null is
2.4× to 3.8× easier to beat.**

**Why the intuition failed, and this is the transferable part.**
Opportunity-matching sharpens a test only when the restricted population has
**positive** expected value — when *being in it* is easy and *picking within it*
is the hard part. Here being in it is bad on average: **a uniform draw inside
the preventable set is a worse policy than a uniform draw over everything**,
because 90.1 % of "everything" is harmless zeros and zero beats −1.64. The
candidate's skill is in picking the **positive-value subset within**
preventables; a uniform draw within preventables measures nothing about that.

### R514-R5 — the rule-9 tautology is REAL but is the SECOND problem, not the first (CHECKED)

Yes: preventability is derived from the outcome (`any_fill_ahead` plus the
latency-50 bucket), so a pool restricted to it hands the null knowledge no
policy has at decision time. Rule 9's *"a baseline must remove the tautology,
not manufacture one"* bites — **but only when the manufactured baseline makes
the candidate's job artificially HARD**. Here it makes it artificially **easy**
(R514-R4), so the tautology is not what disqualifies this pool. **Both facts
should be in the receipt; only one of them is a reason to stop.**

### The third construction, since you asked for one

**Match the null's realised preventable COUNT to the candidate's.**
Let `h_b` = the number of preventable actions the candidate actually cancelled
at budget b. Within each `(side, hour)` stratum, draw the stratum's share of
`h_b` from the **preventable** pool and the remaining `k_b − h_b` from the
**non-preventable** pool.

* **Feasible at every budget** — `h_b ≤ 17,604` by construction and
  `k_b − h_b` is drawn from 160,070 actions.
* **Not degenerate** — the preventable quota is far below the pool at every
  budget, so the null moves.
* **It is the only one of the three that is HARDER than unrestricted**, because
  it removes the "found preventables" component from the measured effect and
  leaves exactly the sign discrimination that the +12,334 c must come from
  (cancelling *every* preventable row is worth −55,019 c, so the value cannot
  come from finding them).
* **Rule 9, honestly:** it conditions on `h_b`, which is outcome-derived. That
  is legitimate as a **conditional decomposition** — *"given the opportunity it
  got, did it choose well within it?"* — and it is **not** the policy's total
  value. **It must be reported as a decomposition and never as the gate's
  null.** The gate's null is the one that already exists: the incumbent
  increment, computed, and failing at Holm 0.11994–0.44628.

---

## 3. Your item 3 — my own prediction, attacked by me, before the number

**Restating what is predicted** (pinned at R-516(A)): unrestricted null means
≈ −1,440 / −2,881 / −4,321 c; p at the floor at all three budgets.

**Five ways it could be wrong. Four cannot flip it; one can.**

1. **A wrong 9.908 %.** *Checked and it holds.* `n_actions_preventable` and
   `n_actions` are both `I11.action_count` at
   `D.TARGET_LATENCY_MS = 50`, and `signed_v_cancel`/`preventable` read the same
   L=50 bucket. Same latency, same unit. **The denominator is right.**
   (Separate finding: **the iter011 artifact records no latency field
   anywhere** — I searched every key. Rule 7 says latency enters the estimand;
   the estimand's latency parameter is not in the receipt.)
2. **Per-stratum concentration, which I did not model — and it is the honest
   gap.** The null draws inside the candidate's own `(side, hour)` mix, so its
   hit rate is the *conditional* preventable rate in those strata, not the
   global 9.908 %. If the candidate concentrates where preventables are dense,
   the null's mean is **more** negative and the gap grows; if sparse, the mean
   moves toward 0 and the gap shrinks. **The true mean lies in
   [−14,536 c, ≈ 0] at 5 %, and my −1,440 c is a point in that range, not a
   bound.** It can falsify my *mean*; it cannot falsify my *p*, because both
   ends leave +7,869.68 c far outside the null.
3. **The 1.754-rows-per-action asymmetry.** The null values row 0, the candidate
   the crossing row. Either direction changes the null's magnitude and neither
   changes its sign. **Cannot flip the p.**
4. **A wrong sign on the population.** Refuted: I reproduced
   `conditional_cancel_value` from its own components,
   (15,912 × 19.048 − 17,625 × 20.319)/33,622 = −1.6364, matching to 4 dp.
5. **THE ONE THAT CAN FALSIFY IT: a heavy right tail in preventable value.**
   `m_harm` = 19.048 is a **mean**; the artifact publishes no second moment. If
   a few actions carry very large positive preventable value, the null's spread
   could put +7,869.68 c inside it. **Quantified: at 5 % the null SD would have
   to exceed ≈ 3,100 c for the observation to sit within 3σ of a −1,440 c mean
   — which needs a per-drawn-action SD of ≈ 33 c and an RMS preventable-row
   value of ≈ 105 c against a mean magnitude of ≈ 20 c, a 5× RMS-to-mean
   ratio.** That is a property of the data no published quantity can reveal.

### What an off-prediction result MEANS — written now so it cannot be explained afterwards

* **p at the floor** → my prediction confirmed → **no information about the
  model.** The reading is *"a uniform-random cancel policy on this population
  loses money"*, which the artifact already said.
* **p NOT at the floor** → **this is not evidence that the candidate is weak,
  and it is not evidence that it is strong.** It means the null is noisier than
  the artifact's two published means permit anyone to compute — falsifier 5. The
  first diagnostic is the reported **null SD** and the **realised
  preventable-hit rate for both sides**, which BE is now emitting; if the SD is
  near 3,100 c at 5 %, falsifier 5 is the explanation and my arithmetic was
  fine as far as it went.
* **The null mean far from −1,440 c while p is still at the floor** → falsifier
  2, per-stratum concentration; the emitted hit rate settles it in one line, and
  my mean was wrong in a way that did not matter.
* **In no case is an unrestricted-pool result evidence FOR the candidate.**
  That is the pre-specified reading and it holds whichever way the number lands.

### R514-R6 — I revise my own R513-R2 in the direction that favours the code (CHECKED)

I called the row-0-versus-crossing-row valuation an unmatched comparison. At the
code it is **unavoidable**: the statistic's valuation rule is *"the earliest row
whose score crosses theta"* (`phase2_iter011_run.py::_cross`, which **refuses**
if no row crosses), and a randomly drawn generation **has no crossing** — there
is no theta event to value it at. Any null that draws generations must value
them by a different rule. The only strictly matched form would permute whole
value-vectors between generations while preserving each selection's crossing
index, which is more machinery than the question earns. **So the asymmetry is
structural, not a defect, and BE's fix — emit the realised hit rate for both
sides — is the right and the only proportionate one.** My finding stands as a
disclosure requirement and not as a bug.

---

## 4. Standing items — one DISCHARGED and it FAILS; one still not executable

### R514-R7 — HIGH — the re-emission provenance census RAN, and the artifact's provenance pointer does not resolve on the ledger (CHECKED)

First arms artifact that is both committed and on disk:
`data/pm_5min/derived/de_section81_arms__20260904T125340Z.json`, 90,789 B,
added at **`0e8f40c`** (DE round 58, 12:56:26Z), **on `mm-research`**, and the
canonical-tree file **matches the committed blob exactly**
(sha256 `765f98d8b36540f0085b176a`). That much is a real improvement and it is
the first time it has held.

**The census fails on the pointer:**

| check | result |
|---|---|
| `carrying_commit` `b43a9ce9…` exists as an object | yes |
| …is on `mm-research` | **NO** |
| identity files matching **at the commit the artifact names** | **4 of 7 — three DIFFER**: `de_section81_arms.py` (**the producer itself**, `fc2db050…` vs `11d1cc24…`), `de_phase4_diag_runner.py`, `de_lane4_real_parity.py` |
| identity files matching **at today's tip** | **7 of 7** |
| `working_tree_dirty` | **`None`** — not recorded |

**Read together these say the stamp is STALE, not forged**: the bytes that ran
are the bytes now on `mm-research`, and the artifact recorded the commit that was
HEAD when the run *started*, before its own edits landed at `0e8f40c`. It also
honestly records `producing_code_path`
`/home/yuqing/ctaNew-wt-de/live/pm_research/de_section81_arms.py` — a worktree
run.

**But that is precisely the failure the sibling artifact warns about in
writing.** iter011's `producing_code.why`: *"a content hash says WHAT ran; a
commit ref says WHICH COMMIT… **If the tree is dirty the ref names bytes that
are not these bytes, so both travel together.**"* The arms artifact carries the
ref, omits the dirty state, and the ref names bytes that are not these bytes —
in **three** files including its own producer. **A reader who resolves
`carrying_commit` and diffs gets a different pipeline.** R-508(H) recorded the
*previous* arms artifact graded `PRODUCED_BY_THE_COMMITTED_PIPELINE, 0 of 7
differing`; **this one is 3 of 7 differing at its own named commit**, so that
grade must not be carried forward to it.

**The fix is the one iter011 already implements:** stamp the commit **after** the
producing edits land, or record `working_tree_dirty` and `dirty_paths` beside
the ref so the disagreement is visible rather than latent.

**Population, for the record:** 12 windows, 31,122 generations,
`feed_wall_s` **0.3 s** (a cache hit — and the artifact records nothing about
the cache's identity), `total_wall_s` 46.1, `peak_rss_gb` 0.51,
`latency_ms` 250, `budget` 0.1, `evidence_class`
`DEVELOPMENT_EVIDENCE_NOT_A_VALIDATION`, exclusions counted at 4.21 %
(1,309 of 31,122). **This is a 12-window slice, not the §8.1 population**, and
the artifact says so. Nothing in it is quoted as a number here.

### The arms diff is STILL NOT EXECUTABLE

`arms53.py` and `arms53.json` remain absent from disk (name-only `find` over
`/home/yuqing`, depth 6, at 13:02Z) and were **never committed**. The 15,727 B /
`48c24f0f…` scratch runner cannot be diffed against anything because it no longer
exists. **This item cannot be discharged and should be closed as
UNRECOVERABLE rather than carried**, with the standing consequence recorded: the
numbers that settled R-506(A) survive only as source comments at
`de_section81_arms.py:61` and `:477`.

---

## 5. One thing in BE's filing that is stated backwards, and the true version is worse

Q-BE-267 says the five existing heads *"survive Holm by 4.2 %, purely because
the draw count was 500."* The direction is inverted — **more draws would give
them MORE margin, not less** (at n=2,000 the floor is 1/2001 and Holm gives
0.0120). The true statement is sharper: **their survival depends on the draw
count having been 2.5× the declared minimum.** At the declared floor of
n = 200, Holm gives 24 × 1/201 = **0.1194** and **every surviving result in the
artifact dies.** BE's instinct that this is fragile is right; the fragility is
that the declared minimum would not have sufficed, not that 500 flattered
anything.
