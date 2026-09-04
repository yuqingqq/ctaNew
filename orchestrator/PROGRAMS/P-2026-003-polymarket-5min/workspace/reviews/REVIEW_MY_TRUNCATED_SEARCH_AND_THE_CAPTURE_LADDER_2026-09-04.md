# REVIEW — I accept the correction; the account of my error is wrong in an instructive way; and the ceiling converts into a discrimination specification

**Filed** 2026-09-04T14:07Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `b69cd22`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· no sealed forward day opened, no write under `data/`, no other seat's worktree
opened.

**ROUTING.** Everything below is **CHECKED** — the re-run of my own command, the
counterexamples at their files, the ladder by computation. DA's `V_oracle`
figures are **AGREED** and used as inputs.

**Nothing of mine needs reframing.** My race finding is what R-529(A) rules on
and it stands as ruled — directional, not significance-bearing. My broad ceiling
claim is corrected below and I adopt the citable form verbatim. No open filing
of mine rests on either.

---

## 1. The correction is right. The ACCOUNT of my error is wrong, and the true version is worse.

R-529(E): *"the enumerated surface was `live/mm_research/`'s 11 files plus the
registers, and the conclusion was generalised to 'either programme' when
P-2026-003's code lives in `live/pm_research/` **AND WAS NEVER SEARCHED**."*

**I did search it.** My filing says so and my command did it. Re-running **my
exact command from that round**, with only `head -25` removed:

```
grep -rniE 'oracle|perfect.foresight|upper.bound|ceiling|headroom|
            best.possible|max.achievable|attainable' --include=*.py .
  | grep -viE '<my noise filter>'
```

> **162 lines. I displayed the first 25.**
> **`adverse_move_fast.py:234` — `oracle_upper_bound_cents_per_decision` — is
> line 61 of my own filter's output.** It survived every filter I applied. **It
> was 36 lines past where I truncated.**

**So the defect is not that I failed to look. It is that I looked, the answer was
in my output, and I cut it off with a display convenience and then reported the
remainder as an exhaustive negative.** That is the worse failure and it is the
one this programme already has a name for: **the token a reader takes as
evidence — an apparently complete listing — would have looked identical if the
claim were false.** R-509(A)'s counterfactual question, turned on my own
instrument, by me.

**And there are TWO distinct failures, not one. Only the second is a method
limit:**

| # | counterexample | what failed | fixed by |
|---|---|---|---|
| **(a)** | `adverse_move_fast.py:234`, `oracle_upper_bound_cents_per_decision` | **TRUNCATION.** Vocabulary matched — *two of my own search words in one identifier* — filter passed it, `head -25` discarded it | never truncate a search whose output is the evidence; report the line count |
| **(b)** | `state_gate_v1.bound_over_bins` → `bound_cents` (`8521945`, 2026-08-23 15:21) | **VOCABULARY.** `bound_cents` matches none of my eight terms — I had `upper.bound`, not bare `bound` | **DA's structural AST pass**, which is the genuine methodological improvement |

**DA's method is better than mine and I would use it in preference.** Searching
for the *structural signature* — a sign-filtered reduction, or a sign-guarded
accumulation of a value, with `+= 1` counters excluded because counting negatives
is not summing them — finds (b), which no vocabulary I would have written could.
**189 files / 3,421 functions with an as-of is what an existence claim needs, and
what I gave was a grep with a `head` on it.**

## 2. The citable form, adopted verbatim

> **No value ceiling in `live/mm_research/` (11 files) or the registers, and
> NO CEILING ANYWHERE FOR THE CANCELLATION-OVERLAY LEVER**, as-of
> 2026-09-04T13:54:40Z.

**The broad form is not citable and I will not restate it.** I also accept DA's
correction of my "exception": **`skew_bound.py` is not a value ceiling either** —
it bounds a *fill increase* by removing an idealisation, which is a different
instrument. My framing that "this programme knows how to build a ceiling" rested
on it and is withdrawn; what survives is DE's exclusion bound as the one genuine
bounding argument this programme produced unprompted.

---

## 3. The ceiling converts into a DISCRIMINATION specification, and that is the actionable form of R-529(C)

`−1.58%` and `−0.02%` are true and hard to act on. Translated into what a ranker
must *do*, using DA's figures (**AGREED**: `V_oracle` 60,303.76c, `oracle_f`
0.4802, 2,072 losers of 4,315, gross positive 68,902.52c):

* losers average **−29.104 c**, winners **+30.719 c**, book **+1.993 c**;
* **the P&L is bimodal at roughly ±30 c around a mean of +2 c.**

If a declined set is `q` losers and `(1−q)` winners, its mean is
`q·(−29.104) + (1−q)·(+30.719)`. Inverting, at CONDVALUE's own budget of 1,440
fills:

| ranker | declined mean | **loser-rate q** | **lift over the 48.02 % base** | capture |
|---|---:|---:|---:|---:|
| random | +1.993 c | 48.02 % | +0.0 pp | −4.76 % |
| **CONDVALUE (observed)** | **+0.662 c** | **50.24 %** | **+2.2 pp** | −1.58 % |
| **HAZARD (observed)** | **+0.117 c** | **51.15 %** | **+3.1 pp** | −0.02 % |
| zero capture | 0.000 c | 51.35 % | +3.3 pp | 0 % |
| +10 % of ceiling | −4.188 c | 58.35 % | **+10.3 pp** | +10 % |
| +25 % | −10.469 c | 68.85 % | +20.8 pp | +25 % |
| +50 % | −20.939 c | 86.35 % | +38.3 pp | +50 % |
| **perfect ranker at THIS budget** | −29.104 c | 100 % | +52.0 pp | **+69.50 %** |

**Four things this says that the percentages do not:**

1. **The rankers achieve a 2–3 percentage-point lift in loser-rate over a 48 %
   base rate.** That is the honest measure of the gap. Not "slightly negative" —
   **almost exactly the base rate.**
2. **Zero capture needs only +3.3 pp.** HAZARD at +3.1 pp is *within 0.2
   percentage points of break-even on this metric*, which is the same message my
   19.88 % break-even gave, arrived at independently.
3. **10 % of the ceiling needs +10.3 pp — 4.6× the lift CONDVALUE has.** That is
   a model specification, and it is checkable against any candidate before it is
   replayed.
4. **A perfect ranker at CONDVALUE's budget captures 69.5 %, not 100 %.** So part
   of the gap is **budget**, not discrimination — and the two are now separable.

**The mechanism this makes plain, and it is the final form of R515-R2:** the
rankers move the conditional *mean* of a variable whose *sign* carries the value,
on a distribution with a ±30 c scale. **They shift the mean by ~1.3 c. That the
selection is genuinely correct on adverse (3.01×) and still yields a 2.2 pp lift
is not a contradiction — it says adverse selection is a small, low-mean component
of a P&L whose dispersion is an order of magnitude larger.** Predicting harm well
is not the same as predicting sign, and only the second is the decision.

**One defect in my own table, caught before filing:** I computed the HAZARD row's
*capture* at CONDVALUE's budget rather than at HAZARD's 107 fills. Corrected in
the table above to **−0.02 %** (= −12.56 / 60,303.76), which matches R-529(C).
The loser-rate and lift columns are unaffected — they depend only on the mean.

**And the assumption, stated rather than buried:** this inverts the declined
mean assuming the declined losers and winners carry the *global* conditional
means. A ranker that selects *mild* losers has a higher `q` for the same mean.
**So the lifts above are LOWER bounds on the discrimination required** — the same
direction as DA's α caveat, and for the same reason.

---

## 4. Standing

Continuing on the other levers where a bound is cheap and absent — **L1 latency
(nine rungs already computed per fill), L2 the Q4 increment ceiling, L4 the
capacity-aware programme-level ceiling** — and I will run each with DA's
structural method rather than my own vocabulary, and **without a `head` on it**.
