# DRAFT FOR USER RULING — the coverage precondition on the day bars

**Status: DRAFT-FOR-USER. No seat may resolve this.** Drafted by DA
2026-09-04 on the coordinator's instruction, in the form used for the
`clob_v4` cite question. Nothing here changes any value or any bar; the
precondition is untouched pending the ruling.

**The question, in one sentence.** 2026-09-03 lost **one window of 288**
(15:20:00Z, all seven coins), and because coverage was therefore not complete
the P1/P2/P3 bars did not return a value at all — `evaluable: false`,
`P1: None` — so the day failed `day_quality_pass` **before its quality was
measured**. Should a day whose coverage is short by a small amount be
refused, evaluated, or evaluated only when the shortfall cannot change the
answer?

**DA states plainly what it may not do here.** The alternative is visible to
me *because* it has just cost a race day, so choosing it is choosing after
seeing (CLAUDE.md rule 11) — the same trap that keeps `LOW_FRAC` and the
tape-density threshold REPORTING rather than governing. **Any change is
prospective-only and is the USER's.**

---

## 1. What the bar does today

`da_forward_day_verify` computes, per coin:

```
_cov = counts.get(coin, 0) > 0 and short.get(coin, 1) <= 0
b    = day_bar_v2(lo, hi, coin, hours, coverage_observed=_cov)
```

and `day_bar_v2` refuses unless coverage is **affirmatively** supplied:
*"only `coverage_observed is True` evaluates; False, None, omitted and
malformed all refuse."* On a closed day `short = 288 - windows_present`, so
**any shortfall at all — one window — makes all three bars unevaluable**, and
`all_pass` is False with no statistic behind it.

**The reason the guard exists is sound and should not be lost in this
discussion:** a bar computed on incomplete coverage *understates* loss,
because the missing windows contribute no lost seconds. Refusing is the
conservative direction. What is at issue is the GRANULARITY — the guard does
not distinguish one missing window from a hundred, and it returns `None`
rather than a bounded answer.

## 2. What it did on 2026-09-03, measured

| quantity | btc | eth |
|---|---|---|
| windows present | **287 / 288** | **287 / 288** |
| `lost_seconds` (from the gap ledger) | 2,294.7 | 215.4 |
| P1 on the observed windows | **95.61** s/hr | 8.97 s/hr |
| P1 if the missing window were **100 % lost** | **108.11** s/hr | 21.47 s/hr |
| P1 bar | 120.0 | 120.0 |
| P2 | 1 window = 0.35 % (bar 5 %) | 0.35 % |
| P3 worst rolling 60 min | 221.5 s (bar 900) | 111.1 s |

**Both readings are under every bar.** The missing window cannot change the
answer: 09-03's quality is a PASS under any assumption about what was in it.
The day's other three conjuncts are all True (FINISHED, AFTER, ADMISSIBLE),
so **the coverage precondition alone is what stopped it accruing.**

The gap ledger holds **zero rows** overlapping 15:20–15:25Z for btc or eth,
so the 300 s is not already counted — the worst case above is the correct
additive bound and not a double-count.

## 3. What each answer would do — computed over the whole record

Days from `DAY_BAR_V2_FROM_DAY` onward, with 08-28 for context. "EVAL" means
the bars return a value; "refuse" means they do not.

| day | short (btc/eth) | **today** | short ≤ 1 | short ≤ 2 | worst-case bound | P1 observed / worst case | verdict |
|---|---|---|---|---|---|---|---|
| 2026-08-28 | 0 / 0 | EVAL | EVAL | EVAL | determined | 114.09 / 114.09 | PASS |
| 2026-08-29 | 0 / 0 | EVAL | EVAL | EVAL | determined | 32.29 / 32.29 | PASS |
| 2026-08-30 | 0 / 0 | EVAL | EVAL | EVAL | determined | 187.61 / 187.60 | FAIL |
| 2026-08-31 | 0 / 0 | EVAL | EVAL | EVAL | determined | 298.52 / 298.52 | FAIL |
| 2026-09-01 | 0 / 0 | EVAL | EVAL | EVAL | determined | 84.40 / 84.40 | PASS |
| 2026-09-02 | 0 / 0 | EVAL | EVAL | EVAL | determined | 73.71 / 73.71 | PASS |
| **2026-09-03** | **1 / 1** | **refuse** | EVAL | EVAL | **determined** | **95.61 / 108.11** | **PASS** |

**Exactly one day in the record changes under any alternative, and it is
09-03.** Every other day has complete coverage, so today's rule and every
alternative agree on all of them. That makes this — like the `clob_v4` cite
question — a decision with a known and bounded consequence rather than one
that trades a result for a rule.

## 4. The three answers

### (a) KEEP the precondition as it is

* **Runner behaviour:** unchanged. A day short by any amount fails
  `day_quality_pass` with no statistic.
* **What it asserts:** that complete coverage is part of what a qualifying
  day *is*, not merely an input to the bars — a defensible position, and the
  one currently in force.
* **Cost, stated:** 09-03 does not accrue, G stays at 2, and the earliest
  G=5 moves from 2026-09-06 to **2026-09-07**. If single-window losses recur
  the race loses a day each time; the observed rate is **1 of 3** era-pure
  days (09-01 and 09-02 were 288/288, 09-03 was 287/288).

### (b) EVALUATE when the shortfall is at most k windows

* **Runner behaviour:** 09-03 evaluates and PASSES; nothing else in the
  record moves at k = 1 or k = 2.
* **What it asserts:** that a bar computed on 287 of 288 windows is close
  enough to the truth to govern.
* **Cost:** k is a threshold chosen after seeing which day it rescues. It
  would need pre-registration against days nobody has looked at, and the
  understatement it permits is unbounded in principle (k windows of arbitrary
  loss).

### (c) EVALUATE ON THE BOUND — report the observed value **and** the value
with every missing window counted as wholly lost, and let the bar govern only
when the two agree

* **Runner behaviour:** 09-03 evaluates, both readings pass, the answer is
  **DETERMINED** and it accrues. A day where the two readings straddle the
  bar stays refused — with a stated reason and two numbers rather than
  `None`.
* **What it asserts:** that a day's answer is knowable whenever the missing
  data cannot change it — which is a property computed per day, not a
  threshold chosen once.
* **Cost:** more machinery than (a) or (b), and it changes what `evaluable`
  means. It is **not** threshold-free in one respect: it still needs the
  convention that a missing window counts as 300 s of loss, which is the
  most adverse assumption available and therefore not a tuned one.
* **In this record it is equivalent to (b) at k = 1** — every day is
  determined — so the difference between them is prospective, not historical.

---

## 5. What DA has done and has deliberately not done

* **Done:** measured the shortfall and its cause (one window, all seven
  coins, permanently absent, no gap row); computed the observed and
  worst-case bars for every day under `day_bar_v2`; established that exactly
  one day changes; and established by cross-venue comparison that the loss
  was **Polymarket-side** (§ Q-DA-224).
* **Not done, and not mine:** touching `coverage_observed`, `day_bar_v2`, or
  the P1/P2/P3 constants. The alternative became visible because it cost a
  day, and adopting it on that basis is rule 11 in one move.

**No recommendation is offered between (a), (b) and (c).** DA's only view is
that the choice should be recorded before the next short day rather than
after it, because after it only one of the answers is still available.
