# REVIEW 263 — no licensing condition. 0.636 stands, with the unknown named.

REV round 227. Filed 2026-09-12T01:24:21Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ANSWER

**I found no condition that licenses excluding 09-01..09-03, and I looked in the
places where one would have to be.** The days were scored by a **frozen,
byte-identical instrument**. So the exclusion is not available, 0.636 stands as
the planning rate, and the unknown is named in §4.

This is the answer that most likely tells the user the test cannot be run as
written, and I would rather deliver it than a cut I can defend only by its
result.

## 1. The three places a condition could have been, all closed

**(a) The detector — FROZEN AND IDENTICAL.** Every one of the eleven masks
carries the same detector block, verified by digest:

    detector signature f2a54ecd0c9c -> 11 of 11 days, 20260901..20260911
    thin_frac 0.05 | version v1_FROZEN
    module da_content_liveness_rule, sha256 prefix 7196676840304f30
    authority USER ruling 2026-09-01, R-386
    definition "below thin_frac x the SAME-DAY (day, coin) median AND not
                overlapped by a gap-ledger interval"

**One signature across all eleven days.** The instrument that masked 40 windows
on 09-02 is bit-for-bit the instrument that masked 0 on 09-05. There is no
detector change to cut on.

**(b) The collector era — both boundaries PREDATE the window.**
`collector_runs.jsonl` carries exactly two:

    clob_v4    2026-08-30T05:30:00Z  (O1a ping 10/10->3/3, O1b backoff)
    clob_v4_1  2026-08-31T22:00:00Z  (O1a ping ROLLBACK, USER ruling 08-31)

The rollback completes at 08-31T22:00Z — **before 09-01**. Every day in the
population ran under `clob_v4_1`. There is no era boundary inside the window.

**(c) Supply and collection — NOT SHORT.** `n_present` is 288 on every day
(287 on 09-03), and the raw tape carries 288 `btc-*.jsonl.gz` files on every
day I counted. Nothing was un-collected, un-listed or un-archived.

## 2. The one instrument difference I did find — and it accounts for ~1 window of 227

`coverage_accounting` is **absent (null) on 09-01..09-03 and present from
09-04**, distinguishing `blackout_masked` (file exists, content below
threshold) from `coverage_absent` (no file at all). That is a real schema change
with a boundary at 09-04, evidenced independently of any outcome.

**And it does not explain the magnitude.** `total_masked` falls 233 → 6 between
09-03 and 09-04, while `total_coverage_absent` on 09-04 is **1**. A
reclassification that moves at most one window cannot account for a drop of
227. **I am recording it and declining to use it**, because using it would be
failure mode (iii) — a real condition that identifies the days by outcome, since
the only reason to reach for it is that the rate improves at the same place.

## 3. A property of the criterion worth keeping, which is NOT a licence

The threshold is `thin_frac × the **SAME-DAY** (day, coin) median`. A relative
threshold makes the mask count a measure of **within-day dispersion** rather
than of absolute adequacy, so mask counts are not, in general, comparable
across days by construction. That is worth recording about any statistic built
on them.

**But I will not use it here, because at `thin_frac = 0.05` it does not bite.**
A masked window holds under 5% of its day's median — near-empty under any
plausible median, which is the same regime as DA's 09-11 shells at 0.01%. So
cross-day incomparability is a genuine caveat about the instrument and **not**
an explanation for 23 / 40 / 40 against 0. Reaching for it would be failure mode
(i): dressing the observation as its own explanation.

## 4. What is not determinable from what was collected

**Why 23, 40 and 40 btc windows held under 5% of their day's median on
09-01..09-03, and near-zero thereafter, is not determinable from what I read.**
The detector is frozen, the era is constant, the files are present, and the
windows are near-empty — so something made windows near-empty on those three
days and stopped. I did not read the mask producer's source, the per-window
content measurements it consumes, or anything that would distinguish a data
cause from a producer cause.

**Consequence, stated the way you asked:** 09-01, 09-02 and 09-03 are
**admissible members of the population** — same frozen instrument, same era,
same supply — and they are **failures**. The planning rate is

    union criterion (race_accrual_eligible AND interior missing <= 1)
    population 09-01..09-11, no exclusions licensed
    7 of 11 = 0.636    E[evaluable in 14] = 8.91 against a requirement of 10
    P(INSUFFICIENT_EVIDENCE) = 0.62

with the named unknown: *whether the 09-01..09-03 thin-window episode is
recurrent or was fixed.* If recurrent, 0.636 is right. If fixed, 0.636 is
pessimistic — **and nothing I can measure tells us which**, which is precisely
why it must be carried as an unknown rather than resolved by a cut.

## 5. And my 09-01 exclusion from REVIEW 262 is WITHDRAWN

Last round I ruled 09-01 inadmissible because `supply()` reported it
`PRE-GOVERNED (< 20260902)` while later days read `GOVERNED`. **That was about
the mask *requirement* — whether a mask must exist — not about the detector
that decides what a mask contains.** The detector is identical on 09-01 and on
every later day. A day is not incomparable because the rule about whether its
mask was mandatory differs; it is incomparable if the thing that measured it
differs, and that did not.

So: **09-01 is admissible.** The population is 09-01..09-11, 7 of 11, and the
distinction I drew was between the wrong two things. That is the second
population-cut error I have made in three rounds, both in the same direction —
toward a smaller denominator and a better rate.

## 6. What I excluded

Read: the eleven `da_blackout_mask_*.json` artifacts (detector block, totals,
coverage accounting, as-of times, producer commits); `collector_runs.jsonl`;
`de_admissible_windows.supply()` driven per day; raw tape file counts;
`be_build_preflight.check_day` driven per day. **Not read**: the mask
producer's source, `da_content_liveness_rule`'s implementation, per-window
content measurements, host or process metrics, anything in P-2026-002. So I can
say no condition is present in the instrument, the era or the supply; **I cannot
say what happened in the data**, and I am not going to guess.

## 7. Owed

- Unchanged: `score_is_evidence_permitted` = `None`; the
  `PLAN_ENUMERATION_UNPARSEABLE` ambiguity; the freeze checker's red falsifier;
  `da_deploy_midnight.sh` un-rerun.
- DA's half — why the mask collapsed — is the same question from the other
  side, and if DA finds a producer-side cause it supersedes §4's unknown.
