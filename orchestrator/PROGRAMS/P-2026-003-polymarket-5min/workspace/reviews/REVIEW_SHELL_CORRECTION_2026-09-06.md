# REVIEW — my A.3 was wrong because my worktree lied to me: both day sets ARE evaluable, Set A holds, and Set B fails on 09-06 for a reason that is not 09-06's fault

**Filed** 2026-09-06T04:05Z (clock read before composing) · reviewer seat
(pm-codex) · HEAD `41cba2c`, an ancestor of tip `0901b61` · no code fixed · no
write under `data/` · nothing sealed opened.

**ROUTING — CHECKED**, every statement below recomputed by me at the symlinked
ledger, and the shell-vs-ledger comparison run exhaustively rather than sampled.

---

## 1. The symlink resolves, and the ledger is four times what I could see

```
data -> /home/yuqing/ctaNew/data          (readlink -f: /home/yuqing/ctaNew/data)
da_dayverdict_20260903.json  67,463 B
da_dayverdict_20260904.json  63,000 B
da_dayverdict_20260905.json  61,371 B
da_dayverdict_20260906.json  46,769 B
```

**The `da_dayverdict_` series is TWELVE files, 08-26 through 09-06.** My worktree
showed **seven**. It was missing not only 09-03..09-06 but 08-26, 08-27 and
08-31 — **seven files invisible to me.**

*(Note for the record: `git status` in this worktree now reports 137 tracked files
under `data/` as deleted. That is git being unable to traverse a symlinked
directory — the files are all present through the link. I have not touched it.)*

## 2. **My A.3 finding is WITHDRAWN. Both sets are evaluable.**

I wrote *"the `da_dayverdict_` series in the repository runs 08-28…09-02 and
STOPS"* and concluded *"the day-quality conjunct has no evaluable input for the
days that matter most."* **That was false, and it was false because I read a
directory listing as evidence of a population.**

Re-run at the ledger — every value read from the verdict files themselves:

| day | `day_quality_pass` | `era_pure` | `counts_toward_race` | previously opened |
|---|---|---|---|---|
| 2026-08-29 | **True** | **True** | — | **YES** (R-502 development read) |
| 2026-08-30 | False | False | — | — |
| 2026-08-31 | False | False | — | — |
| 2026-09-01 | True | True | — | **YES** (interim read, R-549(A)) |
| 2026-09-02 | True | True | — | **YES** (interim read) |
| 2026-09-03 | **True** | **True** | **True** | no |
| 2026-09-04 | **True** | **True** | **True** | no |
| 2026-09-05 | **True** | **True** | **True** | no |
| 2026-09-06 | **False** | True | **False** | no |

**Set A (opened days count) = 08-29 + 09-01..09-05 — SIX DAYS, and it HOLDS.**
All six are `day_quality_pass: True` and `era_pure: True`. Evaluable today,
complete today.

**And two things I argued from the register are now corroborated at the ledger:**
08-29 reads `quality True / era_pure True`, exactly as R-497(F)(1) describes it;
and 08-30/08-31 fail on **quality** as well as era — so their exclusion was the
right answer, and my earlier point stands that the *stated* reason (era boundary)
was not the operative one.

## 3. **NEW FINDING — Set B as R-550(D) names it does not hold, and 09-06's failure is an artifact of when the verdict was written**

R-550(D)'s purist set is *"09-03, 09-04, 09-05 + 09-06, 09-07, 09-08 … complete at
the 09-09 00:06Z verdict."* But **09-06 carries `day_quality_pass: False` and
`counts_toward_race: False`.**

**It is not a bad day. It is a six-minute-old day.** From the verdict itself:

```
/content_liveness_rule/coins/{bnb,doge,hype,xrp}/n_windows = 1
/content_liveness_rule/coins/*/why                         = "1 windows < 20"
/content_liveness_composite/status = CONTENT_LIVENESS_UNJUDGEABLE
/blackout_mask_and_complement/why  = "REFUSED: ... CONTENT_LIVENESS_UNJUDGEABLE
                                      for 20260906 (no coin had enough windows
                                      for a median)"
```

And the file mtimes settle it:

```
da_dayverdict_20260905.json   written 2026-09-06T00:06Z   <- 09-05 COMPLETE
da_dayverdict_20260906.json   written 2026-09-06T00:06Z   <- 09-06 SIX MINUTES OLD
```

**The 09-06 verdict was written six minutes into 09-06.** This is precisely the
`OPEN_DAY_MASK_DEFERRED` case rule 20's classifier exists to distinguish — *"the
frozen detector cannot judge a six-minute-old day."* 09-06 will be re-verdicted at
the 09-07 00:06Z run.

**The consequence for the rule, and it is a trap worth closing before G is fixed:**
applying the `day-quality` conjunct mechanically to today's ledger **excludes
09-06** and pushes the purist set to 09-07/08/09, completing at the **09-10 00:06Z
verdict — one day later than R-550(D) states, for a reason that is not a quality
failure.** The rule must say that **`day-quality` is evaluated only on COMPLETE
days**, and that the current day's in-progress verdict is not an input.

*(This does not disturb the recommendation. Set A is six days and available now;
Set B is six days and its completion date is one day later than stated unless
09-06 is re-verdicted first. Either way six days, so either way a unanimous pass
clears Holm at m = 2.)*

## 4. Every finding in my last three filings that read a path under `data/` — audited exhaustively

**I did not reason about the exposure. I diffed the preserved shell against the
ledger, file by file.**

```
files the shell held      : 137
identical to the ledger   : 137
differing                 :   0
present only in the shell :   0
```

**The shell never served stale or altered content. Its only failure mode was
ABSENCE** — it held 137 files against the ledger's 911,223.

And specifically, the fifteen `data/` artifacts I cited across `be03d4d`,
`89e81d5` and `41cba2c` — DE's design declaration, DA's era artifact, both
`de_section81_arms` comparands, `be_ceiling_null_v1`/`v2`,
`be_cancel_axis_null_v1`/`v2`, the 09-03/09-04 receipts and both RACE_CONTEXT
companions, seam v4, ceiling v2, mutation audit v3 — are **every one
byte-identical** between shell and ledger.

**So the audit resolves cleanly:**

| filing | findings resting on CONTENT | findings resting on ABSENCE / a listing |
|---|---|---|
| `be03d4d` design round | all of them (the `asm` omission is a claim about DE's `object` field; the imported bar is register + DA's artifact) | none |
| `89e81d5` DE 69 / BE 45 | all of them; the two strongest also used `git show`, not the filesystem | none |
| `41cba2c` documents | resource arithmetic, receipt count, the withdrawn sentence, headlines, rule 20 | **A.3 — the only one** |

Three further notes for completeness:

* **The flock drive used the absolute path** `/home/yuqing/ctaNew/data/.heavy_run.lock`,
  not my worktree's — so §B.4 of `41cba2c` was always against the ledger.
* **The classifier probe on BE's `cb9bf8a` diff used `git show`**, not the
  filesystem, so `89e81d5`'s headline finding never touched the shell.
* **One residual risk I did not previously state:** in the DE 67 / DA 55 round I
  used a worktree listing (`ls … | grep de_section81_arms__`) to *locate* a
  comparand. The content I then read was identical, and the comparand was the one
  DE named by digest — but a shell can hide a file, so a listing used for
  discovery is weaker evidence than one used for a count. Both were fine here.

## 5. **Could the shell have made any CHECKED routing wrong? One sentence.**

**A partial shell can only omit, never corrupt — 137 of 137 files it held were
byte-identical to the ledger — so the sole exposure was findings resting on a file
being absent or a directory listing being complete, and exactly one of mine did:
A.3, now withdrawn and replaced above.**

---

## CONTEXT

Far below the 80% reset threshold. **Standing by to file on DE 71 (design v3) and
BE 46 the moment they land — and I will re-read `data/` paths through the symlink
from now on, with the count of what I see stated whenever a population is the
evidence.**
