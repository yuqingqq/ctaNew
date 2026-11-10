# REVIEW 198 — BE's falsifier VARIES its resolution (the good case) but never varies its constants; six of DE's fifteen gates have no cell; and the generation band is a bar fitted to the days it judges

**REV 158, 2026-09-11T10:33Z** (clock read separately). Read-only, no lock; nothing run that
writes. `be_build_preflight.py` read on disk, DE's matrix read from `55fa438`.

## 0. WHERE THE FILES ACTUALLY ARE — CHECKED, AND TWO OF THE FOUR ANSWERS ARE "NOT WHERE SAID"

| file | state |
|---|---|
| `be_build_preflight.py` | **NOT on origin.** On disk in the **SHARED tree** `~/ctaNew/live/pm_research/`, uncommitted, mtime 10:20:18Z. **Not in wt-fwd** |
| `launch_stage2.sh` | **In no repo and no worktree** — it is in a **THIRD session's scratchpad** (`1ac1df02…`) |
| `live/pm_research/launchers/` | **4 files at `55fa438`, and `55fa438` is on `origin/de-freeze-chain-v2` ONLY — `origin/mm-research` has ZERO** |
| `recert_then_val.sh` | **NOT among the four committed launchers.** Still only in DE's scratchpad (`6a11e5b4…`) |

> **The launcher with the measured mid-run edit — the instance REVIEW 197 §b was written
> about — is the one that was not committed.** The four that were are `chain_day.sh`,
> `emit_0908.sh`, `emit_wait2.sh`, `preflight_gate.sh`. **wt-fwd is at `7ed5a90` with a clean
> tree**, so BE's new instrument is not in the worktree it certifies either.

## 1. BE's `be_build_preflight.py` — **THE FALSIFIER DOES NOT SHARE THE RESOLUTION IT TESTS. IT SHARES THE ONE IT DOESN'T.**

**This is the structural opposite of `de_preflight_matrix`, and it is worth saying so:**

```python
env = os.environ.get("BE_WORKTREE")
out.append(("-", "BE_WORKTREE is wt-fwd",
            "PASS" if env == WT_FWD else "WOULD_FAIL:BE_WORKTREE_NOT_WT_FWD(...)"))
```

**`BE_WORKTREE` is COMPARED to a constant, never RESOLVED THROUGH.** A wrong value produces a
named failure; it cannot silently relocate the check. And the falsifier **varies that
variable across three values** — unset, wrong tree, right tree — asserting a different
outcome for each, including *"the RIGHT tree PASSES — the control admits as well as fires."*
**That is rule 16 satisfied explicitly, and it is the pattern `de_preflight_matrix` lacked.**

**BUT THE RESOLUTION IT NEVER VARIES IS THE ONE EVERYTHING ELSE RESTS ON:**

```
PIN    = "7ed5a9015f75…"    WT_FWD = "/home/yuqing/ctaNew-wt-fwd"
D      = Path("/home/yuqing/ctaNew/data/pm_5min/derived")      PINNED = (...)
```

**Every git read is `git -C WT_FWD`; every artifact read is under the absolute `D`; the five
digest checks are against `PIN`. The falsifier varies the env and holds all four constant** —
so a defect in `WT_FWD`, `PIN` or `D` turns the instrument and its falsifier green together.
Cell 3 would "pass" by setting the env to the same wrong constant it compares against.

**AND ONE LIVE SEAM, LATENT TODAY:** the module inserts `parents[2]` on `sys.path` — the
**SHARED** tree, since that is where the file lives — and then imports `flow_intensity`,
`be_era_for_day`, `be_gate1_fragment`. **So it CERTIFIES wt-fwd's pinned blobs while EXECUTING
the shared tree's copies.** Measured now:

```
flow_intensity      shared e0b0c578… pinned e0b0c578…  SAME
be_era_for_day      shared 6b5afd71… pinned 6b5afd71…  SAME
be_gate1_fragment   shared d87c8208… pinned d87c8208…  SAME
```

**They agree today, and nothing checks that they do — a control that is correct by
coincidence.** One cell closes it: for every module the preflight IMPORTS, assert the
shared-tree bytes equal the `PIN` blob (or import from `WT_FWD` explicitly).

## 2. DE's FIFTEEN GATES — **SIX HAVE NO CELL**

Gates: `book, book_builder_commit, book_era, book_generations, book_newer_than_tape,
book_receipt, book_windows, certification, comparator_digest, frozen_params,
gap_windows_artifact, mask, mask_referenced, receipt, verify_run_inputs`.

`falsify()` exercises `frozen_params`, the cascade, `book_era`, `book_windows`, `book`,
`receipt`, `book_receipt`, `verify_run_inputs`, and — behaviourally, through the old-cert
argument rather than by name — `certification` and `comparator_digest`.

**With no cell at all: `book_builder_commit`, `book_generations`, `book_newer_than_tape`,
`gap_windows_artifact`, `mask`, `mask_referenced`.**

**THE ONE NOBODY WOULD NOTICE: `book_generations`.** Every other gate's subject is a
presence, identity or ordering fact whose silence breaks something visible downstream. **A
generation-count gate that always returned PASS would look exactly like a gate that was
working** — the book still exists, parses, carries the right era and 288 windows, and the
count it no longer checks appears nowhere else. **THE ONE WITH THE WORST CONSEQUENCE:
`mask_referenced`** — a book built without consulting the blackout mask would pass, and
nothing downstream re-derives it.

**The cells, and the shape is already in the file** — the era cell fabricates a receipt with
`clob_v3_1` and asserts `WOULD_REFUSE:BOOK_ERA_NOT_DECLARED`. **Clone it:**
a synthetic receipt with `n_reference_generations` **outside** the band must
`WOULD_REFUSE:BOOK_GENERATION_COUNT_OUT_OF_BAND`, **and one inside must PASS** — both
directions, or the new cell has the defect it was added to catch. Same two-armed cell for
`mask_referenced`, `book_newer_than_tape` and `book_builder_commit`.

## 3. THE GENERATION BAND — **A SMOKE BAR FITTED TO THE DAYS IT JUDGES**

```
GENERATION_BAND = (241_000, 429_000)
source: 09-07 n_reference_generations 321,925 and 09-08 342,942, widened 25% either side
```

**IT CAN CATCH:** an order-of-magnitude build failure — a near-empty book, a 10× book, a
wrong-day book; the **09-08 class** (a fragment that died 13 minutes in on a missing tape and
produced almost nothing). **That is a real and cheap catch, and the label SMOKE is honest.**

**IT CANNOT CATCH, and these are the ones that matter:**
1. **Anything inside the band — which is 1.78× wide (429/241).** A **20 % generation loss
   passes silently**, and generations are what the estimand is summed over. A gap-handling
   change that dropped one coin's windows, or 15 % of a day's generations, is invisible here.
2. **Composition at constant count.** The same number of generations with **different
   membership** — precisely what the era fix does — passes exactly. **The band is blind to
   the change this population was rebuilt for.**
3. **n = 2.** Two observations give no spread estimate; ±25 % is a guess, not a tolerance. And
   the two days are **09-07 and 09-08 — members of the population it will judge.** Days 3–7
   are measured against a bar fitted to days 1–2, so **a systematic drift already present on
   those two days is inside the band by construction and can never be detected.**
4. **Trend.** It has no per-day expectation, so monotonically falling generations across the
   seven days pass every day individually.
5. **It is not a correctness control at all.** A book can sit mid-band and be wrong in each of
   the ways the other fourteen gates exist to catch.

**What would make it a control rather than a smoke test:** a **per-day expected count derived
from that day's own supplied-window and admissible-generation counts** — quantities the
fragment already computes — with the band a tolerance around *that*, not around two
historical totals. Until then it should keep saying SMOKE, and **no receipt should cite it as
evidence the book is right.**
