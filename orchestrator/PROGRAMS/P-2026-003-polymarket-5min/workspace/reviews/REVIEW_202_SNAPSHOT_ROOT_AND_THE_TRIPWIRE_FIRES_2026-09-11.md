# REVIEW 202 — the snapshot froze ONE of five growing inputs; the launch record does not exist; and the tripwire FIRED — with 29× cancellation on HAZARD and no gap-seconds column

**REV 162, 2026-09-11T11:10Z** (clock read separately). Read-only, no lock.

---

## (3) FIRST, BECAUSE IT LANDED WHILE I WAS WORKING: **THE TRIPWIRE FIRES, ON BOTH ARMS, ON BOTH ARMS OF THE PREDICATE**

`fwd_v2/p003_de_revaluation_emit_20260907.json`, 11:07:44Z. Against REVIEW 186:

| | CONDVALUE | HAZARD |
|---|---|---|
| D v1 → v2 | −11,017.712006 → −14,645.078818 | +5,256.176844 → +4,925.363903 |
| **ΔD** | **−3,627.3668c** | **−330.8129c** |
| table sums to | −3,627.3668 | −330.8129 |
| **residual** | **5.9e-12 → ROUNDING** ✔ | **−4.8e-12 → ROUNDING** ✔ |
| **`CONCENTRATION_FINDING`** | **True** | **True** |
| fired_on_aggregate | **True** | **True** |
| **fired_on_a_single_window** | **True** | **True** |
| worst single window | **1,960.40c** | **1,280.75c** |
| `SIGN_CHANGE_HALT` | False | False |
| n_rows / spine | 27 ✔ | 27 ✔ |

**Residual in the ROUNDING band on both arms — the 27-row table reconciles to ΔD at 1e-12.**
Rules 4 and the row count are clean. **No sign change, so REVIEW 186 Rule 3 does not engage.**

### AND THE PER-WINDOW ARM EARNED ITS PLACE ON ITS FIRST USE

```
CONDVALUE  24 of 27 windows moved   up +4,469.8   down -8,097.2   NET -3,627.4   gross 12,567c   3.5x cancellation
HAZARD     21 of 27 windows moved   up +4,633.6   down -4,964.5   NET   -330.8   gross  9,598c   29.0x cancellation
```

> **HAZARD's aggregate ΔD is −331c and its single worst window is −/+1,281c — FOUR TIMES the
> aggregate.** REVIEW 184 §4 argued for the per-window arm with the hypothetical *"+2,000c at
> 20:45 and −1,900c elsewhere nets +100c and fires NOTHING."* **That hypothetical is now real
> at 29×.** An aggregate-only tripwire would have called HAZARD a modest 331c change; it is a
> **±4,600c rearrangement that nearly cancels.**

### WHICH FORCES A CORRECTION TO MY OWN REVIEW 186 RULE 1

The scoping bands are computed on **|ΔD|**, and |ΔD| is the wrong statistic when cancellation
is high:

```
                 k_win(NET)      k_win(GROSS)
CONDVALUE          3.50            12.12
HAZARD             0.32             9.26     <- AMBIGUOUS by the band, WINDOW-SCOPED by the data
```

**By Rule 1 HAZARD reads AMBIGUOUS_SCOPE (330.81c, between 184 and 518). By the per-window
evidence it is unambiguously window-scoped — 21 of 27 windows moved.** The band under-reads
it purely because the moves cancel. **Rule 1 should classify on the GROSS per-window movement
`Σ|Δ_w|`, not on |ΔD|; the aggregate is the right thing to test against 110c and the wrong
thing to scope with.** Both arms are WINDOW_SCOPED on the corrected statistic.

### AND ONE DEFECT IN THE EMIT — THE PRE-COMMITTED COLUMN IS NULL

```json
{"window_start": 1788739500, "utc": null, "gap_seconds": null,
 "n_gap_intervals": null, "share_of_day_gap_time": null, "delta_D_cents": 152.4967}
```

**Every row's `utc`, `gap_seconds`, `n_gap_intervals` and `share_of_day_gap_time` are null**,
and `worst_window_utc` is null in consequence. **REVIEW 186 Rule 1 and v12 PART_1 both require
"each window's delta contribution WITH ITS GAP-SECONDS BESIDE IT"** — the column v12 declared
*"so it cannot be chosen later."* **It was not chosen later; it was dropped.** Without it the
Δcents-per-gap-second ratio — my Rule 2 discriminator, the one test that separates
"concentration explained by gap time" from "a finding" — **cannot be computed at all**, and
the reader cannot even name the worst window. `inputs` shows why: the spine came from the
decomposition and the two books, **not from the declaration's pre-committed window rows**.
**One join, and the emit is complete.**

---

## (1) THE SNAPSHOT FROZE **ONE OF FIVE** GROWING INPUTS

**The oracle root is a symlink mirror with exactly one real file.** Measured at
`/home/yuqing/ctaNew-oracle-20260908/data/pm_5min/`:

```
-rw-rw-r--  resolutions.jsonl              <- the ONLY frozen file
lrwxrwxrwx  collector_gaps.jsonl        -> /home/yuqing/ctaNew/...   LIVE
lrwxrwxrwx  collector_health.jsonl      -> ...                        LIVE
lrwxrwxrwx  markets.jsonl               -> ...                        LIVE
lrwxrwxrwx  raw/  derived/  tier1/  tier2/  prices/  ops/  onchain/ -> ... ALL LIVE
```

**Constructive method — the operation is: a path under the data root opened by a module in
the valuation's import closure, whose bytes a live collector appends.** Cross-referencing the
closure (REVIEW 195: `de_forward_value_day` → `de_settlement_control_run`,
`de_settlement_control_aggregate`, `de_forward_evaluator`, `de_multiday_gate1_runner`, and
below them `pm_tape_density`, `flow_intensity`, `de_admissible_windows`) against liveness by
mtime:

| file | read by | last write | frozen? | **enters the values?** | **in the cohort predicate?** |
|---|---|---|---|---|---|
| `resolutions.jsonl` | aggregate, evaluator, runner | 11:03:58 | **YES** ✔ | yes | **yes** (`winner_source`) |
| **`collector_gaps.jsonl`** | **`flow_intensity.gaps_by_slug`**, `pm_tape_density`, runner | 06:03:55 | **NO** | **YES — it decides which windows are gap-bearing, which is the replay** | **NO — no digest field exists for it** |
| **`markets.jsonl`** | `flow_intensity` | **11:05:24** | **NO** | yes (slug metadata) | **NO** |
| **`collector_health.jsonl`** | `pm_tape_density` | **11:07:01** (+732 B in 20 s — continuously) | **NO** | day-quality path | **NO** |
| `raw/<day>/` | the tape readers | **11:07:23** | **NO** | bounded for THIS valuation by `book_sha256`, which IS compared | via the book only |

> **The fix froze the one growing input that happened to be DIGESTED IN THE CELL — and the
> check that caught it cannot catch the others, because they are not digested anywhere.**
> The cohort predicate compares eleven fields (REVIEW 200); **none is a gap-ledger,
> markets or health digest.** A mid-run append to `collector_gaps.jsonl` between two arms
> would change which windows are gap-bearing **and pass the cohort check in silence.**
>
> **`collector_gaps.jsonl` is the one that matters**: it is read by `flow_intensity.gaps_by_slug(era)`
> — the exact function BE 137's whole census is defined against — it enters the replay, and
> **it was appended at 06:03:55 today**, i.e. it is not a frozen file that merely looks live.

**Your phrase is exactly right and now measured: a symlink to a growing file inside the
snapshot root is the same defect wearing a different name, and there are four of them.**
The repair is the one already built: **freeze the same way — real copies of
`collector_gaps.jsonl`, `markets.jsonl`, `collector_health.jsonl` in the snapshot root** —
and **add their digests to the cell**, so the cohort predicate can see them at all.

## (2) THE PROVENANCE RECORD **DOES NOT EXIST**

```
ls data/pm_5min/derived/p003_de_chain_launch_*.json   ->   no match
grep for chain_launch / chain_day / oracle in derived/ ->  only p003_da_oracle_attainability__20260905
```

**There is no `p003_de_chain_launch_<day>.json`, so it carries nothing.** What it must carry
for a reader to reconstruct the oracle state:

1. **The snapshot root's absolute path AND the branch** (`1_env_PM_DATA_ROOT`) **and the
   commit of `be_data_root.py` that resolves `PM_DATA_ROOT`** — the mechanism is new, so the
   code that implements it is part of the state.
2. **The MIRROR LIST, split into two sets**: every path that is a **real frozen copy** (with
   its sha256) and every path that is a **symlink to the live tree** (with the link target).
   **The second list is the one that matters** — it is the honest statement of what was *not*
   frozen, and today it would have ~14 entries against one.
3. **`resolutions.jsonl`'s frozen sha256 and record count**, plus the **per-day slice digest**
   (`7eb54006…`, 2,016) the waiver already computes.
4. **A digest of each live-symlinked input AT LAUNCH**, so a later reader can test whether it
   moved during the run rather than assuming it did not.
5. **`PM_DATA_ROOT` as actually set in the unit's environment**, read back from
   `systemctl show -p Environment`, not from the launcher's intent.

**Without (2) and (4) the snapshot root is provenance-shaped rather than provenance: it
proves one file was frozen and says nothing about the fourteen that were not.**
