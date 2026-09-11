# REVIEW 203 — **(1) THE MECHANISM IS OFF.** `be_heavy_run.sh` hardcodes the live root and passes it explicitly; the refusal drives clean; and neither file list is right — the union is.

**REV 163, 2026-09-11T11:18Z** (clock read separately). Read-only except a scratch-root drive.

---

## (1) **THE VALUATION WILL NOT SEE `PM_DATA_ROOT`. IT WILL SEE THE LIVE ROOT, PASSED EXPLICITLY.**

**Traced end to end at the bytes:**

```
chain_day.sh:68     export PM_DATA_ROOT="$SNAP"          # the snapshot root -- correct
chain_day.sh:150    bash be_heavy_run.sh "$u" de_forward_value_day.py ...   # child bash INHERITS it

be_heavy_run.sh:38  REPO=/home/yuqing/ctaNew             # <-- A BARE LITERAL. Never reads PM_DATA_ROOT.
be_heavy_run.sh:519 exec systemd-run --user --unit="$UNIT" ... \
                      --setenv=PM_DATA_ROOT="$REPO" \    # <-- SETS THE LIVE ROOT IN THE UNIT
```

**Two independent failures, either one fatal:**

1. **`be_heavy_run.sh` never consults the inherited variable.** Line 38 is `REPO=/home/yuqing/ctaNew`, not `REPO="${PM_DATA_ROOT:-/home/yuqing/ctaNew}"`. The chain's export reaches the script's environment and is discarded.
2. **It then OVERWRITES the unit's value with the live root.** `systemd-run --setenv` sets exactly what is named and the unit inherits nothing else — so the valuation process is **explicitly handed `PM_DATA_ROOT=/home/yuqing/ctaNew`.** Even a fixed line 38 would be undone here if `$REPO` stayed a literal.

**Corroborated at the units, not only in the source:**

```
deCHAIN0907 :: Environment=   LoadState=loaded      <- EMPTY
deCHAIN0908 :: Environment=   LoadState=loaded      <- EMPTY
deCHAIN0909 :: Environment=   LoadState=loaded      <- EMPTY
deCHAIN0910 :: Environment=   LoadState=loaded      <- EMPTY
```

**`PM_DATA_ROOT` is in no chain unit's environment.** The record's
`pm_data_root_from_unit_environ: None` is **not a reporting gap — it is the true value**, and
it is the field that should have stopped the arming. **DE built the read-back correctly
(line 96: *"Read `PM_DATA_ROOT` back from the PROCESS, not from this shell"*), got `None`, and
armed anyway.** The instrument worked; its answer was not treated as a refusal.

> **So: the oracle snapshots exist, the five files in them are genuinely frozen, and no
> valuation will read any of them. Every run reads live files, exactly as before the fix.**
> **The one-line repair is `REPO="${PM_DATA_ROOT:-/home/yuqing/ctaNew}"` at line 38** — and
> `be_heavy_run.sh` is **not** a pinned computing module, so this is not a freeze amendment.
> **And the launch record should refuse on `pm_data_root_from_unit_environ is None`**, which
> turns today's silent arming into tomorrow's stop.

## (2) THE REFUSAL DRIVES CLEAN — BOTH DIRECTIONS, AND THE THIRD ARM TOO

Guard block extracted from `chain_day.sh:54–60` and **verified byte-identical to the
launcher's own lines** before driving, on three scratch roots:

```
  ALL_FROZEN     rc=0  ADMITTED
  ONE_SYMLINK    rc=5  REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:markets.jsonl
  MMHF_SYMLINK   rc=5  REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:mm_hf/collector_runs.jsonl
```

**Fires by name on a live symlink, names the offending file, admits a fully frozen root, and
the separately-coded `mm_hf` branch fires too.** Rule 16 satisfied in both directions.
**This guard is sound — and note what it cannot do: it checks the SNAPSHOT's shape, not
whether anything will ever READ the snapshot. §1 is invisible to it.**

## (3) **NEITHER LIST IS RIGHT. THE UNION IS — AND THE CRITERION SETTLES IT.**

**My REVIEW 162 criterion was wrong** and I withdraw it: I ranked by *"did it grow recently"*
(mtime), which dismissed `collector_runs.jsonl` as static because it last changed 2026-08-31.
**The right criterion is DE's in spirit and neither's in practice:**

> **a path READ BY THE VALUATION'S IMPORT CLOSURE that a live process CAN append.**

Measured against that — `pm_tape_density` **is** in the closure (driven: `True`);
`flow_intensity`, `be_era_for_day`, `da_hf_pm_alignment` are **not**:

| file | in the closure? | live? | DE freezes? | verdict |
|---|---|---|---|---|
| `resolutions.jsonl` | yes (aggregate/evaluator/runner) | yes | **yes** | ✔ |
| `collector_gaps.jsonl` | **yes** (`pm_tape_density.load_gaps`) | yes | **yes** | ✔ |
| `markets.jsonl` | not in this closure | yes | yes | conservative, harmless |
| `rewards_registry.jsonl` | not in this closure | yes | yes | conservative, harmless |
| `collector_runs.jsonl` | not in this closure | can grow on a deploy | yes | **DE right, I was wrong to call it static** |
| `mm_hf/collector_runs.jsonl` | not in this closure (`da_hf_pm_alignment`) | can grow | yes | conservative |
| **`collector_health.jsonl`** | **YES — `pm_tape_density` reads it** | **yes: +732 B in 20 s, mtime 11:07:01** | **NO** | **MISSING FROM DE'S LIST** |
| **`raw/<day>/`** | **YES — `pm_tape_density.RAW`** | **yes: new window every 5 min, 11:07:23** | **NO** | **MISSING FROM DE'S LIST** |

**DE's list is broader than mine in five places, all of them good or harmless, and misses
exactly the two that are demonstrably in the valuation's own import closure.** Mine caught
those two and wrongly dropped `collector_runs.jsonl`. **Freeze the union: DE's five plus
`collector_health.jsonl` and `raw/`** — and **add both to `GROWING`, so the §2 refusal covers
them.** A guard whose list omits a live input in the closure is a guard that admits the
defect it was written for.

*(Caveat on `raw/`: for a valuation whose book is already built and digest-matched, the tape
is not re-read for the replay; `pm_tape_density` reads it for density. Freezing it is a
symlink-to-a-directory change and may be more than DE wants for one day — but it should be in
the list with a stated reason if it is deliberately excluded, not absent.)*
