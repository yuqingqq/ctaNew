# BE_PROCEDURE — how this seat actually works

Written by BE at 2026-09-09 (BE 109) because MEM 289 found that DA, DE and
MEM each have a procedure file and BE has none: this seat is the one that
would lose its method on a reset. Everything below is what I do, with the
numbers I measured, not what ought to be written. **This file is mine to
maintain.** When a number here stops being true, correct it here in the same
round you learn it.

Read `SEAT_PROTOCOL.md` first — this file assumes it, and never contradicts
it. On any conflict the register (`COORDINATION.md`) wins.

---

## 0. The first five minutes of a round

```
date -u                                  # never write a time you did not read
git fetch origin -q
bash scripts/wt_refresh.sh /home/yuqing/ctaNew-wt-be
```

`wt_refresh.sh`, never a bare `checkout --detach` (R-625). After it,
`git -C /home/yuqing/ctaNew-wt-be status --short` shows **`?? data` and
nothing else**. That line is the ledger symlink and it is CORRECT — `data/`
is a symlink to `/home/yuqing/ctaNew/data`, the tracked data files carry
skip-worktree, and a materialised `data/` directory would be a partial shell
whose uncommitted artifacts are all missing. Check `readlink -f
<wt>/data`. If you ever see more than that one line, stop and look before
you build.

Set `PM_DATA_ROOT=/home/yuqing/ctaNew` when running batteries by hand. It is
the **repo** root, not the data directory (R-562), and the launcher sets it
for real runs. Several checks resolve the ledger through it, and a bare
`python3 -m …` without it once cost me a round chasing a red that was my
shell (BE 90/91).

---

## 1. The heavy-run launcher

`live/pm_research/be_heavy_run.sh` — every heavy step goes through it.

```
bash live/pm_research/be_heavy_run.sh --poll <unit> <module.py> [args…]
```

**The lock goes INSIDE the unit.** The form is a transient *service* the
systemd manager forks, with `flock` inside it:

```
systemd-run --user --unit=<name> --slice=research.slice \
  -p MemoryMax=8G -p CPUQuota=100% -p RemainAfterExit=yes \
  --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
  -- flock -n -E 75 /home/yuqing/ctaNew/data/.heavy_run.lock <cmd>
```

Never `--scope`: a scope registers processes the *caller* forks, so the run
dies with the launching shell. A 35-minute run was killed that way (R-628).
A service survives its watcher being killed — I have seen it happen and
finish normally.

* **`flock -n -E 75`** — refuse immediately with **75**, the DECLARED
  conflict code, if another heavy run holds the lock. Without `-E` a refusal
  and a payload crash are both `1`. No producer may exit 75 for any other
  reason; each producer declares its exit codes in
  `producer_exit_maps_v<N>.json` and asserts 75 is not among them.
* **Never wait on the lock silently.** A refusal is reported.
* **`research.slice` is capped at `CPUQuotaPerSecUSec=2s`** (200 %, verified
  at the running manager) with `MemoryMax=15032385536`; each unit gets
  `CPUQuota=100%` and `MemoryMax=8589934592`. So a light suite can overlap a
  heavy run, but two heavy runs cannot share the box.
* **`-p RemainAfterExit=yes`** or a succeeded transient unit is COLLECTED at
  exit and `systemctl show` returns DEFAULTS — `LoadState=not-found` with
  `success/0` that are not readings (R-648).

### Reading a unit's outcome

Read **five fields WHILE `LoadState=loaded`**, with the **InvocationID**:
`LoadState ActiveState SubState ExecMainStatus Result` + `InvocationID`.
Under `RemainAfterExit` a finished run is `loaded/active/exited` and a
running one `loaded/active/running`, both `ExecMainStatus=0` — **`SubState`
is the discriminator**. A copy without an InvocationID is VOID. After the
receipt lands and the fields are copied, `stop` the unit (and
`reset-failed` a failed one) so the name is free; a loaded name makes the
next `systemd-run` fail.

**Do not trust `--poll`'s RETURN as completion.** It returns once the lock
is taken and the sampler starts. Wait on the unit itself:

```
while systemctl --user show "$U.service" -p SubState | grep -q running; do sleep 30; done
```

I got this wrong at BE 102 and my own driver ran `systemctl stop` on a live
build 15 s in, killing it. It wrote nothing, but it was my error and it is
the single easiest way to destroy an hour.

**The peak of record is SAMPLED WHILE THE RUN LIVES.** `--capture` cannot
read a released cgroup leaf. `memory.peak` is monotonic, so the largest read
is a high-water mark up to the last read, and its time is recorded so a
bound is never mistaken for the peak. An absent peak is `ABSENT`, never 0
and never systemd's `MemoryPeak` property, which is a different number
(BE 74).

**A peak equal to the cap is a FLOOR, not a measurement.** Read
`memory.events`: `max` counts reclaim events, `oom_kill` counts deaths.
09-04 at L=250 sat at exactly 8,589,934,592 with **`max 775`, `oom_kill 0`** —
it wanted more than 8 GiB and the kernel held it there. The number cannot be
read as its demand.

**The journal is NOT the record.** It rotates within hours; measure the
retention state (the oldest entry it holds) when you quote it, and copy any
line you need into an artifact at the moment you read it, filtered on the
InvocationID with BOTH `_SYSTEMD_INVOCATION_ID` and `USER_INVOCATION_ID`.

---

## 2. The day-book build

```
be_heavy_run.sh --poll <unit> be_daybook_build.py --day <YYYYMMDD> \
    [--placement-latency-ms 250] [--artifact-revision EV20]
```

A book is a pickle with `{fr, asm, header}`:

* **`fr`** — the QR_SKEW_ONLY reference: quotes, generations, tranches,
  fills. Built by `de_phase4_diag_runner.build_reference`. **It is a pure
  function of (tape, spec, placement latency, selector) — no model, head,
  score or threshold reaches it at any depth** (measured by AST
  reachability, BE 106). That is why a scoring correction does not require a
  rebuild from the tape in principle.
* **`asm`** — the per-generation/per-row scores from the heads. This is
  where `load_lgbm`, `generation_scores`, `booster` and the thresholds live.
* **`header`** — BE 101: protocol, day, coin and the placement-latency
  block, so a reader knows which L produced the fills without the receipt.

**Revisioned output (rule 13).** `artifact_paths(day, coin, L, revision)`
gives `be_daybook_<day>_<coin>[__L<N>ms][__<REV>].pkl` and the matching
receipt, and `assert_artifacts_absent` refuses to overwrite. Every landed
book stays as provenance. Never write a variant to the base path.

**The receipt records** the book's path/bytes/sha256 with a readback, the
selection and era, the reference's windows/generations/statuses, the
assembly evidence, the resources (`wall_s`, `peak_rss_gb`, per-stage peaks),
the producing code's import-closure digests and HEAD, and — from BE 101 —
`placement_latency`: the value, its **source** (`the caller` vs
`PLACEMENT_LATENCY_MS_DEFAULT`), and `TRANCHE_BEFORE_PLACEMENT_LATENCY`.
`source` matters: passing nothing records the default, which is how a build
made without the argument stays byte-identical to a pre-parameter one.

**A book that cannot say its L is refused by name.** `placement_latency_of`
returns `RECORDED_IN_THE_HEADER` / `IN_THE_REFERENCE_ONLY` or raises
`BOOK_RECORDS_NO_PLACEMENT_LATENCY`. An unrecorded L and an L of zero are
the same number and opposite facts. The four books built before the
parameter existed are GRANDFATHERED by name in
`be_daybook_structure_v6.json`, computed from their own receipts.

### Measured costs (mine, at L=250 unless noted)

| day | wall | in-process peak | cgroup leaf | tape rows |
|---|---|---|---|---|
| 09-03 | 2,040 s (34 min) | 5.34 GB | 8,430,505,984 (98.1 % of cap) | 544,286 |
| 09-04 | 2,914 s (49 min) | 4.955 GB | pinned at cap, `max 775` | **638,602** ← heaviest |
| 09-05 | 2,588 s (43 min) | 3.574 GB | — | 500,821 |
| 09-06 | 3,561 s (L=0) | 3.766 GB | — | 511,778 ← lightest |
| 09-07 | 2,855 s (L=0) | 4.32 GB | 6,975,811,584 | 593,338 |

Five days serial ≈ **13,959 s ≈ 3.9 h**. **Build time does not fall with L**
— 09-05 took 2,588 s at L=250 against 1,696 s at L=0.

**The leaf tracks TAPE ROWS, not book bytes.** From the one un-thrashed
measurement, ≈ **15,489 B per tape row**; that predicts 09-04 at 9.89 GB
(consistent with it pinning the cap) and 09-07 at 9.19 GB. If a build must
be given more room, propose the value from that arithmetic — **and note that
rules 8 and 20 say caps are never raised (R-174), so raising one needs a
register amendment, not a dispatch.** I have never raised it myself.

---

## 3. Why concurrent builds are ruled out

Two reasons; the second matters more.

**Memory.** Box is 16 cores / 30 GB, ~18–22 GB free. Two heavy days need
18.4–19.8 GB (2 × 09-07's implied 9.19, 2 × 09-04's 9.89) — over the free
pool. Three fit on no day. Only two *light* days fit (2 × 8.43 = 16.9 GB),
with about a gigabyte to spare.

**Attributability, which is the real reason.** One-at-a-time exists so
`peak_rss_gb` and the leaf peak in every receipt mean something. The leaf is
dominated by **page cache** (09-03: 8.43 GB leaf against 5.34 GB
in-process), and two builds share that cache — so concurrency makes every
recorded peak un-attributable and can push both into reclaim, which is
exactly how 09-04 spent 775 events at the ceiling. Concurrency would buy
~2 h across five days and spend the ability to say what a build costs.
R-551 set the rule from the same observation.

DE's runs are minutes where mine are ~40, so **interleave**: build, release,
let DE take the lock for its point estimate, build the next. Announce each
book the moment it lands rather than at the end of a batch — and when
yielding, hold a grace period after a `de*` unit clears before taking the
lock again, or a gap between two of DE's runs reads as "DE finished".

---

## 4. The collectors

Four, all long-lived, none mine to restart without saying so:

```
collect_pm_prices.py                       (pid 1049)
collect_hf.py        live/mm_research/     (pid 30901)   } P-2026-002
collect_hl.py        live/mm_research/     (pid 30902)   }
collect_pm.py --heartbeat-mode control-v4-slow (pid 1108125)
```

Alive check: `pgrep -af "collect_pm_prices|collect_hf|collect_hl|collect_pm\.py"`.
Do this after anything that could have killed a process — I have twice used
a broad `pkill -f` that matched my own shell, and the check is how I proved
nothing else died. **Capture PIDs before killing, never pattern-match at
kill time.** Collector surface changes are coordinator-owned (R-110) and
deploy only at a UTC day boundary with an era stamp.

---

## 5. The null's contract on my surface

`live/pm_research/be_cancel_axis_null.py`. I own the cascade module; DE owns
the runner and `harmful_stateful_policy`.

* `load()` reads the cached book ONCE — one buffer, hashed and unpickled
  from the same object, because two reads can differ.
* `flagged_stream(rows, flagged)` turns a flag set into a score stream;
  `_alloc(by_side, pools)` checks an allocation is realisable and refuses by
  name; `draw_flags(pools, by_side, rng)` draws it.
* `replay(bk, scores, theta)` calls DE's `HSP.replay_policy` and DE's
  `R.received_fills`. `mechanics()` DELEGATES to `R.cancel_mechanics` — this
  module keeps no second copy of the decomposition (Q-BE-271 was the round I
  caught myself doing exactly that).
* `reproduction_gate()` refuses unless the baseline AND both arms reproduce
  their filed numbers. A null is only on the arms' machinery if the arms
  come out of it.

**`bk["rows"]` is BOTH the decision stream and the control's sampling pool.**
Since BE 107 (`272dfb3`) rows are **PER SCORED ROW**, not one per
generation: the old expression walked the reference and looked each
generation up at `float(g["t0"])`, so it was first-row-only and dropped
every generation whose first scored row was not at its start. Two shapes
exist and both are handled — `PER_ROW_SCORES` (values are dicts with `gen`)
and `PER_GENERATION_SCORES` (values are bare floats, every pre-fix book);
the dispatched one-expression version raises `TypeError` on the second.
Exclusions are named and counted: `GENERATION_NOT_SCORED`,
`FIRST_SCORED_ROW_NOT_AT_GENERATION_START`, with the reference/scored
generation counts beside them.

**UNRESOLVED, WITH THE USER:** a per-row `rows` changes what a draw draws,
so the matched action count stops being the arm's cancel count. **No control
is drawn on a corrected book until that is ruled.** BE 107 made `rows`
faithful to the book; it did not re-specify the matching.

---

## 6. The settlement method (BE 99, R-801)

The ruled P&L, per slug and per path, in cents:

* **trades leg** = Σ_sells px×size − Σ_buys px×size. `BUY_UP` is a BUY of
  the UP share (cash out, shares in); `SELL_UP` a sell.
* **residual leg** = net shares at window close × settle, settle = 100 iff
  Up won else 0.
* **total** = the two, which equals Σ sgn×(settle−px)×size — reconciled per
  slug to ≤ 6e-12.

The winner comes from `resolutions.jsonl`'s `winners` dict on records with
`closed: true`. **Verified per window against Chainlink** with
`exp_m6_settlement.py`'s own loader and readers: **`S60(T) >= S60(t0)` agrees
288/288 on both days tested, zero unavailable, zero unresolved.** Its other
three conventions disagree on **10–44 windows a day**, so the convention is
load-bearing and the module scores a grid and chooses none — neither do I.

Two things to keep saying: `resolutions.jsonl` carries `source: "clob"`, so
it is the venue's record of the outcome; what names Chainlink is
`markets.jsonl`'s declared resolution source. And **no null exists under
this valuation** — the draws keep no fills (`NULL_DRAW` rows carry only
`arm, cancels, i, value`), so there is no Z and no p.

**Placement latency is a declared assumption, not a measurement.** 250 ms
was chosen as the arms' own cancel latency. The data bounds only the inbound
leg (venue→us p50 45–51 ms, p95 424–672, p99 ~1.8 s; ~355 book updates/s);
us→venue is unbounded because no owned-order acknowledgement exists anywhere
on disk. A symmetric round trip would be ~90 ms median — which brackets 250
rather than confirming it.

---

## 7. What I have learned the hard way

* **`?? data` is correct**; anything more is not. See §0.
* **A stale cache can make a green battery red for a reason unrelated to
  what it tests.** `be_cancel_axis_null`'s battery cannot complete at the
  tip: DE's `ASSEMBLY_PREDATES_CAUSAL_SCORING` refuses
  `de_section81_cache_12.pkl`, which predates causal scoring. I proved it
  pre-existing by running HEAD's bytes with my change absent. **Always
  measure whether a failure is yours before reporting it as yours — and
  before assuming it isn't.**
* **Derived files drift the moment a new artifact class appears.**
  `be_heavy_peaks.jsonl` lost `be64book` from its roll-up as soon as an
  L-variant record existed for the same (day, stage), because the derivation
  took the record branch and never emitted the receipt-only row — the drift
  REV 89 found, returning through a door my own L=250 build opened. Fixed at
  BE 103. **When you add an artifact shape, re-run every derivation that
  keys on the old one.**
* **A landing that touches a pinned surface re-pins in the SAME round.**
  BE 101 added the `header` key and did not supersede
  `be_daybook_structure`; the 09-07 structure step refused the book by name
  and was right to. That cost a round.
* **A label must not assert what it does not compute.** One of my cells said
  "zero violations" as a literal beside a computed test.
* **A count literal pinned to a moving thing is a guard, not a bug** —
  `EXPECTED_CHECKS` has caught a cell of mine that silently did not run
  because I inserted it after the count guard.
* **Verify at the artifact, not at the report** — including another seat's
  report about my own module. BE 107's dispatched one-line change was right
  about the defect and would have raised `TypeError` on every existing book.
* **Never `pkill -f` on a shared box.** Twice it matched my own shell.
* **Land the register row with `--row`**: write the row to a file, then
  `scripts/land_register_row.sh --row <rowfile> '<id>' <msgfile>`; dry-run
  first. Never hand-edit `COORDINATION.md`.
