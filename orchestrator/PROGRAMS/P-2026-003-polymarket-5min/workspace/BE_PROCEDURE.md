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

**RULED 2026-09-09 (R-837), was UNRESOLVED:** the null's sampling unit is
**B -- match on CANCELS**. Under the ruled first-crossing rule a generation
yields at most one cancel, so the cancel IS the action and B is the only
option matching on the decision variable (rule 7); A (rows) is biased in the
arm's favour because 39.7 % of rows begin after their generation's start.
The accepted cost is seed reproducibility, replaced rather than absorbed:
DE 160 persists the DRAWN CONTROL SET as an artifact. theta is NOT re-fitted,
by ruling. BE 107 made `rows` faithful to the book; the matching is now
specified elsewhere.

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

## 6b. WHICH OF MY FILES ARE PINNED (read this before editing one)

Learned at BE 112, and it is the fact that decides the SHAPE of a fix on
this seat, not just its risk. `de_multiday_gate1_params` pins BE's cascade
in TWO places (`be_module` = the entry point, `be_cascade.modules` = all
ten), and `de_multiday_gate1_design` pins the params file BY DIGEST. So
touching a pinned module costs a params version AND a design version, and
until both land **no day can run** -- that is R-835's blocker, three pin
pairs in one day.

**PINNED (the params HEAD's cascade -- v23 at BE 112, v24 at BE 113; read
the head, never a version literal):**
`be_cancel_axis_null.py` (the ENTRY POINT), `be_data_root.py`,
`de_head_scoring.py`, `de_matched_random_control.py`,
`de_phase4_diag_runner.py`, `de_rho_estimator.py`, `de_score_stream.py`,
`harmful_stateful_policy.py`, `phase4_generation_tables.py`,
`pm_tape_density.py`.

**NOT PINNED, so free to edit:** `be_daybook_build.py`,
`be_generation_count_derivation.py`, `be_rule22.py`, `be_score_coverage.py`,
the gate1 builders, `be_heavy_peaks.py`, `be_race_feed_pins.py`.

**THERE IS A SECOND PIN SURFACE AND IT IS NOT THE CASCADE (BE 113).**
`de_phase4_diag_runner.pin_statuses` walks `fit_manifest.json`'s
`fit_code_files` -- **twelve files, and BOTH `harmful_exposure_rows.py` and
`flow_intensity.py` are in it** -- comparing each against the FIT-COMMIT
bytes. A file the runner imports whose CALLED function moved, and which
nobody declared additive in `DECLARED_ADDITIVE`, is **BLOCKING**:
`verify_called_code()` raises and no day runs. Three functions of
`harmful_exposure_rows.py` carry declarations (`select_v2_era`,
`_era_or_refuse`, `_refuse_empty_selection`), each with a REASON that is a
statement about what the function does -- so editing one both risks the
block and falsifies its declaration.

**The census, run at BE 113:** `flow_fill_development.py`,
**`flow_intensity.py`**, `harmful_action_eval.py`,
`harmful_candidate_manifest.py`, **`harmful_exposure_rows.py`**,
`harmful_fast_compute.py`, `harmful_hazard_model.py`,
`harmful_state_features.py`, `phase2_arms.py`, `phase2_declaration.py`,
`phase2_embargo.py`, `phase2_state_schema_freeze.py`. None of `be_*.py` is
in it, which is why a new BE-owned module is always the cheap way in.

**`producer_exit_maps` is NOT a source pin.** It declares each producer's
EXIT-CODE MAP (`be_daybook_build`: 0/1/2, 75 never used) and carries a
`block_sha256` of that block, not of the file. Editing a producer needs no
re-pin unless its exit codes change.

**The consequence for a shared fix:** when a defect sits in a pinned module
AND an unpinned one, do NOT convert the pinned module to import the shared
implementation. Put the shared implementation in a new unpinned module, let
the unpinned sites import it, and DRIVE the pinned module's own copy against
it in the shared module's falsifier -- two implementations that meet in a
cell do not drift, and no pin moves. That is what `be_score_coverage.py`'s
seam cell does against `be_cancel_axis_null.load()`.

**Check before you edit, not after:**
`be_rule22.assert_pin_sites_agree(json.load(open(<params head>))["doc"],
root=<tree>)` recomputes all ten against disk and refuses by name.

## 6d. THE DAY'S ERA IS A PROPERTY OF THE DAY (BE 113)

`flow_intensity.ERA` is a **literal**, `clob_v3_1`, and
`harmful_exposure_rows._era_or_refuse(fi, None, …)` returns it whatever day
it is asked about. Its era closed **2026-08-30T05:30:01Z**.

**Measured, over every collected day:** 08-22..25 and 08-27..29 resolve
`clob_v3_1`; **every September day resolves `clob_v4_1`**; 08-19/20/21/26/30/31
resolve nothing (collector outages and era transitions). So the literal was
right for the days it was written for and wrong for every day in the queue --
which is how a literal survives review and then goes stale under a running
collector.

**What it cost:** `gaps_by_slug("clob_v3_1")` and `gaps_by_slug("clob_v4_1")`
are **disjoint** (1,143 slugs against 728, zero in common), so
`gaps.get(slug, [])` was `[]` for every September window. On 09-03, **160 of
247 windows and 2,294.7 s of tape** were assembled as if continuous, and all
twelve landed book receipts record `selection.era: clob_v3_1`.

**Use `be_era_for_day.resolve(fi, day, slugs)`** in any day-scoped selector.
It has no default and refuses by name: `NO_ERA_SPANS_IN_THE_LEDGER`,
`NO_WINDOWS`, `WINDOW_START_UNPARSEABLE`, `WINDOWS_IN_NO_ERA`,
`DAY_STRADDLES_ERA_BOUNDARY`, `WINDOW_IN_MORE_THAN_ONE_ERA`. Pass its answer
INTO `_era_or_refuse` rather than editing that function -- see §6b for why.

**And `float("inf")` is not JSON.** `fi._eras()` gives a live era an `inf`
end and `json.dumps` writes `Infinity`, which strict readers reject. Any
receipt field taken from `_eras()` carries `None` plus a field saying it is
open-ended.

## 6e. THE 09-03 WINDOW COUNT HAS THREE VALUES AND ONLY TWO ARE WINDOWS

Reconciled at the code, BE 113, after MEM 294 filed it as open:

| value | where | what it is |
|---:|---|---|
| **287** | `AW.supply(...)["counts"]["btc"]["n_present"]` | windows present in the ledger for the day |
| **247** | same block's `n_supplied` = 287 − 40 masked | **the day's window count**; `day_slugs` returns it and `reference.windows` records it |
| **246** | `economic_settlement.arm_legs.n_slugs` | `len(per)` over **FILLS** in `settlement_legs_by_slug` -- slugs with ≥1 valued fill in ONE replay |

**246 is not a window count.** It equals the supplied count on 09-04/05/06
(288 each, measured at DE's point-estimate artifacts) and is one short on
09-03 only, where exactly one supplied window produced no valued fill. It
moves with the arm, the latency and the policy. Quote **247**.

The 287→247 half now travels in the receipt (`selection.mask`) with its
arithmetic CHECKED, and the 246 claim is a battery cell, not prose.

## 6c. The shared-falsifier convention

Every importer runs a shared module's own falsifier as ONE cell of its
battery, spawned as a SUBPROCESS, through `be_rule22.shared_falsifier(prog=…)`
(REV 84 §3.2). The helper reads the summary line and requires it to END with
`0 failures`, so **a shared module must print `<n> cells, <k> failures`** --
`"26 checks passed"` makes the cell report `ok=False` forever, which is a
cell that can only fail. `EXPECTED_CHECKS` in the shared module must count
the cells the failure branch also prints, or the count guard fires.

## 6f. RULE 28 ON THIS SURFACE (BE 114)

*The pipeline records the right thing and leaves the check that would make it
load-bearing switched off.* `be_rule28_sweep.py --sweep` is the census;
`--falsify` proves it can fire. **It grades PRESENCE, not IDENTITY** -- what
happens when a caller passes nothing -- so a clean `on_omission` column is
not a clean surface: two of BE 114's three fixes were about whether the value
PASSED was the right one, and the census classifies both as SUBSTITUTES,
correctly and unhelpfully.

**Three that were real, all mine, two of them one round old:**

1. **A supply must name its own day.** `day_slugs(supply=...)` and
   `mask_block(sup, …)` read only `windows`/`counts`; the supply carries
   `day`. Harmless while `supply=` was falsifier-only; **BE 113 made it a
   production argument.** Now `SUPPLY_IS_FOR_A_DIFFERENT_DAY`,
   `SUPPLY_DOES_NOT_NAME_ITS_DAY`, `MASK_SUPPLY_IS_FOR_A_DIFFERENT_DAY`.
2. **`if x is False` is not `if not x`.** `mask_block` refused only on
   `closes is False`, so a supply MISSING the counts gave `closes = None` and
   emitted three nulls beside `arithmetic_closes: null`. Now
   `MASK_COUNTS_ABSENT`.
3. **THE SELECTOR'S SECOND RETURN VALUE IS A STATUS FIELD.**
   `build_reference` does `selected, n_bn_gap = selector(...)` ->
   `statuses["BINANCE_GAP_EXCLUDED"]`; `build_rows` -> the fragment's
   `windows_excluded_binance_gap`. `select_v2_era` MEASURES it; both of my
   day selectors returned a hardcoded `0`. **Measured 2026-09-09: 3 of
   09-03's 247 windows fail `binance_continuity_ok`** (three real Binance
   gaps, 409 s to compute), so the zero is not harmless by coincidence. The
   zero stays -- changing the population is not this seat's call -- and a
   DISCLOSURE travels beside it so a reader can tell a selector property from
   a measurement.

**When you add a producer that returns evidence, add the consumer's refusal
in the same change, and default the safe call, not the short one.**

## 6g. THE GAPS ARE AN INPUT TO `fr`, AND THE RECEIPT CANNOT SHOW IT (BE 116)

Asked whether a rebuilt `fr` could be byte-identical to the landed L=250
books'. **It cannot**, and the chain is worth keeping because it needs no book
load:

* the selector's 5th element goes straight into the replay --
  `HER.replay_with_recorder(ent[1], ent[2], ent[3], ent[4], spec)`,
  `de_phase4_diag_runner.py:571`;
* what that replay produces is stored IN `fr` -- the generation `t1` at
  `:649`, the terminal mark's `ended_in_gap` at `:675`;
* the landed books used `gaps=[]` for every window (all twelve receipts say
  `selection.era: clob_v3_1`, and **0** of 09-03's 247 windows are gapped
  under that era against **160** under `clob_v4_1`);
* driven on four spread-sampled gapped windows: **`n_gens` identical,
  `sum(t1)` moves by −12.1 to −32.1 s.** THE GAPS TRUNCATE GENERATION
  LIFETIMES -- same population, different bounds.

**THE TRAP, and it is the part to remember: no receipt count shows this.**
`n_gens`, `n_fills` and `n_tranches` are unchanged, and
`TERMINAL_MARK_ENDED_IN_GAP` -- the status one would reach for -- is already
63/247 with `gaps=[]` and flipped in **0 of 10** sampled gapped windows (only
the day's widest gap, 111.3 s, flips). **A corrected receipt could match the
landed one on every number in its `reference` block while the reference
underneath differs.**

**And do not load a landed book to answer a question like this.** A day book
is ~290 MB on disk; unpickling it exceeds the 1 GiB that makes a step HEAVY by
rule 20's own definition and would need the lock.

## 6h. A STATUS ADDED TO A COUNTS DICT IS A SIBLING KEY, NEVER A REPLACEMENT

DA 147 asked for a STATUS where `statuses["BINANCE_GAP_EXCLUDED"]` published a
hardcoded `0`. **The literal form breaks another seat's verifier:**
`da_book_verify.py:742` computes `sum(st.get(k, 0) for k in excl)` over that
exact key, so a string there raises `TypeError` -- and that file is DA's, which
this seat reads and never edits (R-235). So the count keeps its type and
`BINANCE_GAP_EXCLUDED_STATUS` sits beside it in the same block, with a
known-bad in the battery driving BOTH directions (the merged block still sums;
the string-in-the-count-slot raises). **Before changing the TYPE of anything in
a shared dict, grep for who sums it.**

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
* **A landing race is normal and the outcome is read at ORIGIN, not at the
  local sha.** BE 112's copy-land committed onto a tip that moved during the
  commit; `push` was refused non-fast-forward, the commit was stranded
  (rule 21: LEAVE it, report it, never rebase it yourself), and it was
  rebased and pushed within a minute -- **with a NEW sha**. So verify a
  landing by `git fetch` then a BYTE COMPARE of `git show <origin sha>:<path>`
  against the worktree's file, never by the local commit id you remember.
  Run the pre-copy guard INSIDE the act
  (`git -C <shared> diff --name-only <wt HEAD>..HEAD -- <paths>` EMPTY);
  an afterwards-diff is how you learn you were lucky.
* **DA reads my receipt's coverage block, and two of its predicates assume
  the PRE-FIX shape.** `da_book_verify.py:683` flags
  `n_scored_keys_equals_n_covered` and `:906` recomputes coverage as
  `len(keys)/n_gen`. Both are TRUE only under `PER_GENERATION_SCORES`; on a
  per-row book `len(keys)` is a ROW count. DA's surface, R-235 (read, never
  edit) -- but a receipt-field change of mine lands in DA's verifier, so say
  so in the same round.
* **Land the register row with `--row`**: write the row to a file, then
  `scripts/land_register_row.sh --row <rowfile> '<id>' <msgfile>`; dry-run
  first. Never hand-edit `COORDINATION.md`.
