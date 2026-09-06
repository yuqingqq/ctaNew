# HANDOFF — P-2026-002 HF Market Making

Updated: 2026-09-06T06:24Z (DA seat). **THE ADMISSION LEG IS RE-DECLARED AS
AN OUTAGE DETECTOR (declaration v6).** DA 61's census8 showed the inherited
gap-fraction leg selects on *how often the best quote changes* — exclusion
monotone in activity, ICP cut from 16 structurally complete days to 1 — so
E2-A was dropping precisely the thin cell it exists to resolve. Admissibility
is now a property of **the collector being live**; book activity is REPORTED,
never gated. Nothing has been read for a gate under v6: the reviewer files on
v6 (REV 38) before the smoke re-runs, and **the ICP smoke's reading stays
SEALED.** Read this section, then E2.0's below it.

## v6 — admission is the COLLECTOR, not the book

**The leg v1–v5 carried** gated a day on the intra-day bookTicker gap
fraction, inherited from E2.0 where it guarded against collector *outage*.
Measured over eight symbols it selects on activity:

| | DOGE | BNB | ADA | FIL | LTC | AVAX | AAVE | ICP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| admissible (v5) | 16 | 16 | 16 | 14 | 14 | 13 | 11 | **1** |
| median decision-time age | 68 | 102 | 116 | 169 | 194 | 222 | 261 | **517 ms** |

**v6's predicate** — (a) 24 hour-files on all three streams, unchanged; and
(b) **the collector was live**: no gap between consecutive heartbeats beyond
2× the collector's own *measured* modal cadence, and no restart inside the day.

**The bar is not chosen on this data.** The collector emits a heartbeat line
per minute; its cadence is measured as the modal inter-heartbeat interval
(60 s) and the bar is 2×. Measured separation on the real ledger:

| day | max heartbeat gap | reading |
|---|---:|---|
| every clean day 08-20…09-05 | **61 s** | one cadence — live |
| 2026-08-24 | **158 s** | the hf_ws_v2 era-boundary restart |
| 2026-08-26 | **4,656 s** | the reboot |

The bar sits between them by construction; the tightest separation is 2.6×.
*And the leg independently rediscovers the era boundary* — 08-24 is the day
CLAUDE.md rule 5 already marks as the sub-second stamp boundary.

**REPORTED, never gated:** the gap fraction, the gap run-length profile, and
decision-time quote age per symbol-day, under `REPORTED_not_gated` — so a
reader still sees the staleness a thin name carries into placement.

**What measuring it consumed.** The *decision* to replace the leg was informed
by seeing which days v5 excluded, so **eight symbols over 08-20…09-05 are
CONSUMED for any further re-choice of an admission predicate** (rule 11). The
new leg itself is not tuned on them: different quantity, bar from the
collector's own cadence. A *third* predicate proposed on this evidence would
need days not yet examined.

## The partial-fill bracket now fires (rule 16, self-reported in DA 61)

At `partial_share = 0.000` the two R-570(C)(2) pricings coincided and the
straddle rule could not fire. The battery now carries an episode built to be
partial — queue 100, order 10, 104 units through → **filled 4 of 10** — on
which the pricings give **eff_RT 6.0 vs 10.0** and the straddle rule **fires**,
their mean 8.0 being exactly the pass it exists to refuse. The R-570(B)
ordering falsifier is split in the battery as it already was in prose:
**quantity per episode, cost at the aggregate gate row.**

## `e1_markout_scan.py` — the resolver adopted, ruled a portability change

`REPO = Path(__file__).parents[2]` → `de_data_root.resolve()`, **imported not
copied**, with a fall back to the code tree. Closes the gap filed twice: from a
per-seat worktree `day_files()` returned `[]` and `tick_size()` raised on an
empty argmax. **Falsifier `--e1-resolver-parity`**, both halves driven every
run — *parity* where the old resolution was already right (same tree ⇒ the
twelve symbols' `tick_size` and file counts must be identical), and *the fix*
where it was wrong (worktree ⇒ old root zero files, new root the real count).
A parity check that could only ever pass is rule 16's shape, which is why the
second half is there.

## Rule 20 finally has an instrument

`live/pm_research/heavy_slice_audit.py` — lists every transient scope in
`research.slice` with RSS, wall and command; names the lock holder; **REFUSES
with rc 2, offenders named**, when a scope over 1 GiB or 60 s runs without the
lock held by *its own process tree*. It keys on the **slice**, because the
calling shell is itself in a transient scope and "is there a scope" is true
either way. **Its first live run found 3 unlocked heavy scopes.**

> **And its own selftest passed its known-bad for the wrong reason.** The first
> version asserted on the audit's global `refused` flag and fired on *another
> seat's* real heavy run that happened to be in the slice, then failed its
> positive control for the same reason. Each planted scope now carries an
> explicit `--unit` and every assertion names it. A second defect in the same
> test: terminating the `systemd-run` **client** does not stop the **scope**,
> so the planted scope survived teardown and held the lock through two later
> controls.

## READ FIRST — E2-A, 2026-09-06

### What was built

`live/mm_research/e2_a_runner.py`, against declaration **v5**
(`p002_e2_a_declaration_v5.json`, sha256 `90a9a99f6cb2a2cc`, carrying
`8e6b753`; v1–v4 untouched, superseded in band).

| checklist item (REVIEW_DA60 §3) | state |
|---|---|
| real-book placement from bookTicker at t0⁻ | built — the touch and the decision mid, integer tick indices everywhere |
| depth20 queue-ahead at placement | built — `p@q\|…`×20 parsed by byte-translation into a flat 84-column CSV |
| both fill sims wired to episodes, ordering predicate computed | built |
| partial fills, `filled_qty` a quantity | built, both pricings |
| chase leg at the taker fee + realised drift | built — crosses the **real** touch at T_p; `ES_day` is gone from E2-A entirely |
| falsifiers both directions | 24, all green, 0 tape paths |
| E1-A reproduction control runs FIRST and gates | built — and it **passed and gated** the smoke |
| `require_canonical` on every emission | inherited |
| one symbol first with resources | ICP: 33.55 s, 1,124 MiB |

### The smoke: REFUSED on the population — and there is no reading to seal

`p002_e2a_smoke_ICPUSDT__20260906T055050Z.json` (sha256 `ad722557872b837a`,
carrying `b5bf558`). The inherited control ran first and reproduced E1-A
exactly — sweep **6.264472967929728**, touch **3.4484893715577347**, matching
`e1a_gate_summary.csv` row `tp_s=600` to full printed precision — so E2-A is
superseding E1-A's estimator and not a different one.

Then: **1 admissible day against the declared minimum of 14.** All 16 complete
days carry 24 hour-files on **all three** streams, so the depth20 requirement
is met everywhere. What excludes 15 of them is the bookTicker gap-fraction leg
(0.109–0.219 against a 0.05 bar). **No `eff_RT` exists in that receipt, no
overlay verdict, nothing to seal — the gate was never reached.**

### Why, measured — and the control that makes it readable

`p002_e2a_census__20260906T055752Z.json`.

| | ADAUSDT | ICPUSDT |
|---|---:|---:|
| complete days | 16 | 16 |
| **admissible** | **16** | **1** |
| quotes/day | 1.55–5.83 M | 0.64–1.67 M |
| gap fraction | 0.0003–0.0030 | 0.0378–0.2190 |
| gap runs/day | 12–249 | 2,853–11,997 |
| median gap run | 1 s | 1 s |
| missing seconds in runs ≥ 60 s | 0.0000 (one day 0.2510) | **0.0000** on 15 of 16 |
| **quote age at the 24 decision times, p50** | **60–230 ms** | **206–902 ms** |
| p90 | 252–326 ms | **940–3,299 ms** |
| max | ≤ 827 ms | up to **7,185 ms** |

**ADA reproduces E2.0's own admissible set exactly**, so the predicate is not
broken as code. **ICP's missing seconds are one-second holes, not outages** —
zero of them sit in runs of a minute or more on 15 of 16 days. The gap leg,
inherited from E2.0 where it guarded against *collector outage*, is measuring
*quietness* on a thinner symbol.

**But it is pointing at something real.** E2-A places at the touch from the
last bookTicker strictly before t0. On ICP that quote is a half-second to a
second old at the median and up to 3.3 s old at p90 — 3–10× ADA. *That is a
limit on what "place at the real touch" means for a thin name, and no
threshold change removes it.*

### The population, across 8 symbols — E2-A has one, and it is the ACTIVE names

`p002_e2a_census8__20260906T055936Z.json` (sha256 `9fa3e218a929ab8a`). Every
symbol whose bookTicker is under 2 GB; **the cut is by measured input size,
declared before the run, not by outcome.** SOL 3.6, XRP 3.2, BTC 8.6, ETH
8.9 GB are queued with their sizes.

| symbol | complete | **admissible** | excl. by gap | median decision-time age p50 |
|---|---:|---:|---:|---:|
| DOGE | 16 | **16** | 0 | 68 ms |
| BNB | 16 | **16** | 0 | 102 ms |
| ADA | 16 | **16** | 0 | 116 ms |
| FIL | 16 | **14** | 2 | 169 ms |
| LTC | 16 | **14** | 2 | 194 ms |
| AVAX | 16 | 13 | 3 | 222 ms |
| AAVE | 16 | 11 | 5 | 261 ms |
| ICP | 16 | **1** | 15 | 517 ms |

Against the declared 14-day minimum: **DOGE, BNB, ADA, FIL, LTC clear; AVAX,
AAVE, ICP do not.** *The exclusion is monotone in activity* — the two-symbol
finding confirmed across eight. **E2-A has a population, and it is the
population of active names; the thin ones it was built to resolve are the ones
it drops.** All eight over 08-20..09-05 are now CONSUMED for any re-choice of
the predicate.

### The real-book path EXECUTES — mechanism check, all costs redacted

`p002_e2a_mechanism_ICPUSDT__20260906T060417Z.json` (sha256
`29c04cf58f9524f5`). ICP's single admissible day, 15.61 s / 1,165 MiB. **No
`eff_RT`, no verdict, nothing to seal** — it answers one question: does the
path meet a real tape without breaking?

- 144 attempted, **138 RESOLVED**, 6 `NO_QUOTE_BEFORE_T0` (hour 0, both
  directions, all three T_p — no quote exists strictly before 00:00:00 in a
  day-scoped read; counted, not dropped)
- **`QUEUE_AHEAD_UNDEFINED`: 0.** The depth20 top-20 carried the bookTicker
  touch at *every* placement — so "depth-aware" is a property here, and one of
  the declaration's five named design-refuters does **not** fire.
- **Ordering violations: 0**, per episode, on quantity. Aggregate fill rate
  0.935 RiskAverse vs 0.978 ProbQueue-f3.
- depth20: **499,175 snapshots, 0 ragged.** Trades: **402 zero-quantity prints
  excluded and counted** — E2.0's collected-tape defect is on ICP too.
- queue-ahead at placement: p50 683.5, p90 2,021.5, max 2,717, **zero zeros**.

**One honest consequence, because it makes a declared rule vacuous.**
`partial_share = 0.000`. With q = one lot against a queue of ~700, RiskAverse
fills fully or not at all, so φ ∈ {0,1} and **the two partial-fill pricings of
R-570(C)(2) coincide — the straddle rule cannot fire on the min-size arm.** It
is implemented and controlled both ways in the fixture, but on the arm that
*runs* it is guarding nothing; it binds only when order size is comparable to
the volume arriving in the window, i.e. the **size-aware** arm, which refuses
for want of a declared rebalance notional. Rule 16's shape, reported rather
than left looking like a guard.

### THE RULING NEEDED — and what I have consumed by measuring

**The declared gap leg excludes the symbols E2-A exists to resolve.** E1-A
already left ICP UNRESOLVED at 72% episode skips; E2-A now cannot admit its
days. The declaration named this class itself: *"a systematically empty
admissible-day set once all three streams are required — that refutes the
population predicate."*

Three readings, none of them chosen here:

1. **Keep the bar.** E2-A resolves the queue question only for the active
   names and reports the thin ones UNRESOLVED on the population leg. Honest,
   and it leaves the ICP cell exactly where E1-A left it.
2. **Replace the gap leg with an OUTAGE predicate** (max contiguous run, or
   the share of missing seconds in runs ≥ 60 s) — which is what the leg was
   always meant to measure.
3. **Admit on decision-time quote age**, the quantity E2-A actually rests on.

**2 and 3 are threshold choices made after seeing, so I cannot make them.**
2026-08-20..09-05 on **ADA and ICP** is now **CONSUMED** for any re-choice of
this predicate (rule 11): I have seen both profiles. A new predicate declared
on this evidence must be validated on symbols or days not yet examined, and
that constraint travels with whichever option is picked.

### Two defects the runner's own fixture caught before any tape was opened

1. **A short depth20 row is NaN-PADDED by the CSV reader, not rejected.**
   `on_bad_lines="skip"` only catches rows with *more* fields than names. Left
   alone it would have entered the queue simulation as a book with zero-size
   levels — an **invented queue position**, the same defect class as reading an
   *absent* level as an *empty* one. Now excluded and counted both ways.
2. **The quantity-step estimator.** v4 declared q as "the same function E1-A
   uses for the price tick". Driven, that is wrong — see below.

A third was caught by a control rather than a fixture: a hand-typed fixture
day-boundary constant was **four days out**. The runner now derives `day0` from
the day *string*, never from the first row read.

### R-570(D), the record defect — at the MECHANISM, and my first note was wrong

`p002_e2a_tick_diagnosis__20260906T054915Z.json`:

| symbol | distinct prices | modal diff (the FIX) | frac. integer-multiple | fallback fires? | `tick_size()` returns |
|---|---:|---:|---:|---|---:|
| **FILUSDT** | 1,371 | **1e-4** | **0.909489** | **YES** (< 0.999) | **1e-6** |
| ADAUSDT | 586 | 1e-4 | 1.000000 | no | 1e-4 |

**The mode-of-diffs fix IS present and DOES produce the 1e-4 the record
claims.** What returns 1e-6 is the **retained GCD fallback**, firing because
only 90.9% of FIL's price diffs are integer multiples of the modal one — the 81
off-grid prints the audit itself named. *The fallback supersedes the fix on
exactly the input the fix was written for.* ADA is the other direction: the
diagnosis can fail to fire.

**This supersedes my own first note (commit `0718fea`)**, which followed the
reviewer's proposed wording *"fix DESIGNED post-audit and NOT LANDED"*. That is
not right. Two independent implementations returning 1e-6 established the
**value**; only executing the intermediates established the **cause** — and the
difference matters, because "not landed" points a reader at a missing edit that
is in fact present. E1-A's operative CSV pair is untouched; the fallback is not
removed.

### Open items routed, not absorbed

- **`e1_markout_scan.py` has NO data-root resolver.** From a worktree
  `day_files()` returns `[]` and `tick_size()` raises on an empty argmax — the
  E2.0 result review's §6 gap, still open in the module that **produced E1's
  published numbers**. The reviewer hit it independently. Not edited: it is
  E1's producing code.
- **THE ORPHANED COMMIT — my DA-61 account of it was WRONG, and the reflog
  says so.** I reported that "the main tree's HEAD moved back to an earlier
  commit". It did not. Both reflogs, read at 06:11Z:

  | | |
  |---|---|
  | shared tree `HEAD` | 30ddbf3 → 13e978d → 26c4c08 → 91da6f2 — **monotone; it never moved backwards, and `11cf710` never appears in it at all** |
  | `~/ctaNew-wt-da` `HEAD` | 05:52:44 `checkout … to mm-research` (30ddbf3) → **05:56:52 `commit: E2-A census: report QUOTE AGE …` = `11cf710`** → 05:56:53 `checkout … to mm-research` |

  **The commit was made in MY OWN detached worktree.** The mechanism is two
  independent failures composing:

  1. **A bare `git` command inherits a `cd` from earlier in the same compound
     shell command.** That call began `cd ~/ctaNew-wt-da && …`, so the
     `git commit` at the end of the same line ran in the *worktree*, on its
     detached HEAD, not in the shared tree.
  2. **`git push origin mm-research` from a detached worktree pushes the
     BRANCH, not your HEAD.** The branch had just advanced to another seat's
     commit (26c4c08, 05:56:14), so git pushed *that*, printed
     `13e978d..26c4c08`, and exited 0. My commit was never referenced.

  **Rule 21 as drafted forbids a mechanism that did not occur here.** No
  `checkout`/`reset`/`rebase` in the shared tree was involved — the shared-tree
  detachment was the *other* incident, the backtick expansion of R-567(B). What
  prevents *this* one is rule 21's **positive** form, which the rule already
  states: always `git -C /home/yuqing/ctaNew …`, never a bare `git` whose tree
  depends on the shell's cwd. What it does **not** yet cover is (2): a push that
  reports success for someone else's commit. **The missing step is verifying
  that the pushed tip is your commit** — `git -C … rev-parse HEAD` equal to the
  remote's new tip — which no rule currently asks for.

---

## READ FIRST — E2.0, 2026-09-06

### The verdict: `NOT_VOIDED+DEAD`, and the two legs say different things

**ADAUSDT, 16 admissible UTC days (2026-08-20..09-05), τ\* = 30 s (16 of 16
days fast, threshold 13), 2,077,916 sweeps.** Receipt
`data/mm_hf/e1/p002_e2_0_smoke_ADAUSDT__20260906T044455Z.json`
(sha256 `6942eb5ff574a42e`, carrying_commit `2cf2029`), declaration v2
`054ff4e7536291ee`.

| leg | quantity | threshold | result |
|---|---:|---:|---|
| **VOID** | Δ_rs(τ\*) on the intersection, eq-weighted | > +1.0 bps voids | **−0.0066 bps → does NOT void** |
| **SETTLE** | notional-weighted rs(τ\*) on true mids | < 1.8 bps kills | **−0.5552 bps → DEAD** |

**The proxy was fine.** Δ_rs is seven thousandths of a basis point: E1's
two-sided last-print mid agrees with the real book. *E1 measured correctly, and
the +2.44 was never a mid artifact.* **ADA dies on the WEIGHTING**, exactly as
the results audit predicted — now confirmed against real bookTicker rather than
inferred from prints.

**The kill is interval-robust.** CI95 `[−1.86, +0.12]`, entirely below the
1.8 bps fee. All four readable §1.5 gates fail:

| gate | bar | value | |
|---|---:|---:|---|
| 1 point ≥ fee + c_safe | 2.3 | −0.5552 | FAIL |
| 2 CI-lo ≥ fee | 1.8 | −1.8612 | FAIL |
| 3 positive on ≥ 70% of days | 12 of 16 | 6 of 16 | FAIL |
| 4 both sides > 0 | — | maker-bid −1.1462 / maker-ask +0.0834 | FAIL |

### The mechanism, monotone and per-dollar

Eq-weighted **+1.3639** against notional **−0.5552** — a **1.92 bps** weighting
gap (E1 measured 2.76 on its own window). Day-clustered rs by **notional
quintile**:

| quintile | n | notional | rs (notional-w) |
|---|---:|---:|---:|
| 0 smallest | 415,433 | $2.7 M | **+2.29** |
| 1 | 415,727 | $9.2 M | +1.98 |
| 2 | 415,577 | $39.4 M | +1.73 |
| 3 | 415,584 | $212 M | +0.98 |
| 4 largest | 415,593 | **$2,951 M** | **−0.69** |

**The top quintile carries 92% of the $3.21 bn.** Small prints earn the
half-tick; the dollars are adversely selected. Even the smallest quintile only
just clears the 2.3 gate, and it is 0.08% of the notional.

### What this does NOT settle — stated in the declaration, not discovered after

- **The window is 2026-08-20..09-05, not E1's 2026-07-18..08-17.** The two do
  not intersect (the collector started 08-19), so Δ_rs was computed with BOTH
  mids on the SAME collected days. A dead cell here is dead **on this period**.
- **H1 is measured by the notional weighting, not removed by it.** rs is still
  a population statistic over all sweeps. Only E2's queue replay puts a
  *marginal* order at the back of the queue.
- **No queue position, no own impact.** E2.0 can kill or fail to kill; it
  cannot establish economic sign.
- **8.60% of true-mid-valid events (178,659) the proxy could not see.** Δ_rs
  therefore speaks about the mid *on the events both mids see*; that share
  bounds the reach of the void reading.

### Two defects this step found

1. **The collected `@trade` stream carries zero-quantity prints** — rows of the
   form `trade_id,0,0`. **22,639 across ADA's 16 days.** A sweep made only of
   them collapses to Q = 0 and price 0, and `mo = −q(m₁−p)/p` explodes: the
   first real day came back at **4.7e12 bps** and the §1.3c identity check
   failed. **E1's Vision aggTrades contain none** (0 of 582,765 rows over 8 ADA
   days), so **E1's published numbers are not exposed** — this is a property of
   the surface E2.0 introduces. Excluded and counted; four falsifiers drive it
   both ways.
2. **`git checkout --detach` destroys the worktree's `data` symlink** and
   leaves a near-empty shell. The first smoke read **zero days and refused** —
   the instrument worked, but the standing refresh procedure silently undoes
   the fix. Restored twice this session. See the Q-DA row.

## E2-A: the inherited control PASSES, v3 lands the reviewer's condition, the smoke has NOT run

**The inherited control PASSES exactly.** `p002_e2a_e1a_reproduction__20260906T051626Z.json`
(sha256 `e76e3226b1cf603e`, carrying_commit `0d7fae6`): re-measuring E1-A's
episode design with new code on E1-A's own aggTrades, 12 symbols × 31 days,
T_p = 600 s, gives **touch 3.4485 against a published 3.4485 and sweep 6.2645
against a published 6.2645** — errors 0.000011 and 0.000027 bps, and the
bootstrap CIs reproduce to 4 dp. 45.86 s wall, 1,274 MiB RSS.

> **RECORD DEFECT FOUND, routed and not a blocker.** `E1_RESULTS.md`'s
> corrections queue quotes a second pair — *"tick_size() FIXED post-audit;
> corrected aggregate 3.36/6.28"*. **It is not reproducible from the committed
> code.** E1's OWN `tick_size('FILUSDT')`, executed directly, returns **1e-6**
> — the pre-fix value — and an independent transcription returns 1e-6 too. Two
> implementations agree, so the repository does not carry the FIL = 1e-4 the
> corrections queue describes and cannot produce 3.36/6.28. **A reader
> reaching for that pair is reaching for a number nothing on disk can make.**
> E1-A's operative number is untouched by this.

**Declaration v3** (`p002_e2_a_declaration_v3.json`, sha256 `6383d781c7bbeaa6`)
carries the reviewer's interior controls — **and they caught a defect in v2's
own formula before any run**: v2 declared ProbQueue-f3's fill probability as
`f(front)/(f(front)+f(back))`, which makes it RISE as the queue ahead grows.
Hand-computed, v2 gave 0.073 for an order nearly at the HEAD and 0.927 for one
nearly at the BACK. v3 states `f(back)/(f(front)+f(back))`, pins the interior
value 343000/370000 = 0.927027027, and drives the direction and monotonicity as
further controls.

**THE SMOKE HAS NOT RUN.** The reviewer's approval is landed
(`REVIEW_P002_E2A_DESIGN_2026-09-06.md`, APPROVED) and v3 meets its condition,
so **the gate is OPEN** — but the E2-A *runner* does not exist yet. What exists
is the episode machinery on the proxy mid (`e2_a_episodes.py`) and the declared
queue models. Still to build: real-book placement from bookTicker, depth20
queue-ahead at placement, the two fill simulations wired to episodes, partial
fills, and their falsifiers. **Nothing was run against a half-built runner.**

## The supersession sidecar convention (write this down — it is now used twice)

When a receipt is superseded by a later one that **adds** a field rather than
changing a result, the earlier file is **never edited** (rule 13). A sidecar
`<earlier>.superseded_by.json` sits beside it naming the operative receipt by
path and sha256, and the claim that both carry the same result is **computed
field by field**, not asserted — the emitter **refuses** to describe it as a
supersession-by-added-diagnostic if any gate-bearing field moved. Both files
stay in git when the earlier one was **read** before the addition: deleting an
artifact a decision saw is worse than keeping it, and the sidecar is what stops
it resolving as current. Emitted by
`e2_0_true_mid.py --supersede EARLIER OPERATIVE OUT`.

## E2-A's two escalations — RULED (R-567(C))

Both stand as declared: the closed-form models proceed with no dependency
installed, and the size-aware arm REFUSES with its status while the min-size
arm runs labelled NOT the gate. A size-aware arm needs a declared notional
source and that question is with the USER.

## E2-A was DECLARED — and here is what needed the ruling

`live/mm_research/declarations/p002_e2_a_declaration_v2.json`
(sha256 `6567a25f04d7fb89`, carrying_commit `0cbaba6`; v1 `405ddb7ab10486c2`
superseded in band, untouched), **no data touched under either**. v2 adds the
data-root discipline the E2.0 result review requires: the P-002 surface
resolves through the same imported resolver as P-003, every receipt records the
root and branch, and a result-bearing run off the canonical ledger REFUSES —
with the falsifier being a **partial** root (a real tape holding 2 days of 19),
which must refuse rather than report a smaller census.
Gate: `eff_RT <= 8 bps` under **RiskAverse** at T_p = 600 s, interval binding
on the PASS side. A bracket that **straddles** the threshold is a FAIL, never
averaged. **ProbQueue-f3 costing more than RiskAverse refutes the INSTRUMENT**,
not the symbol. ICP: above a **50% episode-skip bar** a symbol is UNRESOLVED
and excluded, aggregate reported both ways — the bar is declared before any
census and applies to all twelve (E1-A measured ICP at 72%).

> **RULING NEEDED (a): `hftbacktest` is NOT installed on this box.** The plan
> names it for the bracket. Installing a dependency is an environment change
> and not the seat's to make, so both queue models are declared in closed form
> and implemented in the runner — which makes their **correctness mine**. Each
> ships a falsifier, and the ordering property is a computed predicate. Either
> the direct implementation is accepted, the dependency is authorised, or E2-A
> waits.
>
> **RULING NEEDED (b): the XS rebalance notional does not exist.** "Depth-aware
> sizes at the XS book's actual rebalance notionals" needs a per-symbol
> notional this programme has never pinned — E1-A's episodes were explicitly
> *min-size, notional-free*. Its absence **REFUSES the size-aware arm** and
> reports the min-size arm labelled NOT the gate, because a min-size answer is
> E1-A's answer with a better fill model.

**The reviewer files on the declaration before any run.**

## Next steps (in order)

1. **E2.0 on the remaining 15 symbols.** The smoke was one symbol by dispatch.
   ADA cost **67 s wall / 657 MiB RSS** for 16 days at one CPU under 8G.
   **BTC is ~12× ADA's daily volume** (480 MB/day gz bookTicker vs 41 MB) — a
   resource check before it runs, and the cap is never raised: a symbol that
   exceeds it refuses. Expected reading: the E1 audit found the same eq→notional
   flip on SOL/DOGE/XRP/ATOM/ICP, so the 15 are a *confirmation* set, not a
   search for a survivor.
2. **E2-A (overlay bracket on real books)** — still the decision-bearing step
   for the programme's live track, and it resolves the fragile ICP cell.
   Untouched by this result.
3. **E2 proper** (hftbacktest queue replay, RiskAverse vs ProbQueue-f3).
4. **E1x is MOOT.** It existed to re-measure ADA's population statistic; E2.0
   supersedes it and the cell is dead.
5. **HL Variant-B screen** still ~2026-09-18.

## Watch out for

- **Never gate on eq-weighted.** This step is the second independent
  demonstration; the quintile table is the picture to show anyone who asks why.
- All E1-A numbers remain maker-optimistic upper bounds; only E4-pessimistic-
  queue establishes economic sign.
- The 16-sym pilot is not PIT (H5); conclusions attach to named symbols.
- Collector restart (do NOT restart without the coordinator — collector
  surface, R-110):
  `nohup python3 live/mm_research/collect_hf.py > data/mm_hf/collector.log 2>&1 &`

---


Updated: 2026-08-19 end of session 1. **E1 COMPLETE — program is overlay-only.**

## Session 1 outcome (full pipeline ran: R → S → D → I → V)

- **Docs** (all in `live/mm_research/`): R1–R4 research briefs →
  STRATEGY_SKETCH.md (canonical architecture, 7 components, variants A/B) →
  EXPERIMENT_PLAN.md (pre-registered E1–E5 ladder) → E1_CODE_REVIEW.md (blind
  code audit + results audit) → **E1_RESULTS.md (the verdict — read this
  first)**.
- **E1-B (standalone Binance MM): no real pass.** Majors NEGATIVE pre-fee
  (BTC −0.19, ETH −0.23 bps at 30 s). ADA's +2.44 screen pass is an H1
  fat-tick artifact: notional-weighted it is **−0.32 bps** (small prints earn
  the half-tick, the dollars are adversely selected; same flip on every
  wide-tick name). Pre-read amendment for E1x/E2.0: gate quantity =
  notional-weighted rs on true mids.
- **E1-A (passive-execution overlay for the XS book): PASS, audit-robust.**
  T_p=600 s bracket: touch 3.45 [3.11,3.79] / sweep 6.26 [5.76,6.75] ≤ 8 bps
  capstone threshold; stale-shadow 7.20; excl-ICP 6.15; per-symbol max 7.53.
  Still maker-optimistic (H1; no queue position) — E2-A/E4-A decide for real.
- **Infra**: collect_hf.py LIVE since ~12:45 UTC (16 syms × bookTicker +
  depth20@100ms + trade; @aggTrade dead on URL-subscribed fstream — @trade
  used; combined /stream?streams= endpoint required; fapi REST geo-blocked
  from this box → Tokyo VPS before any order). 31 d tick aggTrades on disk.

## Next steps (in order)

1. **E2.0 + E2/E2-A when 14 d of L2 exist (~2026-09-03)**: true-mid recompute
   (notional-weighted, per amendment) voids/settles ADA; E2-A resolves the
   overlay bracket + the ICP cell with real books under the queue-model
   bracket (RiskAverse vs ProbQueue-f3; sign-flip = fail).
2. **E1x (optional, ADA only)**: 12-month quarterly confirmation with
   notional-weighted gate + fixed tick_size(). Expectation: kill. Low priority
   given E2.0 supersedes.
3. **Hyperliquid forward collection: RUNNING since 2026-08-19 ~13:20 UTC**
   (`live/mm_research/collect_hl.py` → `data/mm_hf/hl_raw/`, log
   `data/mm_hf/hl_collector.log`). bbo + l2Book(top-10) + trades, 16 coins
   (Binance-pilot equivalents; all HL-listed). HL market-data WS and /info
   are NOT geo-blocked from this box. App-level ping required ({"method":
   "ping"}) — handled. Trade side field ("B"/"A") recorded raw; side
   semantics must be verified empirically vs prevailing bbo before the
   screen. HL E1-style screen (notional-weighted gate) readable ~2026-09-18.
   Restart: `nohup python3 live/mm_research/collect_hl.py >
   data/mm_hf/hl_collector.log 2>&1 &`
4. Keep the collector alive (restart cmd below); check heartbeats when
   session starts.

## Watch out for

- All E1 numbers are maker-optimistic upper bounds (H1 population-vs-marginal,
  H2 staleness) — fails final, passes provisional; only E4-pessimistic-queue
  establishes economic sign.
- Weighting matters more than anything: eq-weighted markout is a per-EVENT
  statistic; economics are per-DOLLAR. Never gate on eq-weighted again.
- Queue bracket rule everywhere: sign-flip across RiskAverse/ProbQueue = fail.
- The 16-sym pilot is not PIT (H5); conclusions attach to named symbols.
- Collector restart:
  `nohup python3 live/mm_research/collect_hf.py > data/mm_hf/collector.log 2>&1 &`
  then `pgrep -af collect_hf` + check heartbeat lines.
