# Review — Phase-4 ESTIMAND questions at `6d04833` against the frozen protocol and the addendum (a reading)
reviewer: claude (pm-codex seat) · requested by the coordinator (pm-co)

**Read at:** `live/pm_research/de_phase4_diag_runner.py` at **`6d04833`** (sha16
`dd3331887b02cf44`, verified), `DE_PHASE4_PROTOCOL_DRAFT.md` (**FROZEN**, sha256
`ab07fd71c9fc2bff…`, verified unedited), the addendum (`35e8aba1381cfa4e…`),
`harmful_stateful_policy.py`, `harmful_exposure_rows.py`, `phase4_generation_tables.py`,
`STATEFUL_HARMFUL_CANCEL_TODO.md`, and the fits under `data/pm_5min/derived/phase2_fits/`
(read-only).
**Request of record:** `REQUEST_DE_ESTIMAND_2026-09-02.md`. **Composed 2026-09-02T18:58:08Z.**
One filing, per R-377. **I estimate; the USER decides (rule 14).**

**Discipline.** Reading only, at `--detach 6d04833` in `~/ctaNew-wt-rev`; `~/ctaNew-wt-de` and
`~/ctaNew-wt-be` never read; the main tree's WIP never read; the runner's `--run` **never
invoked**; the only execution was in-process arithmetic on synthetic fills through
`de_rho_estimator`. Nothing written anywhere; no plan file edited; no unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run; worktree clean.

## Summary

| # | question | ruling |
|---|---|---|
| 1 | rho's denominator | **INCONSISTENT** — `DRAFT:150` read with `DRAFT:212-213` |
| 2 | the horizon | the **value** is consistent (it uses the feed `DRAFT:68` prescribes); the **receipt's 1 s declaration** is INCONSISTENT with it |
| 3 | the three parameters | **two NEW NUMBERS** (USER-level); the third is named by the frozen text, not silent |
| 4 | the control's design | the cancel set **must** be the drawn generations (frozen); **repost parity is not in the frozen text** — an addendum residual |

| number the runner declares | verdict |
|---|---|
| `HALF_SPREAD_CENTS = 0.5` (`:97-101`) | **NEW NUMBER (USER-level) if kept — and it need not exist**: the quantity is measurable one call away |
| `theta_repost = theta_cancel / 2` (`:188`) | **NEW NUMBER (USER-level)** — the source of record fixes only the inequality |
| `REPOST_DWELL_S = 2.0` (`:96`) | **NEW NUMBER (USER-level)** — the source of record requires a *declared* dwell and none is declared |
| `MAX_CANCELS_PER_MINUTE = inf` (`:102`) | **NAMED BY THE FROZEN TEXT** (`DRAFT:71`): declarable per cell, so `inf` is admissible — but the frozen reporting identity it comes with is not met, and the runner's stated reason ("the frozen protocol names no rate limit") is **false** |

## 1. rho's denominator — INCONSISTENT (EST-R1)

**What the frozen text means by `spread`.** Three places, and together they settle it:

- `DRAFT:32` — the estimand's ledger carries "**lost spread capture**" as a term of the value;
- `DRAFT:150` — the comparison metric is "cost-adjusted value in cents, and `rho = adverse /
  spread`";
- `DRAFT:212-213` — the reporting list requires, for **every** cell, "fill and share retention,
  **spread capture**", and then "**retained-book adverse-cost / spread-capture ratio (`rho`)**".

So `spread` is **the spread capture measured on the retained book** — a quantity the protocol
requires the run to report in its own right at `:212`, of which rho at `:213` is the ratio. It is
not a modelling constant of the diagnostic.

**What the code computes.** `:484` synthesises `mid_cents_at_fill = level ∓ HALF_SPREAD_CENTS`, so
the estimator's denominator is `Σ size·|px − mid_at_fill| = 0.5·Σ size`. Driven on synthetic fills
through the real estimator:

```
rho = 0.426667      adverse 80.00      spread 187.50
size-weighted mean markout per share = -0.213333 c
-mean_markout / H  = 0.426667  ==  rho            (exact)
spread_cents       = H × total_size = 0.5 × 375   (exact)
```

`rho ≡ −(size-weighted mean markout per share) / H`: a **rescaled mean markout**, and the "spread
capture" `:212` asks for is `0.5 × shares` — a restatement of retention. **A declared constant
cannot satisfy `:212-213`**, because the ratio's denominator then measures nothing.

**And the constant is decision-bearing.** Under it the frozen reading rule (`rho ≥ 1` at every rung
⇒ the route CLOSES) is exactly *"mean adverse markout per share ≥ H cents ⇒ the route closes"*. At
H = 0.5 a book losing 0.7 c/share closes the route; at H = 1.0 the same book does not. The verdict
turns on a number nobody measured.

**Is any number left to declare? No.** The feed already calls `wf.mid_at(...)`:
`harmful_exposure_rows.py:308` reads the mid at `t + MARKOUT_S` to build
`markout_cents_per_share` (`:309-312`, level-referenced). The same object answers
`wf.mid_at(f["t"])` (`edge_layer1.py:108`) — **the mid at the fill instant, one call away in the
loop that already runs**. Carry it per tranche and the denominator becomes measured per fill,
`HALF_SPREAD_CENTS` disappears, and `:212`'s "spread capture" becomes a real reported quantity.

**Verdict: INCONSISTENT.** `0.5 c` is a **NEW NUMBER (USER-level)** if kept — zero occurrences of a
half-spread or a `0.5 c` in either plan file (the frozen document's single `0.5` is
`retention_share_fraction ≥ 0.50` at `:263`, an unrelated threshold). My recommendation is that it
not be kept: this is a measurement, not a declaration, and the cheapest closure removes the number
rather than ruling on it.

## 2. The horizon — the value is right, the receipt is not (EST-R2)

**(a) What the frozen protocol names.** Not ambiguous once `§2`'s parameter table is read.
**`DRAFT:68` (row 5, "feed")**: *"**generation-level tranche tables**
(`harmful_exposure_rows.generation_table` shape), **never per-row latency labels**. Any cell that
consumes the per-row labels DECLARES the 1 s cap…"* — which is Cap 2 (`:41-45`) stated as a
conditional: the cap attaches to a cell **built on the per-row labels**, and the protocol's own feed
is the generation-level table.

**What the cell at `6d04833` is built on — the prescribed feed.** `build_reference` (`:204`, tranches at `:261-265`)
carries the generation's tranches, the `generation_table` shape `:68` names; the replay
(`:499`) and `cost_adjusted_value_cents = harm − sac − queue_reset_cost_total`
(`harmful_stateful_policy.py:1134-1145`) run over that. So **the over-the-hold number is the one
the frozen feed prescribes**, and the per-row table produced at `:1068` correctly decorates only
the receipt's `feed` block (`:1086-1091`) instead of feeding the value.

**(b) The receipt.** It carries `fill_horizon_s = FILL_HORIZON_S` (`:362`) as a **binding field**
(`:107`) with an `estimand_note` (`:364`) stating every cell "estimates VALUE PREVENTABLE WITHIN
ONE SECOND". That is the declaration `:68` attaches to the OTHER feed. **Yes — a mis-declaration**,
and the one that matters most, because a binding receipt field is what a later reader resolves
(rule 13) and this one names an estimand the number does not have.

**(c) Which closure is consistent without a new number.** Reversing what one might assume from Cap
2 alone:

- **Declare the horizon the number actually has** (the generation's hold) and drop or re-word the
  1 s claim. This is the **frozen-consistent** closure: `:68` prescribes exactly this feed, no new
  number is introduced, and nothing in `§1`'s ledger (`:28-34`) imposes a 1 s bound. Because the
  addendum declares **no** horizon today, saying so is a change to the document the run is measured
  against → a **dated addendum v2 before the run**, not a row statement (rule 6 puts the estimand in
  the declaration; rule 13 makes a superseding declaration the way to change one).
- **Cap the value at 1 s of the cancelling decision row.** Also introduces no number (1.0 s is
  frozen at `:42` and mirrored at `phase4_generation_tables.py:23`) and would make the receipt's
  current words true — but it moves the cell onto per-row-label semantics, which `:68` treats as the
  exception requiring the declaration, not the protocol's feed.

Either is defensible; **the first is the one the frozen text prescribes**, and the second is the
exception route. What is not defensible is the present pairing: the prescribed feed under the
exception's declaration.

One asymmetry worth recording either way: Cap 1's lower edge **is** applied (the policy's
`cancel_effective_latency_ms`), so the cell is `[t + L, end of hold]` — a well-defined window; it
is simply not the window the receipt names.

## 3. The three parameters (EST-R3, EST-R4)

**`theta_repost = theta_cancel / 2` (`:188`) — NEW NUMBER.** The source of record is
`STATEFUL_HARMFUL_CANCEL_TODO.md:381-382` — the ledger the frozen document scopes itself to
(`DRAFT:19`, `:33`) — which says: *"`HELD`: no repost until score is below `theta_repost` for a
declared dwell time. **Require `theta_repost < theta_cancel`**"*. That fixes the **constraint**
(which `harmful_stateful_policy.py:218-221` also enforces) and **no value**: every point in
`(0, theta_cancel)` satisfies it. Zero occurrences of `theta_repost` in the frozen document, the
addendum, `fit_manifest.json` or any of the 15 files under `phase2_fits/`. Halving is a choice
inside the admissible interval → **USER-level**, declared in a dated addendum v2 before any run,
with the sensitivity the run round reports.

**`REPOST_DWELL_S = 2.0` (`:96`) — NEW NUMBER.** The same TODO line requires the dwell to be
**declared**; no plan document declares one. A constant in a runner is not a declaration in rule 6's
sense — the declaration is what the run is measured against. Same route.

**`MAX_CANCELS_PER_MINUTE = inf` (`:102`) — the frozen text is NOT silent, and the runner's reason
is false (EST-R4).** `DRAFT:71` (row 8 of the `§2` parameter table, authority TODO) reads:
*"**rate limit** — `max_cancels_per_minute` **declared per cell**; requested / effective(passed) /
suppressed counted separately and **reported as the identity `requested = passed + suppressed`**"*.
So:

- `inf` is an **admissible declared value** — the row asks for a declaration per cell, not for a
  particular limit — but it is a declaration the protocol requires, not an absence the protocol
  leaves open;
- the runner's comment (`:193-197`) says *"the frozen protocol names no rate limit — its axes are
  latency, reset cost, budget, repost model, protection mode, reset semantics, reduce and coin
  (:88-99)"*. **Both halves miss**: the protocol names it at `:71` (as a per-cell declaration, not
  as a grid axis), and the axes table is at `:99-108` — `:88-96` are `§3`'s validation and
  cluster-unit paragraphs;
- **the reporting identity is not met.** The policy already counts `cancels_requested`,
  `cancels_rate_passed` and `cancels_suppressed_rate_limited` (`harmful_stateful_policy.py:1042-1043`),
  and the runner's per-arm output carries only `cancels_issued` (`:511`). Under `inf` the identity
  is trivial, which is precisely why reporting it is cheap and why its absence is a reporting gap
  rather than a modelling one.

## 4. The control's estimand (EST-R5)

**What the frozen text fixes.** `DRAFT:147-156`: "**matched random cancellation on identical
opportunities**: draws are matched on action count, side and hour"; "the control's action count is
*determined* by the treated arm and is never a caller-chosen number"; the pool is totally ordered
before the RNG; a demand exceeding the eligible pool **REFUSES** rather than clamping; the
comparison is on the DECISION metric.

**So the cancel set must be exactly the drawn generations — fixed, not a preference.** The draw is
matched at the granularity of the pool it draws from, and the pool at `:586-589` is keyed
`slug|side|gen`. The control then **discards the generation**: `:601-604` splits the key, throws
`_gen` away, and emits `{"t": 0.0, "slug": …, "side": …, "score": 1.0}`. Two consequences, both
breaking the frozen matching: the arm cancels **whichever generation is live at t = 0**, not the one
drawn; and two draws inside one `(slug, side)` collapse to one cancel, so the **action count is not
preserved** — the one property `:152-154` says must be determined by the treated arm.

**Repost parity: the frozen text is SILENT.** `§6` constrains the draw and the comparison metric,
not the control's policy configuration, and `§8`'s reporting list says nothing about it. As built
the control cannot repost at all: eligibility requires a later score event below `theta_repost`
(`harmful_stateful_policy.py:903`, and `:40` "the score has been < theta_repost continuously"), and
the control emits exactly one event per key, at score 1.0. So the contrast measured today is
"cancel-and-never-repost vs cancel-and-repost", not "which generations were cancelled".

**Ruling on the design the fix must satisfy.** (i) the cancel set is **exactly the drawn
generations**, matched on count within `(side, hour)` strata — **required by the frozen text**;
(ii) **repost parity** — the control carrying the treated arm's hysteresis so the only difference is
which generations were cancelled — is **required by the estimand's logic but not fixed by the frozen
text**, so it is a **residual for the addendum v2**, stated as a design requirement. It introduces
no number: it reuses whatever `theta_repost` and dwell the USER rules under `§3`.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| EST-R1 | **HIGH** | runner `:97-101`, `:484`; `DRAFT:150`, `:212-213` | rho's denominator is a declared constant, so "spread capture" is never measured and the reading threshold IS the constant |
| EST-R2 | **HIGH** | runner `:107`, `:362-364`; `DRAFT:68` | the cell uses the feed the protocol prescribes, under a binding receipt field declaring the other feed's 1-second estimand |
| EST-R3 | **MEDIUM** | runner `:96`, `:188`; TODO `:381-382` | `theta_repost` and the repost dwell are new numbers — the source of record fixes the inequality and requires a *declared* dwell, and none is declared |
| EST-R4 | **MEDIUM** | runner `:193-197`, `:511`; `DRAFT:71` | the rate limit's justification asserts a silence that is not there, cites the wrong lines, and the frozen reporting identity `requested = passed + suppressed` is not carried |
| EST-R5 | **MEDIUM** | runner `:601-604`; `DRAFT:147-156` | the control discards the drawn generation and collapses same-`(slug, side)` draws, so neither the cancel set nor the action count is the matched one |

## What this filing is and is not

It is a reading of four estimand questions at one blob; it is **not** a round on DE's code, and DE
round 34 was not read. Three findings (EST-R1, R3, R5) restate at the estimand level what R-464's
DE33-C6/C3 name at the code level; what the reading adds is the frozen line that settles each
question — `DRAFT:212-213` for rho's denominator, **`DRAFT:68`** for the horizon (which reverses
the obvious reading of Cap 2 alone: the over-the-hold value is the prescribed one and the receipt is
what misstates it), `TODO:381-382` for the hysteresis, **`DRAFT:71`** for the rate limit (the
protocol is not silent), and `DRAFT:147-156` for what "matched" binds. Two numbers
(`theta_repost`, the dwell) are USER-level by my reading; `0.5 c` is USER-level only if the run
keeps it, and my recommendation is that it be measured instead; `inf` is a declaration the frozen
text asks for, with a reporting duty attached.
