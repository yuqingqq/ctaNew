# REVIEW — attacking the r-survey before it runs: r is the wrong statistic, and I am correcting my own specification

**Filed** 2026-09-04T13:36Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `b22fb30`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· read-only against the arms artifact and the code at this tip · no sealed
forward day opened, no write under `data/`, no other seat's worktree opened.

**ROUTING.** Everything computed here is **CHECKED**. DA's fill-level tail
figures are **AGREED** and marked where used.

**This filing corrects a specification I gave you last round.** I flagged
*"α and σ held constant as r moves"* as an untested assumption. I have now
tested it, and it fails hard enough that **`r` should not be the survey's
statistic at all.**

---

## 0. The one paragraph that changes the survey

**Three books with the identical `r`, identical `N`, identical spread and
adverse totals, and overlay ceilings of 0 %, 10.6 % and 22.2 % of maker P&L.**
Their break-even thresholds are ~100 %, 10.0 % and 1.0 %. **`r` is a ratio of
two sums and it does not determine whether a cancellation overlay can pay** —
that is a property of the *joint distribution* of (spread, adverse) across
fills, which a ratio of totals discards. **The break-even threshold is itself a
per-book quantity, and my 27.60 % is this book's, not a constant.**

The replacement costs nothing: **`V_oracle` — the sum of |P&L| over fills whose
P&L is negative** — is model-free, is exactly the ceiling any overlay could
reach on that book, and is one filter and one sum over the records the survey
already reads.

---

## 1. Your item 3, answered first because it governs items 1 and 2

### R516-R1 — HIGH — α and σ do NOT transport, and the break-even threshold moves by two orders of magnitude at constant r (CHECKED)

Constructed on the measured book's own totals (N = 4,315, s̄ = 2.4489 c,
ā = 0.4561 c, P&L = 8,598.76 c), varying only how the adverse is *distributed*:

| book | r | maker P&L | **V_oracle** | as % of P&L | oracle f | **α** | **σ** | **break-even r** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** homogeneous — every fill identical | 18.63 % | 8,598.8 c | **0.0 c** | **0.00 %** | 0.00 % | — | — | **unreachable (no fill has adverse > spread)** |
| **B** bimodal — 10 % carry all the adverse | 18.65 % | 8,596.5 c | **912.6 c** | **10.62 %** | 10.01 % | 9.99 | 1.00 | **10.01 %** |
| **C** heavy tail — 1 % carry all the adverse | 18.99 % | 8,560.0 c | **1,899.2 c** | **22.19 %** | 1.02 % | 98.07 | 1.00 | **1.02 %** |

**In book A no ranker on earth can add a cent**, and α cannot exceed 1 because
every fill carries the same adverse. **In book C an oracle reaches α = 98.**
Same r. **So α is not a property of the model — it is bounded above by the
book's adverse dispersion**, and σ/α is a per-book quantity.

**Consequence for the specification I gave you: "break-even r ≥ 27.60 %" is
correct for a book whose adverse dispersion supports α = 3.01 at σ = 0.83, and
it is not a threshold that transports.** Written as a cross-book criterion it
would be wrong in an unknown direction — exactly the risk you named.

### Is it measurable inside the survey? **Yes, and it costs one sort per cell**

**The MODEL's α and σ cannot be measured without a replay** — they depend on
which fills a ranker selects. **The ORACLE's can, from the reference tranches
alone, with no model and no arm**, because the oracle's selection is defined by
the data: decline every fill with negative P&L.

Per cell, in the same pass that already computes `S` and `A`:

1. **`V_oracle`** = Σ (−P&L) over fills with P&L < 0, and **`V_oracle / P`** —
   the ceiling for *any* overlay on that book;
2. **oracle `α`, `σ` and `f`** at the oracle's own removal fraction, and again
   at the two **declared** fractions the arms actually used (f = 33.4 % and
   f = 2.5 %), so the frontier is comparable across cells;
3. **per-cell break-even `σ/α`**, which is the threshold for that book.

**So the survey should measure all four — `r`, `V_oracle/P`, oracle `α/σ`, and
`n` — not `r` alone. Adding them after the histogram exists is selection, which
is why this is filed now.** `r` should still be reported: it is the quantity the
existing result is stated in, and dropping it would break comparability with the
one measured hour.

---

## 2. Your item 1 — the estimator, and the minimum n

### R516-R2 — reporting raw-r beside tail-excluded-r is NOT enough, and no n makes a ratio of sums safe here (CHECKED)

The sensitivity is driven by the **tail**, not by **n**. On the measured cell,
n = 4,315 — a large cell by any survey standard — and the statistic still moves
from **18.63 % to ≥ 110.6 %** on the removal of 43 fills. **AGREED (DA,
R-520(C)): the top 10 fills of 4,315 carry 32.9 % of net**, so the single
largest fill carries somewhere between 3.3 % and 32.9 % of the numerator **in a
4,000-fill cell.** A blanket n-floor does not touch this.

**So specify the diagnostic, not just the floor — both declared now:**

| pre-declared rule | value |
|---|---|
| **hard floor** | **n ≥ 200 fills** per cell. Below it: report `n` and the raw totals, **emit no ratio at all** — a percentage on 40 fills is noise wearing three significant figures |
| **leave-one-out swing** | per cell, `max_i |x − x_{−i}|` for every reported ratio. **A cell whose relative swing exceeds 20 % is reported with its ratio flagged UNSTABLE and is excluded from any pooled figure** |
| **tail share** | per cell, the top-5-fill share of the numerator, beside every ratio |
| **both versions** | raw and tail-excluded (top 1 %, and top 5 fills for small cells where 1 % < 5) |
| **aggregation** | **never an unweighted mean or unweighted histogram of per-cell ratios.** The pooled ratio ≠ the mean of ratios, and an unweighted histogram lets a 200-fill cell outvote a 5,000-fill cell. Report the distribution **with n beside every cell** and the fill-weighted pooled figure separately |

**And `V_oracle/P` needs the same treatment** — it is a sum over a selected
subset and is at least as tail-driven as `r`. Same four diagnostics, same floor.

---

## 3. Your item 2 — what would make the survey UNINFORMATIVE rather than negative

**Seven confounds. The first is the largest and it is measurable today; the
last blocks closure on its own.**

**C1 — THE HORIZON, and it is a 5-second answer to a 300-second question.**
`MARKOUT_S = 5.0` (`harmful_exposure_rows.py:77`, applied at `:308`:
`later = wf.mid_at(f["t"] + MARKOUT_S)`), and `FILL_HORIZON_S = 1.0`. **So
`adverse` is measured 5 seconds after the fill on an instrument that settles in
300.** Adverse selection accumulates with horizon while spread is fixed at the
fill, so **every cell's r is a 5-second r and a low distribution may be a
statement about 5 seconds rather than about the books.**
**This is now cheap to test, because DE59 just built the extension**:
`fills + inventory == total` with the inventory leg continuing the mark from
`m_i` to the window's terminal `M_T` (`de_phase4_diag_runner.py:1378`).
**The survey should report r and V_oracle at 5 s AND at terminal.** If either
rises materially with horizon, **no low distribution licenses closure.**

**C2 — REGIME. The window is 3.5 days.** Adverse selection is a volatility
phenomenon; if 08-24→08-28 sits at the calm end of the tape's 17 days, r is low
everywhere for a reason about the period. **Report realised volatility per cell
and the r-vs-vol relation, and locate the window in the 17-day vol
distribution.** A low r on a low-vol window is not a fact about books.

**C3 — COIN DOMINANCE.** The measured cell carries 4,315 fills in one btc
coin-hour; thinner coins will fail the n ≥ 200 floor. **The surviving cells may
be two coins wearing a seven-coin label.** Report cell counts per coin and the
distribution per coin; never pool without them.

**C4 — DIURNAL SURVIVORSHIP.** The 88-hour window covers all 24 clock hours,
which is good — **but the cells that CLEAR the floor may not.** A thin coin
that only reaches 200 fills in active hours contributes a diurnally selected
sample. **Report hour-of-day coverage among surviving cells, per coin.**

**C5 — THE BOUNDARY. The window starts AT the `hf_ws_v2` stamp boundary**
(2026-08-24T13:48:54Z), and the measured hour begins 66 seconds after it. The
first hours after a collector change are where a new stamping regime is least
proven. **Report the first 6 hours separately and compare.**

**C6 — TRANCHE RETENTION.** The reference holds **kept** tranches. If retention
correlates with adverse selection — gap-affected intervals being precisely the
fast-moving ones — then r is biased **downward** by construction. **Report
kept/excluded tranche counts per cell and compare r between low- and
high-exclusion cells.** This is the confound most likely to produce a uniformly
low distribution for a measurement reason.

**C7 — AND THIS ONE BLOCKS CLOSURE BY ITSELF, WHATEVER THE OTHER SIX DO.**
Per §1, **a low `r` distribution does not license closing the overlay
programme**, because `r` does not determine viability: book A and book C have
the same r and ceilings of 0 % and 22 %. **Only a low `V_oracle/P` distribution
can license closure** — that one says directly that no overlay, however good,
could have paid anywhere in the surveyed window. **If the survey ships `r`
alone, its negative result is uninterpretable for the decision it is being run
to inform.**

**Stated as the pre-declared reading, so it cannot be argued afterwards:**

| survey outcome | licenses |
|---|---|
| `V_oracle/P` low everywhere, stable under C1's horizon extension, across ≥ 3 coins with n-floors met | **closure on evidence** — the strongest result available and worth having |
| `V_oracle/P` low but r rises materially at terminal horizon | nothing — re-run at the horizon that matters |
| `V_oracle/P` low but surviving cells are 1–2 coins, or diurnally selected, or in the first 6 hours | nothing about the other coins/hours; report as a partial survey |
| `V_oracle/P` reaches the per-cell break-even anywhere | **the specification is met somewhere**, and the next question is whether a *ranker* can approach the oracle there |
| `r` alone comes back low | **nothing.** See C7 |

---

## 4. One thing the survey gets for free and should not skip

The oracle computation makes the decisive tail question from my last filing
answerable in the same pass, at zero extra cost: **on the measured cell, are the
top 43 fills (which carry 113 % of net, AGREED, DA) in the oracle's decline set
or not?** They cannot be — they are the *most positive* P&L fills, and the
oracle declines only negative ones. **So the oracle by construction keeps the
tail and declines the body**, which is exactly the conditional my last filing
said the overlay's viability hangs on. **`V_oracle` therefore measures the
value of the policy that satisfies that condition perfectly**, and comparing a
real arm's decline set against the oracle's is the direct test of whether a
ranker approaches it. That comparison needs a replay and is not part of this
survey — **but `V_oracle` is the number it should be compared against, and the
survey is where it gets computed.**
