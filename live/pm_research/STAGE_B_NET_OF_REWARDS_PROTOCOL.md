# STAGE_B_NET_OF_REWARDS — does the net cross zero, at size

REVISION: 2

**Status: CANCELLED BY USER DIRECTIVE — R-125, 2026-08-24, before any
cell was computed.** Rewards are OUT OF SCOPE; the whole rationale of
this protocol was the reward term entering the objective, so without it
there is no objective to search — "a search in want of a question."
Never frozen, never run; no receipt exists. **DA's rewards measurement
stands on the record exactly as filed** (deferred, not deleted): Q-DA-51
becomes DEBT with the trigger "the user reopens the rewards question,"
and this draft is the design that trigger revives. Q-DE-15 (the freeze
ask) is WITHDRAWN. The draft below is retained unedited as the record of
what was designed and why.

*(Superseded draft header follows:)*
**Status: DRAFT FOR COORDINATOR FREEZE — nothing runs before the freeze;
drafted blind (no net cell has been computed; every number cited below is
DA's published measurement or a Stage-A receipt, by value, with as-of).**
A NEW protocol per R-124 item 4 — NOT an amendment: R-111's Stage-A grid
freeze holds and a running search may not grow toward a result. Surface
authorization: R-124 itself.

**What changed and what did not (R-123/R-124, carried verbatim):** Stage A
STANDS for what it measured — no policy makes the FILL economics pay — and
CANNOT be read as "no policy pays": its objective omitted the revenue term
(the R-116 mis-specification, now measured rather than suspected). DA
measured the Liquidity Rewards Program from the market list: **$550k/month
to 5-minute markets, BTC $300k (as-of August)**; the reward **splits by
arm AGAINST R-109's ranking** (100 % of pool covers the loss on 9/10 JOIN
coin-days, 0/10 FRONT); at Stage A's tested configuration it covers
**3.1 %** of the loss; and the score share is **strongly CONCAVE in
resting size** (v=3¢: 0.69 % @ 5 sh, 6.5 % @ 50, 40.9 % @ 500, 66.0 % @
1,400; robust across v=2/3/5¢) against a roughly LINEAR loss — while real
depth rests **698–1,382 shares** within 3¢ and our replay rested 5.

## §0. Forbidden forms, and THE question

Concave revenue against linear loss means the ratio IMPROVES WITH SIZE BY
CONSTRUCTION — **improvement is not the question and may not be reported
as a finding**. The question is whether NET crosses ZERO at any size, on
holdout days, at the day unit. The size ladder below is FROZEN; no rung
added after any receipt is read; all cells reported; the headline is the
crossing verdict, never the best cell. DA's own caveat is binding: its
figures are score shares, NOT P&L.

## §1. The objective — NET of rewards, per window

    NET(size) = reward_share(size, v) × pot_per_window − |TotalM5PnL(size)|

- **`TotalM5PnL(size)`**: DE's harness at the size ladder (Stage A's
  objective, unchanged semantics).
- **`reward_share(size, v)` and `pot_per_window` (per coin)**: DA's
  measured inputs, consumed BY VALUE with their as-of — **Q-DA-51 is the
  BLOCKING input** (the fill-versus-resting-size response is the one
  input DE cannot supply itself; §3 says why).
- Reward requires RESTING: abstention and halt forfeit share pro rata;
  v1 uses DA's share curve as measured (two-sided resting within v).

## §2. The grid — size PRIMARY, arm fixed by measurement

**Cells: JOIN × size {5, 50, 150, 500, 1400} — five cells.** JOIN is
fixed by DA's split measurement (9/10 JOIN vs 0/10 FRONT coin-days
covered; FRONT excluded WITH ITS RECEIPT CITED, not by taste). r_cut=0,
skew off, cancel off (Stage A: abstention only scales; the reward term
punishes it directly). **Anchors: WAIT-only (zero) and Stage A's
JOIN:r0:s5 (continuity).** Total 7 rows, all reported, train/holdout as
Stage A (day-grouped sampler; TRAIN 08-20/21/22; HOLDOUT = every
complete later day at run time; partials beside, never deciding).

## §3. VALIDITY BOUNDS — where the simulation stops being one

The no-impact assumption (maker never affects the tape) DEGRADES WITH
SIZE and at 500–1,400 shares we would BE 36–100 %+ of the real resting
band. Declared handles, frozen now:

1. **Saturation cap (computable from tape, stamped per window)**: fills
   cannot exceed the window's aggressive volume reaching our level;
   replay fills above the cap are impossible and the cell is flagged.
2. **Depth-share flag**: any cell whose resting size exceeds **20 %** of
   DA's measured band depth (698–1,382 sh) is labeled
   `SIMULATION-DEGRADED` — reported, never promoted on replay numbers
   alone; DA's empirical fill-vs-size response (Q-DA-51) is the
   correction path and the blocking input.
3. **Capital feasibility, stamped not hidden**: resting notional vs the
   SP operative $1,000 budget per cell (1,400 sh two-sided ≈ $1,400 —
   INFEASIBLE-AT-OPERATIVE-CAPITAL; measured anyway, labeled; deploying
   above budget would need an SP re-freeze that explicitly invalidates
   nothing retroactively).

## §4. Verdict — the crossing question, day unit

Per coin: **CROSSES iff NET(size) > 0 on EVERY complete holdout day at
some frozen rung whose cell is not `SIMULATION-DEGRADED`-unconfirmed**
(a degraded rung can cross only when DA's response measurement confirms
the fill side). Signs and points, no intervals below supporting G
(R-109 standard). NOT-CROSSING at every valid rung is a real answer:
the reward term, measured, does not rescue the fill economics at
simulable size — and the degraded rungs' status is stated, not
extrapolated. Improvement-with-size is reported as arithmetic, never as
the verdict (§0).

## §5. Controls (before any cell is read)

Null-point: the s5 cell must reproduce Stage A's JOIN:r0:s5 numbers
exactly (same engine, same population ⇒ identical receipts). Saturation
comparator MUST-FAIL: a constructed window where replay fills exceed
tape aggressive volume must flag. Reward-join hand case: share × pot −
loss on constructed numbers. Doctored-input must-fail: a share curve
scaled ×10 must flip the crossing verdict on constructed cells. All
§4.3 engine controls inherited (determinism, lag).

## §6. Sequencing

1. Draft → freeze ask (Q-DE-15, blocks the reshaped decision-path
   item 2). Build under seal meanwhile (size-ladder replay + saturation
   diagnostics run; no NET cell read).
2. DA lands Q-DA-51 (fill-vs-size response) + pot/share inputs by
   value. The reward join runs only then.
3. On freeze + inputs: run, read against §4, report all rows — cells
   first, crossing verdict second, degraded labels everywhere they
   apply, every population with n and as-of.
