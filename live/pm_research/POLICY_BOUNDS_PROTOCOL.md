# POLICY_BOUNDS_PROTOCOL — the three unopened levers, bounded

**Status: FROZEN per Ruling R-45, 2026-08-23, with three coordinator
amendments (landed in §2/§5/§6 below and marked [R-45]). From this point the
document is APPEND-ONLY (R-28): corrections are annotations beside, never
edits.** Authored by the DE session 2026-08-23 under Ruling R-44. **Drafted
blind**: at the freeze no per-bin, per-size or per-depth markout had been
computed; every number below is a previously published figure, cited as
derivation basis exactly as R-1's `f*` and R-14's MDE were.

**Origin, recorded because the method note demands it:** this gap was found by
the USER asking whether the capture-vs-adverse-selection trade-off had been
optimised — not by a review loop, not by the coordinator, not by DE. The loops
audit what exists; nobody audited what was never attempted. R-44's state,
verified: every placement arm ever tested sits AT THE TOUCH; quote size has
never left the 5-share pin; no time-conditional quoting arm exists — while the
programme had already measured the terminal minute as a different regime
(`f_r` collapse, few-and-large flow at 15.5→24.0 USDC/arrival, amplitude 6–7×
the body) and never used it as a quoting decision.

**What stays closed and is NOT re-opened here:** the cancellation family.
`ww_v1` bounded the whole family — 15.3 % of btc's damage carries ANY warning
against a 30.9 % break-even, and unwarned damage carries no signal to
discriminate on — with one stated limit: a fourth signal outside
flow/depth-depletion/mid-move would be outside the envelope. This protocol
tests DIFFERENT levers on the same trade-off; nothing in it revives
cancellation.

---

## §0. THE FORBIDDEN FORM, named first

**FORBIDDEN: a grid over depth × size × time-gate reporting the best cell.**
On four era days that manufactures an edge out of selection, and it is exactly
what a tired researcher would do. Also forbidden: promoting ANY cell not
pre-registered below (a second gate found in the descriptive tables is a NEW
protocol, drafted blind, frozen before its run); pooling across arms; pooling
across the pinned and swept sizes in any downstream use. Every lever below is
ONE pre-registered directional hypothesis with its own bar, plus a
ww_v1-shaped BOUND whose negative closes the lever's whole family. All cells
are reported; none is selected.

## §1. Population and discipline (house rules, unchanged)

Day-series selection (4 era days × 30 windows/coin; 08-19 pre-era excluded);
verdict coins btc/eth, others descriptive; R-DUAL; share-weighted primary
with per-fill beside; within-day window-clustered CIs only (day-clustered not
computable at 4 clusters); `M_5` primary with `M_T` reported beside (Layer 2
measured `bridge(5) ≈ 0` — damage by t+5 s sticks to resolution — so an
`M_5`-positive policy is expected `M_T`-consistent; reported, not assumed);
VOID floor 500 fills per (coin, day) cell per arm; exclusions ledgered;
conformance-locked replays; engine variants land under EV-Replay §4.3
(perturbation controls + parity at defaults); receipts stamp the operative SP
set and `days_sampled`. Baseline legs, published: btc h=5 spread +0.642 /
drift −1.175 / markout −0.532; eth +0.778 / −2.021 / −1.243.

## §2. LEVER T — selective quoting (the time gate)

**Pre-registered hypothesis (ONE gate, named now):** BODY-ONLY — quote
`r ∈ [60, 300)`, stand down inside `r < 60`. Mechanism: the terminal minute
is a measured different regime; standing down forfeits terminal capture to
skip terminal adverse selection.

- **Bar (cell):** body-only share-weighted `M_5 ≥ 0` with the within-day CI
  excluding zero from above → `POSITIVE`; symmetric `NEGATIVE`; else
  `UNDETERMINED`. Roll-up per the R-14 pattern: `GATE_RESCUES` at ≥75 % of
  era days POSITIVE and zero NEGATIVE (min 4 era days); `GATE_FAILS`
  symmetric; else UNDETERMINED (calendar).
- **The BOUND (ww_v1-shaped):** per-fill `M_5` aggregated on the FROZEN
  non-uniform grid — body 4×60 s + terminal 12×5 s, chosen upstream for
  independent reasons (V5), so the binning embodies no selection. The
  most ANY time-gate could keep is `Σ_bins max(0, w_b · M_b)`. **If no bin
  is positive on a verdict coin, ALL time-gates are dead on that coin at
  once** — the lever's family verdict, not one rule's. The per-bin table is
  DESCRIPTIVE; §0 governs what may be promoted from it (nothing).
- **[R-45 amendment 1] THE BOUND IS A ONE-WAY INSTRUMENT — verdict
  semantics, not a side-rule.** `Σ max(0, w_b·M_b)` selects bins by their
  IN-SAMPLE sign, which is exactly right for a falsifier and nothing else:
  a NEGATIVE bound closes the lever's family (`ALL_GATES_DEAD` — if even
  the in-sample maximum is negative, no gate survives); a POSITIVE bound
  is an in-sample maximum and BOUNDS NOTHING — the family is merely
  `NOT_CLOSED`, nothing is adopted, and any specific gate needs its own
  blind-drafted protocol. The asymmetry lives here, in the verdict, because
  this is where it will actually be read.

## §3. LEVER S — quote size

**The venue fact that shapes this lever, measured (SP §4):** `min_size = 5 =
the pin` — ZERO downward headroom. So the sweep has two tiers with different
standings:

- **Deployable tier {5, 10, 15} shares** — pre-registered hypothesis:
  per-share `M_5` does NOT improve above the pin (larger quotes absorb more
  of each one-sided burst). **Bar:** the lever is `ALIVE` on a coin iff some
  deployable size beats size-5 per-share `M_5` with the within-day CI of the
  PAIRED same-window difference excluding zero, on ≥75 % of era days with
  zero contrary; else `DEAD_DEPLOYABLE`.
- **Counterfactual tier {1, 2, 3} shares** — below the venue floor;
  MECHANISM ONLY, no verdict, labeled counterfactual in every table. It
  answers whether the venue's floor is what binds (a fact about the venue,
  not a policy), and doubles as the Class-B robustness probe of the
  5-share pin that every published number is conditioned on (R-6/SP §6).

## §4. LEVER D — placement depth

**Pre-registered hypothesis:** DEPTH-1 — rest one tick behind the touch
(bid−1 tick / ask+1 tick), both sides, same book otherwise. Two published
mechanisms compete and the measurement arbitrates: an extra tick of capture
plus skipping touch-only fills, versus conditioning fills on deeper sweeps
(plausibly MORE informed). The 1-tick modal book limits the room — an
argument, not a measurement, until this runs.

- **Bar (cell):** depth-1 share-weighted `M_5 ≥ 0` with CI excluding zero
  from above → `POSITIVE`; roll-up as §2 (`DEPTH_RESCUES` /
  `DEPTH_FAILS` / UNDETERMINED). The capture/drift decomposition is
  reported so the mechanism that wins is named, not inferred.

## §5. Power, declared before the run (R-14 amendment-2 form)

Effects of interest here are CAPTURE-SCALE — the distance from −0.53 ¢ to
zero is ~0.5 ¢, from −1.24 ¢ ~1.2 ¢ — against per-cell within-day CI
half-widths of ±0.3–0.9 ¢ observed on the same population (the Layer-2 day
cells). **btc cells are adequately powered for the effects that would matter;
eth cells are marginal** (half-widths near the effect size — expect some
UNDETERMINED cells on eth even under a true effect). Body-only shrinks btc n
by ~16 % (the terminal-minute fill share; the h=60 truncation measured 1,611
of 10,294 fills there) — the VOID floor is not threatened. Stated so an
UNDETERMINED is read against the declared power, not dressed either way.

**[R-45 amendment 2] AN ETH UNDETERMINED IS UNINFORMATIVE AND MUST BE
REPORTED AS SUCH.** Since eth half-widths sit near the effect size,
UNDETERMINED is the EXPECTED eth outcome even under a TRUE effect. It
carries no information in either direction and must not be written up as
"eth might work" — failure-to-reject-is-not-equivalence, applied
prospectively for once instead of in hindsight. Every eth UNDETERMINED cell
and roll-up carries the literal label `UNINFORMATIVE (declared-power)` in
tables and prose.

## §6. Verdict semantics, and what a triple negative means

Each lever verdicts independently; there is no cross-lever selection. If all
three read FAILS/DEAD on both verdict coins, **the passive-maker policy
space measured by this programme is closed at every axis anyone has named**:
placement (touch and depth-1), size (deployable range), time (any gate, by
the bound), cancellation (ww_v1, by the bound), carry (Layer 2). A triple
negative is therefore a programme-level answer, not a disappointment — and
any FOURTH axis proposed later starts as its own blind-drafted protocol.

**[R-45 amendment 3] A TRIPLE NEGATIVE CLOSES THE MARGINAL SPACE, NOT THE
INTERACTION SPACE.** Each lever is tested MARGINALLY, and §0 rightly
forbids the grid — so a genuine INTERACTION (depth-1 AND body-only
together when neither works alone) is out of scope BY CONSTRUCTION. A
triple negative is a strong programme-level answer about the levers AS
LEVERS; it is not a bound on their combinations, and that difference will
matter the first time someone reads the result as final. Any interaction
hypothesis is a new pre-registered protocol, one named combination, its
own bar, drafted blind.

## §7. Sequencing

1. This draft goes to the coordinator (§0a row filed). **Freeze before any
   receipt is read**; the R-1 sealed pattern (build-allowed /
   read-forbidden, auditability mandatory) is available if the freeze is
   pending.
2. Engine variants (time-gate, depth-1, size parameter exposure) land under
   the EV-Replay gates BEFORE the measurement arms run: parity at defaults,
   perturbation controls per §4.3, conformance to the reference loop.
3. On freeze: run, read against the frozen bars, report per (coin, day) —
   the R-9/R-17 reporting shape, cells first, roll-ups second, bounds
   beside.
