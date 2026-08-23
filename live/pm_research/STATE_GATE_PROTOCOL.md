# STATE_GATE_PROTOCOL — stand down by MARKET STATE: the family union bound

**Status: FROZEN per Ruling R-51, 2026-08-23. APPEND-ONLY from this point
(R-28); R-38 clause (d) applies — an amendment buys an obligation to
re-measure, never a verdict. PROMOTED TO FIRST PRIORITY per R-51: one of
two live channels in the mitigation space, and the only one not facing an
actuation wall.** Authored by the DE session 2026-08-23 under Ruling R-48
(Channel 5 — "new and nobody has ever proposed it"). **Drafted blind**: at
the freeze no per-state-bin markout had been computed. Every constant below
was pre-declared before any receipt.

**What this tests:** every stand-down arm ever considered is indexed by the
CLOCK (`r` — Lever T's body-only gate, the terminal-minute regime). None is
indexed by STATE: quote only when the spread is wide, or flow is thin, or
realised vol is low. The programme has MEASURED all three quantities and
never used any as a quoting condition — the R-44 pattern, found a third
time. This protocol bounds the whole state-conditional family in ONE
measurement, R-48's union-bound rule: nothing tunable, so nothing
p-hackable; a negative kills every rule in the family at once; a positive
is merely NOT_CLOSED and adopts nothing (R-45 amendment-1 semantics,
inherited verbatim).

## §0. Forbidden forms

No threshold sweeps; no gate promotion from any table in this protocol —
a specific gate, if the family survives, is its own blind-drafted protocol
with ONE pre-registered predicate. No pooling across variables. **The
family is SINGLE-VARIABLE gates by construction** — joint predicates
(spread wide AND flow thin) are out of scope exactly as lever interactions
were under R-45 amendment 3, and a negative here does not bound them; that
sentence appears in the verdict whenever the verdict is quoted.

## §1. Population and estimand — inherited from POLICY_BOUNDS

Same day-series population, BASE arm (touch, pinned size), share-weighted
`M_5` in cents with `M_T` beside, VOID floors, exclusion ledgers,
conformance-locked replays, SP-operative stamp. The bound is era-pooled
per verdict coin (the Lever-T bound's shape); per-day tables ride beside,
descriptive.

## §2. The three state variables — pre-declared, all previously measured

At each fill, read at the engine's standard 250 ms knowledge lag
(as-knowable, nothing new):

- **V1 `spread_at_fill`** — ask − bid (price units) from the lagged book
  state at the fill.
- **V2 `flow_60`** — count of PM taker prints (complement-folded,
  deduplicated — the replay's own trade stream) in the trailing 60 s at
  the fill.
- **V3 `rvol_60`** — standard deviation of 1 Hz log-returns of the
  deployed underlying feed (`crypto_prices` relay) over the trailing 60 s
  at the fill, receipt-anchored.

The 60 s trailing window is a single pre-declared constant (no ladder, no
sweep). The deployed-feed caveat from WW_EBX §3 applies to V3 verbatim:
its verdict binds the feed we have.

## §3. The union bound

Per verdict coin, era-pooled: partition fills into **equal-share bins of
each variable** — deciles primary, ventiles as the sharpness control (a
threshold gate can cut inside a coarse bin; the kill must survive the
finer grid too). Both granularities are pre-declared here; no third is
computable later. For each variable `v` and binning `g`:

    bound(v, g) = Σ_bins max(0, w_b · M_b)     (share-weighted M_5, cents)

In-sample binning is ADMISSIBLE for a one-way bound — generosity
strengthens the negative and a positive adopts nothing. **Degeneracy is a
finding, not a failure**: the 1-tick modal book may collapse V1's deciles
(most fills at one spread value); a variable with fewer than 3 distinct
populated bins is reported DEGENERATE — meaning the state variable has no
room to gate on, which answers the family question for that variable by
itself.

## §4. Verdict semantics — one-way, per R-45 amendment 1

Per verdict coin: **STATE_GATES_DEAD** iff NO bin of ANY of the three
variables is positive at BOTH granularities (deciles and ventiles);
**NOT_CLOSED** otherwise — nothing adopted, any specific gate needs its
own blind protocol. eth UNDETERMINED-scale caveats inherit R-45
amendment 2's power framing where CIs are quoted (the bound itself is a
point construct; per-bin CIs ride beside, descriptive). Scope sentence,
mandatory wherever the verdict is quoted: *single-variable gates, deployed
feed, marginal — joint predicates and direct-feed state variables are
unbounded by this result.*

## §5. Dependency and sequencing (R-48)

1. **Bar to the coordinator before the run** (register row Q-DE-12);
   build-under-seal permitted meanwhile.
2. **The OPS-τ gate is SATISFIED (R-49)** — and the ruling confirms what
   §5's note anticipated: a stand-down is a PRIOR decision that never
   races the 160 ms warning, so achievable-τ does not touch this family.
   R-49 promotes this protocol and `policy_bounds_v1` to FIRST PRIORITY:
   ex-ante selectivity (clock, state, depth, size) is now the primary
   line, not the fallback.
3. `policy_bounds_v1` continues in parallel; this probe reuses its BASE
   arm fills, annotated with V1–V3 at fill time — no new engine arms, so
   the conformance surface does not grow.

---

## §6. RUN AND ANSWERED — 2026-08-23 (appended per R-28; §§0–5 untouched)

**Receipt:** `derived/state_gate_v1.json`. 240/240 windows conformant;
determinism control identical on both repeat replays (fills AND captured
state); V3 exclusions ZERO after the pre-read hour-boundary repair (ledger
carries only the standard 281 gap/tick + 51 truncated). btc n=40,273 rows,
eth n=7,466. No variable DEGENERATE (the 1-tick modal book still yields 4
populated spread bins on btc, 5 on eth).

**btc — STATE_GATES_DEAD.** No positive bin on ANY variable at EITHER
granularity; bound exactly 0.0 everywhere, including all four per-day
tables. The single-variable state-gate family is CLOSED on btc.

**eth — NOT_CLOSED, on one 55-fill bin.** Spread decile 7 (range
(0.01, 0.02]): n=55, 0.74 % of share, +0.21 ¢ point, descriptive CI
[−2.98, +3.43] — the same bin at ventiles (same 55 fills). Bound
+0.0016 ¢/share against −0.9..−2.9 ¢ baseline losses. Per amendment-1
one-way semantics NOTHING IS ADOPTED; DE recommends no specific-gate
protocol against a noise-shaped ceiling three orders under the loss.

**The folklore inverts, twice — the finding beyond the verdicts:**
- *"Quote when the spread is wide"*: WIDE-spread bins carry the WORST
  fill quality on both coins (btc −0.93 thinnest → −2.22 widest; eth
  −0.92 → −2.64). The spread widens exactly when informed flow arrives —
  a wide book is distress, not opportunity.
- *"Quote when vol is low"*: LOW-rvol bins are WORSE than high on both
  coins (btc −1.57 low vs −0.87 high; eth −3.81 low vs −1.53 high).
  Both pre-declared mechanism directions are contradicted by their own
  data; a state gate built on either folklore would have SELECTED INTO
  the damage.

**M_T beside:** slightly larger positive residue (btc up to +0.12 ¢
ventile-v3; eth up to +0.42 ¢) — still 1–2 orders under M_T losses, no
truncation, noisier; descriptive only.

**Scope (mandatory sentence):** single-variable gates, deployed feed,
marginal — joint predicates and direct-feed state variables are unbounded
by this result.
