# STATE_GATE_PROTOCOL — stand down by MARKET STATE: the family union bound

**Status: DRAFT FOR COORDINATOR FREEZE — nothing here is frozen.** Authored
by the DE session 2026-08-23 under Ruling R-48 (Channel 5 — "new and nobody
has ever proposed it"). **Drafted blind**: no per-state-bin markout has been
computed. Every constant below is pre-declared here, before any receipt.

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
