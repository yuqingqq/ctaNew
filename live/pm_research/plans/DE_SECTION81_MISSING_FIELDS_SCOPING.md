# §8.1's three unproducible fields — what each would take

DE, 2026-09-04. Scoped answer for a USER decision. **No build is
proposed and none has been started.**

§8.1 requires an arm's replay output to include complete maker P&L,
spread capture, post-fill markout, fill/share retention, `rho`,
effective/stale/unresolved cancels, hold/repost/queue-reset traffic,
terminal and peak inventory, and inventory loss — and closes
*"`net_cancel_cents` alone is not a strategy-P&L verdict."*

Twelve of sixteen declared fields are filled by the current replay.
**Three are not, and they are exactly the three that separate a
strategy-P&L verdict from an overlay increment.** The fourth gap,
latency × cost, is only unrun: it needs the frozen 9-rung axis looped,
which is compute rather than capability.

A correction to my own earlier claim: in round 52 I reported all three as
not producible by anything in this repository. **For two of them that was
wrong**, and the error was that I described what the replay *reports*
rather than checking what its inputs *support*.

---

## 1. `spread_capture_cents` — BUILD. Inputs exist. Works on sealed days.

**What it is.** The maker's edge at entry: the price posted at, against
the prevailing mid at the moment of the fill, signed by side, times
shares.

**Inputs, and where they are.** All four are already on **every tranche**
of the reference:

| input | field | provenance |
|---|---|---|
| fill price | `level` | `harmful_exposure_rows` generation table |
| mid at fill | `mid_at_fill` | added by EST-R1, `wf.mid_at(t["t"])` |
| shares | `shares` | same |
| side | generation's side | same |

**Construction.** The producer already computes
`markout_cents_per_share = sgn * (later - level) * 100.0`, where `later`
is the mid at `t + MARKOUT_S`. Spread capture is *the same formula at the
fill's own time*: `sgn * (mid_at_fill - level) * 100.0 * shares`.

**Cost.** One function plus its falsifiers. **No producer change, no
re-feed**, and it works on artifacts already sealed, because the fields
are in them.

**Caveat that must be a status, never a zero.** `mid_at()` returns `None`
before a window's first quote, so `NO_MID_AT_FILL` is a real exclusion
and must be counted (rule 4).

---

## 2. `maker_pnl_cents` — BUILD. Same inputs, same function.

**What it is.** Spread captured at entry minus adverse selection after
the fill, per fill, summed.

**Inputs.** The four above plus `markout_cents_per_share`, also already
carried.

**Cost.** The same function, plus two things that are declarations rather
than code: a **declared sign convention**, and a **reconciliation against
`received_markout_cents`** so the new number and the existing one cannot
silently disagree.

**Scope limit that must travel with the number.** This is maker P&L **on
received fills within the reference's tranche population**. It is not a
book-wide P&L over unfilled quotes — there is nothing to value there —
and it excludes any position left at window end, which is field 3.

---

## 3. `inventory_loss_cents` — PRODUCER SHAPE. Inputs discarded.

**What it is.** Mark-to-market on the residual position at window end.

**Inputs.**

| input | state |
|---|---|
| terminal net position | **EXISTS** — `inventory.terminal_net`, in shares |
| terminal mark | **DOES NOT EXIST ANYWHERE** |

**The shape.** `wf.mid_at()` is live inside `build_reference`'s loop and
is called **twice** already — once for `mid_at_fill`, once for the
markout's `later` — but **no window-end mid is ever stored**. The value
exists for the length of one function and is discarded before anything
downstream sees it. That is the same shape BE established for tranche
identity: built by one function, thrown away before any consumer.

**But the consequence differs from BE's, and this is what the decision
turns on.** BE's tranche identity is discarded inside a producer over
*captured* data, so days already sealed cannot recover it and a fix has a
lead time. **This producer is the feed builder — a derivation over
RETAINED RAW CAPTURE.** Re-running it reconstructs the terminal mid for
any day whose window archive still exists. **There is no lead time and
sealed days are not lost; the binding constraint is ARCHIVE RETENTION,
not collection.**

**Cost.** One field beside `mid_at_fill`, plus a re-feed: ~250 s per 12
windows measured, so **~28 minutes for the full 471-window §3
population**. Every existing reference artifact lacks the field, so it is
a re-generation rather than a patch.

**Ruling needed, not a build.** What "terminal" means when a window ends
in a gap. `cross_window_correlation` records a case where a
`terminal_mid` defaulted to exactly 0.5 — *"a default, not an
observation"* — which is the failure mode to avoid.

---

## The decision, in one line

Two of the three are a function I can write against already-sealed data.
The third changes what the producer keeps and costs a ~28-minute re-feed,
recoverable for any day whose archive is retained.
