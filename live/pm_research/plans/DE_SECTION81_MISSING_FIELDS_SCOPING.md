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

---

## Correction and outcome — DE58, 2026-09-04

Both buildable fields were built, and then **the build was wrong in a way
that doubled the answer**. Recorded here in-band because this document is
what the ruling was made from.

### The double-count

The first build returned `maker_pnl = spread_capture + markout`. Read at
the producer (`harmful_exposure_rows.py:307-312`, not from memory):

```python
sgn = 1.0 if f["side"] == "BUY_UP" else -1.0
later = wf.mid_at(f["t"] + MARKOUT_S)
"markout_cents_per_share": sgn * (later - f["level"]) * 100.0
```

The markout is struck **from `level`**, not from the mid at the fill, so
it **already contains the entry edge**. Adding the spread to it counts
that edge twice. On the real 12-window fragment it reported
**19,165.71 cents where the figure is 8,598.76**.

The correct decomposition, all three in the markout's own convention:

| quantity | formula | 12-window value (cents) |
|---|---|---|
| spread capture | `sgn*(mid_at_fill − level)*100` | 10,566.95 |
| adverse selection | `sgn*(mid_at_markout − mid_at_fill)*100` | −1,968.19 |
| **maker P&L** | `sgn*(mid_at_markout − level)*100` | **8,598.76** |

**The total was already in the artifact under another name.** Per share,
`markout_cents_per_share` *is* the maker P&L at `t + MARKOUT_S`, so
`maker_pnl_cents == post_fill_markout_cents`. What the new fields add is
not a new total but **its split** into entry edge and post-fill drift —
which is the part §8.1 could not previously see. The identity is checked
in code, not asserted, and the double-count is a named regression
falsifier in the suite.

### The caveat, measured

`NO_MID_AT_FILL` — flagged in §1 as a real exclusion of unknown size —
is **0 of 4,315 tranches** on this population (`de_section81_mid_census`,
as-of 2026-09-04T12:43:47Z). Not 0.1%, not 20%. `da_population_audit`
returns `NOTHING_EXCLUDED`, a status rather than a pass, so no second
selective filter stacks on DE53's duration axis here.

The zero is consistent with a mechanism rather than luck: a fill requires
`buy.level`/`sell.level` to be set, which happens only inside
`resync()`, which calls `record_mid()` at that same instant — so
`mid_t[0] <= t_fill` for any fill. The census is that claim's falsifier
and would catch it on a larger population.

The two legs' denominators were checked and **do** agree here — but only
because the rate is zero. The first build accumulated both legs inside
one `mid is None` guard, which would have silently truncated the P&L leg
to the decomposition's denominator. Dormant, not absent; removed so it
cannot wake.

### `inventory_loss_cents` — the blocker moved

Not a capability gap and no longer a re-feed question first: the terminal
mark is one stored line at a site that already calls `wf.mid_at()` twice.
**It awaits a ruling on which mark is meant**, and the two candidates are
not interchangeable:

| | (A) mid AT `t1` | (B) last observed mid BEFORE `t1` |
|---|---|---|
| in a gap | `NOT_AVAILABLE`, counted | always present |
| what it marks | the window's end instant | a price the market has already left |
| fails when | the residual is riskiest | the gap is long |

They differ **exactly when the gap matters**. Also unruled: whether the
loss is **per-slug or summed** — the emission calls the summed terminal
net "reporting-only, carries no decision meaning", which points at
per-slug but was never confirmed. Both are carried in
`SECTION_8_1_FIELDS` and emitted with the field, so a reader sees what
the ruling is *between* rather than only that it is missing.

### What the fields then answered

With the decomposition in place, §8.1's closing sentence becomes
checkable for the first time. See `cancellation_economics` in
`de_section81_arms__20260904T125340Z.json`.
