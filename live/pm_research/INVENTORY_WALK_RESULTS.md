# Does `net` random-walk? — `inventory_walk_v1`

> For current programme state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).
> This document reports one measurement. Where it conflicts with that page, that
> page wins.

Status: **DEVELOPMENT**, not decision-eligible. Research only, no forward-day
claim, no profitability claim. The first test named by
`plans/BE_INVENTORY_MODEL_PLAN.md` §5/§6.

## Decision rule — pre-registered before the measurement

A resting two-sided quote of 5 shares per side is replayed over the tape and
`net(t) = Up_shares − Down_shares` recorded on a 1 s grid.

- **`SELF_BALANCING`** — variance-scaling exponent upper CI `< 0.70`, OU
  reversion slope CI excluding 0 (negative), half-life `< 100 s`, **and**
  terminal `|net|` p95 within `3 ×` quote size. The dump mechanism and the
  switching rule of plan §2 are then unnecessary machinery.
- **`DRIFTING`** — variance ~linear (CI covers 1.0) and no reversion. Inventory
  control is load-bearing.
- **`WEAK_REVERSION`** — reversion present but terminal `|net|` outside the band.
- **`UNRESOLVED`** — fewer than 20 windows, or intervals that do not separate.
  **Underpowered defaults to `DRIFTING`** — removing risk control on thin
  evidence is the dangerous direction.

Probe `inventory_walk.py`, 24 self-test checks, including the three controls
that would otherwise make the verdict vacuous: a synthetic pure random walk must
return `DRIFTING`, a synthetic OU series must return `SELF_BALANCING`, and
one-sided quoting must produce monotone `|net|` growth.

## Result — `JOIN_BBO`, 60 windows/coin, `clob_v3_1`

| coin | n | beta | beta CI95 | OU slope | half-life s | p95 \|net\| | p95 $ at risk | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---|
| btc | 60 | 0.652 | [0.46, 0.83] | −0.00254 | 2726 | 191.8 | 121.80 | `UNRESOLVED` |
| eth | 60 | 0.289 | [0.17, 0.54] | −0.01080 | 639 | 92.0 | 58.92 | `WEAK_REVERSION` |
| sol | 60 | 0.422 | [0.21, 0.64] | −0.01326 | 519 | 40.0 | 30.08 | `WEAK_REVERSION` |
| xrp | 60 | 0.281 | [0.14, 0.46] | −0.00469 | 1473 | 32.5 | 18.37 | `UNRESOLVED` |
| hype | 60 | 0.112 | [0.07, 0.18] | +0.00698 | — | 19.7 | 4.19 | `UNRESOLVED` |
| bnb | 60 | 0.393 | [0.22, 0.66] | −0.00790 | 874 | 15.0 | 5.43 | `UNRESOLVED` |
| doge | 60 | 0.588 | [0.34, 0.95] | +0.00615 | — | 15.0 | 9.20 | `UNRESOLVED` |

**No coin returns `SELF_BALANCING`. The dump mechanism is NOT deleted.**

Every measured half-life (519–2726 s) exceeds the 300 s window, so even where a
reversion slope is detected it is **too slow to matter inside a market**.

## The sub-linear variance is real — a confound I expected, and did not find

`beta` sits well below 1 everywhere, which would ordinarily read as reversion.
The obvious confound is this programme's own measured terminal collapse: if
fills simply *stop* near expiry, variance flattens with no reversion at all.

Tested by recomputing `beta` over the **body only** (`t ≤ 240 s`, `r > 60`,
where intensity is roughly flat):

| coin | beta full | beta body |
|---|---:|---:|
| btc | 0.652 | 0.568 |
| eth | 0.289 | 0.230 |
| sol | 0.422 | 0.359 |
| hype | 0.112 | 0.047 |

**Body-only `beta` is SMALLER on every coin, not larger.** The terminal collapse
is not manufacturing the sub-linearity — if anything it masks a stronger effect.
The suspicion was wrong and the sub-linear scaling stands.

## What actually drives the imbalance — and it splits by liquidity

The informative statistic was not in the pre-registered rule. Comparing terminal
`|net|` under two-sided quoting against the **one-sided control** on the same
windows:

| coin | fills/window | two-sided ÷ one-sided | reading |
|---|---:|---:|---|
| btc | 317.6 | **0.101** | second side removes 90 % of the imbalance |
| eth | 62.5 | **0.199** | removes 80 % |
| sol | 15.9 | 0.449 | removes 55 % |
| xrp | 10.3 | 0.663 | removes 34 % |
| bnb | 2.2 | 1.075 | **no benefit** |
| doge | 2.2 | 1.173 | **worse than one-sided** |
| hype | 1.9 | **1.752** | **75 % worse than one-sided** |

**The second side offsets in proportion to fill frequency, and on thin coins
two-sided quoting is actively counterproductive for inventory.** With ~2 fills
per window the sides rarely pair, so adding a second quote adds unpaired fills
rather than offsetting ones.

That is a mechanism-level finding the plan did not anticipate: §2 treats `skew`
as a control that always helps. On btc/eth it does most of the work; on
bnb/doge/hype it does not work at all.

## `NEW_BBO` — front-of-queue is a PURE RANDOM WALK, at 9.4x the risk

Secondary policy, same 60 windows/coin:

| coin | beta | OU slope | p95 \|net\| | p95 $ at risk | verdict |
|---|---:|---:|---:|---:|---|
| doge | 0.996 | +0.02720 | 88.3 | 10.15 | `DRIFTING` |
| btc | **0.973** | +0.03798 | **1805.1** | **541.76** | `DRIFTING` |
| sol | 0.927 | +0.01836 | 207.8 | 48.88 | `DRIFTING` |
| bnb | 0.908 | +0.01548 | 72.6 | 16.41 | `DRIFTING` |
| eth | 0.539 | +0.01688 | 323.7 | 80.03 | `UNRESOLVED` |
| xrp | 0.478 | +0.00775 | 148.9 | 46.41 | `UNRESOLVED` |
| hype | 0.332 | +0.01789 | 96.8 | 7.34 | `UNRESOLVED` |

**`beta` is at or near 1.0 on four coins and EVERY OU slope is positive** — no
reversion anywhere. Front-of-queue inventory is a textbook random walk.

**And the cost is large.** On btc, terminal `|net|` p95 goes **191.8 → 1805.1
shares** and cash at risk **$121.80 → $541.76** — a **9.4x** increase moving from
`JOIN_BBO` to `NEW_BBO`.

Mechanically this is what you would expect and it had not been stated: at the
front you fill on *every* reaching trade, so you absorb small one-sided bursts
in full. Behind the displayed queue you fill only once a level is swept, and the
queue acts as a **filter** that skips exactly the small directional flow which
accumulates.

**This is a trade-off the policy comparison could not see.** That test measured
fill rate and markout, and found `NEW_BBO` ahead on fills (94.6 % vs 76.9 %).
It did not measure inventory. Front-of-queue buys fill rate **and pays for it in
inventory risk at roughly 9x** — so "which placement is better" cannot be
answered on fills and markout alone, and the earlier `UNRESOLVED` verdict there
was, if anything, generous to `NEW_BBO`.

## A defect in my own pre-registered rule — raised, not re-cut

**The terminal-band criterion cannot distinguish *balanced* from *inactive*.**
`bnb` and `doge` show p95 `|net|` = 15.0 shares — exactly the 3-quote band — and
would have passed that clause. But their two-sided ÷ one-sided ratio is ~1.1:
their `|net|` is small because **almost nothing trades** (2.2 fills/window), not
because the quote self-balances.

A rule keyed on terminal size alone rewards illiquidity. The two-sided ÷
one-sided ratio is the statistic that separates the two, and it was not in the
rule. The verdicts above stand **as computed under the rule as written**; this
is recorded as a defect for the next pre-registration, not applied retroactively.

## Does this delete the dump mechanism?

**No — but it narrows where it is needed, and reverses one of the plan's
assumptions.**

- Terminal `|net|` p95 runs **3 to 38 quote-sizes** with **$4–$122** at risk.
  Inventory control is load-bearing on every coin.
- On **btc/eth** the second side already removes 80–90 %; the dump mechanism
  handles a residual, not the bulk.
- On **bnb/doge/hype** the second side does not help at all, so `skew` is not a
  control there and only `dump` or `do not quote` remain. Plan §2's menu is
  wrong for the thin coins.
- Measured reversion half-lives all exceed the window, so nothing self-corrects
  on an operationally useful timescale.

## Scope and limits

- 60 windows/coin, `clob_v3_1`, one era, inside two collected days. **No
  day-clustered interval is computable**; window-clustered intervals understate.
- Quote size fixed at 5 shares; `|net|` in quote-sizes is the comparable unit,
  and absolute figures scale with it.
- `JOIN_BBO` is primary because it is the conservative bound and always
  available. `NEW_BBO` is an **upper bound on itself** — being first assumes
  winning a race whose outcome depends on latency not present in this tape.
- **No edge or PnL claim.** Cash-at-risk is worst-case loss on the residual, not
  an expected loss, and the maker-edge sign is `+0.173 [−0.251, +0.596]` —
  undetermined. Nothing here is conditioned on profitability.
- Risk is reported **side-aware** per plan §0.3: long Up at `p` risks `p` per
  share, long Down risks `1 − p`. Identical `|net|` and identical `p(1−p)` give
  very different worst cases.

## Against `FLOW_MODEL_STATE.md`

Nothing contradicted. The measured terminal liquidity collapse is used here and
is consistent. The §3 correction that **queue position is a policy output** is
supported again: both placements were simulable with no queue parameter assumed
anywhere — `JOIN_BBO` reads displayed depth, `NEW_BBO` sets it to zero.
