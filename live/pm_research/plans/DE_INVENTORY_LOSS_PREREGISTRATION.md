# `inventory_loss_cents` — the ruling it needs, and my prediction, declared BEFORE the number exists

DE, 2026-09-04. Written and committed **before any inventory number has
been computed**, at coordinator request (round 59, item 4), so that a
confirmed prediction can be told from a fitted one.

Population as of this writing: the 12-window `v3_4_consumed_fragment`
btc fragment, 4,315 tranches, `de_section81_arms__20260904T125340Z.json`
(committed at `0e8f40c`). Nothing about inventory P&L has been computed
on it.

---

## 1. My two candidates collapse — and the real choice is a different one

I brought (A) *the mid at `t1`* and (B) *the last observed mid before
`t1`*, and said they differ exactly when the gap matters. **Read at the
implementation, they are the same function.** `edge_layer1.py:108-113`:

```python
def mid_at(self, t):
    """Prevailing mid at time `t`. None before the first known quote."""
    if not self.mid_t or t < self.mid_t[0]:
        return None
    i = bisect.bisect_right(self.mid_t, t) - 1
    return self.mid_v[i]
```

It is a **step function held forward**. It returns `None` only *before
the first quote of the window* — never inside a gap. Inside a gap it
returns the last pre-gap observation, because `record_mid()` early-returns
whenever `touch()` is `None` and `advance()` clears state at a gap start.
And `advance(WINDOW_S)` is the last call in the replay, so
`mid_t[-1] <= WINDOW_S` always: **`mid_at(WINDOW_S)` IS the last observed
mid before window end.** (A) and (B) are one expression.

So the distinction I drew was real in intent and empty in code, and my
own NOT_AVAILABLE worry attached to a branch that cannot be reached at
this call site. **The choice that actually exists is about STALENESS, not
about which lookup:**

| | what it does | what it implies |
|---|---|---|
| **(A′) HELD-FORWARD, ALWAYS** | take `mid_at(WINDOW_S)` whatever its age | never `NOT_AVAILABLE` after the first quote; marks the residual at a price the market may have left minutes ago, and *silently*, since nothing in the value says how old it is |
| **(B′) REFUSE WHEN STALE** | `NOT_AVAILABLE`, counted, when the window ends inside a gap (`wf.touched(mid_t[-1], WINDOW_S)`) or the mark is older than a declared bound | refuses to value the residual exactly in the windows where the residual is riskiest — the gap is *when* the position became dangerous |
| **(C′) HELD-FORWARD PLUS ITS AGE** | take the mark and carry `staleness_s` and `ended_in_gap` beside it, and let the admissibility predicate be applied at read time | costs one more stored number; makes (A′) and (B′) both computable from one re-feed instead of two |

**A third thing was never ruled and is not mine to choose either:** what
"terminal" indexes. Inventory is a per-**window** residual, so the mark
is at the window's end (`WINDOW_S`), not at any generation's `t1`. The
word `t1` in the question is the window's, and I am naming that because
`t1` is also a generation field in this codebase.

## 2. Per-slug versus summed — still unruled, and the emission already argues one way

The emission calls the summed terminal net *"reporting-only, carries no
decision meaning"*, which points at per-slug. It matters because
`inventory_loss` is **not additive across windows the way a share count
is**: each window has its own mark, and a summed net position across
windows is a quantity no book ever held. I will emit **per-slug and the
sum of per-slug values**, which is well-defined, and will not emit a
mark-to-market on a cross-window net position, which is not.

## 3. The construction, so this field cannot repeat round 58's defect

Round 58's defect was two quantities struck from the same origin and
added. The same trap is here: valuing the residual **from entry** would
re-count what the markout leg already valued. The non-overlapping split
is exact and is an identity, not a convention:

```
total marked to terminal  = Σ sgn_i · shares_i · (M_T     − level_i)
fills leg  (round 58)     = Σ sgn_i · shares_i · (m_i     − level_i)
inventory leg (this)      = Σ sgn_i · shares_i · (M_T     − m_i)
```
where `m_i` is the mid at `t_i + MARKOUT_S` (= `level_i + sgn_i·mo_i/100`,
so it needs **no re-feed**) and `M_T` is the window's terminal mark. The
inventory leg **continues the mark from where the markout stopped**, so
`fills + inventory == total` is a CHECKED identity and nothing is
double-counted. Sign convention is the markout's: **positive is in the
maker's favour**; `inventory_loss_cents` is §8.1's name, not a claim
about the sign.

**Only `M_T` is new. That is the one stored line.**

---

## 4. THE PREDICTION, declared before the build

**P1 — the baseline's inventory leg is NEGATIVE.**
Mechanically, not as a bet: a resting two-sided quote ends net **long**
precisely in the windows where price fell through its bid, and net short
where price rose. So the signed flow is anti-correlated with the mid's
deviation from the terminal by construction, and
`inventory = −Σ sgn_i·shares_i·(m_i − M_T)` is systematically negative.
This is inventory/trend risk — the same phenomenon as adverse selection,
seen at a longer horizon than `MARKOUT_S = 5.0 s`.

**P2 — cancelling IMPROVES the inventory leg, for both acting arms.**
`inventory(arm) − inventory(baseline) > 0`. Cancels fire in the states
where the declined fill would have left an adverse residual, so the leg
they avoid is the one that hurts.

**P3 — the magnitude is decision-relevant, not a rounding correction.**
`|inventory(baseline)|` between **500 and 10,000 cents**, point estimate
**~1,500 c** (|terminal net| 146.74 shares × a window-scale mid move of
order 10 c). The fills-leg gap it must overcome is **−953.92 c**.

**P4 — THE HEADLINE: inventory REVERSES the fills-leg conclusion for
CONDVALUE_X_SKEW.** Fills + inventory, delta vs baseline, **> 0**. The
fills leg alone says cancelling forgoes more entry edge (2,933.58 c) than
it saves in adverse selection (1,979.66 c); I predict the residual
position it also avoids is worth more than the 953.92 c gap.

### What would make me wrong, named now rather than after

**The cancel policy does not REDUCE the residual — for CONDVALUE it
REVERSES it.** From the committed artifact, before any P&L:

| arm | cancels | terminal net |
|---|---|---|
| QR_SKEW_ONLY | 0 | **+146.74** |
| HAZARD_OVER_SKEWED_REF | 48 | **+94.93** |
| CONDVALUE_X_SKEW | 333 | **−147.28** |

HAZARD cuts the magnitude by 35 %. **CONDVALUE flips the sign at
essentially identical magnitude.** A sign flip at equal size is a
*directional bet*, not risk reduction: its payoff depends on the realised
path, not on the mechanism P2 rests on. So:

* P2 for **HAZARD** rests on the mechanism, and its improvement should be
  **small** (~+100 to +500 c) because the residual and the gap
  (−12.56 c) are both small.
* P2 for **CONDVALUE** rests on the mechanism *and* on a path. **If
  CONDVALUE's improvement is large, that is the reading I distrust**, and
  I am saying so before seeing it: 12 windows cannot separate "the policy
  avoids adverse residuals" from "the policy happened to be short in a
  fragment that fell." A large confirming number here is **not** a
  confirmation of P4; it is a candidate for held-out windows.

**P5 — the falsifier I would accept as decisive against P1:** a
baseline inventory leg that is positive, or smaller in magnitude than
500 c. Either would say the residual carries no systematic cost at this
horizon and that §8.1's inventory fields are reporting furniture.
