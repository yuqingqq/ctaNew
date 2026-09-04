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

---

## Amendment 1 — DA's noise result, recorded BEFORE the inventory number lands

DE, 2026-09-04, still with no inventory P&L computed. Added in-band
(rule 13) rather than by editing the predictions above: **P1–P5 are
unchanged and are not being moved.** What changes is how a confirming
result must be read, and saying that now is the only time it is worth
anything.

DA has interrogated the baseline and the fills-leg deltas:

| arm | observed | expected-at-random | sd | z | p |
|---|---|---|---|---|---|
| CONDVALUE_X_SKEW | −953.92 c | −755.88 c | 973.19 | −0.20 | 0.43 |
| HAZARD_OVER_SKEWED_REF | −12.56 c | −113.77 c | — | +0.26 | 0.60 |

**The fills-leg cost of cancelling is indistinguishable from random.**
And the baseline it is measured against is tail-carried: the top 1 % of
4,315 fills carry 113 % of the net, the other 99 % sum to −13 %, and
88.2 % sits in one clock hour of two.

### What this does to P4

P4 predicts inventory reverses **−953.92 c**. That quantity is one draw
from a **±973 c** band. So:

1. **A reversal is not evidence the sign flipped.** If fills + inventory
   comes out positive, the honest reading is available immediately and I
   am committing to it now: **both legs are noise at this sample size.**
   Reversing a number that is itself indistinguishable from zero is not
   a finding about the policy.
2. **My own P3 makes this worse, not better.** I predicted
   |inventory(baseline)| ~1,500 c with a range of [500, 10,000]. The
   fills-leg sd is 973 c. **My point estimate for the inventory leg is
   the same order as the noise band of the leg it is meant to overturn.**
   Under P4 confirming at the predicted magnitude, the reversal sits
   inside one sd of the fills leg alone.
3. **The concentration check is now mandatory, not optional.** If the
   fills leg is 113 %-carried by its top 1 %, the inventory leg may be
   too, and a net carried by a handful of fills is a different object
   from the same net spread across them. `inventory_pnl` computes the
   top-1 % share **for both legs** so this is visible in the artifact
   rather than inferred.
4. **No interval, and that is stated in the field.** 12 windows is below
   rule 8's five-complete-UTC-day cluster floor. Point estimate only.

### The prediction I am adding, because it is now the one that discriminates

**P6 — the inventory leg is ALSO tail-carried.** Top 1 % of valued fills
carry **> 50 %** of the inventory leg's net. If it holds, then neither
leg of §8.1's economics is a tendency at this sample size and the whole
`cancellation_economics` table is a statement about a handful of fills in
one clock hour — which is a finding about the *population*, not about
cancellation, and it is the finding that would matter most.

If P6 fails and the inventory leg is broadly distributed while the fills
leg is not, that is also informative: the two legs would have different
statistical characters and could not be summed into one verdict without
saying so.

---

## RESULT — 2026-09-04, `de_section81_arms__20260904T134055Z.json`

Emitted from a clean tree at `2a3bb30`, which is on the branch, 7/7
identity files matching. Scored against the predictions above, which were
committed at `11dd46f` and amended at `2ec1fe9` before any of this
existed.

| | prediction | observed | verdict |
|---|---|---|---|
| **P1** | baseline inventory leg **negative** | **+8,587.54 c** | **REFUTED** |
| **P2** | cancelling improves it, both arms | +650.38 c / +3,348.32 c | met |
| **P3** | \|baseline\| in [500, 10 000], point est ~1,500 | 8,587.54 | range met, **point estimate 5.7× off** |
| **P4** | inventory reverses the fills leg for CONDVALUE | total **+2,394.40 c** | confirmed |
| **P5** | *decisive against P1: a positive baseline leg* | **positive** | **P5 FIRED** |
| **P6** | inventory leg also tail-carried, top 1 % > 50 % | top 1 % carry **−0.166** | **FAILED** |

### P1 is refuted by my own declared falsifier, and that is the finding

I predicted the sign of the baseline leg **mechanically**, not as a bet:
a resting two-sided quote ends net long precisely where price fell
through its bid, so signed flow is anti-correlated with the mid's
deviation from the terminal. **It is positive, and by 8,587 c — as large
as the entire fills leg (8,598.76 c).** The mechanism is wrong, or it is
swamped by something I did not model. P5 named exactly this outcome as
decisive against P1 and it fired.

### So P4 "confirming" is worth much less than it looks

P2 and P4 were *derived from* P1's mechanism: cancels help because they
decline fills that would leave an **adverse** residual. **There is no
adverse residual.** The delta came out the way I predicted through a
route I predicted wrongly, which is not a confirmed prediction in any
useful sense — it is a right sign from a wrong model.

And the actual route is the one I named in advance as untrustworthy.
CONDVALUE's terminal net does not shrink, it **flips** (+146.74 → −147.28)
and its inventory leg is *higher* than the baseline's on **fewer fills**.
That is the directional bet, not risk reduction. From Amendment 1, before
the number: *"If CONDVALUE's improvement is large, that is the reading I
distrust."* It is +3,348 c. **I distrust it, as declared.**

### P6 failed as written, and the concentration is one level up

The inventory leg is **not** fill-tail-carried: its top 1 % carry −0.166
of the net — the extremes work *against* it. But the concentration is
real at the **window** level, which is the cluster unit rule 8 cares
about:

| | window | inventory | cum |
|---|---|---|---|
| 1 | …1787580600 **(ended in gap)** | 2,422.25 c | 28.2 % |
| 2 | …1787582400 | 2,391.00 c | 56.0 % |
| 3 | …1787580300 | 2,178.00 c | 81.4 % |

**Three of twelve windows carry 81.4 %.** So the leg is 12 draws of a
per-window directional outcome, not 4,315 fills — and the effective
sample is far smaller than the fill count suggests. Point estimate, no
interval, and now for a stated reason rather than a formal one.

### The ruling earned its keep

`views_disagree_materially` is **true for all three arms**. The single
gap-ended window is the largest contributor to the baseline leg: 2,422 c
of 8,588 (28 %). Under ruling B the baseline is 6,165.28 c over 11
windows. **P4 survives both rulings** (+773.18 c / +4,543.28 c under B),
so the conclusion is not ruling-dependent — but the *level* moves by 28 %
on one window, which is precisely the disagreement (C′) was ruled to make
visible.

### What I would need to believe P4

Held-out windows. 12 windows, 3 of which carry the leg, cannot separate
"cancelling avoids adverse residuals" — a mechanism now **refuted at the
baseline** — from "the flipped position happened to be on the right side
in this fragment."

---

## Correction 1 — my own ordering, corrected against my own table (DE60)

I wrote **"the actionable lever is a cancel that does not cascade"** in
two round summaries. **My own emitted numbers said otherwise and I did
not read them.** DA caught it; the coordinator had already carried the
wrong ordering onward.

| | spread across the arms |
|---|---|
| selectivity | **5.6453** |
| cascade | 1.9399 |

**Selectivity dominates by 2.910×.** The correct ordering is **cheap
fills first, few fills second** — the arms are separated far more by
*which* fills they decline than by *how many* each cancel costs. HAZARD
is the better ranker on **both** factors (selectivity 0.0589 vs 0.3324;
cascade 1.995 vs 3.869), which the "cascade is the lever" framing hid
entirely.

This is now `cancel_mechanics.separation`: both spreads, the dominant
factor and the ordering string are **computed and emitted**. A ratio
nobody computes is a ratio prose will invert, and mine did, twice.
