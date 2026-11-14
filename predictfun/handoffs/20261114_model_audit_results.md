# MODEL AUDIT: are the guard and the keep / cancel model reasonable for our task? (2026-11-14, session ce68ebe0)

USER: "Can you check if the model is good for our task, or check if the model is reasonable at all? We shouldn't have triggered so many pulls, is our training target aligned with our goal, or do we train it properly".

**POST HOC and descriptive.** Script `pf_model_audit.py` (a1b88d8b, committed before the run). Run 16:23Z on the claude slot, peak 458 MiB. Data: the 9 consumed replay days, first 60 s; the test days for the model's decisions.

## Answer

**The way we train is sound** (chronological folds, nested out-of-fold chains, look-ahead controls, no leak found). **What we train for does not match our goal.** Our goal is to keep queue place, collect the good fills and step out only when a move is about to hit the order. Neither model's target sees the value of the queue place.

### The guard: what triggers the pulls

- **What it predicts** (pm_ladder_v1's frozen guard, pf_badfill_info M1):
  - The target is the 1 s markout of being swept: a trade going through the best price within the next 500 ms, else 0.
  - It is built from a neutral grid (no own orders, no queue).
  - It does not count the fills that stop at our price (the profitable ones) or the queue place.
  - G2: 88.6% of its training targets are exactly 0, 9.2% negative, 2.2% positive; mean -0.226 c/sh. Its predictions are above 0 only 3.2% of the time. By the declared rule that is NOT-COST-ONLY, because of the 2.2% positive targets; read from the code, the target only scores sweeps.
- **When it pulls:**
  - The threshold is a quantile, not a break-even. c60 is the 60th percentile of its own predictions (-0.086 c/sh), and the ladder pulls at c60 - 0.10, i.e. a predicted sweep cost <= -0.186 c/sh.
  - Nothing weighs what staying earns, or what a pull costs in queue place.
- **The result is flicker** (G1): the guard is on in 33.9% of 50 ms steps, with 54.5 alarm onsets per side-minute. On spells last a median 100 ms (p75 350 ms) and off spells a median 150 ms (p75 650 ms). It flips every few hundred ms.
- **Every alarm pulls every rung.** The target is about the best price only, and most alarms see no move (PULL-DEPTH: 60% no move within 1 s, 20% one tick).
- **It was built for a different setup:** pm_ladder v1/v3's single order at the best, where stepping out is cheap.

### The keep / cancel model (ORDER-HARM v3)

- **Pulls are free in training** (K1, FREE-PULLS).
  - Within the label's horizon (the clear + 2 s), keep-all and pull have the same value in most rung-episodes:

    | Rung | 0 | 1 | 2 | 3 | 4 | 5 |
    |---|---|---|---|---|---|---|
    | Ties, 2 s tail | 66% | 75% | 81% | 84% | 87% | 87% |
    | Ties, 5 s tail | 55% | 63% | 68% | 73% | 76% | 77% |

    Rungs 2-5 pooled: 84.8% with the 2 s tail.
  - No fill on either branch means pulling costs nothing in training. In reality a pull throws away the queue place, and that cost shows up after the horizon.
- **Most decisions have no consequence in the target** (K3).
  - 94.6% of the model's pull decisions and 89.3% of its keeps have realized G = 0: the choice changes nothing within the horizon.
  - Only 1.7% of pulls were costly (keeping was better). 3.7% were right.
  - So most pulls are decided where the target carries no information.
- **Decisions sit at the threshold** (K2).
  - 39% of pull decisions have A_hat between -0.01 and 0.
  - Decisions flip on 8.1% of consecutive steps (STABLE by the declared 10% rule).
  - With a decision every 50 ms the pulls still compound. An order kept at the onset meets a pull later in the same episode in:

    | Episode length | Share pulled later |
    |---|---|
    | 100-150 ms | 8% |
    | 200-300 ms | 23% |
    | 0.35-1 s | 42% |
    | > 1 s | 66% |
- **Other misalignments, from the design:**
  - The pull branch is pull + re-post at the first step after the clear. "Wait" is not an option.
  - It decides only inside the guard's episodes, so it inherits the guard's flicker.
  - Its labels come from a single-rung lane at M 50, while the replay is the joint ladder.
  - The policy iteration did not converge (21% of decisions flip at iteration 3).

## Previous evidence to weigh

QR2 (FINDINGS: "Keeping orders through guard flicker is worse per share") and KEEP-PLACE found that keeping more was worse in the replay. Those tests used the same guard and the same queue model, and their losses also sat on E2-timed trades (TIMING-SPLIT). The audit does not overturn them. It shows the models could not have learned the value of keeping, because their targets do not contain it.

## What an aligned target needs (proposal, USER's call)

1. **The value of the queue place.** Value keep vs pull over the order's life, until it fills or the window ends (60.2 s, as R652). The pulled branch re-enters at the back, so a pull's cost is in the label rather than a free tie.
2. **Both sides of staying.** Score every fill the order would get (STOP fills at our price included), not only the sweep cost.
3. **An economic decision rule.** Pull when the expected loss before the cancel lands exceeds the queue value lost by pulling. Add hysteresis so a decision is not re-made every 50 ms on noise.
4. **Rung-specific risk.** A rung is at risk only if the move reaches its price.
5. **On-policy labels.** Labels from the joint ladder the policy runs in.

## Files

- **Outputs:** `data/predict_fun/research/model_audit_20261114/` (`model_audit.txt`, `model_audit.json`).
- **Review copies:** `predictfun/reviews/model_audit_20261114/`.
