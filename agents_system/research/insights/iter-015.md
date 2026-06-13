# Research insight — iter-015 (ALPHA track): is the BULL momentum signal improvable by a better formulation?

**Directive:** the bull sleeve = 100% of HL70 PnL (+6.03 Sharpe in-regime). iter-004 found the current
selection signal `mom30` has ~0 bull cross-sectional IC and that swapping to `pred` is catastrophic
(kills the beta engine). UNTESTED: does a BETTER momentum formulation have genuine bull XS-IC and
improve the bull capture? STEP 2 = a cheap PRE-CHECK (bull XS-IC of a momentum battery) BEFORE building.

**Outcome: NO-CANDIDATE.** No momentum variant has robust bull cross-sectional IC. The one variant
with a non-trivial *HL70* bull IC (`mom_180d`, +0.029 / t=+2.3) has the **OPPOSITE sign** on EXT and S44
(both negative, t≈−2). The sign flips across universes → it is an HL70-2026-specific artifact, not a
transportable selection edge. **The bull regime's PnL is irreducible net-long-BETA capture; it is not
improvable by a better cross-sectional momentum SELECTION signal.** This closes the alpha-improvement
question for the bull sleeve on the same wall iter-004 hit, now confirmed across a full momentum battery
and three universes. No build, no gate run — the pre-check is decisive and gating per the AGENT.md rule.

Script: `research/convexity_portable_2026-05-20/scripts/X129_bull_mom_xsic.py` (227s, seed-free; reuses
X123 `load_close`/`WIN` verbatim; reads only cached preds + klines; modifies nothing). Artifacts:
`results/X129_bull_xsic_{HL70,EXT,S44}.parquet`.

---

## Method (PIT)
For BULL cycles only (BTC trailing-30d > +10%, PIT/lagged — the exact production regime gate), compute
the per-cycle cross-sectional **Spearman rank-IC** of each momentum variant vs the forward target, then
average across cycles with a cycle-level t-stat and a per-fold breakdown. Two targets:
- **`return_pct`** — the RAW 4h-fwd return (what the bull long/short book actually monetizes).
- **`alpha_A`** — the 4h-fwd alpha-residual (ret − β·BTC) (beta-neutral selection skill).

All momentum variants built from 4h closes and **`.shift(1)`** (no current bar). Universes HL70 (prod) +
EXT (2021-26) + S44. Variants: multi-TF {7,14,30,90,180}d; vol-adjusted (mom30/vol, Sharpe-like) +
Barroso-scaled (mom30 × const-vol-target/vol); rank-composite (mean XS rank across 7/14/30/90d);
**residual momentum** (trailing-30d Σ of beta-stripped ret − β·BTC); **trend-quality** (mom30 × Kaufman
efficiency ratio = |net move| / Σ|bar moves|).

## The headline: bull XS-IC vs `return_pct` (the bull book's real target)
| variant | HL70_IC | HL70_t | EXT_IC | EXT_t | S44_IC | S44_t | verdict |
|---|---|---|---|---|---|---|---|
| mom_7d | −0.0094 | −1.05 | −0.0155 | −3.12 | −0.0172 | −3.22 | ~0/neg all |
| mom_14d | +0.0007 | +0.08 | −0.0193 | −3.81 | −0.0140 | −2.53 | sign-flip |
| **mom_30d (current)** | **−0.0016** | **−0.17** | −0.0227 | −4.45 | −0.0167 | −3.04 | ~0 HL70 (confirms iter-004) |
| mom_90d | +0.0149 | +1.66 | −0.0125 | −2.54 | −0.0143 | −2.62 | **sign-flip** |
| **mom_180d** | **+0.0287** | **+2.30** | **−0.0113** | **−2.24** | **−0.0102** | **−1.73** | **HL70-only, sign-flips EXT/S44** |
| mom_voladj | +0.0097 | +1.15 | −0.0160 | −3.29 | −0.0110 | −2.09 | sign-flip |
| mom_barroso | +0.0024 | +0.27 | −0.0216 | −4.29 | −0.0146 | −2.70 | ~0/neg |
| mom_rankcomp | +0.0021 | +0.24 | −0.0225 | −4.49 | −0.0181 | −3.33 | ~0/neg |
| res_mom_30d | +0.0026 | +0.30 | −0.0176 | −3.51 | −0.0089 | −1.62 | ~0/neg |
| mom_trendq | +0.0043 | +0.48 | −0.0207 | −4.15 | −0.0142 | −2.65 | ~0/neg |

(vs `alpha_A` target the pattern is identical — mom_180d HL70 +0.0305/t=+2.47, EXT −0.0113, S44 −0.0106;
every other variant ~0 on HL70 and negative on EXT/S44. Full tables in stdout / the parquets.)

## Reading the result
1. **mom_30d (current) confirmed ~0 in bull on HL70** (+/−0.002, |t|<0.6 vs both targets) — reproduces
   iter-004 exactly. The current selection signal has no bull cross-sectional skill.
2. **Only longer-horizon momentum (90d, 180d) shows positive HL70 bull IC** (mom_180d +0.029, t=+2.3).
   That is meaningfully above mom_30d's ~0 and is NOT just a beta tilt (controlling cross-sectionally for
   trailing beta, mom_180d's HL70 bull IC stays +0.0197; beta's own bull IC is −0.017, and corr(mom_180d,
   beta) is mildly NEGATIVE −0.14). So in HL70 it is a genuine (weak) selection signal.
3. **But it is universe-specific and sign-inconsistent — the disqualifier.** mom_180d (and mom_90d) flip
   to **negative** bull IC on BOTH EXT (t=−2.2) and S44 (t=−1.7). A selection signal that predicts +IC in
   one universe's bull and −IC in two others' bulls is not a transportable edge. Per the contract this
   FAILS G7 (cross-universe robustness) before it is even built — exactly the iter-014 pattern (an
   HL70-only in-sample improvement that does not transport = universe-overfit).
4. **The HL70 positive IC is fold-concentrated in the 2026 bull** (mom_180d fold ICs: f5 +0.163, f6 +0.060
   dominate; early folds ~0). MEMORY notes the 2026 regime is HL70-healthy and 44-sym-decayed — i.e. this
   is the known HL70-2026 composition artifact, not a stable bull-momentum law. A G5/LOFO build would
   collapse on dropping f5/f6, the same one/two-fold signature that has killed every prior in-sample win.

## Verdict — the bull engine is irreducible beta capture (not a stock-selection problem)
- **No momentum formulation has robust bull cross-sectional IC.** Across 10 variants × 2 targets × 3
  universes, the only positive bull IC is long-horizon momentum **on HL70 only**, and it flips sign on
  EXT and S44. There is no variant that is same-signed-positive and significant across universes.
- This **mechanistically confirms iter-004**: the bull +6.03 Sharpe is net-long-BETA capture as crypto
  rises, not cross-sectional alpha. mom_30d is already the right lever to *express* the long-beta tilt
  (high-momentum names carry the beta long, laggards short) — replacing or reformulating the *selection*
  signal cannot add bull alpha because there is no robust bull selection alpha to add. (And iter-004 showed
  the other direction — ranking by the beta-stripped `pred` — is catastrophic because it removes the beta
  engine.) Both directions now closed.
- A pure SIZING / vol-scaling of the bull book is NOT proposed: that is the iter-001 vol-scaling family,
  which failed G4 on HL70 (p0 — the throttle did no better than matched-random de-gross), and it is not a
  selection improvement; out of scope for "improve the selection signal."

## Why this is a useful (closing) result, not a dead end
The directive's STEP 4 explicitly authorizes NO-CANDIDATE as the decisive closure if every momentum
variant is ~0 in bull. That is the result: the bull engine is irreducible beta capture. Combined with the
already-closed prediction axis (iters 5–10) and reaction axis (iters 11–12), the **alpha-improvement
question for the profit engine is now answered on the data** — the only honest levers left are out of
this loop's scope: (a) a different *factor* (paid leading data — flagged, not free); (b) accept the
bull=beta nature and manage the −57% DD with the iter-012 portable reactive stop (already characterized);
(c) a model retrain on fresh data with proper winsorization (fixes the universe-expansion clip, an infra
task, not a per-iteration construction tweak). None is a bull-selection change, which is what this
iteration was scoped to find — and there is none.

### Honest note (not an ADOPT, not even offered as a risk dial)
On HL70 *alone*, biasing the bull-sleeve ranking toward a longer lookback (mom_90d/180d) would raise the
in-sample bull IC. It must NOT be adopted: it fails G7 (opposite sign EXT/S44) and is fold-concentrated in
the 2026 bull (G5/LOFO would collapse). Documenting it here forecloses a future "just use longer-horizon
momentum in bull" proposal — it is HL70-2026-overfit, not a robust bull selection edge.
