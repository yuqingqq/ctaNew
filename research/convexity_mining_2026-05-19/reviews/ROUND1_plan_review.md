# Convexity-Mining Plan v1 — 3-Agent Review (2026-05-19)

Verdicts: Methodology **NEEDS-REVISION** · Profitability **NEEDS-REFOCUS** ·
Red-team **DO-NOT-PROCEED**. Convergent fatal findings:

1. **Selection-endogeneity (all 3, decisive).** The convex-winner label in
   `record_forensic.py` is defined ONLY over legs the OLD production strategy
   entered (`rec[rec["traded"]]`; 33/51 symbols, ~5–7/group). OOS-*symbol*
   holds out symbols but every test row is still an old-selector pick → the
   AUC 0.68 is **selector-echo conditioned on rolling-IC/conv_gate/refill**,
   not an independent convexity property. C1/C2 re-select by this signature →
   closed loop. The "first OOS-portable signal" claim is false in the sense
   that matters.
2. **Volatility detector (red-team verified on data in hand).** Raw `atr_pct`
   alone: AUC 0.70 (big-positive leg), **0.68 (big-negative leg)**, 0.75
   (big-|abs|). The 12-feat signature ≈ this. C0's own kill-condition
   (AUC pos≈neg≈|abs|) is already satisfied. `atr_pct`/`idio_vol` are
   WINNER_21 features shown portable-negative in B★.
3. **Re-derives closed work.** The cohort = VVV/PENDLE/AVAX, the same rotating
   meme tail R1c diagnosed non-portable and R3c showed ports to −0.33.
4. **Friction model is a tunable fiction.** Perps have no equity borrow; real
   short cost = observable funding (already in panel). "∝ idio_vol pctile"
   double-counts vol and is a free goalpost knob.
5. **C1 baseline trap.** "≥ in-universe baseline − 0.10" certifies a
   cost-saving on a −0.33 non-portable meme bet.
6. **Power.** +0.5 LCB>0 gate unreachable on ~5-group/0.74y machinery
   (mean −0.36 ± 0.72); n_eff ~25× overstated by a duplicate-timestamp join.

## Decision
Do NOT run C0–C2 as written (closed-loop, re-derives closed negative on an
artifact). Run the single **mandatory precondition** the reviewers all named:
the convex-winner label/signature rebuilt on the FULL 51-name panel (every
name every cycle, both sides), vol-orthogonalized, OOS-symbol — does a
*positive-convexity-specific*, *vol-independent*, *off-old-selector-manifold*
signal exist at all? Outcome decides: line closed-by-measurement, or plan
rebuilt and re-reviewed. Agent ids: meth `a655a9b43608fccc7`, prof
`a1d395c83f98c69ce`, red `ad8e55006bf5a4f04`.
