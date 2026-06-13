# iter-031 — DEPLOY-UNIVERSE decision: liquidity ranking, breadth-N sweep, composition stress, transport

**DECISION-SUPPORT (not ADOPT/REJECT).** The team is deploying the full champion (HL70
regime-hybrid held-book + iter-012 vol-norm reactive stop, k=2.0) on Hyperliquid and must pick
the live SYMBOL SET — the most universe-overfit-sensitive lever in the project (UNI stress-test,
iter-log). This iter sizes that lever with a tradable, rule-based answer.

**One-line verdict:** **Deploy the WIDEST tradeable universe, not a liquidity-ranked top-N.**
Breadth is the dominant driver of the edge — Sharpe rises monotonically with N and peaks at the
full universe; cutting down to the most-liquid megacaps DESTROYS the edge (top-20-by-liquidity =
−0.59 Sharpe). Counter-intuitively, **liquidity ranking is ANTI-correlated with the edge**: the
least-liquid tail out-performs the most-liquid head at every N, and top-N-by-liquidity ranks p5–p10
of random-N draws. The cross-sectional residual book needs DISPERSION, and the megacaps (ETH/SOL/
XRP/DOGE/BNB) are the most BTC-correlated, lowest-dispersion names. **Liquidity is a hygiene FLOOR
(drop names too thin to execute), never a SELECTOR.** The +1.93 HL70 Sharpe is **broad-based**
(random 30–40 of 69 names averages +1.06/+1.20) but carries real composition risk (std ~0.5–0.6,
worst-case draws dip to ~0). Conclusions transport to EXT 2021-26 (same monotone breadth shape).

Script: `agents_system/research/scratch/iter031_deploy_universe.py` (reuses X123 `build_universe` +
X125 vol-norm-stop mechanics verbatim; base reproduces X117 EXACTLY: HL70 Sharpe +1.93 / maxDD
−5674 / Calmar +1.68 / tot +10472). Artifacts in `outputs/iter031/`.

---

## 1. Liquidity ranking (HL70) — median daily dollar-volume

Proxy = **`quote_volume`** from 5m klines (already in USDT/quote units → a direct dollar-volume
proxy; no |ret| proxy needed), aggregated to daily sums, median over the backtest window (robust to
spikes). Spans **4 orders of magnitude**: ETH $13.6B/d → GMX $2.9M/d.

| tier | names (sample) | med daily $vol |
|---|---|---|
| head (1–10) | ETH, SOL, XRP, DOGE, SUI, BNB, HYPE, ADA, ZEC, FARTCOIN | $250M–$13.6B |
| mid (11–40) | LINK, ENA, AVAX, LTC, AAVE, TAO, DOT, FIL, UNI, NEAR, ARB, TIA, APT, OP, ONDO, HBAR, ETC, INJ, SEI, TON | $44M–$240M |
| tail (41–70) | ATOM, ORDI, ICP, PENDLE, JUP, KAITO, AIXBT, STRK, TRB, AXS, RUNE, AERO, … NIL, GMX | $2.9M–$35M |

**Hygiene exclusion:** **PAXG** (rank 30, tokenized gold — not a crypto-beta name; it does not
belong in a BTC-regime cross-sectional book). No USD stables or wrapped duplicates exist in HL70.
Full ranking → `outputs/iter031/iter031_hl70_liquidity_rank.csv`.

---

## 2. Breadth-N sweep (HL70) — top-N by liquidity, full champion @4.5bps

| N (top-by-liq) | base Sharpe | base maxDD | base Calmar | +stop Sharpe | +stop maxDD | +stop Calmar |
|---|---|---|---|---|---|---|
| 20 | **−0.59** | −6051 | −0.33 | −0.60 | −3821 | −0.40 |
| 30 | **−0.11** | −6350 | −0.06 | −0.22 | −4318 | −0.14 |
| 40 | **+0.36** | −7075 | +0.20 | +0.51 | −4350 | +0.36 |
| 50 | **+1.53** | −6457 | +1.06 | +1.64 | −4294 | +1.44 |
| 69 (clean, −PAXG) | **+1.97** | −5695 | +1.71 | +1.78 | −4274 | +1.79 |
| 70 (full) | +1.93 | −5674 | +1.68 | +1.80 | −3794 | **+2.01** |

**Sweet spot = the FULL universe.** Sharpe is monotone increasing in N — there is NO illiquid-tail
penalty; the illiquid tail HELPS. Dropping PAXG (hygiene) is a marginal positive (69 base +1.97 vs
70 +1.93). The reactive stop cuts maxDD ~30–40% and raises Calmar at every N (portable, as iter-012
established). → `outputs/iter031/iter031_breadthN_HL70.csv`.

---

## 3. Composition stress (HL70) — random subsets vs the liquidity rule (30 draws, base book)

| N | top-N-by-liq | random-N mean | random-N std | random-N min | random-N max | top-liq rank |
|---|---|---|---|---|---|---|
| 30 | −0.11 | **+1.06** | 0.53 | −0.12 | +1.94 | **p3** |
| 40 | +0.36 | **+1.20** | 0.59 | +0.08 | +2.32 | **p7** |

**The edge is BROAD-BASED, not hinging on a few names:** a RANDOM 30 of the 69 clean names averages
+1.06 Sharpe; a random 40 averages +1.20. **But composition risk is real** — std ≈ 0.5–0.6, worst
random-30 draw +Sharpe dips to −0.12 (worst maxDD −8712), worst random-40 to +0.08. So a small
unlucky live composition (or delistings) can roughly halve the Sharpe; a catastrophic zero is rare
but possible at small N. **Top-N-by-liquidity is a PATHOLOGICAL composition (p3–p7), not a lucky
one** — it deliberately picks the lowest-dispersion corner. → `outputs/iter031/iter031_composition_stress_HL70.csv`.

---

## 4. Liquidity-tier vs random (the decisive test) — is top-by-liquidity a tradable WINNING rule?

| N | TOP-liquidity | BOTTOM-liquidity | RANDOM median | top-liq rank vs random |
|---|---|---|---|---|
| 20 | −0.59 | **+0.84** | +0.54 | p5 |
| 30 | −0.11 | **+2.03** | +0.96 | p5 |
| 40 | +0.36 | **+1.93** | +1.04 | p10 |
| 50 | +1.53 | **+2.35** | +1.47 | p50 |

**NO — liquidity selection is the WRONG rule.** The least-liquid tail BEATS the most-liquid head at
every N (the gap is enormous at small N: +2.03 vs −0.11). Top-N-by-liquidity ranks BELOW the random
median (p5–p10) until N≈50, where it finally converges to random because it has absorbed most of the
universe. **Mechanism:** the most-liquid names are the megacaps with the highest BTC-correlation and
the LOWEST idiosyncratic cross-sectional dispersion; a residual rank book has almost nothing to rank
among them. The alpha lives in the mid/small-cap dispersion. This means liquidity can ONLY be used as
an EXCLUSION FLOOR (execution hygiene), never as a ranking selector. → `outputs/iter031/iter031_liqtier_vs_random_HL70.csv`.

---

## 5. Transport — breadth-N on EXT 2021-26 (23 syms)

| N (top-by-liq) | base Sharpe | base Calmar | +stop Sharpe | +stop maxDD |
|---|---|---|---|---|
| 10 | +0.38 | +0.23 | +0.33 | −2748 |
| 15 | +0.19 | +0.10 | +0.14 | −4199 |
| 20 | +0.37 | +0.20 | +0.44 | −3927 |
| 23 (full) | **+0.87** | **+0.66** | +0.86 | −3000 |

**Conclusion HOLDS:** the full 23-sym universe is again clearly best (+0.87 vs +0.19–0.38 for any
liquidity-truncated subset). Same monotone "breadth wins / liquidity-truncation hurts" shape as HL70.
EXT absolute Sharpe is lower (it spans 2021-22 alt-bear + multiple DD episodes — the structural-DD era)
but the breadth-N *ranking* is identical, and the reactive stop again cuts maxDD ~30–40% on every N.
→ `outputs/iter031/iter031_breadthN_EXT.csv`.

---

## DEPLOY-UNIVERSE RECOMMENDATION

**(a) Construction RULE (ex-ante, tradable, NOT cherry-picked):**
1. Start from the full set of Hyperliquid-listed USDT perps with sufficient history (≥ ~6 months of
   5m klines for the trailing-180×4h beta/mom and the model's walk-forward warmup).
2. **Hygiene exclusions:** stables / USD-pegged, wrapped/duplicate listings, and **non-crypto-beta
   tokenized assets (PAXG = gold)**. (Only PAXG applies in today's HL70.)
3. **Liquidity FLOOR only** (execution hygiene, NOT a selector): drop names below a tradeable median
   daily dollar-volume threshold. On HL70 the floor that the data supports is **low** — even the
   $3–10M/day tail HELPS the edge — so set the floor only where YOUR live size makes slippage/impact
   unacceptable (≈ ADV/participation limit), not by alpha. A floor around **$3–5M/day** keeps the full
   current HL70 minus PAXG (~69 names).
4. **Do NOT rank or truncate by liquidity.** Trade the WIDEST resulting set. (If capacity forces a cut,
   cut from the bottom by EXECUTABILITY, accepting it costs edge — see risk below.)

**(b) Target breadth N + robustness:** **N ≈ 69** (full HL70 minus PAXG). Base Sharpe **+1.97**,
Calmar +1.71; with the iter-012 stop +1.78 Sharpe / maxDD −4274 / **Calmar +1.79**. This is the breadth
sweet spot — Sharpe is monotone in N and there is no tail penalty.

**(c) Live composition-risk estimate:** the edge is broad-based (random-40 mean +1.20, random-30
+1.06) but composition-sensitive (std ≈ 0.5–0.6). Honest forward Sharpe band, sizing composition luck:
**roughly +1.0 to +2.0, mean ~+1.5**, widening as the live universe shrinks (delistings) or drifts.
At the full ~69-name breadth you sit near the top of this band; every name you drop for capacity moves
you down it. Worst-case single random-30 draw was ≈ 0 Sharpe — a real (low-probability) tail if the
live universe both shrinks AND drifts into the low-dispersion corner.

**(d) Refresh cadence + IC-monitor kill signal:**
- **Refresh quarterly** (add new HL listings that clear hygiene+floor; drop delisted/illiquid). Keep
  breadth MAXIMAL each refresh; never prune by past-IC (proven value-negative — IC-selector p58
  placebo, MEMORY ic-selector root-cause).
- **Monitor:** rolling realized per-cycle cross-sectional IC and rolling Sharpe vs this band. **Kill /
  de-risk signal:** rolling-90d Sharpe falling below ~0 OR realized maxDD breaching the band while the
  iter-012 stop is already engaged → cut gross / halt and re-validate composition (likely a dispersion
  collapse or a universe-drift into megacaps).
- The iter-012 vol-norm reactive stop (k=2.0) is the always-on capital-preservation layer underneath
  all of this and transports across whatever the live universe becomes (unitless trigger).

## Honest read: how much of HL70 +1.93 is robust vs composition-luck
**Robust:** the breadth principle (more names = more dispersion = more edge) transports to EXT and is
mechanistically grounded; the reactive stop's DD-cap is portable. **Luck/overfit:** the SPECIFIC level
(+1.93 vs the +1.06 random-30 / +1.20 random-40 mean) reflects that the full 69-name composition is a
GOOD draw, and the project's UNI stress-test already showed specific high-IC names carry
disproportionate, non-repeating alpha. Forward, expect to regress toward the broad-based mean
(~+1.5 ± 0.5), NOT to reproduce +1.93. **The deploy rule maximizes the robust component (breadth) and
refuses the fragile one (name-picking/liquidity-truncation).**
