# Review — iter-007 (fix-round 1)

**Script:** `research/convexity_portable_2026-05-20/scripts/X122_altbear_gate.py`
**Feature under audit:** parameter-free 2-axis alt-bear SIDE gate (F1) — the ONE new branch
`side AND (alt30 < btc30) -> FLAT`. The rest is the validated X121/X117 held-book engine.

## Verdict: **PASS** (G1 satisfied — no look-ahead, evidence is trustworthy)

The result is a pre-registered REJECT; the Review job here is to certify the **evidence** is clean so
Evaluation can rely on the in-sample HL70 +4.73 not being a leak and on the multi-episode / LOFO logic
being sound. It is.

---

## 1. Look-ahead audit on `alt_index_30d` (the only new feature) — CLEAN

**Source (X122:144-153):**
```
altcols   = [c for c in ret4.columns if c not in ("BTCUSDT","ETHUSDT")]
altidx    = ret4[altcols].mean(axis=1)            # eq-weight 4h log-return of own alts
alt_cum   = altidx.cumsum()
alt30_full= (alt_cum - alt_cum.shift(WIN)).shift(1)   # trailing-30d cum log-ret, LAGGED
b30_lag   = (b4/b4.shift(WIN)-1).shift(1)              # btc30 lagged the SAME way
flag      = (rg=="side") and isfinite(a30) and isfinite(b30) and (a30 < b30)   # X122:163
```

- **Trailing + `.shift(1)`:** `alt30` = `(cum − cum.shift(180)).shift(1)` → a strictly trailing 30d
  (180×4h bars) cumulative log-return ending at *t−1*. No current/future bar enters the window.
- **Matched lag:** the comparison `btc_30d` (`b30_lag`, X122:151) is `.shift(1)`-lagged with the SAME
  180-bar window. Apples-to-apples; no lagged-vs-contemporaneous mismatch that could leak. (The regime
  *label* still uses contemporaneous `b30` as in X117 — unchanged, and that path is the validated engine,
  not the new feature.)
- **Independent recompute (I rebuilt alt30/btc30 from raw klines):** **max diff = 0.0** vs the emitted
  `X122_percycle_EXT.parquet` for BOTH `alt30` and `btc30`. The PIT construction is exactly what the
  parquet contains.
- **Per-universe isolation:** HL70 alt30 vs EXT alt30 on 2404 overlapping dates differ by up to 0.26
  (built from each panel's OWN alts: HL70=68 alts, EXT=21 alts, S44 from 44) → **no cross-universe carry**.
  `btc30` matches exactly across universes (same BTC source), as expected.
- **No full-sample normalization, no forward window.** The `cumsum`/`shift(180)` differencing is a
  trailing window; `mean(axis=1, skipna)` over alts only averages symbols live at *t*. Warm-up cycles
  are already removed upstream by `.dropna(subset=["b30"])` (b30 needs 180 bars), so the parquet has
  **0 NaN** alt30/btc30; the explicit `np.isfinite` NaN-guard (X122:163) is the belt-and-braces net.

**Conclusion: G1 PASS on the new feature. No mechanism inflates the HL70 in-sample +4.73.**

## 2. Parameter-free — confirmed; G3 legitimately WAIVED

The gate boundary is the structural ±0 **relative** comparison `alt30 < btc30` (X122:163). There is **no
swept or selected scalar** anywhere (the iter-006 `−0.10` is gone). Like the existing ±10% BTC regime rule,
this is a definition, not a tuned parameter → G3 waiver is correct per the contract.

## 3. Regime-DEFINITION change (not a sizing overlay) — confirmed

A flagged side cycle appends `{}` to `cyc_w_f1` (X122:169) exactly like a bear cycle (X122:167); it does
NOT rescale any leg. Prior bull/unflagged sleeves age out via the normal `heldbook` HOLD=6 decay
(X122:202-217, identical to X121:163-177). Verified the F1 weight-build branches (X122:171-188) reproduce
X121's structure with the flag substituted for the side-condition.

## 4. Correctness of the decisive outputs — VERIFIED

- **Base reproduces X117:** recomputed from `X122_percycle_HL70.parquet` → Sharpe **+1.93**, maxDD
  **−5,674**, Calmar +1.68, totPnL +10,472. **Exact match** to the spec target (+1.93 / −5,674). The
  base-arm construction (X122:171-188 sans flag) is verbatim X121.
- **Per-cycle parquets** for all 3 universes emit the contracted columns
  (`open_time, fold, regime, is_side, is_active_base, alt30, btc30, flag, side_flat, pnl_base/f1 @
  {010,030,045}` + 4.5bps aliases). Flag-integrity check: in ALL universes **0** cycles are
  `flag & ¬side`, **0** are `flag & ¬(alt30<btc30)`, **0** are `flag & NaN`. Counts match handoff
  (HL70 1101/1455, EXT 3523/5232, S44 2596/3650). **0 NaN** in any pnl column (totPnL +nan bug guarded
  at X122:214).
- **G4 placebo re-derives held-book PnL under random masks (NOT row-zeroing):** `heldbook_flatlist`
  (X122:220-225) builds a fresh held book emitting `{}` for masked cycles and runs the full decay engine.
  I verified the placebo's F1 reconstruction (`heldbook_flatlist` with the real flag mask) equals the
  direct `cyc_w_f1` build to **0.0** — so the placebo holds the decay machinery constant and only varies
  WHICH side cycles are FLATted. Matched count (`size=n_flag`), drawn `replace=False` from the side pool,
  200 seeds (≥100). Correct.
- **Episode definitions + episode-LOFO + fold-LOFO:** episode windows are sane — EXT spans 2021-08→2026-05
  and each of the 4 episodes is well-populated (luna 547 / ftx 547 / summer 727 / q4 727 cycles, all >5)
  with real flag counts (35/240/371/319). `episode_report` (X122:272-313) computes per-episode maxDD
  improvement (`ddimp>0.5`, ≥3/4 bar) then episode-LOFO drops each episode's cycles (`keep=~m`) and
  recomputes full-series Calmar lift — correct subset recomputation. `fold_lofo_report` (X122:252-269)
  likewise drops each fold and recomputes `calmar_of` on the retained series. LOFO Calmar is recomputed
  on the dropped-subset series (cumsum/maxDD over the kept cycles) — the standard, correct LOFO design.
- **NaN guards:** explicit (X122:163 flag, X122:214 cyc). No silent NaN→0 hiding missing data.
- **RNG seeded:** `np.random.default_rng(12345)` (X122:442), shared deterministically across
  HL70/EXT placebo + paired CIs. Bootstrap 2000 draws blocked by fold (X122:347-363). Reproducible.

## Minor notes (non-blocking)
- The single `rng` is threaded sequentially through 4 consumers; deterministic and seeded, so
  reproducible, but each run's later consumers depend on earlier draw counts. Acceptable for this report.
- `EXT_EPISODES` (X122:59-64) intentionally omits the 2021_blowoff window (documented deviation #1); the
  pre-registered G5 bar is the 4 alt-bear episodes, all present.

## Bottom line for Evaluation
The HL70 in-sample +4.73 Calmar is **real (no leak)** — it is simply a single-episode (fold-5 / 2025-Q4)
artifact, which the LOFO and placebo correctly expose. The multi-episode EXT test and both LOFO routines
are **sound and trustworthy**. Evaluation can adjudicate the REJECT on the honest numbers.
