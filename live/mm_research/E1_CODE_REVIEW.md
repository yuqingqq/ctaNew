# E1_CODE_REVIEW — adversarial audit of `e1_markout_scan.py` vs EXPERIMENT_PLAN.md §1

Reviewed 2026-08-19, blind to gate numbers (`e1_gate_summary.csv` /
`e1a_gate_summary.csv` values NOT read; only headers, `e1a_universe.csv`
(written pre-cost by design), and validity/intensity diagnostic columns).
Provenance: outputs written 13:17–13:18 UTC by the code mtime-stamped 13:15:06 —
the reviewed code is the code that produced the outputs. Scan completed in
~2m40s, so re-running after patches is cheap.

Test artifacts: `/tmp/claude-1001/.../scratchpad/test_e1.py`, `test_e1a.py`
(snippets inlined below).

---

## A. Sign conventions end-to-end — PASS

- `sweeps()` L87: `sign = np.where(mkr_buy, -1.0, 1.0)` — `is_buyer_maker=True`
  ⇒ taker sold ⇒ q=−1 ⇒ maker-bid fill. Matches §1.1 exactly.
- Markout L182: `mo = -sgn*(m1-ps_)/ps_*1e4` = MO_j(τ) = −q·(m̂(t+τ)−p)/p ✓
  (sketch §2(g) canonical form, denominator p).
- Λ L183: `sgn*(m1-m0)/m0` ✓; es L175: `sgn*(ps_-m0)/m0` ✓ (both /m per (g)).
- Side masks L193: `makerbid = sgn<0` (taker sold hits resting bid) ✓,
  `makerask = sgn>0` ✓.
- Flip pair L220: `espair = sgn[:-1][sel]*(ps_[:-1][sel]-ps_[1:][sel])` =
  q_j(p_j−p_{j+1}) ✓ (buy→sell gives ask−bid ≥ 0; verified symmetric for the
  deterministic buy-before-sell ordering of same-ms opposite sweeps — ES_pair
  value is identical under either ordering, no bias).
- E1-A L310-313: buys filled by taker-SELL sweeps (`sells = sgn<0`, uses
  `pmin`), sells by taker-buy sweeps (`pmax`) ✓. Chase L350:
  `p_x = mTp + sign*es_day/2` (buy crosses to ask) ✓. Fill cost L339:
  `sign*(L-mhat0)/mhat0 + fee_maker` ✓ (§1.5.4).

Synthetic test (4 sweeps, mixed sides): sign vector `[+1,−1,+1,−1]` as
expected. No sign slip found anywhere.

Minor: `as60` L343 = `sign*(m60-L)/L` is post-fill **markout** (positive =
favorable), i.e. −AS, while the sketch's E1-A "+AS" term is a cost (positive =
bad). Descriptive-only column; see NOTED-6.

## B. Look-ahead — PASS (code); one prereg-sanctioned look-ahead flagged as a PLAN defect

- `mid()` L105-106: `searchsorted(t, u, "left") - 1` = last sweep strictly
  before u. Verified empirically: `mid(100)` with a sweep AT t=100 returns
  `ok=False` (nothing strictly before) — **a sweep is never in its own m̂(t_j⁻)**.
  Same-ms opposite-side sweep also excluded (strictly-before applies to both
  side arrays). m̂(t_j+τ) may include sweep j itself — that is the last-print
  proxy by design, declared as H2.
- 10 s validity boundary: print exactly 10 000 ms old still valid
  (`>= u-VALID_MS`, L109) — inclusive edge, consistent with "within trailing 10 s".
- Day-boundary: per-day MidProxy means m̂(t+τ) past day end uses only that
  day's prints — extra staleness, never look-ahead; validity rule kills most
  such cells. Affects ≤ τ_max/86400 ≈ 0.35% of events. Neutral-to-pessimistic.
- **ES_day same-day median (E1-A placement + chase, L472/L326/L350): the
  episode at 00:00 uses the full day's median flip-bounce — in-day look-ahead.
  The prereg §1.5 step 1 explicitly says "ES_day = that day's median
  flip-bounce", so this is a PREREG DEFECT, not a code bug.** Direction:
  weakly optimistic (day-level parameter fitted in-sample, placement and chase
  half-spread "correctly sized" per realized day); it does not peek at
  directional returns. Log as a plan-defect note; E2-A supersedes with real
  books. Empirical guard check: `es_med_price_1s` minimum across ICP/LTC/AAVE
  × 31 d = +1 tick — never ≤ 0, so the unguarded "negative ES ⇒ order placed
  through the mid at maker fee" hazard does NOT fire on this window (no guard
  in code though; see NOTED-8).

## C. Validity rule / <50% cell exclusion / proxy_incoherent — FAIL (one real bug, see G/MUST-FIX-1)

- 10 s two-sided validity: `val = v0 & v1` (mid valid at t_j AND t_j+τ), all
  markout stats on `sel = mask & val` ✓. Dropped fraction reported as
  `valid_frac` per (symbol, day, τ, side) ✓.
- <50% cells: rows ARE written to `e1_markout_daily.csv` (plan says "not
  reported"; reported-but-excluded is more transparent — harmless deviation),
  and the gate filters `valid_frac >= 0.5` (L386) for the **mean/sign-frac
  path only**. **The per-side means (gate cond 4, L389-392) and the bootstrap
  bins (gate cond 2, L396) do NOT apply the <50% exclusion** — the plan's
  exclusion rule (§1.2) is violated for two of the five gate conditions. This
  is not a corner case (empirical, from validity columns only):

  | symbol | days <50% valid at τ*=30 | days in mean_rs | days in CI/per-side |
  |---|---|---|---|
  | GMXUSDT | 31/31 | 0 (mean=NaN → auto-fail) | 31 |
  | ICPUSDT | 29/31 | 2 | 31 |
  | ATOMUSDT | 28/31 | 3 | 31 |
  | LTCUSDT | 20/31 | 11 | 31 |
  | APTUSDT | 10/31 | 21 | 31 |
  | ARBUSDT | 3/31 | 28 | 31 |

  Gate conds 1/3 and conds 2/4 are computed on different day populations for
  6 of 16 symbols. **MUST FIX 1.**
- NaN propagation: `valid_frac` NaN (empty side) fails `>= 0.5` → excluded;
  `_wm` returns NaN on empty selections and pandas `mean(skipna=True)`
  drops them — no silent NaN → 0 coercion found.
- `proxy_incoherent`: implemented per logged deviation D-b (flag if identity
  gap > 0.2 bps at ANY τ, eq-weighted all-side, L189-191); excluded from all
  gate paths via `excl` ✓. Empirically fired on 0 of 496 symbol-days. Note the
  0.2 bps threshold also absorbs the plan's own denominator mismatch (MO is
  /p, es−Λ is /m; cross-term ≈ MO_bps·es_bps·1e-4 ≪ 0.2 bps here) ✓.

## D. Sweep collapse — PASS

Verified on a constructed array with duplicate timestamps at start and end
(`test_e1.py`): groups `(100,F)×2, (100,T)×1, (200,F)×1, (300,T)×2` →
qty-weighted prices `[10.5, 12.0, 10.5, 9.125]`, `Q=[2,2,1,4]`,
`n=[2,1,1,2]`, correct pmin/pmax per group. `lexsort((m.int8, t))` gives
contiguous (t, side) groups; `reduceat` boundaries from
`flatnonzero(key_change)` include index 0 and run to array end via
`np.diff(np.append(starts, len(t)))` ✓. First expression at L57-58 of
`load_day` is dead code (immediately overwritten) — see I.

## E. Statistical machinery — CONCERN (three items, none a code bug per se)

- **Day-clustered t-stat: never computed anywhere.** Plan §1.4 declares it
  "co-primary", but no §1.5 gate condition and no §1.6 schema column uses a t.
  So the code faithfully implements the gates; the t-stat is a prereg gap
  (declared machinery with no consumer). NOTED-3; do not add a column post-hoc
  without logging an amendment.
- Stationary bootstrap L143-163: correct Politis–Romano — uniform random
  block starts, `Generator.geometric(1/exp_block)` lengths (verified mean 7.99,
  support ≥ 1), circular `(start+arange) % n`, percentile CI, fixed seed
  20260819 (reused across calls — deterministic, harmless). Constant-series
  CI degenerates to the mean ✓.
- **Estimand mismatch (plan-sanctioned):** the bootstrap CI is on the pooled
  bin-weighted mean (ratio of sums), gate cond 1 on the mean of day-means.
  Adversarial synthetic (4 heavy bins @+10, 4 light @−10): pooled = +9.8,
  day-clustered = 0.0, CI = (8.7, 9.97). §1.4 itself prescribes both, so this
  is a PLAN inconsistency to keep in mind when cond 1 and cond 2 disagree on
  symbols where markout correlates with activity. NOTED-4.
- **30-min-bin path exists only for τ=30** (bins stored at `tau == 30`,
  L206-212). The `ts != 30` fallback (day-level bootstrap, exp_block=3,
  L399-402) deviates from §1.4's "48 bins, expected block 8" and is NOT in
  the logged deviations D-a/b/c. **Empirically latent: all 16 symbols get
  τ* = 30** (replicated `tau_star()` blind from `t_opp_med_s`), so the
  fallback never fired in this run — but E1x reuse will hit it. SHOULD FIX
  (store bins at 60/300 too, or log as D-d).
- E1-A bootstrap: day-mean sequence, exp_block=3, logged as D-c ✓.

## F. E1-A episode mechanics — CONCERN (silent skip bias, see MUST-FIX-2)

- Touch vs sweep: buy tested against opposite-sweep `pmin` (`<= L` touch,
  `< L` strict), sell against `pmax` — correct sides and strictness (L328-333).
  Fill search `searchsorted(tt, t0, "right")` excludes a sweep exactly at t0
  (could not have known our order) ✓; `argmax(seg)` = first hit ✓.
- Tick snapping L327: buy `floor`, sell `ceil` — away from market ✓.
- Chase L350: `p_x = m̂(t0+Tp) + sign*ES_day/2` ✓ (§1.5.4); drift reported ✓.
- **T_p=3600 day boundary: the `else 23` branch (L315) is DEAD CODE —
  `3600*1000 <= 3_600_000` is True, so ALL T_p get `range(24)`.** However this
  is self-consistent and unbiased: the hour-23 window [23:00, 24:00) lies
  entirely within the day file, and the chase mid at exactly next-midnight
  uses strictly-before (same-day) prints. Nothing dropped, no bias, no
  reporting needed; the dead branch is cosmetic. The audit-brief premise
  "hour 23 dropped" is false in the code — and correctly so.
- **n_episodes accounting: skips are counted internally (`rec["n_skip"]`,
  L317/321/347) but NEVER written to any output**, and the prereg schema has
  no skip column. Two silent drop paths:
  1. t0-mid invalid → both directions skipped (symmetric);
  2. **no-fill AND chase-mid invalid at t0+Tp → the episode vanishes from the
     CHASE branch only (L345-347)** — the fill branch can never be skipped.
  Empirical (test_e1a.py, hourly grid, full 31 d): t0-invalid = **69.6% of
  hours on ICPUSDT**, 51.3% LTC, 37.9% AAVE; chase-mid invalid at t0+600 s =
  **43.4% of started hours on ICP**, 40.9% LTC, 23.4% AAVE (upper bounds on
  the affected no-fill fraction — quiet tape ⇒ mostly no-fill ⇒ mostly
  dropped). This deletes exactly the quiet-tape winner's-curse branch the plan
  says is "captured mechanically" (§1.5.4), biases fill_rate UP and eff_RT
  DOWN = **maker-optimistic beyond declared H1/H2**, and silently deviates
  from the mandated 24-episodes/day grid. ICP and LTC ARE in the XS-overlap
  set (see H/J). **MUST FIX 2.**
- Fee constants: maker 1.8/1.44, taker 4.5, c_safe 0.5 — match §0 ✓. E1-A
  uses VIP0 as gated ✓. `eff_RT = 2×eff_leg` ✓ symmetric-leg per prereg.

## G. Gate logic §1.5 — FAIL (inherits C's inconsistency); logic otherwise faithful

- All 5 screen conditions implemented (L406-410): mean ≥ fee+0.5 ✓, ci_lo ≥
  fee ✓, sign_frac ≥ 0.70 ✓ (fraction-of-used-days reading; 22/31 = 0.71 ✓),
  both sides > 0 ✓, VWAP sign agreement ✓ (day-clustered mean of
  `mo_vwapmid_bps`, sign vs mean_rs). But conds 2 and 4 read unfiltered day
  populations (bug C) — the five conditions are not evaluated on a common
  sample for 6/16 symbols.
- Tier logic: pass_vip0/pass_vip1 separate ✓; `e1b_final = "pending_e1x"`
  when any tier passes, never "pass" ✓ (L419); `e1x_quarters_pos` left
  empty ✓.
- τ* rule (L371-376): `(day-median t_opp ≤ 30).sum() >= 24` → 30; else
  `min(300, 2·median)` with `60 if cand <= 60 else 300` — correct round-UP
  behavior at the 60 boundary (cand=60→60, 60<cand≤300→300). Day-median
  pools the two side rows (median of 2 = their mean) — reasonable reading of
  "day-median time-to-next-opposite". NaN days count as failing ≤30 —
  conservative ✓.
- NaN flow: GMX has days_used=0 → mean_rs=NaN → all conds False → "fail".
  Numerically safe, but this is a **no-estimate**, not a measured fail — and
  §1.2 says "fails are final". GMX is precisely the wide-spread Variant-B
  archetype; auto-failing it on mid-validity starvation rather than economics
  is a potential false kill. SHOULD FIX (label, not logic) — see J.

## H. Schema fidelity — PASS

All six CSV headers verified against §1.6 character-for-character (headers
read; values not read). `es_med_price_1s` is correctly absent from the
public `e1_spread_daily.csv`: `sp.drop(columns=[...]).to_csv(...)` (L483)
chains `to_csv` on the copy that `drop` returns — correct pandas idiom; the
in-memory `sp` keeps the column, which the gate path never reads beyond
flags. `mo.drop(columns=[], inplace=True)` (L482) is a no-op oddity.
`e1a_universe.csv` is extra vs §1.6 but mandated by §1.5 ("WRITTEN INTO the
output before any cost is computed") and was written at 13:15:27, before any
markout/cost rows existed ✓ prereg order honored. Intensity side labels
`takerbuy/takersell` — plan does not pin labels; fine.

## I. Performance/correctness traps — PASS (with cosmetic debris)

- **int64 ms conversion (L57-63): correct final behavior, dead first branch.**
  The L57-58 ternary (which WOULD double-divide ms by 1e6) is unconditionally
  overwritten by the L60-63 block whenever dtype kind is "M". Verified on real
  data: `datetime64[ms, UTC].astype(int64)` → 1.784e12 (ms), `max > 1e16`
  False → no division ✓; an ns-unit column at 1.78e18 would be divided ✓. A
  µs-unit column (1.78e15) would slip through undivided — not present in this
  dataset (all files ms). Cosmetic: delete L57-58.
- searchsorted arrays: `tb`/`ts` strictly increasing by construction (same
  (t, side) collapsed), `ta`/`t_s`/`t_b` non-decreasing from the lexsort —
  all sorted ✓.
- `np.clip` guards (L108, L252): clipped indices only used where the `ok`
  mask already carries the invalid flag through `&` — garbage values never
  escape unguarded ✓ (all e1a mid uses check `ok[0]` first).
- Q modal (L254-256): exact-float `np.unique` on decimal-string-derived
  floats — identical decimals hash identically; modal and
  `|ratio−round(ratio)| < 1e-9` are sound. Modal computed per side, plan says
  "day's modal Q" (§1.3d) — micro-deviation, defensible (the block is
  per-side).
- tick_size: integer-multiple tolerance is 1e-6 relative vs plan's 1e-9 —
  pragmatic float64 accommodation, unlogged micro-deviation; GCD fallback at
  1e8 scaling safe for USDM tick grids.
- `pinned` adds an undocumented `len > 10` minimum-pairs guard — sane,
  unlogged.

## J. Hazard consistency — CONCERN

Optimistic beyond declared H1/H2:
1. **E1-A chase-skip selection (F): the dominant undeclared optimism.** Only
   the no-fill branch can be dropped; drops concentrate on quiet tape (wide
   effective spread, worst chase). Gate quantities `fill_rate`, `eff_rt_bps`
   biased favorably, materially for ICP/LTC/AAVE-class names.
2. ES_day in-day median (B) — prereg-sanctioned, weakly optimistic.
3. E1-A estimates come only from mid-valid (busy) hours — 24/day grid
   silently thinned to the busiest ~30-60% on thin names; busy hours have
   tighter spreads/faster fills.

Pessimistic (false-kill risks):
4. GMX (and nearly ICP/ATOM) auto-fail E1-B via validity starvation, not
   economics (G). The <50% rule is prereg'd, so this is a plan-accepted
   pessimism — but "fails are final" should be read as "no-estimate" for
   days_used ≈ 0 names; E1x/HL should not treat GMX as an economic negative.
5. Gate cond 2's CI includes the low-validity days the mean excludes (C) —
   direction ambiguous per symbol, but any disagreement between conds 1 and 2
   on ATOM/LTC/ICP/APT/ARB is currently uninterpretable. Fix before reading.

Also noted: XS-overlap resolved to 12 pilot names **including ICPUSDT at ADV
rank exactly 40** under the D-a stale (2026-05-30) ranking — a boundary call
that fresh ADV could flip; E1-A aggregate should be read with/without ICP in
mind (ICP is also the worst skip-bias name).

---

## Verdict table

| # | Severity | Item | Where |
|---|---|---|---|
| 1 | **MUST FIX** | <50%-validity day exclusion missing from per-side means (gate cond 4) and bootstrap bins (gate cond 2); 6/16 symbols affected, ICP mean on 2 days vs CI on 31 | `build_gate_summary` L389-402 |
| 2 | **MUST FIX** | E1-A silent episode drops: skips counted but unreported; no-fill chase branch selectively deleted on invalid chase-mid (up to ~43% of started hours on ICP); undeclared optimistic bias on the gate quantity | `e1a_day` L317-353 |
| 3 | SHOULD FIX | τ*≠30 day-level bootstrap fallback is an unlogged deviation (latent this run — all τ*=30 — but live for E1x); store bins at 60/300 or log D-d | L206, L399-402 |
| 4 | SHOULD FIX | `e1b_final` conflates no-estimate (days_used≈0, GMX) with measured fail | L419 |
| 5 | SHOULD FIX | chase-branch sensitivity: book skipped chases at last stale mid as a shadow column to bound the MUST-FIX-2 bias | `e1a_day` |
| 6 | NOTED | `as60_fill_bps` is +markout (−AS); sketch's AS is a cost — document sign in any readout | L343 |
| 7 | NOTED | ES_day same-day median = prereg-sanctioned look-ahead (plan defect, §1.5.1); E2-A supersedes | plan §1.5 |
| 8 | NOTED | no guard against ES_day ≤ 0 (did not fire: min = +1 tick over ICP/LTC/AAVE × 31 d); add `max(es_day, tick)` if reused on E1x | L326 |
| 9 | NOTED | bootstrap CI estimand (pooled weighted) ≠ cond-1 estimand (day-clustered) — plan-sanctioned; caution if conds disagree | plan §1.4 |
| 10 | NOTED | day-clustered t declared co-primary in §1.4, computed nowhere, needed nowhere — prereg gap | plan §1.4 |
| 11 | NOTED | dead `else 23` branch; hour-23+3600 s is fully within-day, unbiased | L315 |
| 12 | NOTED | dead first conversion expression in `load_day`; µs-unit input would break the guard (none present) | L57-58 |
| 13 | NOTED | `<50% cells` written to CSV rather than suppressed (transparent; plan said "not reported") | L199-205 |
| 14 | NOTED | ICP in XS-overlap at ADV rank exactly 40 under stale D-a ranking — boundary-sensitive inclusion | `e1a_universe.csv` |
| 15 | NOTED | tick tolerance 1e-6 vs plan 1e-9; per-side modal Q; `pinned` len>10 guard; §1.3e's "pinned is an E1-A placement input" not implemented (§1.5 episode design governs — plan-internal inconsistency) | L132, L254, L228 |

## Minimal patches (for the orchestrator; NOT applied)

**MUST FIX 1** — common day-exclusion set for all five gate conditions:

```diff
 def build_gate_summary(mo, sp, intens, bins):
     rows = []
     for sym in SYMS:
         ts = tau_star(intens, sym)
         excl = set(sp[(sp.symbol == sym) & (sp.proxy_incoherent_flag)]["date"])
+        # §1.2: <50%-valid (symbol, day, τ) cells are excluded from ALL aggregation,
+        # not only the mean/sign-frac path.  ~(x >= 0.5) also catches NaN.
+        cell = mo[(mo.symbol == sym) & (mo.side == "all") & (mo.weighting == "eq")
+                  & (mo.tau_s == ts)]
+        excl |= set(cell[~(cell.valid_frac >= 0.5)]["date"])
         sub = mo[(mo.symbol == sym) & (mo.side == "all") & (mo.weighting == "eq")
                  & (mo.tau_s == ts) & (mo.valid_frac >= 0.5) & (~mo.date.isin(excl))]
```

(per_side at L390-391 and `bs = bins[...]` at L396 already filter on
`~mo.date.isin(excl)` / `~bins.date.isin(excl)` — enlarging `excl` fixes both
with no further edits.)

**MUST FIX 2** — surface the skip accounting (episode logic unchanged; log as
amendment D-e: one extra column appended to `e1a_overlay_daily.csv`, and
all-skip symbol-days now emit a row instead of vanishing):

```diff
             nf, nc = len(rec["fill"]), len(rec["chase"])
-            if nf + nc == 0:
+            if nf + nc == 0 and rec["n_skip"] == 0:
                 continue
             fr = nf / (nf + nc)
+            fr = nf / (nf + nc) if nf + nc else np.nan
             cf = float(np.mean(rec["fill"])) if nf else 0.0
             cc = float(np.mean(rec["chase"])) if nc else 0.0
-            eff_leg = fr * cf + (1 - fr) * cc
+            eff_leg = fr * cf + (1 - fr) * cc if nf + nc else np.nan
             rows.append({
                 "symbol": sym, "date": day, "tp_s": tp, "fill_rule": rule,
                 "n_episodes": nf + nc, "fill_rate": fr, "cost_fill_bps": cf if nf else np.nan,
                 "chase_frac": 1 - fr, "cost_chase_bps": cc if nc else np.nan,
                 "drift_nofill_bps": float(np.mean(rec["drift"])) if rec["drift"] else np.nan,
                 "eff_leg_bps": eff_leg, "eff_rt_bps": 2 * eff_leg,
                 "as60_fill_bps": float(np.mean(rec["as60"])) if rec["as60"] else np.nan,
+                "n_skipped": rec["n_skip"],
             })
```

(`build_e1a_summary`'s `groupby("date")["eff_rt_bps"].mean()` skips the new
NaN rows automatically.) For the fuller SHOULD-FIX-5 treatment, additionally
count the invalid-chase-mid episodes into a `n_skip_chase` split and/or book
them at the last available two-sided mid (staleness unbounded) in a shadow
`cost_chase_stale_bps` — that bounds the bias without touching the
pre-registered columns' definitions.

After both patches: re-run the scan (≈3 min), THEN read gates. Gate-reading
caveats even post-fix: E1-A numbers on thin overlap names remain busy-hour
conditioned (skip counts now visible); GMX/ICP/ATOM E1-B rows are
no-estimate, not economic fails; ICP's overlay inclusion is D-a
boundary-sensitive.

---

## Results audit (run 3, 2026-08-19 13:36 outputs)

Amendments D-e..D-j verified implemented correctly in code before this read
(common `excl` set incl. <50%-valid days; D-i integer-tick bracket confirmed
live: touch fill-rate > sweep fill-rate on all 12 symbols, no eff_rt
collapse; D-j literal ≥22-day count correctly demotes ATOM's 3-day sample;
D-f labels GMX/ICP/ATOM `no_estimate`).

### 1. ADAUSDT E1-B pass — verdict: **ARTIFACT-LIKELY as maker economics** (mechanism = H1 eq-weighting, NOT H2 staleness)

(a) **Decomposition** (all/eq, day-clustered, valid days): es_half flat at
2.83 bps for ALL τ; Λ(τ) = 0.21/0.33/0.39/0.39/0.37/0.22 at τ=1..300 s; rs
2.62→2.44→2.61. Λ does NOT grow with τ — it peaks at 15-30 s and DECAYS by
300 s; rs at τ=300 ≈ rs at τ=1. No adverse-selection ramp at any horizon.
Context: ADA tick = 1e-4 on a ~$0.18 price = **5.64 bps tick**, pinned
31/31 days, es_med_1s = 5.63 bps = exactly 1 tick. So es_half ≡ half-tick
mechanically and rs ≈ half-tick − Λ. Same shape on SOL/DOGE/XRP at their
smaller ticks (rs +0.53/+0.48/+0.29, all sub-fee) — ADA "passes" only
because its tick is 5.6 bps wide.

(e) **Staleness test — NEGATIVE (H2 exonerated).** rs(30 s) recomputed from
raw on 5 days with tightened validity: 10 s → 5 s → 2 s gives 2.78/2.77/2.77,
2.80/2.82/2.91, 2.15/2.13/2.05, 2.37/2.38/2.33, 2.59/2.61/2.76. Materially
unchanged; the +2.4 is not stale-mid inflation.

(b) **The kill: weighting flip.** Same file, same days, τ*=30:
eq-weighted +2.443 (31/31 days > 0) vs **notional-weighted −0.322 (only
7/31 days > 0)**; per side notional: makerask −0.60, makerbid −0.09.
Persists at 2 s validity (eq +2.05 / notional −1.33 on 08-03). And it is
universal: SOL −0.53 (0/31 days > 0), DOGE −0.51 (2/31), XRP −0.55 (1/31),
ATOM −0.65, ICP −0.41 under notional weighting. The eq-positive numbers are
the count-dominated tiny-sweep bounce inside a fat tick; **per DOLLAR of
maker fill, markout is negative everywhere in the panel.** In a 1-tick
pinned book the marginal entrant is at the back of an enormous single-level
queue and fills precisely when the level is swept — i.e. in the big sweeps
that carry the negative markout. This is exactly declared hazard H1 made
measurable; the H1 haircut visible in the file (eq − notional ≈ 2.8 bps)
exceeds ADA's margin over fee (2.44 − 2.30 = 0.14 bps) by ~20×.
Burst join: corr(day rs, burst_frac) = −0.65; hi-burst days 2.30 vs
lo-burst 2.59 — adverse selection concentrates in bursts, as expected.

(c) VWAP secondary: mo_vwapmid 2.30 vs 2.44 (gap 0.14, sign agrees) — does
not indict the primary mid; but the VWAP proxy shares the pinned-book
bounce structure, so it cannot arbitrate the weighting question.

(d) rs(30) = 2.44 < es_med/2 = 2.82 — no over-capture flag.

**Prereg defect (log):** §1.5 gate 1 says "day-clustered mean rs(τ*)"
without pinning the weighting; §1.3b mandates BOTH eq and Q·p-weighted be
reported. The code gated on eq (chosen blind, procedurally clean), but the
co-reported notional estimate reverses the verdict. The pass is
weighting-dependent and stands only in the front-of-queue-min-size fantasy.

**What E1x/E2.0 must check:** E1x (cheap, prereg'd) may proceed but cannot
resolve this — it will re-measure the same population statistic. Binding
checks are: (i) E2.0 Δrs(τ*) proxy-vs-true mid (prereg'd, ±1 bp voiding);
(ii) **notional-weighted and size-bucketed rs on true mids** — pre-specify:
if notional-weighted rs(τ*) < fee at VIP0 on the L2 window, the ADA cell
dies regardless of eq numbers; (iii) E2 RiskAverse queue replay on a pinned
fat-tick book (expected to kill it; ProbQueue-f3 upper bracket may not).
Recommend recording (ii) as a protocol amendment BEFORE E2 data is read.

### 2. E1-A PASS (touch 3.45 [3.11,3.79], sweep 6.26 [5.76,6.75]) — verdict: **ROBUST**, with one fragile cell flagged

(a) Per-symbol sweep-rule eff_RT (T_p=600 day-means): ICP 7.53, ADA 7.02,
AVAX 6.55, FIL 6.34, DOGE 6.34, AAVE 6.23, SOL 6.10, XRP 6.05, BTC 5.98,
BNB 5.92, ETH 5.77, LTC 5.35. **No symbol individually exceeds 8** on base
accounting; the aggregate is not majors-dragged (alts sit 6-7.5).
(b) D-h stale shadow: aggregate sweep 6.26 → 7.20 (+0.94), touch 3.45 →
3.79 (+0.34). **PASS holds under the skip-bias bound.** Per-symbol stale
sweep: ICP 11.02, ADA 8.43, AVAX 8.07 — ICP breaches 8 under the bound.
(c) Skip concentration: ICP 72.5% of potential episodes skipped, LTC 54.7%,
AAVE 39.6% — as predicted by the code review; now visible, not silent.
(d) Fill-rate sanity post-D-i: touch > sweep for all 12 (e.g. ADA
0.986/0.855, ICP 0.965/0.847); bracket real everywhere.
(e) Winner's curse clearly visible: drift_nofill 9.8 bps (BTC) → 31.4 bps
(FIL) at T_p=600; chase cost 14-37 bps. The chase branch is doing its job —
fill rates ~0.85-0.94 keep eff_RT inside 8.
(f) Excluding ICP: touch 3.70 / sweep 6.15 — **PASS stands** (also stands
excluding ICP+LTC+AAVE: 3.72/6.23).
Gate-reading caveat: the equal-weighted XS book, if it actually trades ICP,
relies on a cell that is D-a boundary-ranked (ADV rank exactly 40 on
2.5-month-stale data), 72%-skipped, and >8 under the stale bound → treat
ICP as E2-A-must-resolve, not as passed.

### 3. Oddities

- **FIL tick misdetection — confirmed bug, verdict-immaterial.** FIL trades
  ~$0.76 on a 1e-4 grid; 81 of 2.46M prints (3.3e-5) are off-grid 6-decimal
  singles (bust/liquidation-style: single-print rows, large qty), which
  poison the pooled min-diff to 1e-6, and the integer-multiple acceptance
  test cannot reject a too-small tick (all true-grid diffs are multiples of
  it). Re-ran all 31 FIL days with tick=1e-4: touch 4.56→3.55, sweep
  6.34→6.47; aggregate ~3.36/~6.28 — **E1-A verdict unchanged.** Side
  effects: FIL tick_bps (0.014) and pinned_flag (False; truly ~pinned at
  es100=1.05 ticks) are wrong in e1_spread_daily. FIX before E1x: derive
  tick from the MODE of successive-print diffs (or require ≥99.9% of PRICES
  on the candidate grid), not the min pooled diff.
- Pinned pattern: 13/16 symbols pinned ≥29/31 days. BTC and SOL show
  pinned=0 despite es100 ≈ 1 tick exactly — boundary artifact of comparing
  qty-weighted (off-grid) sweep prices against `≤ 1.0·tick`; informational
  only, no gate impact.
- ATOM (3 usable days, rs +2.97, tick 7.2 bps) and ICP (2 days, +0.94,
  4.6 bps) are the same fat-tick family as ADA — correctly `no_estimate`,
  and the notional-weighted sign is negative for both. If ADA's E1x is run,
  do NOT read ATOM/ICP fragments as corroboration; they carry the same H1
  structure.
- LTC: 11/11 usable days rs > 0 but fails D-j's literal ≥22-day count —
  correct prereg behavior (insufficient sample, not economics; rs +0.43
  sub-fee anyway).
