# Freeze receipt: pm_ladder_v1 (P-2026-005, frozen 2026-11-06T06:07Z)

USER 2026-11-06: "We freeze and tell the coding session to implement this version." Research summary: docs/EXPLORATION_2026-11-01.md
sections 55 (result) and 60 (round-i summary). Everything here is from the fixed engines (round i); every earlier replay number is void.

## Identity

- Code commit: 8d5b02a0583b08cc46d7ecc84f10adcbb4d03223 (branch mm-research). The artifacts in this folder were built from that commit by
  `predictfun/research/pf_freeze_pm_ladder_v1.py`, via run_light.sh (tag freeze_v1, 2026-11-06T06:05:27Z, rc 0). A dry run from the same
  file gave byte-identical models.
- Reference implementation: `predictfun/research/pf_step4_ladder.py` (sha256 c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f,
  the hash in every round-i ladder summary).
- Pipeline files (sha256 at the commit):

| file | sha256 |
|---|---|
| pf_step4_ladder.py | c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f |
| pf_step4_live.py | 742637b9a2d5bd6d53c7538f78b2277412aff6b5c40dec05e4d6d2ee68a4884f |
| pf_step5_features_rl.py | 8bb95a0090f03196169d62e92a66b8a01bbdb3845021793655b08b4c0555d553 |
| pf_model_spec.py | ede27dafd28766d4d8de320238c277e0dde838f2cb335460bb66d21006151dd8 |
| pf_step5_guard_preds.py | 15d84613d74a9d0f08c501a279ebd28372e501ae5fb057b65162fa159f6e0f56 |
| pf_preds_startknown.py | 02d55cd721548b2f958a2a75b49290feb343393a369a4a1c12782fa26b76f08d |
| pf_clock_rl.py | 0a1ce1e12e45a73de7a46102385f63be94d065d60ffbcc44aa30b4e1a840bb98 |
| pf_io.py | 057fb052683b715084e175304a3c0bdf4dd9262694992d332056bbc6523e9494 |
| pf_freeze_pm_ladder_v1.py | 446fb0bfb75364d6cbfbc82441582ed7645320dff55086759ec0fcd8eeb603e3 |

## Candidate and control

- Candidate: `MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95`
- Control: `MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06` ("today")
- Multiplicity: one candidate (plus the control) in the forward check. The receipt's choice was made after seeing the 8 read days
  (sections 55-60).

What the candidate does: rule semantics are the engine's docstring and code. Line numbers are in pf_step4_ladder.py at the commit.

- Market: Predict.fun BTC Up/Down 5-minute windows; quotes only in the first 60 s of each window (S60 + G5), then cancels and carries
  the remainder to settlement.
- Touch lanes (bid / ask), MICRO price (`desired`, l. 686): microprice m = (b1 * a1_size + a1 * b1_size) / (b1_size + a1_size), half
  = (a1 - b1) / 2. Bid = floor(m - half), ask = ceil(m + half), kept inside (1..99, not crossing).
- D1 guards: G1 Binance 1 s move, G2 Binance 250 ms leg, G3 gap behind the touch, G4 never quote behind the touch, G5 stop, G6
  inventory taper (caps use the large size, 25: the increasing side is withheld at |net| >= 50 shares; hard 125 including live size). Definitions are in the docstring
  "Rules (D1 pieces)" and "Additions 2026-11-01".
- Guard value v per side (the freeze artifact): v_side = model_side(features) - threshold_c60[side] (params.json). Rules:
  - v is stale after 100 ms.
  - Before the window's start price is known, v = -1 (pf_preds_startknown).
  - The features are SETS['ALL'] in params.json order, built as pf_step5_features_rl does on the Tokyo clock.
- PM90 per-minute pull (l. 758): a touch side is withheld when v_side <= line_90[minute of the window]. line_90 = [0.0492, 0.0623,
  0.0789, 0.0917] c for minutes 0-60 / 60-120 / 120-180 / 180-240; only the first applies at S60.
- GS (l. 761): no fresh v on a side means that side is withheld.
- AM:5:25:95 size (l. 671): 25 shares when v_side >= line_95[minute] ([0.0698, 0.0841, 0.1009, 0.1094]), else 5.
- LD2 ladder (l. 779): a second order per side, present only while that side is not withheld and v_side > line_90[minute]. It is
  placed one tick behind the best (bid b1 - 1, ask a1 + 1) and kept at its price while it rests within 2 ticks behind the best or is
  promoted to the best. It uses the same size rule and G6's inventory check.
- XNA no amend (a size-down is a cancel + new order at the back). XMIN1 skips orders under 1 USD (bid p * size, ask (1 - p) * size).
  XV uses the venue throttle (one bucket of 40 refilling 500 / min, cancels always sent, a place needs tokens >= open orders + 2,
  < 3 unanswered places per side, retry 250 ms doubling to 4 s).
- Cancel-pending orders still count as ours: they are in our price priority, post-only and self-cross checks (round-i fixes).

Replay clock (the assumptions the live system is measured against):
- Tokyo clock: Predict frame known at its measured delay + 2 ms decision.
- Order live 40 ms after sending (PF_RL_LPLACE); cancel effective after 22 ms, or 95 ms when sent while the create is pending.
- Own fills known after 100 ms.
- Env: PF_RL_DDEC=2 PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_TIMING=e2 PF_TXTIME=ws PF_TXKEY=tk.
- Timing J: PF_TXJIT=frame. Timing X (bound): PF_TXSHIFT=-50 PF_QACT=max.

## Artifacts (this folder)

| file | sha256 |
|---|---|
| guard_bid.txt (LightGBM text, 300 rounds) | 0dc09d84bb28e0c0cafac99ee0fa8ff1140cb97ca70571526469f3b42f345e39 |
| guard_ask.txt | 6d2d7149e6c556c6a2a63dad89e68ce4ea6113c76400edf9bd4f7a345ea9fade |
| params.json (thresholds, feature order, lines, LightGBM params, input hashes, parity) | written by the builder |

- Training data: step4/badfill_info_tx_tk/grid.parquet, the 8 read days 10-23..27, 10-31, 11-01, 11-02 (1,071,151 rows). Its sha256 is
  in params.json.
- Thresholds c60: bid -0.08573863448914996, ask -0.08373751096122671. These equal the 11-03 fit of round i.
- Per-minute lines: step4/minute_lines_t60k.json (in-sample cuts of the out-of-fold v, written 2026-11-05T10:54Z). Its sha256 is in
  params.json.
- Parity, checked by the builder: the saved and reloaded models reproduce the 11-03 guard values used by every round-i replay exactly.
  That is 1,728,000 rows, 1,324,002 of them live, float32, with an equal NaN pattern. A threshold moved by 1e-6 fails the check
  (falsifier fired).

## In-sample result (not a validation)

Ladder engine, round i, 8 read days, mark@stop $ / day, no rebate, no own price impact; 11-03 beside.

| | central J | bound X |
|---|---|---|
| candidate | +63.6 (+27.3..+99.9) 7/8, 11-03 -12.2 | +52.0 (+13.5..+90.6) 7/8, 11-03 -33.1 |
| control | +45.5 (+7.4..+83.5) 6/8, 11-03 -8.1 | +17.7 (-18.7..+54.1) 6/8, 11-03 -6.2 |
| ladder on pm (paired) | +18.1 (+9.4..+26.9) 8/8 | +23.8 (+9.9..+37.8) 7/8 |

Lane detail, J / X:
- Touch: 2,872 / 2,297 sh / day; median queue ahead at activation 315 / 417 sh.
- Rung: 1,234 / 1,050 sh / day; median queue ahead 232 / 296 sh.
- About 270 requests / window; about 5,000 throttled places / day.

## Declared forward check (written before any forward read)

- Days: 11-04, 11-05, 11-06, 11-07, 11-08. These are the untouched validation days declared in STATUS (rebuild 3), 5 complete UTC days.
  - The stopped C2 forward job wrote step1 and step5 (rows, preds_c2) for 11-04 on 2026-11-05 02:20-02:22Z.
  - No replay output exists for 11-04 onward, and no report has read those days.
  - The forward run rebuilds every input for these days with this commit's code; those C2-era files are not used.
- Inputs: guard values from the frozen model files above (no refit), the start-known withhold, and the same per-minute lines.
  Replay: pf_step4_ladder.py at this commit, candidate and control, at timings J and X.
- PASS iff all of:
  - (a) the candidate's 5-day mark@stop sum > 0 at J and at X;
  - (b) the 5-day sum of candidate minus control > 0 at J and at X.
- Reported beside, not part of the rule: per-day values, days > 0, a t interval (4 df, 2.776), settle, +1 s c / share against the top
  5 makers.
- Any change to the candidate, artifacts or rule after this receipt is a new version (v2 receipt). This file is not edited.

## Hand-off

The coding session (pmmmrust exec port) implements this version. Going live is the user's decision and is not part of this freeze.
