# Freeze receipt: pm_ladder_v3 (P-2026-005, frozen 2026-11-07T06:18Z)

USER 2026-11-07: "Go ahead to freeze this new session" (this version), after the CAP10 + RO60-240 result (STATUS 2026-11-07 04:34Z).
v3 = pm_ladder_v1 plus a 10-share inventory limit in the first minute and a 60-240 s sell-down that keeps the pull line. Everything
else is v1's: models, thresholds, per-minute lines, rules, clock (predictfun/freeze/pm_ladder_v1/RECEIPT.md, unchanged and in force).

## Identity

- Code commit: 4ecaca69576086e0c76288f286a93d8c30e5b3d5 (branch mm-research), on top of v1 (8d5b02a0 / 38535aed) and the race (4dee5c19 / 636450b5).
- Files (sha256 at the commit):

| file | role | sha256 |
|---|---|---|
| pf_ladder_v3_build.py | generates the v3 engine and params.json | 8c0ea4c4f0b7e2a47de91babd772aaf92b84bcbe3deb389f61eaffda55e92b38 |
| pf_step4_ladder_v3.py | v3 replay engine (generated) | d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17 |
| pf_ladder_inv_build.py | inventory-token patches (unchanged since the in-sample result) | e6f37a43bcb268ec300a69a050273db94fec14df747e4040c4c0d27f7c162461 |
| pf_ladder_fwd_build.py | the shared FORWARD patch (race receipt) | b78fd0e2003ffb2a102f0cd0ea0e5c128b10b8def5392d96ac07782d60c68ea1 |
| pf_step4_ladder.py | frozen v1 engine (the source; unchanged) | c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f |
| pf_step4_ladder_fwd.py | v1 + control forward engine (race receipt) | 4aa075d1fe4c9b1deac206047b35e707d10895db159c5bddb623684acd72f025 |
| pf_v3_engine_check.py | engine parity check | 15a6f891fa0640eeecf30f1d8f10d936cf1ab898a96c3995455888fa9a8f0dc9 |
| run_v3_check.sh | runs the parity check and the report rehearsal | 3d14987b9e02a2eb06574ce2753fb7232443f9f88f4435d746191234e64dae72 |
| run_forward_v3.sh | the forward run | c3205c90b68f8ca3252fe89ec486fda7a363be3dfc22e95711fe7f5a974586d4 |
| pf_forward_v3_report.py | the declared rules as predicates | 413c4dc38e8e98e982cfd5a787ff92a2dcf67f132f6b7d9229990e9a4139a733 |
| ../freeze/pm_ladder_v3/params.json | arm, hashes, v1 artifacts | 486b9152bc5187e19bb73bbc54af0c758112fb6b54d3c6f7ddd9463efbbea801 |

- pf_step4_ladder_v3.py is the frozen engine with 11 anchored changes: pf_ladder_inv_build's 10 (the CAP / RO tokens allowed with the
  ladder and the per-minute lines, and three tokens v3 does not use: SK, RX, RL) plus pf_ladder_fwd_build's FORWARD from PF_FORWARD.
  It equals pf_step4_ladder_inv.py (the engine of the in-sample result) plus the FORWARD line.
- Run summaries record `code_sha256` of the source pf_step4_ladder.py, not of the generated file; run_forward_v3.sh writes every hash.

## Candidate and comparators

- Candidate: `MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95`
- Comparators in its forward check: v1 `MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95` and the control
  `MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06` (v1 receipt).
- Multiplicity: one candidate in its own forward block (11-09..11-13). Selection: CAP10 and RO60-240 were picked from 10 inventory arms
  (report_inv_v1.json) and the spread-capture view (report_spread_capture_v1.json) on the 9 consumed days; the combination was run once
  (report_inv2.json). The 11-04..11-08 race (v1, v2, control; predictfun/freeze/forward_20261104_08) is unchanged; v3 is not in it.

## What v3 does (rules are the engine's code; v1's rules all apply)

- 0-60 s: v1, plus CAP10. A side that does not reduce |known net| (touch and rung) is withheld once |known net| >= 10 shares; the
  reducing side keeps quoting. Known net = own fills known after 100 ms. A resting order can still fill past 10 (sizes 5 / 25).
- 60-240 s (RO60): reduce-only. Only the side that reduces |known net| is quoted, at the touch (MICRO price), size min(size rule,
  |known net|), so the position never flips; no rung. All guards still apply: the per-minute pull line of minutes 1-3 (0.0623 / 0.0789
  / 0.0917 c), GS (no fresh guard value -> withheld), G1-G4. Flat -> nothing quoted. Up and Down are one book: reducing an Up position
  is an Up ask (= a Down bid).
- 240 s (S240): cancel all; the rest is carried to settlement. Guard values exist only for t < 240 s (pf_v1_apply), so a later stop
  with the line would quote nothing after 240 s.

## Artifacts

- params.json (this folder, written by pf_ladder_v3_build.py): arm, comparators, engine hashes, v1's model hashes and thresholds,
  per-minute lines. The builder re-hashes v1's guard_bid.txt / guard_ask.txt and minute_lines_t60k.json and refuses on a mismatch.
- Guard models and lines: predictfun/freeze/pm_ladder_v1 (guard_bid.txt 0dc09d84..., guard_ask.txt 6d2d7149...; c60 bid
  -0.08573863448914996, ask -0.08373751096122671; lines sha256 3852fec9...). No refit.

## Checks run at the commit (all PASS; step4/v3_engine_check.json, logs/run_v3_check.log)

- Engine parity (run_v3_check.sh, 06:14-06:17Z): the generated engine, v3 arm alone, no PF_R60_FILE, is byte-identical to the
  in-sample run (fills / win / requests) and its summary arm equal on 11-03 J (tag i2J), 11-03 X (i2X) and 11-02 J with
  PF_FORWARD=20261114 (i2J). Falsifier: the same compare against its parents RO60-240 and CAP10 differs (fired, 2 / 2).
- Engine hash equals params.json's; the run's env (PF_FORWARD, timing, arms) as declared.
- Report rehearsal on 5 read days (10-27, 10-31, 11-01..03; v1 / control from round i, v3 from i2J / i2X), code path only,
  step4/report_forward_v3_rehearsal.json: 5 / 5 synthetic rule controls give the declared verdicts (one PASS + PREFERRED case, one case
  failing each of a-d); the forward env checks refuse the in-sample runs (falsifier: 71 errors).
- run_forward_v3.sh refuses before 2026-11-14 02Z (run once at hour 2026-11-07 06Z by its own clock read: REFUSED, rc 2).

## In-sample result (not a validation)

9 consumed days (10-23..27, 10-31, 11-01..03), fixed engine, $ over 9 days, J / X (step4/report_inv2.json, STATUS 2026-11-07 04:34Z).
Both parents were picked on these days.

| | v1 | CAP10 | RO60-240 | v3 |
|---|---|---|---|---|
| settle | 639.2 / 370.0 | 815.3 / 461.6 | 782.1 / 549.1 | 905.9 / 592.9 |
| days > 0 (settle) | 5 / 6 of 9 | 6 / 5 | 6 / 7 | 8 / 7 |
| +1 s $ | 551.5 / 398.7 | 452.3 / 354.3 | 642.3 / 452.8 | 541.1 / 409.2 |
| day sd (settle) | 201.7 / 154.5 | 155.4 / 102.5 | 164.0 / 136.4 | 126.4 / 80.8 |
| worst day | -166.3 / -152.9 | -88.6 / -54.8 | -108.8 / -141.8 | -52.3 / -44.5 |
| max drawdown | -457.4 / -402.6 | -233.7 / -181.9 | -332.0 / -351.2 | -169.3 / -138.6 |
| $150 cap halt within 1 day | 26.9 / 9.8% | 11.2 / 2.8% | 15.4 / 10.9% | 2.0 / 0.0% |

v3 - v1 per day (t 8 df, 512 sign flips raw p): settle +29.6 (p 0.40) / +24.8 (p 0.46); +1 s -1.2 (p 0.65) / +1.2 (p 0.59).

## Declared forward check (written before any read of these days)

- Days: 11-09, 11-10, 11-11, 11-12, 11-13 (5 complete UTC days after this freeze; no file of them has been read).
- Procedure: predictfun/research/run_forward_v3.sh at the commit, by hand after 2026-11-14 02:00Z (it refuses earlier; no timer).
  - It regenerates the v3 engine and params and refuses if they differ from the commit.
  - Source coverage: every hour file from D-1 23Z to D+1 01Z, or SOURCE_GAP.
  - Inputs: run_forward.sh's per-day commands (pf_step1, timing, booktime, features, c3 apply, features PF_PEND_KNOWN=frame, txmap,
    pf_v1_apply with the frozen models), then pf_preds_startknown; PF_FORWARD=20261114.
  - Replays, timings J and X with the v1 receipt's clock: v1 + control with pf_step4_ladder_fwd.py (tags f3aJ / f3aX); v3 with
    pf_step4_ladder_v3.py (tags f3bJ / f3bX), no PF_R60_FILE.
  - Report: pf_forward_v3_report.py (refuses before any value on a wrong env / arm / timing / code-hash record).
- Metric: settle $ = summary gross_usd (no rebate). v3 stops at 240 s and the comparators at 60 s, so mark@stop is not
  like-for-like; settle is, since every arm is held to settlement.
- Rules (sums over the 5 days):
  - v3 PASS iff (a) sum v3 > 0 at J and at X, and (b) sum (v3 - control) > 0 at J and at X.
  - v3 PREFERRED over v1 iff v3 PASS and (c) sum (v3 - v1) > 0 at J and at X, and (d) day sd of v3 < day sd of v1 at J and at X.
- Reported beside, not part of a rule: per-day values, days > 0, t interval (4 df, 2.776), +1 s $ (like-for-like trading profit),
  mark at each arm's own stop, worst day, max drawdown, $150 cap halt within 1 day, |net| at stop, fills, shares.
- Power, from the in-sample per-day values (normal approximation; bootstrap of days for (d)): (a) about 0.96 / 0.97; (c) about 0.75 / 0.71
  per timing; (d) about 0.96 / 0.93. (c) will often not resolve in 5 days; not resolving is not evidence against.
- Any change to the candidate, artifacts, days or rules after this receipt is a new version. This file is not edited.

## Live port (not checked here)

Relative to v1 the bot needs: the 10-share add limit on known net (touch and rung); a 60-240 s reduce-only phase (reducing side only,
size capped at |known net|, no rung, minute 1-3 lines, all guards); the stop moved from 60 s to 240 s; guard values to 240 s. Going
live is the user's decision and is not part of this freeze.
