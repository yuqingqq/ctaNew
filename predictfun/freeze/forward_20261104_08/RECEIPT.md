# Forward check receipt: 11-04..11-08, pm_ladder_v1 and pm_ladder_v2 (P-2026-005, declared 2026-11-06T09:23Z)

This receipt declares the one forward race on 11-04..11-08. It supersedes the procedure lines of the "Declared forward check" section of
predictfun/freeze/pm_ladder_v1/RECEIPT.md, because that procedure cannot run as written:
- the frozen engine pf_step4_ladder.py refuses every day >= 11-04 (`FORWARD = '20261104'`, l. 179-180);
- pf_txmap.py has the same bound;
- pf_step5_guard_preds.py always refits, so it cannot apply the frozen models.

v1's candidate, artifacts, days and PASS rule are unchanged. The v1 receipt stays as provenance and is not edited.

## Candidates (multiplicity: 2 candidates + 1 control on the same 5 days)

- pm_ladder_v1: fixed models, thresholds and lines (predictfun/freeze/pm_ladder_v1, frozen 2026-11-06T06:07Z).
- pm_ladder_v2: the same arm with a daily 7-day refit (predictfun/freeze/pm_ladder_v2, frozen 2026-11-06T09:23Z).
- Control: `MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06` (today's rule, from the v1 receipt).

## Data status at this declaration

- No replay, report or model output exists for 11-04 onward.
- The stopped C2 job wrote step1 and step5 (rows, preds_c2) for 11-04 on 2026-11-05 02:20-02:22Z; nothing has read them. The forward run
  rebuilds every input with the commit below.
- 11-05..11-08: no research file.

## Procedure: predictfun/research/run_forward.sh at commit 4dee5c195bbe030203ace928a6021aa57ea7977e

The run starts by hand after 2026-11-09 02:00Z; the script refuses earlier. Step 1 reads hour files up to D+1 01Z. No timer is installed.

### A. Inputs, per day D

Each step is the command that built 11-03 (run_light log tags c2s1 / c2t / c2bt, feat_1103, c3_1103, featpk, txmap, startknown_t60):

1. pf_step1
2. pf_step1_timing
3. pf_step1_booktime
4. pf_step5_features_rl
5. pf_step5_c3_preds apply
6. pf_step5_features_rl with PF_PEND_KNOWN=frame
7. pf_txmap (PF_FORWARD=20261109)
8. pf_v1_apply (frozen models)
9. pf_preds_startknown preds_t60

The source-coverage rule is run_c2_day.sh's: every hour file from D-1 23Z to D+1 01Z, or the run stops (SOURCE_GAP).

### B. v2

- pf_v2_grid_day for 11-03..11-07.
- pf_v2_refit for T = 11-04..11-08. Each window is the 7 admissible days before T; 11-04's window is 10-25..11-03.

### C. Replays, timings J and X (the v1 receipt's clock and env), PF_FORWARD=20261109

- v1 candidate and control: pf_step4_ladder_fwd.py.
- v2 candidate: pf_step4_ladder_v2.py with step5/<D>/preds_v2w7k and lines_v2w7.json.

### D. Report

pf_forward_report.py evaluates the rules below as predicates. It refuses before printing any value if any run's env, arms, refit window or
code-hash record is wrong.

### Files (sha256 at the commit)

| file | sha256 |
|---|---|
| run_forward.sh | c42df0101788d580d9cf72dd36e1c4fd343979cc03d9ccf9f962a0c5623faae7 |
| pf_forward_report.py | d38eea40cb87be0641e7bafdfa669a5d88dbb5579385ccb956bf73716d55eeca |
| pf_step4_ladder_fwd.py (generated: frozen engine + FORWARD from PF_FORWARD) | 4aa075d1fe4c9b1deac206047b35e707d10895db159c5bddb623684acd72f025 |
| pf_step4_ladder_v2.py (generated: + lines from PF_LINES_FILE) | 9e1dc0ed441dd3355ba2e55418e29e7a0f849cd7a165e2bfde8acfc64d08688b |
| pf_v1_apply.py | 810147752dbef7baad326a53543fbec8c0dd3134d6368cc33b64d811b7e95739 |
| pf_v2_refit.py | 2e0127a08b3ecb4b3a6452164dd26f2fdc7fd19310dd35505e02c9b2587af61b |
| pf_v2_grid_day.py | d60efd0dd18ea4b143e718e3a01ee33602324f0d24a23cc9ed12f7f55901e756 |
| pf_txmap.py (FORWARD from PF_FORWARD, default unchanged) | dc31b9a36f1016d97ded1f2e793cfb4bd4ea5a6f8bf4cfeaad4b67455a833ae2 |
| pf_step1.py | fa03ee189a587207ea4484820f6276bef20a3d073ef16b4370ccbfade974cc71 |
| pf_step1_timing.py | 0e669a7c6389027094ef0bf6fd040372b13daccc3dd87962bf4f9715b98c5775 |
| pf_step1_booktime.py | 3e0d2969b744a29f49ab1291569091dd4b936b9878be1092ad8ccfd4cf9990d8 |
| pf_step5_c3_preds.py | 8d06904b5600c0679ac3aa1baafe94e07d02f117a129baf8df3eae65f7f0b0d5 |
| pf_step5_features_rl.py | 8bb95a0090f03196169d62e92a66b8a01bbdb3845021793655b08b4c0555d553 |
| pf_preds_startknown.py | 02d55cd721548b2f958a2a75b49290feb343393a369a4a1c12782fa26b76f08d |
| pf_clock_rl.py | 0a1ce1e12e45a73de7a46102385f63be94d065d60ffbcc44aa30b4e1a840bb98 |
| pf_io.py | 057fb052683b715084e175304a3c0bdf4dd9262694992d332056bbc6523e9494 |
| pf_model_spec.py | ede27dafd28766d4d8de320238c277e0dde838f2cb335460bb66d21006151dd8 |

The generated engines' run summaries record `code_sha256` of the source file pf_step4_ladder.py, not of the generated file.
run_forward.sh writes logs/forward/code_sha256.txt with every file's hash and the commit at launch.

### Why the replacements change nothing in the strategy (checked at the commit, all PASS)

- `FORWARD` only bounds which days' input is admissible. pf_step4_ladder_fwd.py is byte-identical to round i on 11-03 J at the default,
  and on 11-02 J with PF_FORWARD=20261109. Falsifier: the pm arm differs.
- pf_v1_apply.py on 11-03 equals step5/20261103/preds_t60.parquet in every column. Falsifier: c + 1e-6 differs.
- pf_v2_refit, pf_v2_grid_day and pf_step4_ladder_v2 checks: see the v2 receipt.

## Rules (mark@stop $ = summary mtm_stop_usd, no rebate; sums over the 5 days)

- v1 PASS iff sum v1 > 0 at J and at X, and sum (v1 - control) > 0 at J and at X. This is the v1 receipt's rule, unchanged.
- v2 PASS iff sum v2 > 0 at J and at X, and sum (v2 - control) > 0 at J and at X.
- ROLLING ADOPTED iff v2 PASS and sum (v2 - v1) > 0 at J and at X.

Reported beside, not part of any rule:
- per-day values;
- days > 0;
- t interval (4 df, 2.776);
- settle;
- +1 s $;
- shares and fills;
- v2's windows, lines and minute-0 quote share;
- v1's minute-0 quote share at the fixed line.

The +1 s c / share against the top 5 makers (v1 receipt) is computed after the rule with the section 57 decomposition.

Power: 5 days. v2's window shares 6, 5, 4, 3 and 2 of its 7 days with v1's training days on 11-04..11-08 (pf_v2_refit.days_before). A difference between
v1 and v2 that is small relative to day-to-day spread will not be resolved by this race. Not resolving it is not evidence for either.
