"""Assemble the R3 receipt (report.md) from the round's outputs: checks.json / checks_falsifier.json, the receipt's CSV copies (eval/,
replay/), tables.md (r3_tables.py) and the run_light logs. Every verdict sentence is chosen by a predicate computed here; every number is
read from an output (or is a count, sum, difference or ratio of read values). The predicates were written before any Oct 26-27 outcome was
read (2026-11-21 ~03:55Z, while the label build ran). Refuses if an input is missing.
Usage: fill_r3.py -> writes report.md in the receipt dir and prints the values the state hunks reuse (JSON)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/r3_adv2_20261121'); L = R / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
SIDE = {0: 'bid', 1: 'ask'}
COMMITS = dict(freeze1='551185f22cb4f85a0be0385edd47af92f988ce57', freeze2='c8655f75251c71dcdd38eb4f78d3a536d4dec7a3',
               fit='2bec465d2d63de193795a5907626b5a948b0c835', freeze2b='259e6c56244025498b2921ded9561e8261a8b44d',
               review='315979d205e7788daf69e277aa1bd6e0cd881e0e')


def suite(name):
    """A mutation summary log -> (caught, planted, passed)."""
    t = (L / name).read_text(); m = re.search(r'(\d+) / (\d+) planted code errors are caught', t)
    return int(m.group(1)), int(m.group(2)), bool(re.search(r'^mutation check passed', t, re.M))


def jobs(prefix):
    out = []
    for p in sorted(L.glob(f'{prefix}*.log')):
        for ln in p.read_text().splitlines():
            m = RL.match(ln)
            if m and m.group(2).startswith(prefix.rstrip('*')):
                out.append(dict(tag=m.group(2), rc=int(m.group(3)), peak=int(m.group(4)), wall=float(m.group(5)), cpu=float(m.group(6))))
    return out


def rcsv(d, name):
    """A receipt CSV copy: name, or name.gz when r3_collect.sh gzipped it."""
    p = R / d / name
    return pd.read_csv(p if p.exists() else Path(str(p) + '.gz'))


def n_of(xs):
    return f'{sum(bool(x) for x in xs)} of {len(xs)}'


def sections(md):
    out, cur = {}, None
    for ln in md.splitlines():
        if ln.startswith('### '):
            cur = ln[4:].strip(); out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return {k: '\n'.join(v).strip('\n') for k, v in out.items()}


def main():  # noqa: C901
    J1 = json.loads((R / 'checks.json').read_text()); J = json.loads((R / 'checks_v2.json').read_text()); F = json.loads((R / 'checks_falsifier_v2.json').read_text())
    G1 = J1['readings'].get('G', {}); G2 = J['readings'].get('G', {})
    rd = rcsv('eval', 'readings.csv'); sc = rcsv('eval', 'screens.csv'); ct = rcsv('eval', 'control.csv')
    sup = rcsv('eval', 'support.csv'); E = rcsv('replay', 'episodes.csv'); Rc = rcsv('replay', 'reached.csv')
    P = pd.read_csv(R / 'eval' / 'predictions.csv.gz'); RJ = json.loads((R / 'replay' / 'replay.json').read_text()); T = sections((R / 'tables.md').read_text())
    v = {}
    # ---- prediction (offline, the consumed Oct 26-27 sample; group all = every ok row; per day x side x head)
    g = rd[rd.group == 'all']; cells = []
    for (day, s, h), x in g.groupby(['day', 'side', 'head'], sort=True):
        r_ = {c: x[x.candidate == c].iloc[0] for c in ('RICH', 'FAIR', 'MECH')}; z = r_['RICH']
        fb = 0. if h == 'post' else float(z.always1)                                       # the fallback-default policy: never post / always keep
        cells.append(dict(day=day, side=SIDE[int(s)], head=h, beats_zero=z.wmse < z.wmse_zero, beats_mean=z.wmse < z.wmse_train_mean,
                          beats_fair=z.wmse < r_['FAIR'].wmse, beats_mech=z.wmse < r_['MECH'].wmse, ratio_zero=z.wmse / z.wmse_zero if z.wmse_zero else np.nan,
                          value=float(z.value), fallback_value=fb, beats_fallback=float(z.value) > fb, beats_const=float(z.value) > max(0., float(z.always1)),
                          value_fair=float(r_['FAIR'].value), value_mech=float(r_['MECH'].value), above_fair_value=float(z.value) > float(r_['FAIR'].value),
                          above_mech_value=float(z.value) > float(r_['MECH'].value), oracle=float(z.oracle)))
    C_ = pd.DataFrame(cells); v['cells'] = len(C_)
    for k in ('beats_zero', 'beats_mean', 'beats_fair', 'beats_mech', 'beats_fallback', 'beats_const', 'above_fair_value', 'above_mech_value'):
        v[k] = int(C_[k].sum())
    scr = sc.groupby(['candidate', 'verdict']).size().unstack(fill_value=0)
    v['screens'] = {c: {k: int(scr.loc[c].get(k, 0)) for k in ('PASS', 'FAIL', 'INSUFFICIENT_SUPPORT', 'NOT_EVALUABLE')} for c in ('RICH', 'FAIR', 'MECH') if c in scr.index}
    v['screens_total'] = int(len(sc[sc.candidate == 'RICH']))
    # ---- support: the share of evaluation rows outside the candidate support (fallbacks)
    tot = sup.rows.sum(); fb_rows = sup[sup.support != 'supported'].rows.sum(); v['fallback_rows'] = int(fb_rows); v['eval_rows'] = int(tot)
    v['fallback_share'] = float(fb_rows / tot) if tot else None
    v['fallback_by_status'] = {k: int(n) for k, n in sup[sup.support != 'supported'].groupby('support').rows.sum().items()}
    # ---- control (scope day x head)
    cd = ct[(ct.candidate == 'RICH') & ct['head'].notna()]
    v['control_lift_pos'] = n_of(list(cd.lift > 0)); v['control_above_p95'] = n_of(list(cd.realized > cd.draw_p95))
    # ---- replay: paired differences per day
    D = pd.DataFrame(RJ['diffs']); rep = {}
    for day in sorted(E.day.astype(str).unique()):
        e = E[E.day.astype(str) == day]; tot_ = e.groupby('policy').value.sum()
        dd = D[D.day.astype(str) == day]
        def dget(p, b):
            z = dd[(dd.policy == p) & (dd.versus == b)]
            return (float(z.diff_sum.iloc[0]), int(z.positive.iloc[0]), int(z.negative.iloc[0]), int(z.n.iloc[0])) if len(z) else None
        rep[day] = dict(worlds=int(e[e.policy == 'RICH'].shape[0]), totals={p: float(tot_.get(p, np.nan)) for p in tot_.index},
                        rich_vs_pi0=dget('RICH', 'PI0'), rich_vs_pa=dget('RICH', 'ENTRY_PA'), fair_vs_pi0=dget('FAIR', 'PI0'), mech_vs_pi0=dget('MECH', 'PI0'),
                        entry_only_vs_pi0=dget('ENTRY_ONLY', 'PI0'), retain_only_vs_pi0=dget('RETAIN_ONLY', 'PI0'))
    v['replay'] = rep
    # ---- reached states against the neutral sample (per day x head; reached pairs unweighted, one per world / side / head)
    ok = Rc[Rc.status == 'ok']; rr = []
    for (day, h), x in ok.groupby(['day', 'head'], sort=True):
        y = x.Y.astype(float).values; mz = float(np.mean(y ** 2)); mr = float(np.mean((x.pred_RICH.astype(float).values - y) ** 2))
        p_ = P[(P.day.astype(str) == str(day)) & (P['head'] == h)]; yn = p_.Y.values; pn = p_.pred_RICH.values; w = p_.w.values
        nz, nr = float(np.sum(w * yn ** 2) / w.sum()), float(np.sum(w * (pn - yn) ** 2) / w.sum())
        nzu, nru = float(np.mean(yn ** 2)), float(np.mean((pn - yn) ** 2))
        rr.append(dict(day=str(day), head=h, pairs=int(len(x)), reached_ratio=mr / mz if mz else np.nan, neutral_ratio=nr / nz if nz else np.nan,
                       neutral_ratio_unweighted=nru / nzu if nzu else np.nan, reached_nonzero=int((y != 0).sum())))
    RR = pd.DataFrame(rr); v['reached_cells'] = len(RR)
    v['reached_worse'] = int((RR.reached_ratio > RR.neutral_ratio).sum()) if len(RR) else 0
    v['reached_statuses'] = {k: int(n) for k, n in Rc.status.value_counts().items()}
    # ---- cost
    lab, rpl = jobs('r3_eval_'), jobs('r3_replay_')
    lab = [j for j in lab if re.fullmatch(r'r3_eval_\d+', j['tag'])]; rpl = [j for j in rpl if re.fullmatch(r'r3_replay_\d+', j['tag'])]
    v['label_core_hours'] = sum(j['cpu'] for j in lab) / 3600.; v['replay_core_hours'] = sum(j['cpu'] for j in rpl) / 3600.
    v['label_peak'] = max(j['peak'] for j in lab); v['replay_peak'] = max(j['peak'] for j in rpl); v['label_jobs'] = len(lab); v['replay_jobs'] = len(rpl)
    I = J['readings'].get('I', {}); v['round_core_hours'] = I.get('core_hours'); v['round_peak'] = I.get('peak_mib')
    # ---- the document
    ck = J['checks']; falsified = F.get('falsifier_ok') is True
    lines = [f"# R3 ADV2-RICH v1 — the fixed paired-advantage candidate on consumed Oct 26–27: prediction diagnostic, local control and bounded own-state replay",
             '', f"Review {COMMITS['review']} (one next priority). Freeze 1 `{COMMITS['freeze1'][:8]}` (evaluation sample), freeze 2 `{COMMITS['freeze2'][:8]}` "
             f"(candidate, controls, evaluator, replay, declared checks), model hashes `{COMMITS['fit'][:8]}` (committed before any Oct 26–27 label was read), "
             f"freeze 2b `{COMMITS['freeze2b'][:8]}` (the replay summary v2). "
             f"Evidence as of {J['asof']}. Development session Claude aea5bfb9.", '',
             '## Verdict (every sentence chosen by a computed predicate)', '',
             f"- **Declared checks (v2):** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} "
             f"([checks_v2.json](checks_v2.json)); falsifier v2 {'passed' if falsified else 'FAILED'} ([checks_falsifier_v2.json](checks_falsifier_v2.json)). "
             f"v1 ([checks.json](checks.json), frozen r3_checks.py) has "
             + ('all true' if J1['all_true'] else ', '.join(k for k, x in sorted(J1['checks'].items()) if not x) + ' false')
             + f": its G required a binding check on every ok reached pair, and {len(G1.get('bad', []))} ok KEEP pairs at the final tick made none, "
             f"correctly (no later tick, the other side already decided; {G2.get('zero_callback_ok_pairs')} zero-callback ok pairs in all). "
             "v2 (r3_checks_v2.py) changes only that predicate; v1 stays as written.",
             f"- **Scope:** this round measures prediction skill and local decision value on two consumed development days (Oct 26–27) with a model "
             "fitted on Oct 24–25. It is development evidence only: no interval (two days), no validation, no promotion, no refit.",
             f"- **Incremental prediction (offline, all ok rows, weighted MSE):** RICH is below the zero baseline in {v['beats_zero']} of {v['cells']} "
             f"day × side × head cells, below the Oct 24–25 training mean in {v['beats_mean']}, below FAIR in {v['beats_fair']} and below MECH in "
             f"{v['beats_mech']}.",
             f"- **Plan-v2 screens (RICH, {v['screens_total']} day × side × head × action groups):** "
             + ', '.join(f"{k} {n}" for k, n in v['screens'].get('RICH', {}).items())
             + '. FAIR: ' + ', '.join(f"{k} {n}" for k, n in v['screens'].get('FAIR', {}).items())
             + '. MECH: ' + ', '.join(f"{k} {n}" for k, n in v['screens'].get('MECH', {}).items()) + '.',
             f"- **Local decision value (pi0-conditional, never a policy return):** RICH's value exceeds its fallback-default policy (never post on "
             f"the post head, always keep on the keep head) in {v['beats_fallback']} of {v['cells']} cells and the best constant policy in "
             f"{v['beats_const']}; it is above FAIR's value in {v['above_fair_value']} and above MECH's in {v['above_mech_value']}.",
             f"- **Support:** {v['fallback_rows']:,} of {v['eval_rows']:,} evaluation rows ({100. * v['fallback_share']:.1f} %) fall outside the candidate "
             f"support and take the fallback (" + ', '.join(f'{k} {n:,}' for k, n in sorted(v['fallback_by_status'].items())) + ').',
             f"- **Local action-information control (RICH, day × head):** lift above the matched draw mean in {v['control_lift_pos']} scopes; the "
             f"realized value is above the draws' 95th percentile in {v['control_above_p95']}. This is additive local selection, not a p-value.",
             ]
    for day, r_ in sorted(rep.items()):
        def ds(t, lab_):
            return f"{lab_} {t[0]:+,.1f} c ({t[1]} worlds above, {t[2]} below, of {t[3]})" if t else f'{lab_} —'
        lines.append(f"- **Replay {day} ({r_['worlds']} worlds, joint own-state 5 s episodes; diagnostic, not the R4 gate):** RICH total "
                     f"{r_['totals'].get('RICH', float('nan')):+,.1f} c against PI0 {r_['totals'].get('PI0', float('nan')):+,.1f} c and N0 0 c; paired "
                     + '; '.join([ds(r_['rich_vs_pi0'], 'RICH − PI0'), ds(r_['rich_vs_pa'], 'RICH − ENTRY_PA'), ds(r_['fair_vs_pi0'], 'FAIR − PI0'),
                                  ds(r_['mech_vs_pi0'], 'MECH − PI0'), ds(r_['entry_only_vs_pi0'], 'ENTRY_ONLY − PI0'),
                                  ds(r_['retain_only_vs_pi0'], 'RETAIN_ONLY − PI0')]) + '.')
    lines.append(f"- **State shift (reached RICH states, frozen pi0 arms):** RICH's MSE relative to zero is higher at reached states than on the "
                 f"neutral sample in {v['reached_worse']} of {v['reached_cells']} day × head cells (reached pairs: "
                 + ', '.join(f'{k} {n:,}' for k, n in sorted(v['reached_statuses'].items())) + ').')
    lines += ['', '## Which layer limits the candidate (computed readings, no tuning)', '',
              '| day | side | head | MSE RICH / zero | below zero | below train mean | below FAIR | below MECH | value | fallback-default value | oracle |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, c_ in C_.iterrows():
        lines.append(f"| {c_.day} | {c_.side} | {c_['head']} | {c_.ratio_zero:.3f} | {str(bool(c_.beats_zero)).lower()} | {str(bool(c_.beats_mean)).lower()} | "
                     f"{str(bool(c_.beats_fair)).lower()} | {str(bool(c_.beats_mech)).lower()} | {c_.value:.3f} | {c_.fallback_value:.3f} | {c_.oracle:.3f} |")
    lines += ['', '| day | head | reached pairs | reached nonzero | reached MSE RICH / zero | neutral MSE RICH / zero (weighted) | neutral (unweighted) |',
              '|---|---|---|---|---|---|---|']
    for _, x in RR.iterrows():
        lines.append(f"| {x.day} | {x['head']} | {x.pairs:,} | {x.reached_nonzero:,} | {x.reached_ratio:.3f} | {x.neutral_ratio:.3f} | {x.neutral_ratio_unweighted:.3f} |")
    lines += ['', 'Readings: a ratio below 1 means the prediction beats predicting zero; the value columns are sampling-weighted cents per '
              'opportunity (local, pi0-conditional); the oracle is the value of acting on the realized sign. Two consumed days: no interval.', '']
    lines += ['## Tables (r3_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    fin = {t: suite(f'mutants_{t}_summary.log') for t in ('sample', 'adv2', 'eval', 'replay', 'rsummary')}
    r1 = {t: suite(f'mutants_{t}_run1_summary.log') for t in ('adv2', 'eval', 'replay')}; r2 = suite('mutants_adv2_run2_summary.log')
    rs = json.loads((R / 'replay' / 'replay.json').read_text())
    lines += ['## Suites and provenance', '',
              '- **Final mutation checks:** ' + '; '.join(f"{t} {c} / {n}{'' if ok_ else ' (NOT passed)'}" for t, (c, n, ok_) in fin.items())
              + '. The first four are verified per record by check H; the summary v2 (rsummary) by its own summary log, since r3_checks.py is frozen.',
              f"- **Run history (kept as *_run1 / *_run2):** run 1 caught " + ', '.join(f'{t} {c} / {n}' for t, (c, n, _) in r1.items())
              + f"; run 2 adv2 {r2[0]} / {r2[1]}. Every miss was in the test instrument, and no module changed:",
              '  - crashes at positive controls (a valid fit, ENTRY_PA\'s valid states, the declared policies\' episodes, including the fixed-sign '
              'comparisons) now fail those checks;',
              '  - the replay selftest\'s unguarded lookup of a POST decision is now a failed check;',
              '  - the coverage dimensions are compared with the declared list, not the module\'s constant;',
              '  - one equivalent mutant (the fingerprint after inner(v)) is replaced by the other side\'s identity at the same site.',
              f"- **The replay summary v2 (freeze 2b):** the frozen `pf_adv2_replay.summarize` wrote the four CSVs and failed writing replay.json (an "
              f"int count column under sort_keys; [log](logs/r3_replay_summary.log)). The superseding `pf_adv2_replay_summary.py` "
              f"proved all four CSVs byte-identical and wrote replay.json ({rs['worlds']} ok worlds; statuses "
              + ', '.join(f'{k} {n}' for k, n in sorted(rs['statuses'].items())) + ').',
              '- **Generators ([tools/](tools/)):** fill_r3.py wrote this report from the outputs, and fill_state_r3.py the state hunks. '
              'r3_collect.sh made the receipt copies and the sha256 manifest of the outputs left in data/ ([checkpoint/](checkpoint/)). '
              'r3_worker.sh and r3_finish.sh did the scheduling.',
              '- **The launcher:** launch_r3.sh was extended in place while the label queues ran it. The queues\' bytes were restored into their '
              'inode before any queue ended, and they read exactly freeze 1\'s launcher ([evidence](logs/launcher_inode_repair.txt)).', '']
    lines += ['## Review 315979d2 items 1–3', '',
              '- **Item 1 (support):** per side / head and selected-action coverage by source, horizon and the six support dimensions for the '
              'evaluation and training rows (coverage table above; [coverage.csv](eval/coverage.csv), [coverage_train.csv](eval/coverage_train.csv)); '
              "RICH's own-state exposure in the replay against the training rows ([exposure.csv](replay/exposure.csv)). Counts, not calibration.",
              '- **Item 2 (arithmetic residues):** the screens count meaningful nonzero originals (|Y| > 1e-6 c) for the 20-original floor; the '
              'exact count and the floor under it (`exact_count_support`) are in [screens.csv](eval/screens.csv). Labels, weights and targets unchanged.',
              '- **Item 3 (suite identity):** check H runs pf_mutant_summary.judge on every record against the final sources and requires one '
              'record per file; the falsifier shows it false after one copied record.', '',
              '## Cost', '',
              f"- Labels: {v['label_jobs']} partition jobs, {v['label_core_hours']:.2f} core-hours, peak {v['label_peak']:,} MiB. Replay: "
              f"{v['replay_jobs']} partitions, {v['replay_core_hours']:.2f} core-hours, peak {v['replay_peak']:,} MiB. The round (every run_light job "
              f"under logs/, check I): {v['round_core_hours']:.2f} core-hours of the 40 ceiling, peak {v['round_peak']:,} MiB.", '',
              '## Limits', '',
              '- Two consumed days: point estimates only, no interval, no claim of validation. The fit\'s in-sample readings (fit/fit.json) are not '
              'evidence.',
              '- The replay is the bounded diagnostic of review 315979d2 item 11 (≤ 384 worlds), not the R4 gate: same-history NEU2 and the full '
              'economic / quoting qualification stay deferred; PI0 is not NEU2.',
              '- The direct regressors output mean advantages: no probability or tail calibration is claimed.',
              '- The churn and exposure tables are recorded state; no cause is inferred from them.', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v.update(all_true=bool(J['all_true']), falsifier_ok=falsified, checks={k: bool(x) for k, x in ck.items()}, asof=J['asof'], cells_table=C_.to_dict('records'),
             reached_table=RR.to_dict('records'))
    (Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad') / 'r3_values.json').write_text(json.dumps(v, indent=1, default=str))
    print(json.dumps({k: x for k, x in v.items() if k not in ('cells_table', 'reached_table')}, indent=1, default=str))


if __name__ == '__main__':
    main()
