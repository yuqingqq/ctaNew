"""Assemble the R4E receipt (report.md) from the round's outputs: checks.json / checks_falsifier.json, the receipt's CSV copies (eval/,
replay/), tables.md (r4e_tables.py) and the run_light logs. Every verdict sentence is chosen by a predicate computed here; every number is
read from an output (or is a count, sum, difference or ratio of read values). The predicates follow the declaration (declaration.md).
Usage: fill_r4e.py -> writes report.md in the receipt dir and r4e_values.json in the scratchpad."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/r4e_entry_smooth_20261121'); L = R / 'logs'; S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
SIDE = {0: 'bid', 1: 'ask'}
COMMITS = dict(review='371aab568277dea8ed15c0e386b8c07e7bacd340', freeze='602fdfc5c3441e5c58b39dc4ffd01e4b541daf1b', fit='eaea1161dc39e8048655ebcd775fbf6004b18014',
               amendment='5c2ec2ca1aa496591c95d22034547e95ddb03e1b', freeze2b='c69ea5c68738e4453821dceba5027f3deb8058f9')


def rcsv(d, name):
    p = R / d / name
    return pd.read_csv(p if p.exists() else Path(str(p) + '.gz'))


def jobs(pat):
    out = []
    for p in sorted(L.glob('*.log')):
        for ln in p.read_text().splitlines():
            m = RL.match(ln)
            if m and re.fullmatch(pat, m.group(2)):
                out.append(dict(tag=m.group(2), peak=int(m.group(4)), cpu=float(m.group(6))))
    return out


def suite(name):
    t = (L / name).read_text(); m = re.search(r'(\d+) / (\d+) planted code errors are caught', t); return int(m.group(1)), int(m.group(2))


def sections(md):
    out, cur = {}, None
    for ln in md.splitlines():
        if ln.startswith('### '):
            cur = ln[4:].strip(); out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return {k: '\n'.join(v).strip('\n') for k, v in out.items()}


def main():  # noqa: C901
    J = json.loads((R / 'checks.json').read_text()); F = json.loads((R / 'checks_falsifier.json').read_text()); ev = json.loads((R / 'eval' / 'eval.json').read_text())
    rd, pr = rcsv('eval', 'readings.csv'), rcsv('eval', 'predicates.csv'); E, Rc = rcsv('replay', 'episodes.csv'), rcsv('replay', 'reached.csv')
    RJ = json.loads((R / 'replay' / 'replay.json').read_text()); T = sections((R / 'tables.md').read_text()); v = {}
    g = rd[rd.group == 'all']; cells = []
    for (day, s), x in g.groupby(['day', 'side'], sort=True):
        a, b = x[x.candidate == 'SMOOTH'].iloc[0], x[x.candidate == 'RICH'].iloc[0]
        cells.append(dict(day=str(day), side=SIDE[int(s)], ratio_smooth=a.wmse / a.wmse_zero, ratio_rich=b.wmse / b.wmse_zero, below_zero=a.wmse < a.wmse_zero,
                          below_rich=a.wmse < b.wmse, below_mean=a.wmse < a.wmse_train_mean, value_smooth=float(a.value), value_rich=float(b.value),
                          value_above_rich=float(a.value) > float(b.value), value_above_never=float(a.value) > 0.))
    C_ = pd.DataFrame(cells); v['cells'] = len(C_)
    for k in ('below_zero', 'below_rich', 'below_mean', 'value_above_rich', 'value_above_never'):
        v[k] = int(C_[k].sum())
    pq = pr.assign(skill=pr.post_value_positive.astype(bool) & pr.screen_pass.astype(bool) & pr.above_draw_p95.astype(bool))
    for c in ('SMOOTH', 'RICH'):
        x = pq[pq.candidate == c]
        v[f'pred_{c}'] = dict(cells=len(x), admitted=int((x.admissions > 0).sum()), post_value_positive=int(x.post_value_positive.astype(bool).sum()),
                              screen_pass=int(x.screen_pass.astype(bool).sum()), above_draw_mean=int(x.above_draw_mean.astype(bool).sum()),
                              above_draw_p95=int(x.above_draw_p95.astype(bool).sum()), demonstrated=int(x.skill.sum()), admissions=int(x.admissions.sum()))
    D = pd.DataFrame(RJ['diffs']); rep = {}
    for day in sorted(E.day.astype(str).unique()):
        dd = D[D.day.astype(str) == day]; tot = E[E.day.astype(str) == day].groupby('policy').value.sum()
        def dget(p, b_):
            z = dd[(dd.policy == p) & (dd.versus == b_)]
            return [float(z.diff_sum.iloc[0]), int(z.positive.iloc[0]), int(z.negative.iloc[0]), int(z.n.iloc[0])] if len(z) else None
        e = E[E.day.astype(str) == day]
        rep[day] = dict(worlds=int((e.policy == 'SMOOTH').sum()), totals={p: float(tot.get(p, np.nan)) for p in tot.index}, smooth_vs_entry=dget('SMOOTH', 'ENTRY_ONLY'),
                        smooth_vs_pi0=dget('SMOOTH', 'PI0'), entry_vs_pi0=dget('ENTRY_ONLY', 'PI0'),
                        r3_match={p: int(e[e.policy == p].r3_match.astype(str).eq('True').sum()) for p in ('N0', 'PI0', 'ENTRY_ONLY')},
                        generations={p: int(e[e.policy == p].generations.sum()) for p in ('SMOOTH', 'ENTRY_ONLY')})
    v['replay'] = rep
    ok = Rc[Rc.status == 'ok']; rr = []
    for day, x in ok.groupby('day', sort=True):
        y = x.Y.astype(float).values; mz = float(np.mean(y ** 2))
        ms, mr = float(np.mean((x.pred_SMOOTH.astype(float).values - y) ** 2)), float(np.mean((x.pred_RICH.astype(float).values - y) ** 2))
        rr.append(dict(day=str(day), pairs=int(len(x)), nonzero=int((y != 0).sum()), mse_smooth=ms, mse_rich=mr, mse_zero=mz, smooth_below_rich=ms < mr, smooth_below_zero=ms < mz))
    RR = pd.DataFrame(rr); v['reached'] = rr; v['reached_statuses'] = {k: int(n) for k, n in Rc.status.value_counts().items()}
    lab = jobs(r'r4e_replay_\d+'); v['replay_core_hours'] = sum(j['cpu'] for j in lab) / 3600.; v['replay_peak'] = max(j['peak'] for j in lab) if lab else None
    I = J['readings'].get('I', {}); v['round_core_hours'] = I.get('core_hours'); v['round_peak'] = I.get('peak_mib')
    fin = {t: suite(f'mutants_{t}_summary.log') for t in ('entry', 'entry_eval', 'entry_replay')}
    s1, s2 = suite('mutants_entry_replay_run1_summary.log'), suite('mutants_entry_eval_run1_summary.log')
    ck = J['checks']; ps, prr = v['pred_SMOOTH'], v['pred_RICH']
    lines = ['# R4E ADV2-ENTRY-SMOOTH-v1 — one declared entry candidate on the existing consumed Oct 26–27 labels and the bounded own-state comparison', '',
             f"Review {COMMITS['review']} (one next priority). Freeze `{COMMITS['freeze'][:8]}`, model hashes `{COMMITS['fit'][:8]}` (before any evaluation), freeze "
             f"amendment `{COMMITS['amendment'][:8]}` (the reuse check's reader, before any evaluation output), freeze 2b `{COMMITS['freeze2b'][:8]}` (the replay summary v2 and the "
             f"falsifier v2). Evidence as of {J['asof']}. Development session Claude aea5bfb9.", '',
             '## Verdict (every sentence chosen by a computed predicate)', '',
             f"- **Declared checks:** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} ([checks.json](checks.json)); "
             f"falsifier {'passed' if F['falsifier_ok'] else 'FAILED'} ([checks_falsifier.json](checks_falsifier.json)).",
             '- **Scope:** one declared smoother entry candidate, fitted on Oct 24–25 and tested on the existing Oct 26–27 labels (adaptive consumed-day '
             'development: these days informed the design). No interval, no validation, no promotion, no refit, no sweep.',
             f"- **Reuse:** the evaluation's {ev['identity']['rows']:,} POST rows equal R3's in keys, labels, weights and RICH predictions "
             f"({'all equal' if all(ev['identity'][k] for k in ('same_keys', 'y_equal', 'w_equal', 'rich_pred_equal')) else 'NOT all equal'}; check C "
             f"{'true' if ck.get('C_reuse') else 'false'}).",
             f"- **Prediction (weighted MSE, per day × side):** SMOOTH is below the zero baseline in {v['below_zero']} of {v['cells']} cells, below the "
             f"training mean in {v['below_mean']}, and below the old RICH POST head in {v['below_rich']}.",
             f"- **Declared entry predicates (SMOOTH, {ps['cells']} day × side cells; {ps['admissions']:,} admissions in all against RICH's "
             f"{prr['admissions']:,}):** selected POST value > 0 in {ps['post_value_positive']}, screen PASS in {ps['screen_pass']}, above the matched draw "
             f"mean in {ps['above_draw_mean']} and above its p95 in {ps['above_draw_p95']}. All three conjoined (the declared demonstration) in "
             f"{ps['demonstrated']} of {ps['cells']}. RICH: {prr['post_value_positive']} / {prr['screen_pass']} / {prr['above_draw_mean']} / "
             f"{prr['above_draw_p95']}, demonstrated in {prr['demonstrated']}.",
             f"- **Local entry value (pi0-conditional):** SMOOTH's value is above never posting (0) in {v['value_above_never']} of {v['cells']} cells and "
             f"above RICH's in {v['value_above_rich']}.",
             ]
    for day, r_ in sorted(rep.items()):
        def ds(t, lab_):
            return f"{lab_} {t[0]:+,.1f} c ({t[1]} worlds above, {t[2]} below, of {t[3]})" if t else f'{lab_} —'
        nw = r_['worlds']; match = ', '.join(f'{p} {n} of {nw}' for p, n in r_['r3_match'].items())
        lines.append(f"- **Replay {day} ({r_['worlds']} worlds; diagnostic, not the R4 gate):** " + '; '.join([ds(r_['smooth_vs_entry'], 'SMOOTH − ENTRY_ONLY (primary)'),
                     ds(r_['smooth_vs_pi0'], 'SMOOTH − PI0'), ds(r_['entry_vs_pi0'], 'ENTRY_ONLY − PI0')])
                     + f"; generations SMOOTH {r_['generations']['SMOOTH']:,} against ENTRY_ONLY {r_['generations']['ENTRY_ONLY']:,}. The re-run controls equal "
                     f"R3's episodes in {match}.")
    if len(RR):
        lines.append(f"- **Reached SMOOTH entry states (common states; frozen pi0 arms):** SMOOTH's MSE is below RICH's at the same states on "
                     f"{int(RR.smooth_below_rich.sum())} of {len(RR)} days and below zero on {int(RR.smooth_below_zero.sum())} (pairs: "
                     + ', '.join(f'{k} {n:,}' for k, n in sorted(v['reached_statuses'].items())) + '). Few nonzero labels: descriptive only.')
    lines += ['', '## Tables (r4e_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    lines += ['## Review 371aab56 checklist', '',
              '- **Item 1 (entry selection):** this round.',
              '- **Item 2:** the first-review table names the POST advantage at the post and the retention value at the review as different targets. '
              'Hybrid losses are not summed into a decomposition. FINDINGS P92\'s wording is corrected in this commit.',
              '- **Item 3:** strata by generation, source, horizon and other side; SMOOTH\'s own entry exposure; both heads on identical reached states.',
              '- **Item 4:** every episode and reached pair keeps its full trace. Checks F and G recompute ledgers and required callbacks with the frozen '
              'label validator, and the controls\' reproduction of R3 directly from R3\'s parts.',
              '- **Item 5:** calibration bins carry w_sum and are weight-pooled; R3\'s pooled RICH table is recomputed by weights (tables).',
              '- **Operational:** part records are accepted only with this round\'s identity; the summary checks complete membership; this round has its own launcher.', '',
              '## Suites and provenance', '',
              f"- **Mutation checks (final):** " + '; '.join(f'{t} {c} / {n}' for t, (c, n) in fin.items()) + ' (per record: check H).',
              f"- **Run history:** replay run 1 caught {s1[0]} / {s1[1]}: one replacement did not compile (harness fault, fixed; every mutant is now "
              f"compile-checked). The evaluator's first suite ({s2[0]} / {s2[1]}) is superseded by the amendment's ({fin['entry_eval'][0]} / {fin['entry_eval'][1]}).",
              '- **Freeze amendment:** the first evaluation refused at the declared reuse check because pandas\' default CSV float parser is not exact '
              '([log](logs/r4e_eval_refused_default_parser.log)). The reader now parses round-trip; no reading had been computed.',
              f"- **Freeze 2b (two frozen writers failed after the run):** the frozen summarize wrote the four CSVs and failed on a top-level binding_bad that the "
              f"episodes keep in their traces ([log](logs/r4e_replay_summary.log)); `pf_adv2_entry_replay_summary.py` proved the CSVs byte-identical and wrote "
              f"replay.json (mutation check {suite('mutants_rsum_summary.log')[0]} / {suite('mutants_rsum_summary.log')[1]}). The frozen falsifier crashed (its fill "
              f"corruption looked in one world without fills; [log](logs/r4e_checks.log)); `r4e_checks_falsify_v2.py` searches every partition and records whether each "
              f"corruption was applied. The checks A–I themselves (checks.json) are r4e_checks.py's, unchanged.", '',
              '## Cost', '',
              f"- Replay: {len(lab)} partitions, {v['replay_core_hours']:.2f} core-hours, peak {v['replay_peak']:,} MiB. The round (every run_light job, check I): "
              f"{v['round_core_hours']:.2f} core-hours of the 8 ceiling, peak {v['round_peak']:,} MiB.", '',
              '## Limits', '',
              '- Two consumed days that informed the design: point estimates only, no interval, no validation. In-sample fit readings are not evidence.',
              '- The replay is a bounded diagnostic, not the R4 gate; PI0 is not NEU2. Reached-state labels are few: descriptive only.',
              '- One smoothing package was tested; a negative result does not show that no entry model can work.', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v.update(all_true=bool(J['all_true']), falsifier_ok=bool(F['falsifier_ok']), checks={k: bool(x) for k, x in ck.items()}, asof=J['asof'], cells_table=C_.to_dict('records'),
             identity=ev['identity'], suites=fin, run1_replay=s1)
    (S / 'r4e_values.json').write_text(json.dumps(v, indent=1, default=str))
    print(json.dumps({k: x for k, x in v.items() if k not in ('cells_table',)}, indent=1, default=str))


if __name__ == '__main__':
    main()
