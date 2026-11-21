"""Assemble the ENTRY-RETENTION-DISCRIMINATION-v1 receipt (report.md) from checks.json / checks_falsifier.json, outputs/retention.json, the
outputs' CSVs, tables.md, the pilot projection and the run_light logs. Every verdict sentence is the one the committed declaration's reading
map assigns to a computed predicate (parsed from declaration.md, never retyped); every number is read from an output or computed from one.
Writes report.md and retention_values.json (scratchpad)."""
import glob
import json
import re
from pathlib import Path

import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/entry_retention_20261121'); L = R / 'logs'; S_ = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
COMMITS = dict(review='67a0d9f7', freeze='2ac6e084', freeze_2b='0f89182f')
READINGS = ('entry_identifiable', 'review_protective', 'forecasts_fail', 'not_distinguishable')


def reading_map():
    out = {}
    for ln in (R / 'declaration.md').read_text().splitlines():
        c = [x.strip() for x in ln.strip().strip('|').split('|')] if ln.startswith('| ') else []
        if len(c) == 3 and c[0] in READINGS:
            out[c[0]] = c[2]
    assert set(out) == set(READINGS), out.keys()
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
    J = json.loads((R / 'checks.json').read_text()); F = json.loads((R / 'checks_falsifier_v2.json').read_text()); X = json.loads((R / 'outputs' / 'retention.json').read_text())
    P = X['predicates']; RM = reading_map(); T = sections((R / 'tables.md').read_text()); pj = json.loads((L / 'pilot_projection.json').read_text())
    cal = pd.read_csv(R / 'outputs' / 'calibration.csv', float_precision='round_trip'); ch = pd.read_csv(R / 'outputs' / 'changes.csv', float_precision='round_trip')
    peaks, cpu = [], 0.
    for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6))
    st = {}
    for x in X['fork_statuses']:
        st[x['status']] = st.get(x['status'], 0) + int(x['n'])
    ck = J['checks']; holds = [k for k in READINGS if P['readings'][k]]
    c = {nm: cal[(cal.stage == nm) & (cal.day.astype(str) == 'all')].iloc[0].to_dict() for nm in ('entry', 'first_review', 'first_pull') if ((cal.stage == nm) & (cal.day.astype(str) == 'all')).any()}
    ca = ch[ch.day.astype(str) == 'all'].iloc[0].to_dict() if (ch.day.astype(str) == 'all').any() else {}
    sel = P['selection']; fe = sections((R / 'tables.md').read_text()).get('Held initial fills: fill-time execution edge against the END value (c)', '')
    lines = ['# ENTRY-RETENTION-DISCRIMINATION-v1 — PI0\'s frozen forecasts at entry, first review and first pull against matched targets', '',
             f"Review {COMMITS['review']} (one next priority). Freeze `{COMMITS['freeze']}`, committed before any measurement. Outputs as of {X['asof']}. "
             'Development session Claude aea5bfb9. One diagnostic, zero fitted candidates, consumed Oct 24–27.', '',
             '## Verdict (every sentence chosen by a computed or declared predicate)', '',
             f"- **Declared checks:** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} ([checks.json](checks.json)); "
             f"falsifier v2 {'passed' if F['falsifier_ok'] else 'FAILED'}, {len(F['cases'])} cases ([checks_falsifier_v2.json](checks_falsifier_v2.json); the frozen falsifier "
             "raised on its first case, see Provenance).",
             f"- **Population:** {X['forks']:,} manifest forks; {st.get('ok', 0):,} ok; " + (', '.join(f'{v} {k}' for k, v in sorted(st.items()) if k != 'ok') or 'no exclusion') + '.']
    for k in holds:
        lines.append(f"- **Reading `{k}`:** {RM[k]}")
    for nm, lab in (('entry', 'Entry (shadow admission; the selected set is would_WAIT)'), ('first_review', 'First review (the selected set is PULL)')):
        s_ = sel[nm]; cc = c.get(nm, {})
        lines.append(f"- **{lab}:** {s_['n']} eligible rows, {s_['n_sel']} selected; the selected mean value {s_['mean_sel']:+.3f} c against all {s_['mean_all']:+.3f} c "
                     f"and a matched null of {s_['null_mean']:+.3f} c (5–95 % {s_['null_p05']:+.3f} to {s_['null_p95']:+.3f}); percentile {s_['percentile']:.3f}; "
                     f"thin {str(s_['thin']).lower()}; {s_['degenerate_strata']} of {s_['strata']} strata degenerate. "
                     + (f"MAE of V {cc['mae_V']:.3f} c against {cc['mae_zero']:.3f} c for zero and {cc['mae_mech']:.3f} c for edge × E[q]; Brier {cc['brier']:.4f} against "
                        f"{cc['brier_pooled_rate']:.4f} for the pooled rate." if cc else '') if s_['mean_sel'] is not None else f"- **{lab}:** no selected rows.")
    if 'first_pull' in c:
        cp = c['first_pull']; lines.append(f"- **First pull:** {cp['n']} rows; MAE of V {cp['mae_V']:.3f} c against {cp['mae_zero']:.3f} c for zero; fill rate after the effect "
                                           f"{cp['fill_rate']:.3f} against mean P {cp['mean_P']:.3f}.")
    if ca:
        lines.append(f"- **Entry → first review:** {int(ca['forks'])} forks; market_flow inputs changed in {ca['share_market_flow_changed']:.3f}, fair in {ca['share_fair_changed']:.3f}, "
                     f"order_state in {ca['share_order_state_changed']:.3f}, time in {ca['share_time_changed']:.3f}; mean V {ca['mean_V_entry']:+.3f} → {ca['mean_V_review']:+.3f} c; "
                     f"flip share {ca['flip_share']:.3f}" + (f"; on the {int(ca['stable_market_forks'])} forks with unchanged market and fair inputs {ca['stable_market_flip_share']:.3f}."
                                                              if ca.get('stable_market_forks') else '.'))
    import sys as _s
    _s.path.insert(0, '/home/yuqing/ctaNew/predictfun/research'); import pf_r2_label_sample as LS
    D = Path('/home/yuqing/ctaNew/data/predict_fun/research/entry_retention_v1_20261121/parts'); fte = env = n_f = n_a = 0.
    for gp in sorted(D.glob('part_*.jsonl.gz')):
        for r in LS.gz_rows(gp.read_bytes()):
            if r['status'] == 'ok':
                for f in r['fills']:
                    n_f += 1; env += f['end_value']
                    if f['fill_time_edge'] is not None:
                        n_a += 1; fte += f['fill_time_edge']
    St = pd.read_csv(R / 'outputs' / 'stages.csv.gz'); E = St[(St.stage == 'entry') & (St.fork_status == 'ok')]; nw = int((E.admission == 'would_WAIT').sum())
    lines.append('- **Descriptive, computed (not declared readings):**')
    for nm in ('entry', 'first_review'):
        cc = c[nm]; lines.append(f"  - {nm}: mean V {cc['mean_V']:+.3f} c against mean value {cc['mean_value']:+.3f} c (bias {cc['mean_V'] - cc['mean_value']:+.3f} c); the movement term "
                                 f"{'adds' if cc['mae_V'] > cc['mae_mech'] else 'removes'} error (MAE {cc['mae_mech']:.3f} c for edge × E[q] against {cc['mae_V']:.3f} c for V); "
                                 f"mean P {cc['mean_P']:.3f} against fill rate {cc['fill_rate']:.3f}; qm | fill MAE {cc['mae_qm_fill']:.3f} c.")
    lines.append(f"  - PI0's shadow admission is would_WAIT on {nw} of {len(E)} forced entries ({nw / len(E):.3f}): these probes are mostly entries PI0 itself would not make.")
    lines.append(f"  - The held initial fills ({int(n_f)}; {int(n_a)} with a fill-time mid) gain {fte:+.1f} c against the venue mid at their fill times and are worth {env:+.1f} c at the "
                 "END mark.")
    lines += ['- **Scope:** adaptive development description on consumed days; no interval, no significance, no fit, no new target chosen here. POST and RETAIN '
              'scores have different targets, so their difference is not called a forecast reversal.', '', '## Tables (retention_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    lines += ['## Review 73f6cfa6 (nonblocking items, folded in)', '',
              '- **Item 1:** restated. Suppressing every discretionary PULL of the forced initial generation (the entire initial-generation retention policy) loses on '
              'the forced forks; it does not show that the first pull was necessary. The 86 ms was the mean removal-effect delay after activation among the 735 '
              'orders with a request. This receipt reports shadow entry admission, first-review status and first-pull time separately.',
              '- **Item 2:** this measurement compares P, E[q], E[qm] and V with matched neutral-hold targets on full 107-input vectors, against zero and edge × E[q].',
              '- **Item 3:** check B enforces END 5,000, one mark and settings, and the true callback / probe denominators, with wrong-END and missing-callback '
              'falsifiers.',
              '- **Item 4:** stage capture runs in the teacher\'s fn (actual callbacks only); decide has no side effect.',
              '- **Item 5:** a fill\'s END-mark value is named end_value; the fill-time venue-mid execution edge is reported with its denominators (tables).', '',
              '## Provenance', '',
              f"- Mutation checks: the final suite {suite('mutants_ret_summary.log')[0]} / {suite('mutants_ret_summary.log')[1]}; run 1 "
              f"{suite('mutants_ret_run1_summary.log')[0]} / {suite('mutants_ret_run1_summary.log')[1]} (an equivalent mutant on the side test, which the gid and the "
              'stop rule already imply, was replaced; the unavoidable sign was moved into a tested function). The checker\'s synthetic selftest passed before the run.',
              f"- The frozen falsifier raised `TypeError` at its first case before recording any (one line unpacked two paths instead of one argument tuple). Freeze 2b "
              f"`{COMMITS['freeze_2b']}` corrected that line (declaration_v2.md, retention_checks_v2.py); the frozen checks A–H in checks.json are unchanged.",
              f"- Pilot (part 0): {pj['pilot_cpu_s']:.0f} CPU s, peak {pj['pilot_peak_mib']} MiB; projection {pj['projected_core_hours']:.3f} core-hours against the "
              f"{pj['ceiling']:.0f} ceiling, so the rest {'ran' if pj['run_rest'] else 'did NOT run'}.", '',
              '## Cost', '', f"- {cpu / 3600.:.3f} core-hours of CPU over every run_light job under logs/, recursive (the ceiling is 1); {len(peaks)} jobs, peak "
              f"{max(peaks)} MiB; check G {'true' if ck.get('G_resources') else 'false'}.", '',
              '## Limits', '',
              '- Forced original-time neutral mandatory POST probes, not voluntary PI0 admissions; the entry stage reports what PI0 would have done there.',
              '- Unweighted rows within repeated worlds; both side forks of a world are related; four consumed days; adaptive development descriptions.',
              '- Remaining queue priority at END has no value (`pf_branch_engine.finish`); targets are gross at the common END mark.', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v = dict(checks=ck, all_true=bool(J['all_true']), falsifier_ok=bool(F['falsifier_ok']), predicates=P, statuses=st, forks=X['forks'], ok=st.get('ok', 0), core_hours=cpu / 3600.,
             peak_mib=max(peaks), asof=X['asof'], holds=holds, sentences={k: RM[k] for k in holds}, calibration=c, changes=ca, projection=pj)
    (S_ / 'retention_values.json').write_text(json.dumps(v, indent=1, default=str)); print(json.dumps(dict(holds=holds, all_true=v['all_true']), indent=1))


if __name__ == '__main__':
    main()
