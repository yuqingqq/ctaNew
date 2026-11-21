"""Assemble the PI0-ADMISSION-EXPOSURE-v1 receipt (report.md) from checks.json / checks_falsifier.json, outputs/admission.json and CSVs, tables.md,
the pilot projection and the run_light logs. Every verdict sentence is the one the committed declaration's reading maps assign to a computed
predicate (parsed from declaration.md, never retyped); every number is read from an output. Writes report.md and admission_values.json."""
import glob
import json
import re
from pathlib import Path

import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/pi0_admission_20261121'); L = R / 'logs'; S_ = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
COMMITS = dict(review='99fb165d', freeze='6b14953f')
READINGS = ('admitted_longer_beneficial', 'admitted_longer_harmful', 'not_identified')
PREDS = ('review_discriminates', 'mse_beats_zero (entry)', 'retention_v2 entry_mse_beats_zero', 'retention_v2 review_mse_beats_zero')


def maps():
    rd, pr = {}, {}
    for ln in (R / 'declaration.md').read_text().splitlines():
        c = [x.strip() for x in ln.strip().strip('|').split('|')] if ln.startswith('| ') else []
        if len(c) == 3 and c[0] in READINGS:
            rd[c[0]] = c[2]
        elif len(c) == 3 and c[0] in PREDS:
            pr[c[0]] = (c[1], c[2])
    assert set(rd) == set(READINGS) and set(pr) == set(PREDS), (rd.keys(), pr.keys())
    return rd, pr


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
    J = json.loads((R / 'checks.json').read_text()); F = json.loads((R / 'checks_falsifier.json').read_text()); X = json.loads((R / 'outputs' / 'admission.json').read_text())
    P = X['predicates']; RM, PM = maps(); T = sections((R / 'tables.md').read_text()); pj = json.loads((L / 'pilot_projection.json').read_text()); ck = J['checks']
    ec = pd.read_csv(R / 'outputs' / 'economics.csv'); fc = pd.read_csv(R / 'outputs' / 'forecasts.csv'); rv = pd.read_csv(R / 'outputs' / 'retention_v2.csv')
    ct = pd.read_csv(R / 'outputs' / 'crosstab.csv')
    peaks, cpu = [], 0.
    for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6))
    st = {}
    for x in X['statuses']:
        st[x['status']] = st.get(x['status'], 0) + int(x['n'])
    holds = [k for k in READINGS if P['readings'][k]]; e0 = ec[(ec.day.astype(str) == 'all')].iloc[0].to_dict()
    fw = fc[(fc.weighted == True) & (fc.day.astype(str) == 'all')].set_index('stage')                                                                          # noqa: E712
    rva = rv[rv.day.astype(str) == 'all'].set_index('stage'); cta = ct[ct.day.astype(str) == 'all'].set_index('cls'); sel = P['review_selection']
    lines = ['# PI0-ADMISSION-EXPOSURE-v1 — longer exposure of PI0\'s own admitted orders, the corrected mean metrics, the cross-tab and RETENTION v2', '',
             f"Review {COMMITS['review']} (one next priority). Freeze `{COMMITS['freeze']}`, committed before any branch ran. Outputs as of {X['asof']}. Development session "
             'Claude aea5bfb9. One diagnostic, zero fitted candidates, consumed Oct 26–27.', '',
             '## Verdict (every sentence chosen by a computed or declared predicate)', '',
             f"- **Declared checks:** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} ([checks.json](checks.json)); "
             f"falsifier {'passed' if F['falsifier_ok'] else 'FAILED'}, {len(F['cases'])} cases ([checks_falsifier.json](checks_falsifier.json)).",
             f"- **Population:** {X['generations']} selected generations of the 689 PI0 POSTs; " + ', '.join(f'{v} {k}' for k, v in sorted(st.items())) + '.']
    for k in holds:
        lines.append(f"- **Reading `{k}`:** {RM[k]}")
    lines.append('- **Per day (inclusion-weighted mean D_EXPOSURE):** ' + '; '.join(f"Oct {d[-2:]} {v['wmean_D']:+.3f} c over {v['n']} generations" for d, v in sorted(P['days'].items()))
                 + f"; {P['gain_forks']} generations gain selected-gid fills by keeping.")
    lines.append(f"- **Pooled:** weighted mean D_EXPOSURE {e0['wmean_D']:+.3f} c (unweighted {e0['mean_D']:+.3f} c): selected generation {e0['wmean_D_selected']:+.3f}, focal rest "
                 f"{e0['wmean_D_focal_rest']:+.3f}, opposite {e0['wmean_D_opposite']:+.3f} c. P(fill) of the selected generation {e0['factual_fill_prob']:.3f} under PI0 → "
                 f"{e0['keep_fill_prob']:.3f} kept; lifetime {e0['factual_wmean_lifetime']:.0f} → {e0['keep_wmean_lifetime']:.0f} ms (weighted).")
    cells = ec[ec.day.astype(str) != 'all']; nneg = int((cells.wmean_D < 0).sum())
    lines.append(f"- **Concentration (descriptive, computed):** D_EXPOSURE is zero on {int(e0['n_zero'])} of {int(e0['n'])} generations ({int(e0['n_pos'])} positive, "
                 f"{int(e0['n_neg'])} negative, {int(e0['n_severe'])} severe); the top-5 |D| share is {e0['top5_share_unweighted']:.2f}; {nneg} of {len(cells)} day × side cells are "
                 "negative.")
    lines.append(f"- **First review:** {PM['review_discriminates'][0 if P['review_discriminates'] else 1]} PULL on {sel['n_sel']} of {sel['n']} rows; percentile "
                 f"{sel['percentile']:.3f}; {sel['free_strata']} of {sel['strata']} strata ({sel['free_rows']} rows) can change assignment; thin {str(sel['thin']).lower()}."
                 if sel['percentile'] is not None else '- **First review:** no PULL rows.')
    if 'entry' in fw.index:
        f_ = fw.loc['entry']; lines.append(f"- **Entry forecasts at PI0's own admissions:** {PM['mse_beats_zero (entry)'][0 if P['mse_beats_zero']['entry'] else 1]} Weighted MSE of V "
                                           f"{f_['mse_V']:.1f} c² against zero {f_['mse_zero']:.1f} and edge × E[q] {f_['mse_mech']:.1f}; mean error {f_['mean_error']:+.3f} c; "
                                           f"Brier {f_['brier']:.4f} against {f_['brier_pooled_rate']:.4f}.")
    if 'first_review' in fw.index:
        f_ = fw.loc['first_review']; lines.append(f"- **First-review forecasts:** weighted MSE of V {f_['mse_V']:.1f} c² against zero {f_['mse_zero']:.1f} and edge × E[q] "
                                                  f"{f_['mse_mech']:.1f}; mean error {f_['mean_error']:+.3f} c.")
    lines.append(f"- **RETENTION v2 (supersedes the MAE-gated reading of `a44f7dd8`):** {PM['retention_v2 entry_mse_beats_zero'][0 if P['retention_v2']['entry_mse_beats_zero'] else 1]} "
                 f"{PM['retention_v2 review_mse_beats_zero'][0 if P['retention_v2']['review_mse_beats_zero'] else 1]} MSE of V / zero / edge × E[q]: entry "
                 f"{rva.loc['entry', 'mse_V']:.2f} / {rva.loc['entry', 'mse_zero']:.2f} / {rva.loc['entry', 'mse_mech']:.2f}; first review {rva.loc['first_review', 'mse_V']:.2f} / "
                 f"{rva.loc['first_review', 'mse_zero']:.2f} / {rva.loc['first_review', 'mse_mech']:.2f} c². Both-modeled flips {X['retention_v2_flips']['flips']} of "
                 f"{X['retention_v2_flips']['both_modeled']}. The ENTRY-RETENTION selection result reads 'not demonstrated on this forced sample'.")
    lines.append('- **Cross-tab (ENTRY-EXPOSURE by shadow admission, adaptive accounting):** ' + '; '.join(
        f"{c_} {int(cta.loc[c_, 'n'])} forks, mean D_EXPOSURE {cta.loc[c_, 'mean_D']:+.2f} c" for c_ in ('would_POST', 'would_WAIT', 'fallback') if c_ in cta.index)
                 + f"; excluded {int(cta.loc['excluded', 'n']) if 'excluded' in cta.index else 0}.")
    lines += ['- **Scope:** adaptive development description on two consumed days; no interval, no significance, no fit, no profitable-policy claim. KEEP_SELECTED changes '
              'every discretionary pull of the selected generation, not only its first. Generations within a world are related.', '',
              '## Tables (admission_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    lines += ['## Review 99fb165d (findings folded in)', '',
              '- **P1:** the MAE gate is superseded by RETENTION v2 above (MSE, calibration, side / stage detail); MAE stays secondary; the perfect-mean mixture is a selftest '
              'control; the ENTRY-RETENTION receipt and gate stay as provenance.',
              '- **P2 (masks):** transitions use both-modeled forks (RETENTION v2 flips; this round by construction); the cross-tab separates model-valid WAIT from fallback.',
              '- **P2 (new information):** new observations are classified per source from ages (ms) and the PF receipt time (tables); a group-change indicator is not read as new '
              'information.',
              '- **P2 (wording):** "not demonstrated on this forced sample"; this round measures PI0\'s actual admissions.',
              '- **P3:** the pilot projection is write-once (tools/pilot_projection_adm.py refuses an existing record).', '',
              '## Provenance', '',
              f"- Mutation checks: the final suite {suite('mutants_adm_summary.log')[0]} / {suite('mutants_adm_summary.log')[1]}; run 1 {suite('mutants_adm_run1_summary.log')[0]} / "
              f"{suite('mutants_adm_run1_summary.log')[1]} (a thin positive case added), run 2 {suite('mutants_adm_run2_summary.log')[0]} / {suite('mutants_adm_run2_summary.log')[1]} "
              '(before the per-generation guard was set to the contract\'s 120 s). The checker\'s synthetic selftest passed before the run.',
              f"- Pilot (part 0): {pj['pilot_cpu_s']:.0f} CPU s over {pj['pilot_generations']} of {pj['selected_generations']} generations, peak {pj['pilot_peak_mib']} MiB; projection "
              f"{pj['projected_core_hours']:.3f} core-hours against the {pj['ceiling']:.0f} ceiling, so the rest {'ran' if pj['run_rest'] else 'did NOT run'}.", '',
              '## Cost', '', f"- {cpu / 3600.:.3f} core-hours of CPU over every run_light job under logs/, recursive (the ceiling is 1); {len(peaks)} jobs, peak {max(peaks)} MiB; "
              f"check G {'true' if ck.get('G_resources') else 'false'}.", '',
              '## Limits', '',
              '- Two consumed days (Oct 26–27) of one adaptive development population; inclusion weights estimate the 689 PI0 POSTs of the 380 replay worlds, not other days.',
              '- Gross values at the common 5 s END mark; remaining queue priority at END has no value.', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v = dict(checks=ck, all_true=bool(J['all_true']), falsifier_ok=bool(F['falsifier_ok']), predicates=P, statuses=st, generations=X['generations'], core_hours=cpu / 3600.,
             peak_mib=max(peaks), asof=X['asof'], holds=holds, sentences={k: RM[k] for k in holds}, pooled=e0, projection=pj,
             review_sentence=PM['review_discriminates'][0 if P['review_discriminates'] else 1],
             retention_sentences=[PM['retention_v2 entry_mse_beats_zero'][0 if P['retention_v2']['entry_mse_beats_zero'] else 1],
                                  PM['retention_v2 review_mse_beats_zero'][0 if P['retention_v2']['review_mse_beats_zero'] else 1]])
    (S_ / 'admission_values.json').write_text(json.dumps(v, indent=1, default=str)); print(json.dumps(dict(holds=holds, all_true=v['all_true']), indent=1))


if __name__ == '__main__':
    main()
