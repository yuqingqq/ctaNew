"""Assemble the ENTRY-EXPOSURE-v1 receipt (report.md) from checks.json / checks_falsifier.json, outputs/exposure.json, tables.md, the pilot
projection and the run_light logs. Every verdict sentence is the one the committed declaration's reading map assigns to a computed predicate
(parsed from declaration.md, never retyped); every number is read from an output (or a count, sum, difference or ratio of read values).
Writes report.md and exposure_values.json (scratchpad)."""
import glob
import json
import re
from pathlib import Path

R = Path('/home/yuqing/ctaNew/predictfun/reviews/entry_exposure_20261121'); L = R / 'logs'; S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
COMMITS = dict(review='3d8bf5ad', freeze='6ca42d25', freeze_2b='5ece0c98', freeze_2c='46a2eabc')


def reading_map():
    """The declaration's two reading-map tables -> ({diagnosis: sentence}, {predicate: (true, false)})."""
    diag, pred = {}, {}
    for ln in (R / 'declaration.md').read_text().splitlines():
        c = [x.strip() for x in ln.strip().strip('|').split('|')] if ln.startswith('| ') else []
        if len(c) == 2 and c[0] in ('keep_better', 'pi0_better', 'not_identified_days_disagree', 'not_identified_thin', 'incomplete'):
            diag[c[0]] = c[1]
        elif len(c) == 3 and c[0] in ('more_execution', 'extra_fills_harmful', 'initial_carries'):
            pred[c[0]] = (c[1], c[2])
    assert len(diag) == 5 and len(pred) == 3, (diag.keys(), pred.keys())
    return diag, pred


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
    J = json.loads((R / 'checks_v3.json').read_text()); F = json.loads((R / 'checks_falsifier_v3.json').read_text()); X = json.loads((R / 'outputs' / 'exposure.json').read_text())
    J1 = json.loads((R / 'checks.json').read_text()); J2 = json.loads((R / 'checks_v2.json').read_text()); F1 = json.loads((R / 'checks_falsifier.json').read_text())
    F2 = json.loads((R / 'checks_falsifier_v2.json').read_text()); falses = lambda j: [k for k, x in sorted(j['checks'].items()) if not x]
    import gzip
    import numpy as np
    import pandas as pd
    FK = pd.read_csv(R / 'outputs' / 'forks.csv.gz', float_precision='round_trip'); SM = pd.read_csv(R / 'outputs' / 'summary.csv', float_precision='round_trip'); okk = FK[FK.status == 'ok']
    max_dop = float(okk.D_opposite.abs().max()); rq = okk[okk.PI0_req.notna()]; eff = (rq.PI0_remove - rq.PI0_act)
    med_ff = (float(SM.median_first_fill_KEEP.min()), float(SM.median_first_fill_KEEP.max())); neg_m1000 = int((SM.sum_m1000_KEEP < 0).sum())
    import sys as _s
    _s.path.insert(0, '/home/yuqing/ctaNew/predictfun/research'); import pf_r2_label_sample as LS
    opp_any = opp_gid = opp_other = 0
    for gp in sorted(Path('/home/yuqing/ctaNew/data/predict_fun/research/entry_exposure_v1_20261121/parts').glob('part_*.jsonl.gz')):
        for r in LS.gz_rows(gp.read_bytes()):
            if r['status'] != 'ok':
                continue
            sd = int(r['side']); oa = [tuple(f) for f in r['arms']['POST_KEEP_INITIAL']['fill_list'] if int(f[2]) != sd]
            ob = [tuple(f) for f in r['arms']['POST_PI0']['fill_list'] if int(f[2]) != sd]; st_ = lambda L: [(f[0], f[2], f[3], f[4], f[5]) for f in L]
            opp_any += int(bool(oa or ob)); opp_gid += int(oa != ob and st_(oa) == st_(ob)); opp_other += int(st_(oa) != st_(ob))
    T = sections((R / 'tables.md').read_text()); P = X['predicates']; ck = J['checks']; D, PM = reading_map(); pj = json.loads((L / 'pilot_projection.json').read_text())
    peaks, cpu = [], 0.
    for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6))
    st = {}
    for x in X['statuses']:
        st[x['status']] = st.get(x['status'], 0) + int(x['n'])
    nok = st.get('ok', 0); excl = {k: v for k, v in st.items() if k != 'ok'}
    B = J['readings'].get('B', {}); days = P['days']
    day_line = '; '.join(f"Oct {d[-2:]}: {v['mean_D']:+.3f} c over {v['forks']} forks" for d, v in sorted(days.items()))
    lines = ['# ENTRY-EXPOSURE-v1 — the same initial POST under PI0 versus keeping that initial generation', '',
             f"Review {COMMITS['review']} (one next priority). Freeze `{COMMITS['freeze']}`, committed before any branch ran. Freeze 2b `{COMMITS['freeze_2b']}` "
             f"(the override readings superseded) and freeze 2c `{COMMITS['freeze_2c']}` (one checker comparison) were each committed before their own output. "
             f"The receipt reports the v2 outputs (as of {X['asof']}) under the v3 checks. Development session Claude aea5bfb9.", '',
             '## Verdict (every sentence chosen by a computed or declared predicate)', '',
             f"- **Declared checks (v3, governing):** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} "
             f"([checks_v3.json](checks_v3.json)); falsifier {'passed' if F['falsifier_ok'] else 'FAILED'}, {len(F['cases'])} cases "
             f"([checks_falsifier_v3.json](checks_falsifier_v3.json)). Earlier check versions are listed under Provenance.",
             f"- **Population:** {X['forks']:,} manifest forks; {nok:,} ok; " + (', '.join(f'{v} {k}' for k, v in sorted(excl.items())) or 'no exclusion') + '. '
             f"Reproduction of the stored PI0 pair on every evaluable fork: {'yes' if ck.get('B_reproduction') else 'NO'} "
             f"({B.get('rows', '—')} of {B.get('evaluable', '—')} evaluable forks; statuses {B.get('by_status', {})}).",
             f"- **Diagnosis `{P['diagnosis']}`:** {D[P['diagnosis']]}",
             f"- **Per day (both sides, equal-world mean D_EXPOSURE):** {day_line}.",
             f"- **Execution:** {PM['more_execution'][0 if P['more_execution'] else 1]} Pooled P(initial fill) PI0 {P['fill_prob_PI0']:.4f} → KEEP {P['fill_prob_KEEP']:.4f}; "
             f"{P['additional_fill_forks']} additional-fill forks.",
             f"- **What the additional fills cost:** {PM['extra_fills_harmful'][0 if P['extra_fills_harmful'] else 1]} Mean D_initial over the additional-fill forks "
             + (f"{P['mean_D_initial_additional']:+.3f} c, {P['share_D_initial_neg_additional']:.3f} of them negative." if P['mean_D_initial_additional'] is not None else '— (none).'),
             f"- **Components:** {PM['initial_carries'][0 if P['initial_carries'] else 1]} Σ D_initial {P['sum_D_initial']:+.2f} c, Σ D_rest {P['sum_D_rest']:+.2f} c.",
             (f"- **The opposite side:** D_opposite is 0 on every ok fork (max |D_opposite| {max_dop:.1f} c). {opp_any} forks have opposite-side fills in either POST arm. "
              f"In {opp_gid} of them the fill lists differ only in the arm-local gid (time, side, quantity, price and trade are equal), and in {opp_other} they "
              "differ otherwise." if max_dop == 0 else f"- **The opposite side:** max |D_opposite| {max_dop:.3f} c; {opp_other} forks have different opposite-side executions."),
             f"- **Timing:** under PI0 the initial order's removal takes effect a mean {eff.mean():.0f} ms after activation (range {eff.min():.0f}–{eff.max():.0f} ms, "
             f"{len(rq)} forks with a request). Kept, its median first fill per day and side lies between {med_ff[0]:.0f} and {med_ff[1]:.0f} ms after the post. "
             f"KEEP's summed +1,000 ms markouts of the initial fills are negative in {neg_m1000} of {len(SM)} day × side cells.",
             '- **Scope:** a declared diagnostic continuation on consumed dates. It is not a fitted candidate, not a minimum-hold timer and not a '
             'profitable-policy claim; no interval and no significance. More fills alone is not success.', '',
             '## Tables (exposure_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    lines += ['## Review 3d8bf5ad (nonblocking items, folded in)', '',
              '- **Item 1:** the audit\'s "rare execution only" recommendation is superseded by: changing Y to V_initial alone would retain the observed '
              'sparsity under PI0. This is not a market-wide scarcity claim, and sparsity alone does not prove the cause of the model failure. The '
              'audit receipt stays as provenance.',
              '- **Item 2:** the audit\'s `cancellation` flag is an aggregate attenuation screen (Σw|Y| against Σw|V_initial|), not an order-cancellation '
              'measure. The rowwise offset of V_initial by D_following is recomputed above from the audit\'s saved rows, and this experiment\'s within-fork '
              'offset of D_initial by D_rest is shown with the components. Value offsets are not order cancellations.',
              '- **Item 3:** the audit checker is not reused. This round\'s check C reconciles every ok fork from its own raw arms (forced-POST gid, '
              'components from fills, fill windows, contrasts). The falsifier and the checker selftest include a compensating-component corruption '
              'and a wrong-but-valid gid.',
              '- **Item 4:** the cost below uses check F\'s recursive aggregation over every job. The audit\'s `top5_abs_share` is a top-5 share of '
              '**unweighted rows**.', '',
              '## Provenance', '',
              f"- **Check versions.** v1 ([checks.json](checks.json)): {', '.join(falses(J1)) or 'none'} false; falsifier "
              f"{'passed' if F1['falsifier_ok'] else 'failed'}. Check C found the KEEP wrapper's `overrides` counter above its decision log on "
              f"{J1['readings']['C']['failing']} forks: the causal probe's decide calls were counted ([declaration_v2.md](declaration_v2.md)). v2 "
              f"([checks_v2.json](checks_v2.json)): {', '.join(falses(J2)) or 'none'} false; falsifier {'passed' if F2['falsifier_ok'] else 'failed'}. Check H "
              "compared NaN counters elementwise ([declaration_v3.md](declaration_v3.md)). v3 governs.",
              f"- **Override readings (summary v2).** {J['readings']['H']['changed']} forks had probe-inflated counters. The forks with an override number "
              f"{J['readings']['H']['overrides_forks_v2']} in v2 and {J['readings']['H']['overrides_forks_v1']} in v1. Every other column, and the predicates, are "
              f"unchanged (check H: {str(J['readings']['H']['same_rest']).lower()} / {str(J['readings']['H']['same_predicates']).lower()}).",
              f"- Mutation checks: the final exposure suite {suite('mutants_exp_summary.log')[0]} / {suite('mutants_exp_summary.log')[1]}. Earlier suites, "
              f"on earlier sources, are kept: run 1 {suite('mutants_exp_run1_summary.log')[0]} / {suite('mutants_exp_run1_summary.log')[1]} (a harness crash at an "
              f"unguarded gid, now a check), run 2 {suite('mutants_exp_run2_summary.log')[0]} / {suite('mutants_exp_run2_summary.log')[1]}, run 3 "
              f"{suite('mutants_exp_run3_summary.log')[0]} / {suite('mutants_exp_run3_summary.log')[1]}, run 4 {suite('mutants_exp_run4_summary.log')[0]} / "
              f"{suite('mutants_exp_run4_summary.log')[1]}, run 5 {suite('mutants_exp_run5_summary.log')[0]} / {suite('mutants_exp_run5_summary.log')[1]} "
              "(two predicate mutants escaped; two hand cases added). The summary v2 suite: "
              f"{suite('mutants_sum_summary.log')[0]} / {suite('mutants_sum_summary.log')[1]}.",
              '- The manifest was built once before the final source and rebuilt with it before the freeze. Its selection, forks and partitions are '
              'identical, and only the recorded source hash differs. No branch had run.',
              f"- Pilot (part 0): {pj['pilot_cpu_s']:.0f} CPU s, peak {pj['pilot_peak_mib']} MiB; projection {pj['projected_core_hours']:.3f} core-hours "
              f"against the 2 ceiling, so the rest {'ran' if pj['run_rest'] else 'did NOT run'}.", '',
              '## Cost', '', f"- {cpu / 3600.:.3f} core-hours of CPU over every run_light job under logs/, recursive (the ceiling is 2); {len(peaks)} jobs, "
              f"peak {max(peaks)} MiB; check F {'true' if ck.get('F_resources') else 'false'}.", '',
              '## Limits', '',
              '- Consumed days only: Oct 24–25 and Oct 26–27 are development days for the frozen PI0. Equal-world finite-sample means, no interval.',
              '- KEEP_INITIAL changes one action of the initial generation. The opposite side and later generations respond under PI0 (capacity, '
              're-entry), so D_EXPOSURE is the continuation-level effect, not the initial order\'s value alone.',
              '- END is unchanged, and queue priority remaining at END has no continuation value (`pf_branch_engine.finish`).', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v = dict(checks=ck, all_true=bool(J['all_true']), falsifier_ok=bool(F['falsifier_ok']), predicates=P, statuses=st, forks=X['forks'], ok=nok, core_hours=cpu / 3600.,
             peak_mib=max(peaks), asof=X['asof'], diagnosis_sentence=D[P['diagnosis']], projection=pj)
    (S / 'exposure_values.json').write_text(json.dumps(v, indent=1, default=str)); print(json.dumps(v, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
