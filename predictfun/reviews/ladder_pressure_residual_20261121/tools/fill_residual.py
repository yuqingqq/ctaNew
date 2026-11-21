"""PRESSURE-RESIDUAL-TIMING-AUDIT-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and the state hunks (FINDINGS, HANDOFF, STATUS) into
the scratchpad's edits/. Every number is read from a file (the audit's CSVs / audit.json and the checks); formats are explicit per column; the reading is the
audit's computed answer. argv[1] = the time from `date`. Usage: fill_residual.py <time>"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_pressure_residual_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_pressure_residual_v1_20261121')
S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW, FREEZE = sys.argv[1], 'a10eeea3'
O = R / 'outputs'; (O / 'full').mkdir(parents=True, exist_ok=True); (O / 'pilot').mkdir(parents=True, exist_ok=True)
for sc, names in (('full', ('audit.json', 'status_table.csv', 'generation_status.csv', 'generations.csv', 'fills.csv')), ('pilot', ('audit.json', 'status_table.csv', 'generation_status.csv'))):
    for n in names:
        shutil.copy(D / sc / n, O / sc / n)
for n in ('checks_rehearsal.json', 'checks_falsifier_rehearsal.json'):
    shutil.copy(D / 'pilot' / n, O / 'pilot' / n)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
a = json.loads((D / 'full' / 'audit.json').read_text()); rd = a['reading']; F = pd.read_csv(D / 'full' / 'fills.csv', dtype={'day': str}, float_precision='round_trip')
T = pd.read_csv(D / 'full' / 'status_table.csv', dtype={'day': str}, float_precision='round_trip'); GS = pd.read_csv(D / 'full' / 'generation_status.csv', dtype={'day': str})
ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text()); rh = json.loads((D / 'pilot' / 'checks_rehearsal.json').read_text())
proj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()); mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_try1_summary.log').read_text())
nf = sum(1 for v in fa['cases'].values() if v['clean'] and v['refused'])
RLX = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
round_cpu = sum(float(m_.group(6)) for p_ in R.joinpath('logs').rglob('*.log') for ln in p_.read_text().splitlines() for m_ in [RLX.match(ln)] if m_) / 3600.
na = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
cnt = lambda v: '—' if na(v) else f'{int(round(float(v))):,}'
cents = lambda v: '—' if na(v) else f'{float(v):+,.1f} c'
sh = lambda v: '—' if na(v) else f'{float(v):,.1f}'
pc = lambda v: '—' if na(v) else f'{100 * float(v):.2f} %'
sec = lambda v: '—' if na(v) else f'{float(v) / 1e3:,.1f} s'
ORDER = ['pending_pressure', 'pending_inventory', 'pending_structural', 'before_feasible_effect', 'unflagged_later', 'nonevaluable', 'event_or_action_mismatch']


def tbl(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


VAL = ['generations', 'fills', 'shares', 'exposure_ms', 'edge', 'mo50', 'mo50_valid_q', 'mo50_censored', 'mo100', 'mo100_valid_q', 'mo100_censored', 'mo1s', 'mo1s_pos', 'mo1s_neg',
       'mo1s_valid_q', 'mo1s_neg_fills', 'mo1s_censored', 'move_end', 'late_end', 'end']


def agg(by):
    g = T.groupby(by)[VAL].sum().reset_index(); g['_o'] = g.status.map({s: i for i, s in enumerate(ORDER)}); return g.sort_values(['_o'] + [b for b in by if b != 'status']).to_dict('records')


st_rows = agg(['status']); tot = {k: float(T[k].sum()) for k in VAL}; tot.update(status='all')
VC = [('status', None), ('generations', cnt), ('fills', cnt), ('shares', sh), ('exposure_ms', sec), ('edge', cents), ('mo50', cents), ('mo100', cents), ('mo1s', cents), ('mo1s_pos', cents),
      ('mo1s_neg', cents), ('mo1s_valid_q', sh), ('mo1s_censored', cnt), ('move_end', cents), ('late_end', cents), ('end', cents)]
VH = ['status', 'generations', 'fills', 'shares', 'generation live time', 'execution edge', '+50 ms', '+100 ms', 'eligible +1 s', '+1 s > 0', '+1 s < 0', 'valid +1 s shares', 'censored fills',
      'movement to END', 'late / missing END', 'END']
rd_rows = [dict(day=d, tn=v['total']['neg'], tp=v['total']['pos'], ft=v['frac_timing'], fu=v['frac_unflagged'], timn=v['timing']['neg'], timp=v['timing']['pos'], timq=v['timing']['q'],
                timf=v['timing']['fills'], unn=v['unflagged']['neg'], unp=v['unflagged']['pos'], unq=v['unflagged']['q'], unf=v['unflagged']['fills'], oth=v['other']['neg'])
           for d, v in rd['by_day'].items()]
gen = GS.groupby('gen_status').generations.sum().to_dict()
dg = []
for s_ in ORDER:
    g = F[F.status == s_]
    if len(g):
        dg.append(dict(status=s_, fills=len(g), sub_on=float(g.sub_on.astype(bool).mean()), never_live=float(g.T1.isna().mean()), T1_on=float(g.T1_on.fillna(False).astype(bool).mean()),
                       feas=float(g.n_feasible.mean()), last_on=float(g.last_on.fillna(False).astype(bool).mean()), exp=float(g.exposure_ms.median())))
conc = a['concentration']
L = ['# PRESSURE-RESIDUAL-TIMING-AUDIT-v1 — report', '',
     f"{NOW}, session aea5bfb9. This is review `e40bd01d`'s ONE next priority: a saved-evidence timing partition of the full pressure rule's fills.",
     '- **Source arm:** FULL_11 of PRESSURE-ACTION-ATTRIBUTION-v1 (`54eab545`), which equals the accepted pressure arm.',
     f"- **Population:** {cnt(a['reconcile']['markets'])} complete markets, {cnt(a['reconcile']['fills'])} fills and {cnt(a['reconcile']['generations'])} generations. All 96 coverage statuses are kept upstream.",
     f"- **Freeze:** `{FREEZE}`, before any table.",
     '- **Status of the evidence:** consumed Oct 24-27 development days; no replay and no tape load. It is a diagnostic partition: no predictive-skill, avoided-loss, threshold or strategy claim.', '',
     '## Reconciliation and integrity', '',
     f"- **Exact once:** every FULL_11 fill is assigned exactly once. Per market, the fill count, shares, END (= the ledger value), eligible +1 s and edge equal the accepted FULL_11 row (`reconcile` refuses otherwise).",
     f"- **Mismatches:** {cnt(rd['mismatches'])} event or action mismatches. A timely evaluable ON that was kept, or a fill at or after its removal, would be one.",
     f"- **Thin days:** {cnt(len(rd['thin_days']))}.",
     f"- **Status counts:** " + ', '.join(f"{s_} {cnt((F.status == s_).sum())}" for s_ in ORDER) + '.',
     f"- **Generations:** " + ', '.join(f'{k} {cnt(v)}' for k, v in gen.items()) + '.', '',
     '## The reading (declared rule)', '',
     f"**{rd['answer']}.** The rows below are shares of the gross negative eligible +1 s value. Timing-limited means pending_* plus before_feasible_effect; unflagged means unflagged_later. Positive value and quantities are shown alongside.", '',
     tbl(rd_rows, [('day', None), ('tn', cents), ('tp', cents), ('ft', pc), ('timn', cents), ('timp', cents), ('timq', sh), ('timf', cnt), ('fu', pc), ('unn', cents), ('unp', cents), ('unq', sh),
                   ('unf', cnt), ('oth', cents)],
         ['day', 'gross negative +1 s', 'gross positive +1 s', 'timing-limited share', 'timing-limited negative', 'its positive', 'its shares', 'its fills', 'unflagged share', 'unflagged negative',
          'its positive', 'its shares', 'its fills', 'other negative']), '',
     '## By status (all days)', '', tbl(st_rows + [tot], VC, VH), '', '- Generations and their live time are counted once per status: a generation with fills in two statuses counts in both, so the all row is not a distinct-generation count.', '',
     '### By status and side', '', tbl(agg(['status', 'side']), [('side', lambda v: 'bid' if int(v) == 0 else 'ask')] + VC[:1] + VC[1:], ['side'] + VH), '',
     '### By status and birth rung', '', tbl(agg(['status', 'rung']), [('rung', cnt)] + VC, ['rung'] + VH), '',
     '### By status and day', '', tbl(agg(['status', 'day']), [('day', None)] + VC, ['day'] + VH), '',
     '### Concentration and the signal diagnostics', '',
     tbl([dict(status=k, **v) for k, v in sorted(conc.items(), key=lambda kv: ORDER.index(kv[0]))], [('status', None), ('markets', cnt), ('top3_share', pc)],
         ['status', 'markets with negative +1 s', 'top-3 market share of its absolute negative +1 s']), '',
     tbl(dg, [('status', None), ('fills', cnt), ('sub_on', pc), ('never_live', pc), ('T1_on', pc), ('feas', sh), ('last_on', pc), ('exp', sec)],
         ['status', 'fills', 'signal ON at submission', 'never live', 'ON at the first live callback', 'mean feasible ticks', 'ON at the latest feasible tick', 'median fill time after activation']), '',
     '- An ON at submission or at the first live callback is descriptive. The pressure rule acts only through a live callback whose cancel can take effect before the fill.', '',
     '## Review e40bd01d finding', '',
     '- The ablation checker\'s lost/added components and pooled +1 s total are not reused by this audit, so the limitation is retained. The frozen ablation receipt is not edited; any later reuse must extend that checker with the three refusals.', '',
     '## Validity', '',
     f"- **Checks** A / B / F / I: {sum(ck['checks'].values())} / {len(ck['checks'])} true. This checker uses its own status code. Falsifier: {nf} / {len(fa['cases'])}.",
     f"- **Checker rehearsal** on the saved-input pilot: {sum(v for k, v in rh['checks'].items() if k != 'B_membership')} / {len(rh['checks']) - 1} true; B_membership is false by design.",
     f"- **Mutants and selftest:** mutation suite {mut.group(1)} / {mut.group(2)}; selftest groups 1-5 pass.",
     f"- **Cost:** projection {proj['projected_core_hours']:.3f} of 0.12 core-hours. The round used {round_cpu:.3f} core-hours; the ladder line was at {ck['details']['I']['line_core_hours']:.3f} of 2.0 at the checks.", '',
     '## Limits', '',
     '- **What the partition is.** It covers the full rule\'s fills under its own feedback path. Statuses describe when a cancel could have taken effect, not what an earlier cancel would have saved.',
     '- **Replay assumptions.** These carry through from the saved evidence: unchanged external flow, M50 timing, gross zero-cost values and the modelled Binance delay.',
     '- **Coverage.** The population is 19 of 96 markets on consumed days. No day here is a validation day.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
al = rd['by_day']['all']
F_ = (f"| P107 | PRESSURE-RESIDUAL-TIMING-AUDIT-v1 (a saved-evidence timing partition of the full pressure rule's {cnt(a['reconcile']['fills'])} fills on its 19 markets; exact-once reconciliation, "
      f"{cnt(rd['mismatches'])} mismatches; checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). Shares of the gross negative eligible +1 s "
      f"({cents(al['total']['neg'])}): unflagged_later {pc(al['frac_unflagged'])} (" + ' / '.join(pc(rd['by_day'][d]['frac_unflagged']) for d in sorted(rd['by_day']) if d != 'all')
      + f" by day; its positive {cents(al['unflagged']['pos'])}), timing-limited {pc(al['frac_timing'])} (its positive {cents(al['timing']['pos'])}). | Computed reading: {rd['answer']}. A diagnostic "
      f"partition on consumed days, not predictive skill or an avoided-loss estimate. |\n")
Hh = (f"<!-- LADDER_PRESSURE_RESIDUAL_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — PRESSURE-RESIDUAL-TIMING-AUDIT-v1 RESULTS: {rd['answer']} (CONSUMED DAYS); CHECKS {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n"
      f"- [Receipt](reviews/ladder_pressure_residual_20261121/report.md); freeze `{FREEZE}`, results in this commit. {cnt(a['reconcile']['fills'])} fills / {cnt(a['reconcile']['generations'])} generations reconcile exactly; {cnt(rd['mismatches'])} mismatches.\n"
      f"- **Shares of the gross negative +1 s ({cents(al['total']['neg'])}):** unflagged_later {pc(al['frac_unflagged'])} (" + ' / '.join(pc(rd['by_day'][d]['frac_unflagged']) for d in sorted(rd['by_day']) if d != 'all')
      + f"), timing-limited {pc(al['frac_timing'])}; unflagged fills also carry {cents(al['unflagged']['pos'])} positive value; timing-limited {cents(al['timing']['pos'])}.\n"
      f"- **Statuses:** " + ', '.join(f"{s_} {cnt((F.status == s_).sum())}" for s_ in ORDER) + '.\n'
      f"- **Validity:** checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}, mutants {mut.group(1)} / {mut.group(2)}; {round_cpu:.3f} core-hours of 0.12; ladder line {ck['details']['I']['line_core_hours']:.3f} of 2.0.\n"
      f"- **Review e40bd01d finding:** retained (not reused here).\n"
      f"- **Watch:** the ladder line is close to its 2-core-hour allocation; consumed-day evidence only.\n"
      f"- **Next:** the ONE combined review, which names the next step.\n<!-- /LADDER_PRESSURE_RESIDUAL_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_pressure_residual_results_20261121_dev:\n  status: done_saved_evidence_audit\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_pressure_residual_20261121/report.md\n  freeze: {FREEZE}\n"
      f"  fills: {int(a['reconcile']['fills'])}\n  mismatches: {int(rd['mismatches'])}\n  frac_unflagged_pooled: {float(al['frac_unflagged']):.4f}\n  frac_timing_pooled: {float(al['frac_timing']):.4f}\n"
      f"  reading: {rd['answer']}\n  evidence: consumed_days_development_only\n  checks_all_true: {str(ck['all_ok']).lower()}\n  cost_core_hours: {round_cpu:.3f}\n"
      f"  ladder_line_core_hours: {ck['details']['I']['line_core_hours']:.3f}\n  review_findings_e40bd01d: retained_not_reused\n  next: one_combined_review_names_the_next_step\n")
for n, t in (('F_res.md', F_), ('H_res.md', Hh), ('S_res.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'items')
