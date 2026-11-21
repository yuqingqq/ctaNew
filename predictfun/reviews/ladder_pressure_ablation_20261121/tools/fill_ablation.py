"""PRESSURE-ACTION-ATTRIBUTION-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and the state hunks (FINDINGS, HANDOFF, STATUS) into
the scratchpad's edits/. Every number is read from a file (the measurement's CSVs / audit.json, checks v2, the part records); formats are explicit
per column; the readings are the measurement's computed predicates. argv[1] = the time from `date`. Usage: fill_ablation.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_pressure_ablation_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_pressure_ablation_v1_20261121')
RUN = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_rest_v1_20261121'); S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
NOW, FREEZE, CHECKS_V2 = sys.argv[1], '8f618ea6', 'd4b843c3'
ARMS = ('BASELINE_00', 'PULL_ONLY_10', 'WITHHOLD_ONLY_01', 'FULL_11')
O = R / 'outputs'
for sub in ('pilot', 'rest', 'rehearsal'):
    (O / sub).mkdir(parents=True, exist_ok=True)
for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv', 'audit.json'):
    shutil.copy(D / 'measure' / n, O / n)
for n in ('checks_v2_rehearsal.json', 'checks_falsifier_v2_rehearsal.json'):
    shutil.copy(D / 'rehearsal' / n, O / 'rehearsal' / n)
for sub in ('pilot', 'rest'):
    for p in sorted((D / sub).glob('*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, O / sub / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
A_ = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); CT = pd.read_csv(D / 'measure' / 'control.csv', dtype={'day': str})
LA = pd.read_csv(D / 'measure' / 'lost_added.csv', dtype={'day': str}, float_precision='round_trip')
au = json.loads((D / 'measure' / 'audit.json').read_text()); pr = au['predicates']; gm = json.loads((R / 'manifest.json').read_text())
ck = json.loads((R / 'checks_v2.json').read_text()); fa = json.loads((R / 'checks_falsifier_v2.json').read_text()); rh = json.loads((D / 'rehearsal' / 'checks_v2_rehearsal.json').read_text())
proj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()); mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_try1_summary.log').read_text())
nf = sum(1 for v in fa['cases'].values() if v['clean'] and v['refused'])
RLX = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
round_cpu = sum(float(m_.group(6)) for p_ in R.joinpath('logs').rglob('*.log') for ln in p_.read_text().splitlines() for m_ in [RLX.match(ln)] if m_) / 3600.
gzr = lambda p: [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]
stat = pd.Series([r['status'] for d in gm['days'] for r in gzr(RUN / 'days' / f'{d}.jsonl.gz')]).value_counts()
na = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
cnt = lambda v: '—' if na(v) else f'{int(round(float(v))):,}'
cents = lambda v: '—' if na(v) else f'{float(v):+,.1f} c'
sh = lambda v: '—' if na(v) else f'{float(v):,.1f}'
per = lambda v: '—' if na(v) else f'{float(v):+.3f} c / share'
pc = lambda v: '—' if na(v) else f'{100 * float(v):.2f} %'
sec = lambda v: '—' if na(v) else f'{float(v) / 1e3:.2f} s'
ratio = lambda a, b: a / b if b > 0 else np.nan


def tbl(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


days = sorted(A_.day.unique()); rows = []
for day in days + ['all']:
    for arm in ARMS:
        g = A_[(A_.arm == arm) & ((A_.day == day) if day != 'all' else True)]; s_ = lambda c: float(g[c].sum())
        rows.append(dict(day=day, arm=arm, value=s_('value'), q=s_('all_q'), edge=s_('all_edge'), mo50=s_('all_mo50'), mo100=s_('all_mo100'), id1=s_('all_id_mo1s'), idm=s_('all_id_move'),
                         idl=s_('all_id_late'), vq=s_('all_cut_valid_q'), cc=s_('all_cut_censored'), per1=ratio(s_('all_cut_mo1s'), s_('all_cut_valid_q')),
                         ask_per1=ratio(s_('ask_cut_mo1s'), s_('ask_cut_valid_q')), bid_per1=ratio(s_('bid_cut_mo1s'), s_('bid_cut_valid_q')), good=s_('all_good_mo1s'), badv=s_('all_bad_mo1s'),
                         pulls=s_('pulls'), wh=s_('withholds'), red=s_('reducing_suppressed'), gens=s_('generations'), struct=s_('bid_structural') + s_('ask_structural'),
                         inv_req=s_('bid_req_inventory_pull') + s_('ask_req_inventory_pull'), frp=s_('first_review_pulls'), rev=s_('reviewed'), bl=g.bid_life_mean_ms.mean(), al=g.ask_life_mean_ms.mean(),
                         inv=g.mean_abs_inv.mean(), lim=g.share_abs_ge_S.mean(), term=g.terminal_inv.abs().mean(), both=g.both_exact.mean(), bid_up=g.bid_exact.mean(), ask_up=g.ask_exact.mean(),
                         r0q=s_('rung0_q'), r0e=s_('rung0_end'), r1q=s_('rung1_q'), r1e=s_('rung1_end'), qa=g.q_activation_ahead_mean.mean(), qpa=g.q_pull_venue_ahead_mean.mean(),
                         pw=s_('pairs_with_next'), psp=s_('pairs_same_price'), pdiff=g.pairs_ahead_difference_mean.mean()))
T = pd.DataFrame(rows); TA = [r for r in T.to_dict('records') if r['day'] == 'all']; ctl = int(CT.control_all.sum())
NAMES = {'cancel_given_withhold': 'cancellation added to withholding (R11 - R01)', 'withhold_given_cancel': 'withholding added to cancellation (R11 - R10)',
         'cancel_alone': 'standalone cancellation (R10 - R00)', 'withhold_alone': 'standalone withholding (R01 - R00)'}
crow = []
for name, c in pr['contrasts'].items():
    crow.append(dict(name=NAMES[name], ea=c['END']['answer'], et=c['END']['total'], ma=c['mo1s']['answer'], mt=c['mo1s']['total'], pa=c['per_share']['answer'], pt=c['per_share']['total'],
                     b=c['markets_better'], w=c['markets_worse'], e=c['markets_equal'], t3=c['top3_abs_share']))
cday = [dict(name=NAMES[n], **{d: c['END']['by_day'][d] for d in days}) for n, c in pr['contrasts'].items()] + [dict(name='interaction (R11 - R10 - R01 + R00)', **pr['interaction']['END']['by_day'])]
cps = [dict(name=NAMES[n], **{d: c['per_share']['by_day'][d] for d in days}) for n, c in pr['contrasts'].items()]
la_rows = []
for name in pr['contrasts']:
    g = LA[LA.contrast == name]; s_ = lambda c: float(g[c].sum())
    la_rows.append(dict(name=NAMES[name], ln=s_('lost_n'), lq=s_('lost_q'), lp=s_('lost_mo1s_pos'), lneg=s_('lost_mo1s_neg'), le=s_('lost_end'), an=s_('added_n'), aq=s_('added_q'), ap=s_('added_mo1s_pos'),
                        aneg=s_('added_mo1s_neg'), ae=s_('added_end'), dv=s_('delta_value')))
it, an = pr['interaction'], pr['all_arms_negative']; tot = pr['arm_totals']
L = ['# PRESSURE-ACTION-ATTRIBUTION-v1 — report', '',
     f"{NOW}, session aea5bfb9. This is review `7fca0db4`'s ONE next priority: one 2 x 2 ablation of the original nominal-clock pressure rule (`793c706b`, results `40a3ee8d`).",
     '- **Bits:** C = pressure PULL of a would-be-KEEP live order; W = pressure WAIT of a would-be-POST slot.',
     f"- **Population:** the same {cnt(sum(len(v) for v in gm['complete'].values()))} complete markets of 96 (statuses: " + ', '.join(f'{k} {cnt(v)}' for k, v in stat.items()) + ').',
     f"- **Freeze and checker:** freeze `{FREEZE}`, before any ablated replay; checks v2 `{CHECKS_V2}` supersedes v1 (see Validity).",
     '- **Multiplicity:** two new real arms, PULL_ONLY_10 and WITHHOLD_ONLY_01. FULL_11 and BASELINE_00 are controls.',
     '- **Status of the evidence:** consumed Oct 24-27 development days. There is no interval, winner, promotion, threshold, feature or further candidate, and no profitability claim.', '',
     '## Arms and controls', '',
     f"- **FULL_11:** reproduces the accepted pressure arm on {ctl} of {len(CT)} markets: fingerprint, fills, orders, requests, value, log, command array, queue records.",
     '- **BASELINE_00:** the pressure receipt\'s disabled arm, equal to the original ladder (checks v2 C). The selftest proves that the 00 class delegates exactly to LadderInventory.',
     '- **PULL_ONLY_10 / WITHHOLD_ONLY_01:** each acts only through its own component, at ON ticks only. A pull-only arm may submit during pressure, as declared (checks v2 D).',
     f"- **Parts:** pilot (first two complete markets per date) and rest (the other {cnt(sum(len(v) for v in gm['parts']['rest'].values()))}; Oct 27's rest is DAY_EMPTY). They are disjoint and cover the 19 markets.", '',
     '## Arm totals', '', 'Gross, zero cost, in cents. END = cash + inventory × the END mark. The +1 s figures are on the shared cutoff t <= END - 1 s.', '',
     tbl(TA, [('arm', None), ('value', cents), ('q', sh), ('edge', cents), ('mo50', cents), ('mo100', cents), ('id1', cents), ('idm', cents), ('idl', cents), ('vq', sh), ('cc', cnt), ('per1', per),
              ('ask_per1', per), ('bid_per1', per)],
         ['arm', 'END', 'filled shares', 'execution edge', '+50 ms', '+100 ms', 'eligible +1 s', 'movement to END', 'late / missing', 'valid +1 s shares', 'censored', '+1 s per share', 'ask per share',
          'bid per share']), '',
     f"- **all_arms_negative:** {an['total']}. By day: " + ', '.join(f'{d} {v}' for d, v in an['by_day'].items()) + '. Every arm\'s END is below zero; the contrasts below are relative gains only.', '',
     '## The declared contrasts', '', 'Each is classified against zero incremental value.', '',
     tbl(crow, [('name', None), ('ea', None), ('et', cents), ('ma', None), ('mt', cents), ('pa', None), ('pt', per), ('b', cnt), ('w', cnt), ('e', cnt), ('t3', pc)],
         ['contrast', 'END', 'END total', 'eligible +1 s', 'eligible +1 s total', '+1 s per share', 'per-share total', 'markets better', 'worse', 'equal', 'top-3 share of the absolute END contrasts']), '',
     f"- **Interaction** (R11 - R10 - R01 + R00):",
     f"  - END {it['END']['answer']}, total {cents(it['END']['total'])}; by day " + ', '.join(f'{d} {cents(v)}' for d, v in it['END']['by_day'].items()) + ';',
     f"  - eligible +1 s {it['mo1s']['answer']}, total {cents(it['mo1s']['total'])}.",
     '- **Sums:** the standalone effects are never summed as the full effect without the interaction. FULL_11 - BASELINE_00 = cancel alone + withhold alone + interaction.', '',
     '### END contrasts by day', '',
     tbl(cday, [('name', None)] + [(d, cents) for d in days], ['contrast'] + days), '',
     '### +1 s per share contrasts by day', '', 'Each arm\'s pooled valid-share denominator is in the arm table.', '',
     tbl(cps, [('name', None)] + [(d, per) for d in days], ['contrast'] + days), '',
     '### Fills lost and added per contrast', '', 'Fills are matched by the multiset of (t, side, price, quantity); candidate - reference END = added - lost.', '',
     tbl(la_rows, [('name', None), ('ln', cnt), ('lq', sh), ('lp', cents), ('lneg', cents), ('le', cents), ('an', cnt), ('aq', sh), ('ap', cents), ('aneg', cents), ('ae', cents), ('dv', cents)],
         ['contrast', 'lost fills', 'lost shares', 'lost +1 s > 0', 'lost +1 s < 0', 'lost END', 'added fills', 'added shares', 'added +1 s > 0', 'added +1 s < 0', 'added END', 'candidate - reference END']), '',
     '## Breakouts per arm (all days)', '',
     '`withheld proposals` count slot decisions, not orders. Inventory is exact from the fills; uptime is exact from the order intervals.', '',
     tbl(TA, [('arm', None), ('pulls', cnt), ('wh', cnt), ('red', cnt), ('inv_req', cnt), ('struct', cnt), ('gens', cnt), ('rev', cnt), ('frp', cnt), ('bl', sec), ('al', sec), ('good', cents), ('badv', cents),
              ('r0q', sh), ('r0e', cents), ('r1q', sh), ('r1e', cents)],
         ['arm', 'pressure pulls', 'withheld proposals', 'reducing-side', 'inventory PULL requests', 'structural requests', 'generations', 'reviewed', 'first-review pulls', 'bid mean life', 'ask mean life',
          'good fills (+1 s > 0)', 'bad fills (+1 s < 0)', 'rung 0 shares', 'rung 0 END', 'rung 1 shares', 'rung 1 END']), '',
     tbl(TA, [('arm', None), ('inv', sh), ('lim', pc), ('term', sh), ('both', pc), ('bid_up', pc), ('ask_up', pc), ('qa', sh), ('qpa', sh), ('pw', cnt), ('psp', cnt), ('pdiff', sh)],
         ['arm', 'mean |I|', 'time |I| >= 100', 'mean terminal |I|', 'both-side active', 'bid active', 'ask active', 'activation-clock ahead', 'venue ahead at a pull', 'next-same-side pairs', 'same price',
          'ahead difference next - pull']), '',
     '- The queue columns carry review 909c671e\'s labels. The next-same-side pairs are a diagnostic, not a matched replacement or a counterfactual queue cost.',
     f"- The pressure receipt's 500-draw local screen ({au['context_screen']['answer']}, percentile {pc(au['context_screen']['percentile'])}) is context only, not an entry-action or whole-policy null.", '',
     '## Review 7fca0db4 finding', '',
     '- The re-entry rows are exported again. Checks v2 H compares every exported field against the ledger, and its falsifier refuses a changed re-entry price. The lag receipt (frozen) is not edited.', '',
     '## Validity', '',
     f"- **Checks v2** A-I: {sum(ck['checks'].values())} / {len(ck['checks'])} true. Falsifier: {nf} / {len(fa['cases'])} corruptions refused by their family.",
     f"- **Supersession of the checker:** the pre-run rehearsal on the pilot copy passed every v1 check, but its falsifier stopped at the record-rule case: a ** spread of a tuple-keyed dict (`logs/ga_rehearse_checks.log`). v2 (`{CHECKS_V2}`) fixes only that case and was committed before it ran.",
     f"- **v2 rehearsal:** {sum(v for k, v in rh['checks'].items() if k != 'B_membership')} / {len(rh['checks']) - 1} checks true; B_membership is false by design.",
     f"- **Mutants and selftest:** mutation suite {mut.group(1)} / {mut.group(2)} on the first run; selftest groups 1-5 pass.",
     f"- **Cost:** projection {proj['projected_core_hours']:.3f} core-hours (cap 0.35); ladder line projected {proj['line_projected_core_hours']:.3f} of 2.0. The round used {round_cpu:.3f} core-hours; at the checks the line had recorded {ck['details']['I']['line_core_hours']:.3f} core-hours; peak {cnt(ck['details']['I']['peak_mib'])} MiB.", '',
     '## Limits', '',
     '- **Complete policies.** These are complete feedback policies on 19 consumed-day markets. A favourable P&L contrast alone is not predictive skill, and the contrasts are not per-order avoided losses.',
     '- **Replay assumptions.** External flow is unchanged and timing is M50; the Binance clock is the declared median-delay model. Values are gross and zero cost, and END is a pre-expiry mark.',
     '- **Remaining budget.** The original 2-core-hour ladder allocation is nearly used; a further round of this size needs the USER\'s decision.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
ct = pr['contrasts']
F = (f"| P106 | PRESSURE-ACTION-ATTRIBUTION-v1 (a 2 x 2 ablation of the pressure rule's pull C and withhold W on its 19 markets; FULL_11 {ctl} / {len(CT)} equal to the pressure arm; checks v2 "
     f"{sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). END totals: BASELINE_00 {cents(tot['BASELINE_00']['END'])}, PULL_ONLY_10 {cents(tot['PULL_ONLY_10']['END'])}, "
     f"WITHHOLD_ONLY_01 {cents(tot['WITHHOLD_ONLY_01']['END'])}, FULL_11 {cents(tot['FULL_11']['END'])}. Cancellation alone {cents(ct['cancel_alone']['END']['total'])} ({ct['cancel_alone']['END']['answer']}), "
     f"withholding alone {cents(ct['withhold_alone']['END']['total'])} ({ct['withhold_alone']['END']['answer']}), cancellation given withholding {cents(ct['cancel_given_withhold']['END']['total'])}, "
     f"withholding given cancellation {cents(ct['withhold_given_cancel']['END']['total'])}, interaction {cents(it['END']['total'])} ({it['END']['answer']}); all_arms_negative {an['total']}. | "
     f"Computed contrasts only - no winner or promotion; consumed-day evidence; every arm loses at END. |\n")
Hh = (f"<!-- LADDER_PRESSURE_ABLATION_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — PRESSURE-ACTION-ATTRIBUTION-v1 RESULTS: CANCELLATION CARRIES MOST OF THE EFFECT, POSITIVE INTERACTION, EVERY ARM NEGATIVE (CONSUMED DAYS); CHECKS v2 {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n"
      f"- [Receipt](reviews/ladder_pressure_ablation_20261121/report.md); freeze `{FREEZE}`, checks v2 `{CHECKS_V2}` (the rehearsal caught a v1 falsifier crash before the rest ran), results in this commit. FULL_11 {ctl} / {len(CT)} equal to the pressure arm.\n"
      f"- **END totals:** 00 {cents(tot['BASELINE_00']['END'])}, 10 {cents(tot['PULL_ONLY_10']['END'])}, 01 {cents(tot['WITHHOLD_ONLY_01']['END'])}, 11 {cents(tot['FULL_11']['END'])} (all_arms_negative {an['total']}).\n"
      f"- **Contrasts (END; per share):** " + '; '.join(f"{NAMES[n]} {cents(c['END']['total'])} {c['END']['answer']}, {per(c['per_share']['total'])} {c['per_share']['answer']}" for n, c in ct.items())
      + f"; interaction {cents(it['END']['total'])} {it['END']['answer']}.\n"
      f"- **Validity:** checks v2 {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}, mutants {mut.group(1)} / {mut.group(2)}; {round_cpu:.3f} core-hours of 0.35; ladder line {ck['details']['I']['line_core_hours']:.3f} of 2.0 at the checks.\n"
      f"- **Review 7fca0db4 finding addressed:** field-by-field re-entry checks with a changed-price refusal.\n"
      f"- **Watch:** consumed-day evidence; every arm loses at END; the 2-core-hour ladder allocation is nearly used (a further round of this size needs the USER).\n"
      f"- **Next:** the ONE combined review, which names the next step.\n<!-- /LADDER_PRESSURE_ABLATION_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_pressure_ablation_results_20261121_dev:\n  status: done_2x2_ablation\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_pressure_ablation_20261121/report.md\n  freeze: {FREEZE}\n  checks: {CHECKS_V2}\n"
      f"  full_control_equal: {ctl}/{len(CT)}\n" + ''.join(f"  {n}_END: {c['END']['answer']}\n  {n}_END_cents: {float(c['END']['total']):.1f}\n" for n, c in ct.items())
      + f"  interaction_END: {it['END']['answer']}\n  interaction_END_cents: {float(it['END']['total']):.1f}\n  all_arms_negative: {str(an['total']).lower()}\n  evidence: consumed_days_development_only\n"
      f"  checks_all_true: {str(ck['all_ok']).lower()}\n  cost_core_hours: {round_cpu:.3f}\n  ladder_line_core_hours: {ck['details']['I']['line_core_hours']:.3f}\n  review_findings_7fca0db4: addressed\n"
      f"  next: one_combined_review_names_the_next_step\n")
for n, t in (('F_abl.md', F), ('H_abl.md', Hh), ('S_abl.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'items')
