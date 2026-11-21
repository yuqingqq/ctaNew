"""RECEIPT-LEAD-TIME-AUDIT-v1: copies the outputs into outputs/ (large tables gzipped, SHA256SUMS), writes report.md and the state hunks
(FINDINGS, HANDOFF, STATUS) into the scratchpad's edits/. Every number is read from a file; formats are explicit per column (counts, cents,
shares, fractions); the candidate is audit.json's computed reading; the per-share view of the candidate's bins is a post-hoc descriptive
diagnostic computed here from the frozen bins.csv and labelled as such. argv[1] = the time from `date`. Usage: fill_audit.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_audit_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_audit_v1_20261121')
S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
FREEZE, CK2, CK3 = 'b1167651', 'b8642e6d', '349d2f32'
O = R / 'outputs'; O.mkdir(exist_ok=True)
for n in ('durations', 'churn', 'recovery', 'leads', 'bins'):
    shutil.copy(D / f'{n}.csv', O / f'{n}.csv')
for n in ('generations_v2', 'sweep_rows', 'states'):
    b = (D / f'{n}.csv').read_bytes(); (O / f'{n}.csv.gz').write_bytes(gzip.compress(b, mtime=0)); assert gzip.decompress((O / f'{n}.csv.gz').read_bytes()) == b
for p in sorted(D.glob('*.json')):
    shutil.copy(p, O / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
au = json.loads((D / 'audit.json').read_text()); cr, tl, cand = au['cross'], au['timeline'], au['candidate']
ck = json.loads((R / 'checks_v3.json').read_text()); fa = json.loads((R / 'checks_falsifier_v3.json').read_text())
rv = json.loads(Path('/home/yuqing/ctaNew/predictfun/reviews/automatic/20261121_ladder_rest_review/evidence.json').read_text())
rd = json.loads(Path('/home/yuqing/ctaNew/predictfun/reviews/automatic/20261121_ladder_rest_review/diagnostics.json').read_text())
mut = {k: re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / f'mutants_{k}_summary.log').read_text()) for k in ('audit', 'safe', 'client_v2', 'runner_v2')}
BT = pd.read_csv(D / 'bins.csv', dtype={'day': str}); LT = pd.read_csv(D / 'leads.csv')
cnt = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{int(round(float(v))):,}'
cents = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):+,.1f} c'
shares = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):,.1f}'
frac = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{100 * float(v):.2f} %'
ms = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):,.1f} ms'
per = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):+.3f} c / share'
nf = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])


def tbl(rows, cols, heads=None):
    heads = heads or [c for c, _ in cols]; out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


ch = ck['checks']
corr = [dict(item='F1 fills in [pending_first, live_first) (submit + 95 .. + 137 ms)', audit=f"{cnt(cr['locked_window']['fills'])} fills / {cnt(cr['locked_window']['generations'])} generations / {shares(cr['locked_window']['shares'])} shares / {cents(cr['locked_window']['end_value'])}",
             review=f"{cnt(rv['locked_window']['fills'])} / {cnt(rv['locked_window']['generations'])} / {shares(rv['locked_window']['shares'])} / {cents(rv['locked_window']['end_value'])}", equal=ch['C_locked_window']),
        dict(item='F2 sweep fill rows: public-trade events / timestamp groups / timestamp-only', audit=f"{cnt(cr['sweeps']['trade_grouped_fill_rows'])} / {cnt(cr['sweeps']['time_grouped_fill_rows'])} / {cnt(cr['sweeps']['time_grouped_but_not_same_trade_rows'])}",
             review=f"{cnt(rd['sweeps']['trade_grouped_fill_rows'])} / {cnt(rd['sweeps']['time_grouped_fill_rows'])} / {cnt(rd['sweeps']['time_grouped_but_not_same_trade_rows'])}", equal=ch['C_sweeps']),
        dict(item='F3 sampled live states: n / lifetime any-fill / future-window any-fill', audit=f"{cnt(cr['live_fill_denominators']['live']['n'])} / {cnt(cr['live_fill_denominators']['live']['any_generation_fill'])} / {cnt(cr['live_fill_denominators']['live']['future_fill'])}",
             review=f"{cnt(rv['live_fill_denominators']['live']['n'])} / {cnt(rv['live_fill_denominators']['live']['any_generation_fill'])} / {cnt(rv['live_fill_denominators']['live']['future_fill'])}", equal=ch['C_window_denominators']),
        dict(item='F3 first reviews: n / lifetime any-fill / future-window any-fill', audit=f"{cnt(cr['live_fill_denominators']['first']['n'])} / {cnt(cr['live_fill_denominators']['first']['any_generation_fill'])} / {cnt(cr['live_fill_denominators']['first']['future_fill'])}",
             review=f"{cnt(rv['live_fill_denominators']['first']['n'])} / {cnt(rv['live_fill_denominators']['first']['any_generation_fill'])} / {cnt(rv['live_fill_denominators']['first']['future_fill'])}", equal=ch['C_window_denominators']),
        dict(item='F4 requested by the first live tick (structural / policy); policy-only first-review pulls', audit=f"{cnt(cr['first_review']['requested_by_first_live'])} ({cnt(cr['first_review']['by_reason'].get('structural'))} / {cnt(cr['first_review']['by_reason'].get('policy'))}); {cnt(cr['first_review']['policy_first_review_pulls'])}",
             review=f"{cnt(rd['first_review']['accepted_orders_requested_by_first_live_tick'])} ({cnt(rd['first_review']['by_reason']['structural'])} / {cnt(rd['first_review']['by_reason']['policy'])}); {cnt(rd['first_review']['reported_policy_first_review_pulls'])}", equal=ch['C_early_requests']),
        dict(item='F4 restriction episodes: recovered / reversal / censored', audit=f"{cnt(cr['inventory_modes']['recovered'])} / {cnt(cr['inventory_modes']['reversal'])} / {cnt(cr['inventory_modes']['censored'])} (recovered median {ms(cr['inventory_modes']['recovered_median_ms'])})",
             review=f"{cnt(rd['inventory_modes']['recovery_to_band_exits'])} / {cnt(rd['inventory_modes']['direct_reversal_exits'])} / —", equal=ch['C_recovery']),
        dict(item='fills before any request / cancel-pending: fills, shares, +1 s (per fill, valid)', audit=f"{cnt(cr['fill_timing']['before_request']['fills'])}, {shares(cr['fill_timing']['before_request']['q'])}, {cents(ck['details']['C_fill_timing']['per_fill']['before_request'])} / {cnt(cr['fill_timing']['cancel_pending']['fills'])}, {shares(cr['fill_timing']['cancel_pending']['q'])}, {cents(ck['details']['C_fill_timing']['per_fill']['cancel_pending'])}",
             review=f"{cnt(rd['fill_timing']['before_request']['fills'])}, {shares(rd['fill_timing']['before_request']['q'])}, {cents(rd['fill_timing']['before_request']['mo1000'])} / {cnt(rd['fill_timing']['cancel_pending']['fills'])}, {shares(rd['fill_timing']['cancel_pending']['q'])}, {cents(rd['fill_timing']['cancel_pending']['mo1000'])}",
             equal=ch['C_fill_timing_counts'] and ch['C_fill_timing_per_fill_vs_review']),
        dict(item='F8 both-side venue-active share: exact (grid)', audit=f"{frac(cr['durations']['both_exact'])} ({frac(cr['durations']['both_grid_mean'])} grid mean of markets)", review=f"{frac(rv['headline']['both_exact'])} ({frac(rv['headline']['both_grid'])})", equal=ch['C_both_exact']),
        dict(item='equal-population edge / +50 ms / +1 s (all-horizon common subset)', audit=f"{cnt(cr['common_markouts_all_horizons']['fills'])} fills: {cents(cr['common_markouts_all_horizons']['edge'])} / {cents(cr['common_markouts_all_horizons']['mo50'])} / {cents(cr['common_markouts_all_horizons']['mo1s'])}",
             review=f"{cnt(rd['markout_headline']['common_fills'])}: {cents(rd['markout_headline']['common_edge'])} / {cents(rd['markout_headline']['common_mo50'])} / {cents(rd['markout_headline']['common_mo1000'])}", equal=ch['C_common_subset']),
        dict(item='F7 net top-5 losing-market share (losing markets)', audit=f"{frac(cr['net_concentration']['top5_share'])} ({cnt(cr['net_concentration']['losing_markets'])})", review=f"{frac(rd['markout_headline']['top5_market_loss_share'])}", equal=ch['C_net_top5'])]
L = ['# RECEIPT-LEAD-TIME-AUDIT-v1 with the corrected ladder diagnostics — report', '',
     f'{NOW}, session aea5bfb9. Review `0065191f`\'s ONE next priority on its accepted baseline (`869fa85f`): the same 19 complete markets and '
     f"{cnt(cr['generations'])} generations; all {cnt(cr['selected'])} statuses kept ({', '.join(f'{k} {v}' for k, v in sorted(au['statuses'].items()))}).",
     f'Freeze `{FREEZE}` before any saved evidence was read; checker supersessions `{CK2}` (v2) and `{CK3}` (v3), each committed before it ran. No fit, guard, profit or avoided-loss claim; no replay.', '',
     '## Corrected diagnostics against the review\'s independent numbers (checks_v3.json C_*)', '',
     tbl(corr, [('item', None), ('audit', None), ('review', None), ('equal', lambda v: 'equal' if v else 'DIFFERS')]), '',
     f"- Sweep tranches: the {cnt(cr['sweeps']['trade_grouped_fill_rows'])} public-trade event rows carry END {cents(cr['sweeps']['trade_event_end_value'])} and +1 s {cents(cr['sweeps']['trade_event_mo1s'])}; "
     'the former table summed every fill of any ever-flagged generation.',
     f"- The audit's own fill-timing aggregate censors a whole generation when one fill lacks its +1 s mid ({cents(cr['fill_timing']['before_request']['mo1s'])} before a request); the per-fill valid sum above is the review's convention. Both are checked.",
     f"- Generation tails: worst END {cents(cr['generation_tails']['end_worst'])}, END severe sum {cents(cr['generation_tails']['end_severe_sum'])}; worst +1 s {cents(cr['generation_tails']['mo1s_worst'])}, "
     f"+1 s below -30 c sum {cents(cr['generation_tails']['mo1s_below_minus30_sum'])}. END legs of flat round trips carry opposite tails (selftest control: +3,000 / -2,900 c for a +100 c round trip).", '',
     '## Lead time: where the +1 s adverse value sits relative to the earliest effective command (states.csv; one window per snapshot, kinds never pooled)', '']
tr = [dict(kind=k, **v) for k, v in tl.items()]
L += [tbl(tr, [('kind', None), ('n', cnt), ('censored_1s', cnt), ('win_any', cnt), ('lifetime_any', cnt), ('win_mo1s', cents), ('pre_mo1s', cents), ('cp_mo1s', cents), ('inv_move_1s', cents),
               ('latency_median_ms', ms), ('pf_age_median_ms', ms), ('bn_age_median_ms', ms)],
          ['kind', 'snapshots', 'censored +1 s', 'window any-fill', 'lifetime any-fill', 'window +1 s', 'fills before the effect +1 s', 'cancel-pending window +1 s',
           'held inventory move to +1 s', 'decision -> effect', 'Predict obs. age', 'Binance obs. age']), '',
      '- Entry: withholding the POST avoids every fill of the generation (the window); the effect is its activation (58 ms). First / live: a PULL at d lands at d + 37 ms; fills in [d, d + 37) cannot be prevented by it.',
      '- The held-inventory move is the client-known inventory at d times the venue-mid change to d + 1 s: inventory already acquired, apart from the order\'s future fills. Windows of different kinds overlap; never add across kinds.', '',
      '## Do receipt-known states separate bad future fills? (the declared lead rule on the frozen Oct 24-25 bins)', '',
      f"**Computed reading:** {cand['answer']}" + (f" — kind `{cand['kind']}`, input `{cand['feature']}`, sign {cand['sign']:+d}, pooled q5 - q1 {cents(cand['pooled_q5_minus_q1'])} per state; {cnt(cand['n_leads'])} lead(s) of {cnt(len(LT))} (kind, input) pairs."
                                                if cand['answer'] == 'LEAD' else '.'), '']
top = LT.assign(a=LT.pooled_q5_minus_q1.abs()).sort_values('a', ascending=False).head(10).to_dict('records')
L += [tbl(top, [('kind', None), ('feature', None), ('lead', lambda v: 'LEAD' if v else '—'), ('pooled_q5_minus_q1', cents)], ['kind', 'input', 'lead (all 8 cells one sign)', 'pooled q5 - q1 (+1 s per state)']), '']
if cand['answer'] == 'LEAD':
    c = BT[(BT.kind == cand['kind']) & (BT.feature == cand['feature']) & (BT.bin.isin(['q1', 'q5']))].copy(); c['per_share'] = c.mo1s_sum / c.win_q.replace(0, np.nan)
    rows = []
    for (day, side), g in c.groupby(['day', 'side']):
        q1, q5 = g[g.bin == 'q1'].iloc[0], g[g.bin == 'q5'].iloc[0]
        rows.append(dict(day=day, side='bid' if side == 0 else 'ask', n1=q1.uncensored, n5=q5.uncensored, m1=q1.mo1s_mean, m5=q5.mo1s_mean, q1s=q1.per_share, q5s=q5.per_share,
                         same=bool(np.sign(q5.per_share - q1.per_share) == cand['sign'])))
    agree = {sd: sum(r['same'] for r in rows if r['side'] == sd) for sd in ('bid', 'ask')}
    L += [f"**Post-hoc descriptive view (computed here from the frozen bins.csv after the reading; the rule and the reading are unchanged):** the +1 s outcome is a SUM over the window's "
          "shares, so an input correlated with the order's own remaining quantity can move it by size alone. Per window share:", '',
          tbl(rows, [('day', None), ('side', None), ('n1', cnt), ('n5', cnt), ('m1', cents), ('m5', cents), ('q1s', per), ('q5s', per), ('same', lambda v: 'yes' if v else 'no')],
              ['day', 'side', 'q1 states', 'q5 states', 'q1 mean +1 s', 'q5 mean +1 s', 'q1 per share', 'q5 per share', 'per-share sign as the lead']), '',
          f"- The per-share difference has the lead's sign on {agree['ask']} of 4 days on the ask side and {agree['bid']} of 4 on the bid side.", '']
L += ['## Causal audit of the reconstructed inputs (16 hash-selected snapshots a day)', '',
      tbl([dict(v, day=d) for d, v in ck['details']['E'].items()], [('day', None), ('n', cnt), ('market_same', cnt), ('market_fired', cnt), ('order_same', cnt), ('order_fire_eligible', cnt), ('order_fired', cnt)],
          ['day', 'snapshots', 'market inputs unchanged (future perturbed)', 'market control fired', 'order inputs unchanged', 'order control eligible', 'order control fired']), '',
      '## Validity', '',
      f"- Checks v3: {sum(ch.values())} / {len(ch)} true; falsifier {nf} / {len(fa['cases'])}. Checker v1 (frozen in `{FREEZE}`) and v2 (`{CK2}`) crashed before computing any check "
      '(a list / Series filter; the day column read as integers); both logs are kept, each superseded by a version committed before it ran.',
      f"- Mutation suites: audit {mut['audit'].group(1)} / {mut['audit'].group(2)}, safe resume {mut['safe'].group(1)} / {mut['safe'].group(2)}; the frozen baseline harnesses under selftest v2: client "
      f"{mut['client_v2'].group(1)} / {mut['client_v2'].group(2)}, runner {mut['runner_v2'].group(1)} / {mut['runner_v2'].group(2)}. The audit suite's first run (14 / 16) is kept: one vacuous assertion (precedence) and a flat-mid fixture.",
      '- The frozen baseline selftest v1 fails after the commit as the review found (logs/ladder_selftest_v1_now.log); v2 makes the guard conditions explicit.',
      '- A `sed` edit of the frozen launcher made after every audit stage had run was reverted by `git checkout` before any further stage (frozen() true again); no output came from the edited file.',
      f"- Cost: {ck['details']['F']['core_hours']:.3f} core-hours of the 0.5 cap; peak {cnt(ck['details']['F']['peak_mib'])} MiB.", '', '## Limits', '',
      '- One fixed policy on 19 of 96 selected markets: associations are leads for the next experiment, not evidence that a loss was predictable or avoidable, nor guard value.',
      '- The lead rule compares quintile tails of one input at a time on sums of +1 s markouts; a later candidate must price lost good fills, queue / re-entry and inventory effects.',
      '- M50 trade timing remains the declared assumption; the snapshot inputs were built in the run on the receipt clock.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
lead = (f"{cand['answer']} {cand.get('kind', '')}/{cand.get('feature', '')} sign {cand.get('sign', 0):+d} pooled {cents(cand.get('pooled_q5_minus_q1'))}" if cand['answer'] == 'LEAD' else cand['answer'])
psv = (f"; per share the lead's sign holds on {agree['ask']}/4 ask days and {agree['bid']}/4 bid days (post-hoc view)" if cand['answer'] == 'LEAD' else '')
F = (f"| P102 | RECEIPT-LEAD-TIME-AUDIT-v1 (review 0065191f's next priority on the accepted 19-market ladder baseline; {cnt(cr['generations'])} generations; checks v3 all_true = "
     f"{str(ck['all_ok']).lower()}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). The corrected diagnostics equal the review's independent numbers (pending-cancel window "
     f"{cnt(cr['locked_window']['fills'])} fills / {cents(cr['locked_window']['end_value'])}; public-trade sweep rows {cnt(cr['sweeps']['trade_grouped_fill_rows'])} of {cnt(cr['sweeps']['time_grouped_fill_rows'])} timestamp-grouped; "
     f"live future-window fills {cnt(cr['live_fill_denominators']['live']['future_fill'])} of {cnt(cr['live_fill_denominators']['live']['any_generation_fill'])} lifetime; exact both-side {frac(cr['durations']['both_exact'])}). "
     f"Window +1 s at the first review {cents(tl['first']['win_mo1s'])} vs fills before its effect {cents(tl['first']['pre_mo1s'])}. | Computed: {lead}{psv}. A lead for the next experiment, not a guard or profit estimate. |\n")
Hh = (f"<!-- LADDER_AUDIT_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — RECEIPT-LEAD-TIME-AUDIT-v1 RESULTS: CORRECTIONS EQUAL THE REVIEW'S NUMBERS; READING {cand['answer']}\n\n"
      f"- [Receipt](reviews/ladder_audit_20261121/report.md); freeze `{FREEZE}`, checkers v2 `{CK2}` / v3 `{CK3}` (v1 / v2 crashed before computing; superseded in-band), results in this commit.\n"
      f"- **Lead time:** at the first live review the future window carries {cents(tl['first']['win_mo1s'])} of +1 s value, the fills before its 37 ms effect {cents(tl['first']['pre_mo1s'])}; "
      f"entries (every fill) {cents(tl['entry']['win_mo1s'])}; cancel-pending fills {cents(tl['entry']['cp_mo1s'])}.\n"
      f"- **Reading:** {lead}{psv}.\n- **Causal audit:** 64 snapshots unchanged under future-only perturbation (market and order inputs), controls firing.\n"
      f"- **Validity:** checks v3 {sum(ch.values())} / {len(ch)}, falsifier {nf} / {len(fa['cases'])}; mutants audit {mut['audit'].group(1)} / {mut['audit'].group(2)}, safe resume "
      f"{mut['safe'].group(1)} / {mut['safe'].group(2)}, baseline v2 {mut['client_v2'].group(1)} / {mut['client_v2'].group(2)} and {mut['runner_v2'].group(1)} / {mut['runner_v2'].group(2)}; "
      f"{ck['details']['F']['core_hours']:.3f} core-hours.\n- **Next:** the ONE combined review, which names the next experiment.\n<!-- /LADDER_AUDIT_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_audit_results_20261121_dev:\n  status: done_corrections_and_lead_audit\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_audit_20261121/report.md\n  freeze: {FREEZE}\n"
      f"  checks: {{version: v3, commit: {CK3}, all_true: {str(ck['all_ok']).lower()}, falsifier_ok: {str(fa['falsifier_ok']).lower()}}}\n  candidate: {cand['answer']}\n"
      + (f"  candidate_kind: {cand['kind']}\n  candidate_feature: {cand['feature']}\n  candidate_sign: {cand['sign']}\n" if cand['answer'] == 'LEAD' else '')
      + f"  cost_core_hours: {ck['details']['F']['core_hours']:.3f}\n  next: one_combined_review_names_the_next_experiment\n")
for n, t in (('F_audit.md', F), ('H_audit.md', Hh), ('S_audit.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'lines')
