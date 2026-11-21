"""LADDER_REST_INVENTORY_v1: copies the measurement into outputs/ (large tables gzipped, SHA256SUMS), writes report.md and tables.md, and the
state hunks (FINDINGS, HANDOFF, STATUS) into the scratchpad's edits/. Every number is read from a file; every answer is audit.json's computed
predicate; no verdict sentence is printed. argv[1] = the time from `date`. Usage: fill_ladder.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_rest_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_rest_v1_20261121')
MD = D / 'measure'; S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]
FREEZE = '47d2eec1'
O = R / 'outputs'; O.mkdir(exist_ok=True)
SMALL = ('markets', 'summary', 'inventory', 'equity', 'replacements', 'state_map')
BIG = ('generations', 'fills', 'minutes', 'participation', 'state_rows')
for n in SMALL:
    shutil.copy(MD / f'{n}.csv', O / f'{n}.csv')
for n in BIG:
    b = (MD / f'{n}.csv').read_bytes(); (O / f'{n}.csv.gz').write_bytes(gzip.compress(b, mtime=0)); assert gzip.decompress((O / f'{n}.csv.gz').read_bytes()) == b
for n in ('audit.json', 'severe_timelines.json'):
    shutil.copy(MD / n, O / n)
(O / 'days').mkdir(exist_ok=True); (O / 'pilot').mkdir(exist_ok=True)
for sub in ('days', 'pilot'):
    for p in sorted((D / sub).glob('*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, O / sub / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
rd = lambda n: pd.read_csv(MD / f'{n}.csv')
Sm, Mk, Iv = rd('summary'), rd('markets'), rd('inventory')
au = json.loads((MD / 'audit.json').read_text()); ck = json.loads((R / 'checks_v2.json').read_text()); fa = json.loads((R / 'checks_falsifier_v2.json').read_text())
ck1 = json.loads((R / 'checks.json').read_text()); v1_failed = [k for k, v in ck1['checks'].items() if not v]
cov = [dict(r, day=d) for d in ('20261024', '20261025', '20261026', '20261027') for r in json.loads((R / 'logs' / f'coverage_diag_{d}.json').read_text())['markets']]
pj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()); man = json.loads((R / 'manifest.json').read_text()); pr = au['predicates']
mut = {k: re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / f'mutants_{k}_summary.log').read_text()) for k in ('client', 'runner')}
nf = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])
f2 = lambda v, nd=1: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):+,.{nd}f}'
g3 = lambda v, nd=3: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{float(v):,.{nd}f}'
pc = lambda v: '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{100 * float(v):.1f}%'


def tbl(df, cols, heads=None, fmt=None):
    heads = heads or cols; fmt = fmt or {}; out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in df.to_dict('records'):
        num = lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
        out.append('| ' + ' | '.join((fmt[c](r[c]) if (c in fmt and num(r.get(c))) else ('—' if r.get(c) is None or (isinstance(r.get(c), float) and np.isnan(r[c])) else str(r[c])))
                                     for c in cols) + ' |')
    return '\n'.join(out)


A = Sm[Sm.rung.astype(str) == 'all']; both = A[A.side == 'both']; days = list(man['days']) + ['all']
both = both.set_index(both.day.astype(str)).reindex(days).reset_index(drop=True).assign(day=days)
hd = both.assign(severe=both.v_severe)
st = Mk.status.value_counts().to_dict(); comp = int(st.get('complete', 0))
I_ = lambda c: {'value': f2, 'filled_shares': lambda v: g3(v, 0), 'both': pc, 'severe': f2, 'markets': lambda v: str(int(v)), 'generations': lambda v: str(int(v)),
                'fills': lambda v: str(int(v)), 'value_per_market': f2, 'value_per_submitted': lambda v: f2(v, 3), 'value_per_filled': lambda v: f2(v, 3)}.get(c, g3)
L = ['# LADDER_REST_INVENTORY_v1 over full markets — report', '', f'{NOW}, session aea5bfb9. Plan `cf4eee75`, metrics `e3c2b998`, plan review `a8f863af` (ONE next priority).',
     f'Freeze `{FREEZE}` (client, runner, selftest, mutation harnesses, manifest, pilot rule, launcher, checks) before the pilot; the results are in the commit carrying this report.',
     'One fixed strategy on consumed Oct 24-27: descriptive development readings, gross zero-cost pre-expiry marked values in cents, not settlement, validation or deployable P&L.', '',
     '## Population and coverage', '', f"Selected {len(man['selected'])} markets (one per day x opening hour); statuses: " + ', '.join(f'{k} {v}' for k, v in sorted(st.items())) + '.', '',
     tbl(Mk.groupby('day').agg(selected=('m', 'size'), complete=('status', lambda s: int((s == 'complete').sum()))).reset_index(), ['day', 'selected', 'complete']), '',
     'Coverage diagnostic (`tools/coverage_diag.py`, declared after the pilot; it changes no status or reading): per builder status, the distance from the last valid mark to the '
     'supported end (opening + 298 s) and the venue frame gaps above 1 s.', '',
     tbl(pd.DataFrame(cov).groupby('status').agg(markets=('m', 'size'), end_minus_last_mark_median_s=('end_minus_last_mark_s', 'median'), end_minus_last_mark_max_s=('end_minus_last_mark_s', 'max'),
                                                gaps_gt_1s_median=('gaps_gt_1s', 'median'), max_gap_s=('max_gap_s', 'max'), first_gap_rel_s_median=('first_gap_rel_s', 'median')).reset_index(),
         ['status', 'markets', 'end_minus_last_mark_median_s', 'end_minus_last_mark_max_s', 'gaps_gt_1s_median', 'max_gap_s', 'first_gap_rel_s_median'],
         fmt={c: (lambda v: g3(v, 2)) for c in ('end_minus_last_mark_median_s', 'end_minus_last_mark_max_s', 'gaps_gt_1s_median', 'max_gap_s', 'first_gap_rel_s_median')}), '',
     '## Headline (metrics.md 1: final marked value, filled shares, both-side active time, severe-loss contribution)', '',
     tbl(hd, ['day', 'markets', 'value', 'filled_shares', 'both', 'severe', 'value_per_market', 'value_per_filled'], fmt={c: I_(c) for c in hd.columns}), '',
     '## Computed readings (audit.json predicates; patterns, not causal verdicts)', '']
for k in ('strategy_value', 'repeated_fills', 'loss_source'):
    v = pr.get(k, {}); L.append(f"- **{k}**: {v.get('answer')} — " + json.dumps({kk: vv for kk, vv in v.items() if kk != 'answer'}, default=lambda x: round(x, 2) if isinstance(x, float) else str(x)))
L.append(f"- **patterns**: {json.dumps(pr.get('patterns'))} ({pr.get('patterns_note')})"); L.append('')
cols_q = ['day', 'side', 'rung', 'generations', 'accepted', 'any_fill_per_accepted', 'quantity_fill_fraction', 'mean_size', 'repeat_generations', 'mean_events', 'value_repeat_events',
          'median_first_fill_ms', 'median_full_fill_ms', 'censored', 'queue_front_share', 'became_best', 'value_after_best', 'sweep_generations', 'sweep_value']
fq = {c: (pc if c in ('any_fill_per_accepted', 'quantity_fill_fraction', 'queue_front_share') else f2 if c.startswith('value') or c == 'sweep_value' else g3) for c in cols_q}
L += ['## 1. Repeated fills, queue and levels (by birth rung; mechanism description, no ladder-versus-one-level claim)', '',
      tbl(Sm[(Sm.day.astype(str) == 'all')], cols_q, fmt=fq), '',
      '## 2. Value: execution edge, movement, markouts (each horizon on its valid fills; common valid subset), final value', '']
cols_v = ['day', 'side', 'fills', 'filled_shares', 'value', 'edge', 'move_end', 'fills_mid_missing'] + [f'mo{h}' for h in (50, 100, 1000, 5000, 30000)] + [f'mo{h}_missing' for h in (1000, 5000, 30000)]
L += [tbl(A, cols_v, fmt={c: (f2 if c in ('value', 'edge', 'move_end') or c.startswith('mo') and not c.endswith('missing') else g3) for c in cols_v}), '',
      tbl(A[A.side == 'both'], ['day', 'common_fills', 'common_q', 'common_edge', 'common_mo50', 'common_mo1000', 'common_mo5000', 'common_mo30000', 'common_value'],
          fmt={c: f2 for c in ('common_edge', 'common_mo50', 'common_mo1000', 'common_mo5000', 'common_mo30000', 'common_value')}), '',
      '## 3. Inventory management (with no ablation: no causal improvement over no inventory management is claimed)', '']
cols_i = ['day', 'peak_abs_mean', 'mean_abs_inventory', 'share_abs_ge_S', 'restricted_markets', 'transitions', 'episodes', 'episodes_censored', 'recovery_median_ms', 'reversals',
          'reversal_value', 'terminal_abs_mean', 'terminal_gt_R', 'mean_bid_reserved', 'mean_ask_reserved', 'discrepancy_mean', 'max_drawdown_mean']
L += [tbl(A[A.side == 'both'], cols_i, fmt={c: (pc if c == 'share_abs_ge_S' else f2 if c in ('reversal_value',) else g3) for c in cols_i}), '',
      '## Participation and downtime (full supported window; exclusive downtime reason by the declared precedence)', '']
cols_p = ['day', 'side', 'bid_active', 'ask_active', 'bid_best_client', 'ask_best_client', 'bid_best_venue', 'ask_best_venue', 'both', 'either', 'both_at_best']
pp = A[A.side.isin(['bid', 'ask', 'both'])].copy()
L += [tbl(pp, [c for c in cols_p if c in pp.columns], fmt={c: pc for c in cols_p if c not in ('day', 'side')}), '']
dcols = ['day', 'side'] + [c for c in A.columns if c.startswith('down_')]
L += [tbl(A[A.side.isin(['bid', 'ask'])], dcols, fmt={c: pc for c in dcols if c.startswith('down_')}), '', '## Churn (requests by reason; effective removals; first-review pulls)', '']
cols_c = ['day', 'side', 'requests', 'requests_inventory', 'requests_no_book', 'requests_distance', 'requests_cross', 'effective_removals', 'full_fill_before_removal', 'pending_at_end',
          'reviewed', 'first_review_pulls', 'survivors_end', 'replacement_delay_median_ms']
L += [tbl(A, [c for c in cols_c if c in A.columns], fmt={c: g3 for c in cols_c}), '', '## 4. Where losses are concentrated', '']
cols_l = ['day', 'side', 'n_nofill', 'n_gain', 'v_gain', 'n_zero', 'n_ordinary', 'v_ordinary', 'n_severe', 'v_severe', 'n_severe_per_share', 'v_severe_per_share', 'value_locked', 'value_later',
          'severe_locked', 'top5_generation_loss_share', 'worst_generation', 'value_without_severe_expost']
L += [tbl(A, cols_l, fmt={c: (f2 if c.startswith('v_') or c.startswith('value') or c in ('severe_locked', 'worst_generation') else pc if 'share' in c else g3) for c in cols_l}), '',
      tbl(A[A.side == 'both'], ['day', 'losing_markets', 'top5_market_loss_share', 'worst_market'], fmt={'top5_market_loss_share': pc, 'worst_market': f2}), '',
      f"- Value without severe generations is an EX-POST decomposition, never an achievable strategy. Severe timelines: `outputs/severe_timelines.json` ({len(json.loads((MD / 'severe_timelines.json').read_text()))} generations).",
      f"- Minute carry vs new fills by day: carry {json.dumps({k: round(v, 1) for k, v in pr['loss_source']['carry'].items()})}, new {json.dumps({k: round(v, 1) for k, v in pr['loss_source']['new'].items()})}.", '',
      '## Validity', '', f"- Checks A-H (v2): {sum(ck['checks'].values())} / {len(ck['checks'])} true (`checks_v2.json`); falsifier {nf} / {len(fa['cases'])} (`checks_falsifier_v2.json`). "
      f"The frozen v1 (`checks.json`, kept) failed {v1_failed}: its CSV reads were ~1 ulp off (one generation on the -1 c / share boundary); v2 (`42411ab3`, committed before it ran) reads exactly.",
      f"- Mutation suites: client {mut['client'].group(1)} / {mut['client'].group(2)}, runner {mut['runner'].group(1)} / {mut['runner'].group(2)}, each caught by a selftest check.",
      f"- Legacy reproduction: {ck['details']['F']['legacy_worlds']} worlds, engine parity {ck['checks'].get('F_legacy_engine')}, {ck['details']['F']['features_equal']} feature points equal; "
      f"{ck['details']['F']['reconstruct_checked']} snapshots reconstructed from saved events ({ck['checks'].get('F_reconstruction')}).",
      f"- Pilot projection (write-once): {g3(pj.get('projected_core_hours'))} core-hours projected (cap {g3(pj.get('cap'))}), pilot {g3(pj.get('pilot_core_hours'))}; "
      f"run cost {g3(ck['details']['H']['core_hours'])} core-hours, peak {ck['details']['H']['peak_mib']} MiB.", '', '## Limits', '',
      '- One policy: no causal claim about two levels versus one, inventory control, or cancellations; the review\'s patterns are investigation guides.',
      '- Historical replay keeps external flow unchanged by our 100-share quotes; markets and generations of a day share flow and are not independent.',
      '- The supported end is opening + 298 s; the last 2 s and settlement are not modelled; terminal inventory is marked, not liquidated.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
SMp = pd.read_csv(MD / 'state_map.csv')
T = ['# LADDER_REST_INVENTORY_v1 — tables', '', '## Summary (every day x side x rung)', '', tbl(Sm, list(Sm.columns)[:40]), '', '## Decision-state map (bins fixed on Oct 24-25)', '',
     tbl(SMp, list(SMp.columns))]
(R / 'tables.md').write_text('\n'.join(T) + '\n')
h = hd[hd.day == 'all'].iloc[0]; ab = A[(A.day.astype(str) == 'all') & (A.side == 'both')].iloc[0]
cdf = pd.DataFrame(cov); cst = {k: int(v) for k, v in cdf.status.value_counts().items()}
cmed = {k: float(g.end_minus_last_mark_s.median()) for k, g in cdf.groupby('status')}
COV = (f"coverage {cst.get('eligible', 0)} of {len(cdf)} complete: missing_endpoint {cst.get('missing_endpoint', 0)} (valid marks stop a median {cmed.get('missing_endpoint', float('nan')):.1f} s "
       f"before opening + 298 s, no frame gap > 1 s), stale_future_book {cst.get('stale_future_book', 0)} (first gap > 1 s a median "
       f"{float(cdf[cdf.status == 'stale_future_book'].first_gap_rel_s.median()):.1f} s after the opening)")
MK = (f"execution edge {f2(ab['edge'])} c, markouts +50 ms {f2(ab['mo50'])}, +1 s {f2(ab['mo1000'])}, +5 s {f2(ab['mo5000'])} c on their valid fills")
F = (f"| P101 | LADDER_REST_INVENTORY_v1 (one fixed two-level 100-share ladder with inventory hysteresis, full supported markets, consumed Oct 24-27; {comp} of "
     f"{len(man['selected'])} selected markets complete; checks all_true = {str(ck['all_ok']).lower()}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). Final marked value "
     f"{f2(h['value'])} c over {int(h['markets'])} markets, filled shares {g3(h['filled_shares'], 0)}, both-side active {pc(h['both'])}, severe contribution {f2(h['severe'])} c; {MK}; {COV}. | "
     f"Computed: strategy_value {pr['strategy_value']['answer']}, repeated_fills {pr['repeated_fills']['answer']}, loss_source {pr['loss_source']['answer']}; patterns "
     f"{json.dumps(pr.get('patterns'))}. Descriptive; one policy; not validation or deployable P&L. |\n")
Hh = (f"<!-- LADDER_REST_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — LADDER_REST_INVENTORY_v1 RESULTS: {comp} / {len(man['selected'])} MARKETS COMPLETE; CHECKS "
      f"{'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n- [Receipt](reviews/ladder_rest_20261121/report.md); freeze `{FREEZE}`, results in this commit.\n"
      f"- **Headline (all days):** value {f2(h['value'])} c, filled {g3(h['filled_shares'], 0)} shares, both-side active {pc(h['both'])}, severe {f2(h['severe'])} c; {MK}.\n"
      f"- **Coverage (the binding limitation of this run):** {COV}; the builder rules were kept, nothing was bridged or shortened.\n"
      f"- **Checker correction:** v1 (frozen) failed {v1_failed} on ~1 ulp CSV parsing; v2 (`42411ab3`, committed before it ran) reads exactly; v1 kept.\n"
      f"- **Readings:** strategy_value {pr['strategy_value']['answer']}; repeated_fills {pr['repeated_fills']['answer']}; loss_source {pr['loss_source']['answer']}; patterns {json.dumps(pr.get('patterns'))}.\n"
      f"- **Validity:** checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}, mutants {mut['client'].group(1)} / {mut['client'].group(2)} and "
      f"{mut['runner'].group(1)} / {mut['runner'].group(2)}; cost {g3(ck['details']['H']['core_hours'])} core-hours.\n"
      f"- **Next:** the ONE combined review, which names the largest evidenced limitation and one targeted next experiment.\n<!-- /LADDER_REST_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_rest_results_20261121_dev:\n  status: done_one_policy_descriptive\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_rest_20261121/report.md\n  freeze: {FREEZE}\n"
      f"  markets: {{selected: {len(man['selected'])}, complete: {comp}}}\n  checks_all_true: {str(ck['all_ok']).lower()}\n  falsifier_ok: {str(fa['falsifier_ok']).lower()}\n"
      f"  value_cents_all: {float(h['value']):.1f}\n  filled_shares_all: {float(h['filled_shares']):.0f}\n  strategy_value: {pr['strategy_value']['answer']}\n"
      f"  repeated_fills: {pr['repeated_fills']['answer']}\n  loss_source: {pr['loss_source']['answer']}\n  cost_core_hours: {float(ck['details']['H']['core_hours']):.3f}\n"
      f"  next: one_combined_review_names_largest_limitation_and_one_next_experiment\n")
for n, t in (('F_ladder.md', F), ('H_ladder.md', Hh), ('S_ladder.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'lines')
