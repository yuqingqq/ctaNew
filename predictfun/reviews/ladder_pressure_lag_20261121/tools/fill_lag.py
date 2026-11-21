"""PRESSURE-LEAD-SENSITIVITY-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and the state hunks (FINDINGS, HANDOFF, STATUS) into the
scratchpad's edits/. Every number is read from a file (the measurement's CSVs / audit.json, the selection, checks v2, the day records); formats are
explicit per column; the readings are the measurement's computed predicates. argv[1] = the time from `date`. Usage: fill_lag.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_pressure_lag_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_pressure_lag_v1_20261121')
RUN = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_rest_v1_20261121'); S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
NOW, FREEZE, CHECKS_V2 = sys.argv[1], 'b19b7ca0', '9cc6161a'
O = R / 'outputs'; O.mkdir(exist_ok=True); (O / 'days').mkdir(exist_ok=True); (O / 'pilot').mkdir(exist_ok=True); (O / 'rehearsal').mkdir(exist_ok=True)
for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv', 'audit.json'):
    shutil.copy(D / 'measure' / n, O / n)
shutil.copy(D / 'selection.json', O / 'selection.json')
for n in ('checks_v2_rehearsal.json', 'checks_falsifier_v2_rehearsal.json'):
    shutil.copy(D / 'rehearsal' / n, O / 'rehearsal' / n)
for sub in ('days', 'pilot'):
    for p in sorted((D / sub).glob('*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, O / sub / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
A_ = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); CT = pd.read_csv(D / 'measure' / 'control.csv', dtype={'day': str})
LA = pd.read_csv(D / 'measure' / 'lost_added.csv', dtype={'day': str}, float_precision='round_trip'); RP = pd.read_csv(D / 'measure' / 'reentry.csv', dtype={'day': str})
au = json.loads((D / 'measure' / 'audit.json').read_text()); pr = au['predicates']; sel = json.loads((D / 'selection.json').read_text()); gm = json.loads((R / 'manifest.json').read_text())
ck = json.loads((R / 'checks_v2.json').read_text()); fa = json.loads((R / 'checks_falsifier_v2.json').read_text()); rh = json.loads((D / 'rehearsal' / 'checks_v2_rehearsal.json').read_text())
proj = json.loads((R / 'logs' / 'pilot_projection.json').read_text()); mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_try2_summary.log').read_text())
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
pp = lambda v: '—' if na(v) else f'{100 * float(v):+.2f} pp'
sec = lambda v: '—' if na(v) else f'{float(v) / 1e3:.2f} s'
ratio = lambda a, b: a / b if b > 0 else np.nan


def tbl(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


days = sorted(A_.day.unique()); rows = []
for day in days + ['all']:
    for arm in ('baseline', 'lag0', 'lag1'):
        g = A_[(A_.arm == arm) & ((A_.day == day) if day != 'all' else True)]; s_ = lambda c: float(g[c].sum()) if c in g else np.nan
        rows.append(dict(day=day, arm=arm, value=s_('value'), q=s_('all_q'), edge=s_('all_edge'), mo50=s_('all_mo50'), mo50_q=s_('all_mo50_valid_q'), mo50_c=s_('all_mo50_censored'),
                         mo100=s_('all_mo100'), mo100_q=s_('all_mo100_valid_q'), mo100_c=s_('all_mo100_censored'), cut=s_('all_cut_mo1s'), cut_q=s_('all_cut_valid_q'), cut_c=s_('all_cut_censored'),
                         per1=ratio(s_('all_cut_mo1s'), s_('all_cut_valid_q')), ask_per1=ratio(s_('ask_cut_mo1s'), s_('ask_cut_valid_q')), bid_per1=ratio(s_('bid_cut_mo1s'), s_('bid_cut_valid_q')),
                         id1=s_('all_id_mo1s'), idm=s_('all_id_move'), idl=s_('all_id_late'), pulls=s_('pulls'), wh=s_('withholds'), whe=s_('withholds_near_end'), red=s_('reducing_suppressed'),
                         gens=s_('generations'), struct=s_('bid_structural') + s_('ask_structural'), inv_req=s_('bid_req_inventory_pull') + s_('ask_req_inventory_pull'), frp=s_('first_review_pulls'),
                         frpp=s_('first_review_pressure'), rev=s_('reviewed'), bl=g.bid_life_mean_ms.mean(), al=g.ask_life_mean_ms.mean(), inv=g.mean_abs_inv.mean(), peak=g.peak_abs_inv.mean(),
                         lim=g.share_abs_ge_S.mean(), term=g.terminal_inv.abs().mean(), both=g.both_exact.mean(), bid_up=g.bid_exact.mean(), ask_up=g.ask_exact.mean(),
                         qa=g.q_activation_ahead_mean.mean(), qf=g.q_first_review_venue_ahead_mean.mean(), qk=g.q_first_review_known_ahead_mean.mean(), qpa=g.q_pull_venue_ahead_mean.mean(),
                         pn=s_('pairs_pulls'), pw=s_('pairs_with_next'), psp=s_('pairs_same_price'), psr=s_('pairs_same_rung'), pdiff=g.pairs_ahead_difference_mean.mean() if 'pairs_ahead_difference_mean' in g else np.nan,
                         pspd=g.pairs_same_price_ahead_difference_mean.mean() if 'pairs_same_price_ahead_difference_mean' in g else np.nan, gap=g.pairs_gap_ms_mean.mean() if 'pairs_gap_ms_mean' in g else np.nan))
T = pd.DataFrame(rows); TR = T.to_dict('records'); TL = [r for r in TR if r['arm'] != 'baseline']; ctl = int(CT.control_all.sum()); tot = T[T.day == 'all'].set_index('arm')
la_rows = []
for pair in ('lag1_vs_baseline', 'lag1_vs_lag0'):
    for day in days + ['all']:
        g = LA[(LA.pair == pair) & ((LA.day == day) if day != 'all' else True)]; s_ = lambda c: float(g[c].sum())
        la_rows.append(dict(pair=pair, day=day, ln=s_('lost_n'), lq=s_('lost_q'), lp=s_('lost_mo1s_pos'), lneg=s_('lost_mo1s_neg'), le=s_('lost_end'), an=s_('added_n'), aq=s_('added_q'),
                            ap=s_('added_mo1s_pos'), aneg=s_('added_mo1s_neg'), ae=s_('added_end'), dv=s_('delta_value')))
pm = []
for (d, m), g in A_.groupby(['day', 'm']):
    v = {a: g[g.arm == a].iloc[0] for a in ('baseline', 'lag0', 'lag1')}
    pm.append(dict(day=d, m=int(m), b=v['baseline'].value, z0=v['lag0'].value, z1=v['lag1'].value, d0=v['lag0'].value - v['baseline'].value, d1=v['lag1'].value - v['baseline'].value,
                   d10=v['lag1'].value - v['lag0'].value))
v1, v0, rt = pr['lag1_vs_baseline'], pr['lag0_vs_baseline'], pr['retained']; ps = sel['per_side']
rd = [dict(day=d, l1=v1['END_paired']['by_day'][d], l0=v0['END_paired']['by_day'][d], d10=rt['lag1_minus_lag0_END'][d], re=rt['END'][d], rp=rt['per_share'][d],
           p1=v1['mo1s_per_share']['all']['by_day'][d], p0=v0['mo1s_per_share']['all']['by_day'][d]) for d in days]
rd.append(dict(day='all', l1=v1['END_paired']['total'], l0=v0['END_paired']['total'], d10=sum(rt['lag1_minus_lag0_END'].values()), re=rt['END']['all'], rp=rt['per_share']['all'], p1=np.nan, p0=np.nan))
L = ['# PRESSURE-LEAD-SENSITIVITY-v1 — report', '',
     f"{NOW}, session aea5bfb9. This is review `909c671e`'s ONE next priority: the frozen BINANCE-PRESSURE-GUARD-v1 rule (`793c706b`, results `40a3ee8d`) with **one more 50 ms of signal delay**.",
     f"- **Population:** the same {cnt(sum(len(v) for v in gm['complete'].values()))} complete markets of 96 (statuses: " + ', '.join(f'{k} {cnt(v)}' for k, v in stat.items()) + ').',
     f"- **Freeze:** `{FREEZE}`, before any lagged replay. Rule sha256 `{gm['rule_sha256'][:16]}`; parent rule `{gm['parent']['rule_sha256'][:16]}`.",
     f"- **Checker:** checks v2 `{CHECKS_V2}` supersedes v1. It was committed before it ran on any full-run output; see Validity.",
     '- **Status of the evidence:** a timing-margin sensitivity on consumed Oct 24-27 development days. It chooses no lag or threshold and makes no validation, promotion or profitability claim.', '',
     '## Rule and controls', '',
     '- **Rule.** At action tick k the frozen pressure rule reads its signal and status at tick k - 1 on its own side, from the pressure receipt\'s hash-bound saved cache.',
     '  - Tick 0 is warm-up.',
     '  - PULL, withhold and clearing all read the delayed stream.',
     '  - Inventory, structural and all other decisions stay current; there is no timer.',
     f"- **Zero-lag control:** it reproduces the pressure receipt on {ctl} of {len(CT)} markets (fingerprint, fills, orders, requests, value, commands, the parent's log fields, queue records).",
     f"- **Baseline:** the pressure receipt's disabled arm, which equals the original ladder (checks v2 C).",
     f"- **Lagged join:** {cnt(sel['join']['live_states'])} live states; {cnt(sel['join']['zero_lag_mismatched'])} zero-lag mismatches; {cnt(sel['join']['off_tick'])} off-tick; {cnt(sel['join']['warmup'])} at warm-up; {cnt(sel['join']['non_ok'])} non-ok.", '',
     '## Three arms', '', 'Gross, zero cost, in cents. END = cash + inventory × the END mark.', '',
     tbl(TR, [('day', None), ('arm', None), ('value', cents), ('q', sh), ('edge', cents), ('mo50', cents), ('mo50_q', sh), ('mo50_c', cnt), ('mo100', cents), ('mo100_q', sh), ('mo100_c', cnt)],
         ['day', 'arm', 'END value', 'filled shares', 'execution edge', '+50 ms', 'valid shares', 'censored', '+100 ms', 'valid shares', 'censored']), '',
     '### +1 s on the shared cutoff and the END identity', '', 'The cutoff is t <= END - 1 s.', '',
     tbl(TR, [('day', None), ('arm', None), ('cut', cents), ('cut_q', sh), ('cut_c', cnt), ('per1', per), ('ask_per1', per), ('bid_per1', per), ('id1', cents), ('idm', cents), ('idl', cents)],
         ['day', 'arm', '+1 s', 'valid shares', 'censored', 'per share', 'ask per share', 'bid per share', 'eligible +1 s', 'movement to END', 'late / missing']), '',
     '### Paired END, per share and the retained gain', '', 'The retained gain is (lag1 - baseline) / (lag0 - baseline). It is descriptive.', '',
     tbl(rd, [('day', None), ('l0', cents), ('l1', cents), ('d10', cents), ('re', pc), ('p0', per), ('p1', per), ('rp', pc)],
         ['day', 'lag0 - baseline END', 'lag1 - baseline END', 'lag1 - lag0 END', 'retained END', 'lag0 - baseline per share', 'lag1 - baseline per share', 'retained per share']), '',
     '### Fills lost and added', '', 'Fills are matched by the multiset of (t, side, price, quantity); candidate - reference END = added - lost.', '',
     tbl(la_rows, [('pair', None), ('day', None), ('ln', cnt), ('lq', sh), ('lp', cents), ('lneg', cents), ('le', cents), ('an', cnt), ('aq', sh), ('ap', cents), ('aneg', cents), ('ae', cents), ('dv', cents)],
         ['pair', 'day', 'lost fills', 'lost shares', 'lost +1 s > 0', 'lost +1 s < 0', 'lost END', 'added fills', 'added shares', 'added +1 s > 0', 'added +1 s < 0', 'added END', 'candidate - reference END']), '',
     '### Per market', '',
     tbl(pm, [('day', None), ('m', None), ('b', cents), ('z0', cents), ('z1', cents), ('d0', cents), ('d1', cents), ('d10', cents)],
         ['day', 'market', 'baseline END', 'lag0 END', 'lag1 END', 'lag0 - baseline', 'lag1 - baseline', 'lag1 - lag0']), '',
     f"- lag1 - baseline: {sum(1 for x in pm if x['d1'] > 0)} markets gain, {sum(1 for x in pm if x['d1'] < 0)} lose.",
     f"- lag1 - lag0: {sum(1 for x in pm if x['d10'] > 0)} markets gain, {sum(1 for x in pm if x['d10'] < 0)} lose.", '',
     '### Interventions, removals, lifetimes, inventory and uptime', '',
     '`withheld proposals` count slot decisions, not orders. `reducing-side` means bids at I < 0 and asks at I > 0. Inventory is exact from the fills; uptime is exact from the order intervals.', '',
     tbl([r for r in TR if r['day'] == 'all'], [('arm', None), ('pulls', cnt), ('wh', cnt), ('whe', cnt), ('red', cnt), ('inv_req', cnt), ('struct', cnt), ('gens', cnt), ('rev', cnt), ('frp', cnt),
                                               ('frpp', cnt), ('bl', sec), ('al', sec), ('inv', sh), ('peak', sh), ('lim', pc), ('term', sh), ('both', pc), ('bid_up', pc), ('ask_up', pc)],
         ['arm (all days)', 'pressure pulls', 'withheld proposals', 'near END', 'reducing-side', 'inventory PULL requests', 'structural requests', 'generations', 'reviewed', 'first-review pulls',
          'of them pressure', 'bid mean life', 'ask mean life', 'mean |I|', 'mean peak |I|', 'time |I| >= 100', 'mean terminal |I|', 'both-side active', 'bid active', 'ask active']), '',
     '### Queue diagnostics (relabelled, review 909c671e finding 2)', '',
     '- The activation-clock snapshot is the engine\'s queue ahead from the venue frame at activation (submission + 58 ms), evaluated at submission.',
     '- Next-same-side pairs pair each pressure pull with the same side\'s next submission at any price and rung. They are a diagnostic, not a matched replacement or a counterfactual queue cost.', '',
     tbl([r for r in TL if r['day'] == 'all'], [('arm', None), ('qa', sh), ('qf', sh), ('qk', sh), ('qpa', sh), ('pn', cnt), ('pw', cnt), ('psp', cnt), ('psr', cnt), ('gap', sec), ('pdiff', sh), ('pspd', sh)],
         ['arm (all days, market means)', 'activation-clock ahead', 'venue ahead at first review', 'known estimate at first review', 'venue ahead at the pull', 'pulls', 'with a next order',
          'same price', 'same rung', 'gap to the next', 'ahead difference next - pull', 'same-price ahead difference']), '',
     '## Lagged selection screen', '',
     'The frozen screen runs with the lagged mask: strata, 500 draws, seed 20261121, minimum support and censoring are unchanged.', '',
     f"- **Pooled:** {cnt(sel['selected'])} selected of {cnt(sel['states'])} uncensored states; {cnt(sel['movable_strata'])} movable strata with {cnt(sel['movable_selected'])} selected. Selected {cents(sel['J'])} against null mean {cents(sel['null_mean'])} and p05 {cents(sel['null_p05'])}; percentile {pc(sel['percentile'])}. Per valid share {per(sel['J_per_valid_share'])} (null p05 {per(sel['null_q_p05'])}).",
     f"- **Per side (descriptive):**",
     f"  - bids: {cnt(ps['0']['selected'])} selected, {cents(ps['0']['J'])} against p05 {cents(ps['0']['null_p05'])}; percentile {pc(ps['0']['percentile'])};",
     f"  - asks: {cnt(ps['1']['selected'])} selected, {cents(ps['1']['J'])} against p05 {cents(ps['1']['null_p05'])}; percentile {pc(ps['1']['percentile'])}.", '',
     '## Computed readings (measure/audit.json predicates)', '',
     f"- **lag1 vs baseline:**",
     f"  - END_paired {v1['END_paired']['answer']} (total {cents(v1['END_paired']['total'])}; {v1['END_paired']['markets_better']} better, {v1['END_paired']['markets_worse']} worse);",
     f"  - per share all {v1['mo1s_per_share']['all']['answer']}, bid {v1['mo1s_per_share']['bid']['answer']}, ask {v1['mo1s_per_share']['ask']['answer']};",
     f"  - exact active time: both {pp(v1['participation']['both_exact_delta'])}, bids {pp(v1['participation']['bid_exact_delta'])}, asks {pp(v1['participation']['ask_exact_delta'])};",
     f"  - inventory: mean |I| {sh(v1['inventory']['mean_abs_delta'])}; selection {v1['selection']}.",
     f"- **lag0 vs baseline (the reference):** END {v0['END_paired']['answer']} (total {cents(v0['END_paired']['total'])}); per share {v0['mo1s_per_share']['all']['answer']}.",
     f"- **Retained gain:** END {pc(rt['END']['all'])} in total (by day " + ', '.join(f'{d} {pc(rt["END"][d])}' for d in days) + f"); per share {pc(rt['per_share']['all'])}.",
     f"- **Reading (declared rule):** `{pr['reading']}`" + (f"; failed: {', '.join(pr['failed'])}" if pr['failed'] else '') + '.', '',
     '## Review 909c671e findings', '',
     '1. **Checker coverage (finding 1).** Checks v2\'s F family rebuilds exact uptime from each order\'s activation, removal and full fill, and exact inventory from all fills. It also rebuilds the queue fields and pair summaries for all three arms, and compares the participation and inventory readings. The coordinated uptime case (+25 pp with the reading moved) and an inventory case are refused.',
     '   - The pressure receipt\'s claim that every reading field was recomputed (`ladder_pressure_20261121/report.md:192`, frozen) was too broad. This receipt supersedes it in band; that receipt is not edited.',
     '2. **Queue labels (finding 2).** They are applied above and in `reentry.csv`: both gids, prices, rungs and times, the same-price and same-rung flags, the activation-clock ahead and the ahead difference.', '',
     '## Validity', '',
     f"- **Checks v2** A-I: {sum(ck['checks'].values())} / {len(ck['checks'])} true. Falsifier: {nf} / {len(fa['cases'])} corruptions refused by their family.",
     f"- **Supersession of the checker:** the pre-run rehearsal on the pilot copy stopped v1 at F. Its `dict(...)` of the frozen helper's defaultdict dropped zero counters (KeyError `all_edge_censored`, `logs/gl_rehearse_checks.log`). v2 (`{CHECKS_V2}`) fixes only that and was committed before it ran.",
     f"- **v2 rehearsal:** {sum(v for k, v in rh['checks'].items() if k != 'B_membership')} / {len(rh['checks']) - 1} checks true; B_membership is false by design.",
     f"- **Mutants and selftest:** mutation suite {mut.group(1)} / {mut.group(2)}. The first run had one escape (`same_price` from rungs); the fixture was strengthened and those logs are kept. Selftest groups 1-5 pass.",
     f"- **Cost:** pilot projection {proj['projected_core_hours']:.3f} core-hours. The round used {round_cpu:.3f} core-hours (every run_light record under logs/) of the 0.50 cap; peak {cnt(ck['details']['I']['peak_mib'])} MiB.", '',
     '## Limits', '',
     '- **One timing margin on consumed days.** This is a single +50 ms margin on the same 19 consumed-day markets. The feature, its edges and the zero-lag result all came from these markets, so no day here is a validation day.',
     '- **Clock model.** The Binance availability clock is the declared Tokyo median-delay model, not per-message receive stamps (review 909c671e). The lag tests a modest margin; it does not measure real feed-tail latency.',
     '- **Replay assumptions.** External flow is unchanged and timing is M50. Values are gross and zero cost; END is a pre-expiry mark. Coverage (19 / 96) and external-flow realism stay open.']
(R / 'report.md').write_text('\n'.join(L) + '\n')
F = (f"| P105 | PRESSURE-LEAD-SENSITIVITY-v1 (the frozen pressure rule with one more 50 ms of signal delay on its 19 markets; zero-lag control {ctl} / {len(CT)} equal to the pressure receipt; "
     f"checks v2 {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). lag1 vs baseline END {cents(v1['END_paired']['total'])} ({v1['END_paired']['answer']}; "
     f"{v1['END_paired']['markets_better']} better, {v1['END_paired']['markets_worse']} worse); per share {v1['mo1s_per_share']['all']['answer']}; lagged selection screen {sel['answer']} (percentile {pc(sel['percentile'])}; "
     f"bids alone {pc(ps['0']['percentile'])}); retained {pc(rt['END']['all'])} of the zero-lag END gain ({pc(rt['per_share']['all'])} per share). | Computed reading: {pr['reading']}. A consumed-day timing-margin "
     f"sensitivity; no lag chosen, no validation, promotion or profitability claim. |\n")
Hh = (f"<!-- LADDER_PRESSURE_LAG_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — PRESSURE-LEAD-SENSITIVITY-v1 RESULTS: {pr['reading'].upper()} (CONSUMED DAYS); CHECKS v2 {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n"
      f"- [Receipt](reviews/ladder_pressure_lag_20261121/report.md); freeze `{FREEZE}`, checks v2 `{CHECKS_V2}` (the rehearsal caught a v1 crash before the full run), results in this commit. Zero-lag control {ctl} / {len(CT)}.\n"
      f"- **lag1 vs baseline:** END {v1['END_paired']['answer']} (" + ', '.join(f'{d} {cents(v)}' for d, v in v1['END_paired']['by_day'].items()) + f"; total {cents(v1['END_paired']['total'])}); per share {v1['mo1s_per_share']['all']['answer']} (bid {v1['mo1s_per_share']['bid']['answer']}, ask {v1['mo1s_per_share']['ask']['answer']}).\n"
      f"- **Retained:** {pc(rt['END']['all'])} of the zero-lag END gain (" + ', '.join(f'{d} {pc(rt["END"][d])}' for d in days) + f"); per share {pc(rt['per_share']['all'])}.\n"
      f"- **Lagged selection screen:** {sel['answer']} (percentile {pc(sel['percentile'])}; bids alone {pc(ps['0']['percentile'])}, asks {pc(ps['1']['percentile'])}).\n"
      f"- **Validity:** checks v2 {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}, mutants {mut.group(1)} / {mut.group(2)}; {round_cpu:.3f} core-hours of 0.50.\n"
      f"- **Review 909c671e findings 1-2 addressed:** uptime / inventory / queue reconciliation with the coordinated uptime refusal; activation-clock and next-same-side labels.\n"
      f"- **Watch:** consumed-day evidence; the Binance clock is the declared median-delay model; validation needs untouched days (a USER decision).\n"
      f"- **Next:** the ONE combined review, which names the next step.\n<!-- /LADDER_PRESSURE_LAG_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_pressure_lag_results_20261121_dev:\n  status: done_timing_sensitivity\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_pressure_lag_20261121/report.md\n  freeze: {FREEZE}\n  checks: {CHECKS_V2}\n"
      f"  zero_lag_control_equal: {ctl}/{len(CT)}\n  lag1_END_paired: {v1['END_paired']['answer']}\n  lag1_END_delta_cents: {float(v1['END_paired']['total']):.1f}\n"
      f"  lag1_mo1s_per_share: {v1['mo1s_per_share']['all']['answer']}\n  lagged_selection: {sel['answer']}\n  retained_END: {float(rt['END']['all']):.3f}\n  reading: {pr['reading']}\n"
      f"  evidence: consumed_days_development_only\n  checks_all_true: {str(ck['all_ok']).lower()}\n  cost_core_hours: {round_cpu:.3f}\n  review_findings_909c671e: addressed_1_2\n  next: one_combined_review_names_the_next_step\n")
for n, t in (('F_lag.md', F), ('H_lag.md', Hh), ('S_lag.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'items')
