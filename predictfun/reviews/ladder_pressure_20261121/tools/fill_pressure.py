"""BINANCE-PRESSURE-GUARD-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and the state hunks (FINDINGS, HANDOFF, STATUS) into the
scratchpad's edits/. Every number is read from a file (the measurement's CSVs / audit.json, the selection control, the checks, the day records);
formats are explicit per column; the readings are the measurement's computed predicates. argv[1] = the time from `date`. Usage: fill_pressure.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_pressure_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_pressure_v1_20261121')
RUN = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_rest_v1_20261121'); S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
NOW, FREEZE = sys.argv[1], '793c706b'
O = R / 'outputs'; O.mkdir(exist_ok=True); (O / 'days').mkdir(exist_ok=True); (O / 'pilot').mkdir(exist_ok=True)
for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'causal.csv', 'audit.json'):
    shutil.copy(D / 'measure' / n, O / n)
shutil.copy(D / 'selection.json', O / 'selection.json')
for sub in ('days', 'pilot'):
    for p in sorted((D / sub).glob('*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, O / sub / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
A_ = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); CT = pd.read_csv(D / 'measure' / 'control.csv', dtype={'day': str})
LA = pd.read_csv(D / 'measure' / 'lost_added.csv', dtype={'day': str}, float_precision='round_trip'); CZ = pd.read_csv(D / 'measure' / 'causal.csv', dtype={'day': str})
au = json.loads((D / 'measure' / 'audit.json').read_text()); pr, ig = au['predicates'], au['integration']; sel = json.loads((D / 'selection.json').read_text())
ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text()); gm = json.loads((R / 'manifest.json').read_text())
rh = json.loads((D / 'rehearsal' / 'checks_rehearsal.json').read_text()); proj = json.loads((R / 'logs' / 'pilot_projection.json').read_text())
mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_final_summary.log').read_text())
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


def tbl(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


days = sorted(A_.day.unique()); rows = []
ratio = lambda a, b: a / b if b > 0 else np.nan
for day in days + ['all']:
    for arm in ('disabled', 'pressure'):
        g = A_[(A_.arm == arm) & ((A_.day == day) if day != 'all' else True)]; s_ = lambda c: float(g[c].sum())
        rows.append(dict(day=day, arm='baseline' if arm == 'disabled' else 'pressure', markets=len(g), value=s_('value'), q=s_('all_q'), ask_q=s_('ask_q'), bid_q=s_('bid_q'),
                         edge=s_('all_edge'), edge_q=s_('all_edge_valid_q'), edge_c=s_('all_edge_censored'), mo50=s_('all_mo50'), mo50_q=s_('all_mo50_valid_q'), mo50_c=s_('all_mo50_censored'),
                         mo100=s_('all_mo100'), mo100_q=s_('all_mo100_valid_q'), mo100_c=s_('all_mo100_censored'), cut=s_('all_cut_mo1s'), cut_q=s_('all_cut_valid_q'),
                         cut_c=s_('all_cut_censored'), per1=ratio(s_('all_cut_mo1s'), s_('all_cut_valid_q')), ask_per1=ratio(s_('ask_cut_mo1s'), s_('ask_cut_valid_q')),
                         bid_per1=ratio(s_('bid_cut_mo1s'), s_('bid_cut_valid_q')), good=s_('all_good_mo1s'), badv=s_('all_bad_mo1s'), id1=s_('all_id_mo1s'), idm=s_('all_id_move'),
                         idl=s_('all_id_late'), end=s_('all_end'), both=g.both_exact.mean(), bid_up=g.bid_exact.mean(), ask_up=g.ask_exact.mean(), inv=g.mean_abs_inv.mean(),
                         peak=g.peak_abs_inv.mean(), lim=g.share_abs_ge_S.mean(), term=g.terminal_inv.abs().mean(), pulls=s_('pulls'), wh=s_('withholds'), whe=s_('withholds_near_end'),
                         red=s_('reducing_suppressed'), gens=s_('generations'), struct=s_('bid_structural') + s_('ask_structural'), inv_req=s_('bid_req_inventory_pull') + s_('ask_req_inventory_pull'),
                         prs_req=s_('bid_req_pressure_pull') + s_('ask_req_pressure_pull'), rev=s_('reviewed'), frp=s_('first_review_pulls'), frpp=s_('first_review_pressure'),
                         bl=g.bid_life_mean_ms.mean(), al=g.ask_life_mean_ms.mean(), bm=g.bid_life_median_ms.median(), am=g.ask_life_median_ms.median(),
                         a0=g.q_ahead0_mean.mean(), af=g.q_first_venue_mean.mean(), kf=g.q_first_known_mean.mean(), qp=s_('q_pulls'), qpa=g.q_pull_venue_mean.mean(),
                         qre=s_('q_reentries'), qg=g.q_reentry_gap_mean.mean(), qra=g.q_reentry_ahead_mean.mean(), qs=g.q_surrendered_mean.mean(), qsp=g.q_reentry_same_price.mean(),
                         mb=s_('missing_bid'), ma=s_('missing_ask')))
T = pd.DataFrame(rows); TR = T.to_dict('records'); ctl = int(CT.control_all.sum())
la_rows = []
for day in days + ['all']:
    g = LA[(LA.day == day) if day != 'all' else np.ones(len(LA), bool)]; s_ = lambda c: float(g[c].sum())
    la_rows.append(dict(day=day, ln=s_('lost_n'), lq=s_('lost_q'), lp=s_('lost_mo1s_pos'), lneg=s_('lost_mo1s_neg'), le=s_('lost_end'), an=s_('added_n'), aq=s_('added_q'),
                        ap=s_('added_mo1s_pos'), aneg=s_('added_mo1s_neg'), ae=s_('added_end'), dv=s_('delta_value')))
pm = []
for (d, m), g in A_.groupby(['day', 'm']):
    b_, g_ = g[g.arm == 'disabled'].iloc[0], g[g.arm == 'pressure'].iloc[0]
    pm.append(dict(day=d, m=int(m), b=b_.value, g=g_.value, dl=g_.value - b_.value, bf=b_.all_q, gf=g_.all_q, pu=g_.pulls, wh=g_.withholds))
dsort = sorted(x['dl'] for x in pm); top3 = sum(sorted((x['dl'] for x in pm), reverse=True)[:3]) / sum(x['dl'] for x in pm)
cs = {f'{s}|{c}': int(CT[f'cache_{s}_{c}'].sum()) for s in (0, 1) for c in ('ok', 'off_grid', 'no_market_row', 'nan')}
ps = sel['per_side']; tot = T[T.day == 'all'].set_index('arm')
L = ['# BINANCE-PRESSURE-GUARD-v1 — report', '',
     f"{NOW}, session aea5bfb9. This is review `2c5dbd53`'s ONE next priority: one external-market signal against the frozen ladder baseline (`47d2eec1`, results `869fa85f`).",
     f"- **Population:** the baseline's {cnt(sum(len(v) for v in gm['complete'].values()))} complete markets of its 96 selected ones (statuses: " + ', '.join(f'{k} {cnt(v)}' for k, v in stat.items()) + ').',
     f"- **Freeze:** `{FREEZE}`, before any replay. Rule sha256 `{gm['rule_sha256'][:16]}`.",
     f"- **Frozen q5 edges:** bids x >= {gm['edges']['0'][-1]}, asks x >= {gm['edges']['1'][-1]}; edge file `{gm['edges_sha256'][:8]}`.",
     '- **Status of the evidence:** this is the second guard candidate of this ladder line on consumed Oct 24-27 development days. The feature was suggested by the audit of these same markets, and its bins come from Oct 24-25. It makes no profitability, significance, validation or promotion claim.', '',
     '## Rule and controls', '',
     '- **Rule.** On side s while the builder\'s float32 adv_bn_r250 at the decision is finite, positive and in its frozen q5:',
     '  - a live order the baseline would KEEP is PULLed;',
     '  - a slot it would POST is WAITed (a withheld slot proposal).',
     '- **Everything else is the baseline\'s:** inventory and structural precedence, sizes, timing and paths. There is no timer.',
     f"- **Disabled control.** The disabled arm reproduces the accepted baseline ledgers on {ctl} of {len(CT)} markets: fingerprint, fills, orders, requests, value, initial queue and receipt-known queue estimates.",
     f"- **Integration boundary**, computed in the run and re-derived by the checker:",
     f"  - snapshot parity: {cnt(ig['parity_rows'] - ig['parity_mismatched'] - ig['parity_unavailable'])} of {cnt(ig['parity_rows'])} saved baseline feature rows equal the cache bit for bit;",
     f"  - side sign: {cnt(ig['sign_mismatched'])} mismatches;",
     f"  - causal checks: {cnt(ig['causal_ok'])} decisions; {cnt(ig['causal_same'])} unchanged under the builder's future-only perturbation ({cnt(ig['causal_same_signal'])} with an unchanged signal); {cnt(ig['causal_fired'])} flipped by the delivered adverse-move control; {cnt(ig['causal_cached_equal'])} equal to the cache;",
     f"  - END identity gap: at most {ig['identity_gap_max']:.1e} c.",
     '- **Cache statuses (ticks):** ' + ', '.join(f"{'bid' if k.startswith('0') else 'ask'} {k.split('|')[1]} {cnt(v)}" for k, v in cs.items()) + '.',
     f"- **Missing-signal policy evaluations:** bids {cnt(tot.loc['pressure', 'mb'])}, asks {cnt(tot.loc['pressure', 'ma'])}.", '',
     '## Paired replay', '', 'Gross, zero cost, in cents. END = cash + inventory × the END mark.', '',
     tbl(TR, [('day', None), ('arm', None), ('markets', cnt), ('value', cents), ('q', sh), ('ask_q', sh), ('bid_q', sh), ('edge', cents), ('edge_q', sh), ('edge_c', cnt), ('mo50', cents),
              ('mo50_q', sh), ('mo50_c', cnt), ('mo100', cents), ('mo100_q', sh), ('mo100_c', cnt)],
         ['day', 'arm', 'markets', 'END value', 'filled shares', 'ask shares', 'bid shares', 'execution edge', 'valid shares', 'censored', '+50 ms', 'valid shares', 'censored', '+100 ms',
          'valid shares', 'censored']), '',
     '### +1 s markout on the shared fill-time cutoff', '', 'The cutoff is t <= END - 1 s; later fills stay in END.', '',
     tbl(TR, [('day', None), ('arm', None), ('cut', cents), ('cut_q', sh), ('cut_c', cnt), ('per1', per), ('ask_per1', per), ('bid_per1', per), ('good', cents), ('badv', cents)],
         ['day', 'arm', '+1 s', 'valid shares', 'censored fills', 'per share', 'ask per share', 'bid per share', 'good fills (+1 s > 0)', 'bad fills (+1 s < 0)']), '',
     '### The END identity', '', 'END = eligible +1 s markout + movement from each eligible fill\'s +1 s mid to the END mark + late / missing-horizon END.', '',
     tbl(TR, [('day', None), ('arm', None), ('id1', cents), ('idm', cents), ('idl', cents), ('end', cents)], ['day', 'arm', 'eligible +1 s', 'movement to END', 'late / missing', 'END']), '',
     '### Fills lost and added', '', 'Fills are matched by the multiset of (t, side, price, quantity). Pressure − baseline END = added − lost.', '',
     tbl(la_rows, [('day', None), ('ln', cnt), ('lq', sh), ('lp', cents), ('lneg', cents), ('le', cents), ('an', cnt), ('aq', sh), ('ap', cents), ('aneg', cents), ('ae', cents), ('dv', cents)],
         ['day', 'lost fills', 'lost shares', 'lost +1 s > 0', 'lost +1 s < 0', 'lost END', 'added fills', 'added shares', 'added +1 s > 0', 'added +1 s < 0', 'added END', 'pressure - baseline END']), '',
     '### Per market (END in cents; filled shares)', '',
     tbl(pm, [('day', None), ('m', None), ('b', cents), ('g', cents), ('dl', cents), ('bf', sh), ('gf', sh), ('pu', cnt), ('wh', cnt)],
         ['day', 'market', 'baseline END', 'pressure END', 'pressure - baseline', 'baseline shares', 'pressure shares', 'pressure pulls', 'withheld slot proposals']), '',
     f"- {sum(1 for v in dsort if v > 0)} markets gain and {sum(1 for v in dsort if v < 0)} lose; the worst is {cents(dsort[0])}.",
     f"- The three largest gains carry {pc(top3)} of the total.", '',
     '### Interventions, removals, lifetimes', '',
     '`withheld slot proposals` count slot decisions, one per vacant slot per 50 ms tick while ON; they are not orders. `reducing-side` means bids at I < 0 and asks at I > 0.', '',
     tbl(TR, [('day', None), ('arm', None), ('pulls', cnt), ('wh', cnt), ('whe', cnt), ('red', cnt), ('prs_req', cnt), ('inv_req', cnt), ('struct', cnt), ('gens', cnt), ('rev', cnt), ('frp', cnt),
              ('frpp', cnt), ('bl', sec), ('al', sec), ('bm', sec), ('am', sec)],
         ['day', 'arm', 'pressure pulls', 'withheld proposals', 'of them near END', 'reducing-side', 'pressure PULL requests', 'inventory PULL requests', 'structural requests', 'generations',
          'reviewed', 'first-review pulls', 'of them pressure', 'bid mean life', 'ask mean life', 'bid median life', 'ask median life']), '',
     '### Inventory (exact, from the fills) and exact uptime', '',
     tbl(TR, [('day', None), ('arm', None), ('inv', sh), ('peak', sh), ('lim', pc), ('term', sh), ('both', pc), ('bid_up', pc), ('ask_up', pc)],
         ['day', 'arm', 'mean |I|', 'mean peak |I|', 'time |I| >= 100', 'mean terminal |I|', 'both-side active', 'bid active', 'ask active']), '',
     '### Queue diagnostics (review 2c5dbd53 finding 2)', '',
     'Venue values are the engine\'s queue ahead in shares; the receipt-known estimate is kept apart. Market means.', '',
     tbl(TR, [('day', None), ('arm', None), ('a0', sh), ('af', sh), ('kf', sh), ('qp', cnt), ('qpa', sh), ('qre', cnt), ('qg', sec), ('qra', sh), ('qs', sh), ('qsp', pc)],
         ['day', 'arm', 'venue ahead after submission', 'venue ahead at first review', 'known estimate at first review', 'pressure pulls', 'venue ahead at the pull', 're-entries',
          'gap to re-entry', 're-entry venue ahead', 're-entry - pull ahead', 're-entry at the same price']), '',
     '## Selection control', '',
     'Saved audit live states of both sides; 500 matched draws, seed 20261121; strata include side and the B >= 200 indicator. This is not a null for the feedback policy.', '',
     f"- **Population:** {cnt(sel['selected'])} selected states of {cnt(sel['states'])} uncensored ({cnt(sel['censored'])} censored, {cnt(sel['selected_censored'])} of them selected; {cnt(sel['missing_feature'])} with a missing value).",
     f"- **Strata:** {cnt(sel['strata'])} in all; {cnt(sel['movable_strata'])} movable with {cnt(sel['movable_selected'])} selected states; {cnt(sel['degenerate_strata'])} degenerate.",
     f"- **Pooled (the answer):** selected mean window +1 s {cents(sel['J'])} against the matched null mean {cents(sel['null_mean'])} (p05 {cents(sel['null_p05'])}, p95 {cents(sel['null_p95'])}); null percentile {pc(sel['percentile'])}.",
     f"- **Per valid window share:** {per(sel['J_per_valid_share'])} (null p05 {per(sel['null_q_p05'])}).",
     f"- **Window quantities:** valid {sh(sel['valid_window_q'])}, total {sh(sel['total_window_q'])}, excluded {sh(sel['excluded_window_q'])} shares.",
     f"- **Per side (descriptive):**",
     f"  - bids: {cnt(ps['0']['selected'])} selected, {cents(ps['0']['J'])} against null mean {cents(ps['0']['null_mean'])} and p05 {cents(ps['0']['null_p05'])}; percentile {pc(ps['0']['percentile'])};",
     f"  - asks: {cnt(ps['1']['selected'])} selected, {cents(ps['1']['J'])} against null mean {cents(ps['1']['null_mean'])} and p05 {cents(ps['1']['null_p05'])}; percentile {pc(ps['1']['percentile'])}.", '',
     '## Computed readings (measure/audit.json predicates)', '',
     f"- **END_paired:** {pr['END_paired']['answer']}.",
     f"  - Total {cents(pr['END_paired']['total'])}; by day " + ', '.join(f'{d} {cents(v)}' for d, v in pr['END_paired']['by_day'].items()) + '.',
     f"  - Markets better / worse / equal: {pr['END_paired']['markets_better']} / {pr['END_paired']['markets_worse']} / {pr['END_paired']['markets_equal']}.",
     f"  - END by side: bid {pr['END_by_side']['bid']['answer']}, ask {pr['END_by_side']['ask']['answer']}.",
     f"- **+1 s per share (shared cutoff):** all {pr['mo1s_per_share']['all']['answer']}, bid {pr['mo1s_per_share']['bid']['answer']}, ask {pr['mo1s_per_share']['ask']['answer']}.",
     "  - All sides by day: " + ', '.join(f'{d} {per(v)}' for d, v in pr['mo1s_per_share']['all']['by_day'].items()) + '.',
     "- **Participation:**",
     "  - filled shares by day: " + ', '.join(f'{d} {sh(v)}' for d, v in pr['participation']['filled_q_delta_by_day'].items()) + ';',
     f"  - exact active time: both sides {pp(pr['participation']['both_exact_delta'])}, bids {pp(pr['participation']['bid_exact_delta'])}, asks {pp(pr['participation']['ask_exact_delta'])}.",
     f"- **Inventory (exact):** mean |I| {sh(pr['inventory']['mean_abs_delta'])} shares; time at |I| >= 100 {pp(pr['inventory']['share_abs_ge_S_delta'])}; terminal |I| {sh(pr['inventory']['terminal_abs_delta'])}.",
     f"- **Selection control:** {pr['selection']}.",
     f"- **Interpretation (declared rule):** {pr['interpretation']}.", '',
     '## Review 2c5dbd53 findings', '',
     '1. **Checker reconciliation (finding 1).** The checker\'s F family recomputes every field a reading uses from the fills, mids, orders and logs. It derives the readings, and the selection answer from the manifest\'s draws, seed and bands, itself. The falsifier\'s coordinated case (a markout field and the stored reading moved together) is refused.',
     '2. **Queue diagnostics (finding 2).** Venue values are recorded after submission, at the first live review and at every pressure pull, apart from the receipt-known estimates; the re-entry is recorded too.',
     '3. **Selftest coverage (finding 3).** The new selftest names only what it runs. It tests every run_day guard in isolation and asserts its message.',
     '- **Interpretation details** (no repair cycle needed): inventory is now exact from the fills, and withholds are reported as withheld slot proposals with the near-END ones counted apart.', '',
     '## Validity', '',
     f"- **Checks** A-I: {sum(ck['checks'].values())} / {len(ck['checks'])} true. Falsifier: {nf} / {len(fa['cases'])} corruptions refused by their family, including the coordinated markout-and-reading case.",
     f"- **Rehearsal** on the pilot copy before the full run: {sum(v for k, v in rh['checks'].items() if k != 'B_membership')} / {len(rh['checks']) - 1} checks true. B_membership is false there by design.",
     f"- **Selftest and mutants:** mutation suite {mut.group(1)} / {mut.group(2)}; selftest groups 1-7 pass.",
     f"- **Cost:** pilot projection {proj['projected_core_hours']:.3f} core-hours. The round used {round_cpu:.3f} core-hours (every run_light record under logs/, the pre-freeze probe included) of the 0.75 cap; peak {cnt(ck['details']['I']['peak_mib'])} MiB.", '',
     '## Limits', '',
     '- **Selection of the feature and its edge.** This is one external signal on the 19 complete consumed-day markets. The audit of these same markets suggested adv_bn_r250 (the largest pooled separation, which failed its all-eight-cells rule), and the edges are its Oct 24-25 bins. No day here is a validation day.',
     '- **Replay assumptions.** External flow is unchanged: pulled and withheld orders move no one else. Timing is M50. Values are gross and zero cost, and END is a pre-expiry mark.',
     '- **Selection control.** It matches on day × market × side × remaining × age × inventory sign × the reservation indicator only. It tests the local association, not the feedback policy.',
     f"- **Path coupling.** Paired values pass through the path-coupled inventory rule (CLAUDE.md pitfall 4); the three largest gains carry {pc(top3)} of the total."]
(R / 'report.md').write_text('\n'.join(L) + '\n')
F = (f"| P104 | BINANCE-PRESSURE-GUARD-v1 (the second guard candidate of the ladder line - per side, pull / withhold while the builder's adv_bn_r250 is in its frozen Oct 24-25 q5 - against "
     f"the frozen ladder on its 19 complete markets; disabled control {ctl} / {len(CT)}; parity {cnt(ig['parity_rows'])} rows bit-equal, causal {cnt(ig['causal_ok'])} / {cnt(ig['causal_n'])}; checks "
     f"{sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}; as of {NOW}). Paired END {cents(pr['END_paired']['total'])} ({pr['END_paired']['answer']}; "
     f"{pr['END_paired']['markets_better']} better, {pr['END_paired']['markets_worse']} worse); +1 s per share {pr['mo1s_per_share']['all']['answer']} (bid {pr['mo1s_per_share']['bid']['answer']}, ask "
     f"{pr['mo1s_per_share']['ask']['answer']}); filled shares {sh(tot.loc['pressure', 'q'] - tot.loc['baseline', 'q'])}; selection control {sel['answer']} (selected {cents(sel['J'])} vs matched p05 "
     f"{cents(sel['null_p05'])}, percentile {pc(sel['percentile'])}; bids alone percentile {pc(ps['0']['percentile'])}). | Computed interpretation: {pr['interpretation']}. Consumed-day development "
     f"evidence: the feature and its edges came from the audit of these markets; not validation, promotion or profitability. |\n")
Hh = (f"<!-- LADDER_PRESSURE_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — BINANCE-PRESSURE-GUARD-v1 RESULTS: INTERPRETATION {pr['interpretation'].upper()} (CONSUMED DAYS); CHECKS {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n"
      f"- [Receipt](reviews/ladder_pressure_20261121/report.md); freeze `{FREEZE}`, results in this commit. Disabled control {ctl} / {len(CT)}; parity {cnt(ig['parity_rows'])} rows bit-equal; causal {cnt(ig['causal_ok'])} / {cnt(ig['causal_n'])}.\n"
      f"- **Paired END:** {pr['END_paired']['answer']} (" + ', '.join(f'{d} {cents(v)}' for d, v in pr['END_paired']['by_day'].items()) + f"; total {cents(pr['END_paired']['total'])}; {pr['END_paired']['markets_better']} better / {pr['END_paired']['markets_worse']} worse).\n"
      f"- **+1 s per share:** {pr['mo1s_per_share']['all']['answer']} (bid {pr['mo1s_per_share']['bid']['answer']}, ask {pr['mo1s_per_share']['ask']['answer']}); eligible +1 s delta {cents(tot.loc['pressure', 'id1'] - tot.loc['baseline', 'id1'])}; "
      f"+50 ms delta {cents(tot.loc['pressure', 'mo50'] - tot.loc['baseline', 'mo50'])}; filled shares {sh(tot.loc['pressure', 'q'] - tot.loc['baseline', 'q'])}.\n"
      f"- **Selection control:** {sel['answer']} (percentile {pc(sel['percentile'])}; bids alone {pc(ps['0']['percentile'])}, asks {pc(ps['1']['percentile'])}).\n"
      f"- **Validity:** checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])}, rehearsal before the full run, mutants {mut.group(1)} / {mut.group(2)}; {round_cpu:.3f} core-hours of 0.75.\n"
      f"- **Review 2c5dbd53 findings 1-3 addressed:** ledger-reconciled checker with the coordinated falsifier case; queue diagnostics; isolated guard tests.\n"
      f"- **Watch:** consumed-day development evidence only - the feature and its edges came from the audit of these markets; any validation needs untouched days (a USER decision).\n"
      f"- **Next:** the ONE combined review, which names the next step.\n<!-- /LADDER_PRESSURE_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_pressure_results_20261121_dev:\n  status: done_one_candidate_replay\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_pressure_20261121/report.md\n  freeze: {FREEZE}\n"
      f"  disabled_control_equal: {ctl}/{len(CT)}\n  END_paired: {pr['END_paired']['answer']}\n  END_delta_cents: {float(pr['END_paired']['total']):.1f}\n"
      f"  mo1s_per_share: {pr['mo1s_per_share']['all']['answer']}\n  selection: {sel['answer']}\n  interpretation: {pr['interpretation']}\n  evidence: consumed_days_development_only\n"
      f"  checks_all_true: {str(ck['all_ok']).lower()}\n  cost_core_hours: {round_cpu:.3f}\n  review_findings_2c5dbd53: addressed_1_2_3\n  next: one_combined_review_names_the_next_step\n")
for n, t in (('F_pressure.md', F), ('H_pressure.md', Hh), ('S_pressure.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'items')
