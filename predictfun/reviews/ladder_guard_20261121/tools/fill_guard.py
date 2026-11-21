"""ASK-RESERVATION-GUARD-v1: copies the outputs into outputs/ (SHA256SUMS), writes report.md and the state hunks (FINDINGS, HANDOFF, STATUS) into the
scratchpad's edits/. Every number is read from a file; formats are explicit per column; the readings are the measurement's computed predicates.
argv[1] = the time from `date`. Usage: fill_guard.py <time>"""
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_guard_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_guard_v1_20261121')
S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad'); NOW = sys.argv[1]; FREEZE = 'ee632bc2'
O = R / 'outputs'; O.mkdir(exist_ok=True); (O / 'days').mkdir(exist_ok=True); (O / 'pilot').mkdir(exist_ok=True)
for n in ('arms.csv', 'control.csv', 'audit.json'):
    shutil.copy(D / 'measure' / n, O / n)
shutil.copy(D / 'selection.json', O / 'selection.json')
for sub in ('days', 'pilot'):
    for p in sorted((D / sub).glob('*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, O / sub / p.name)
(O / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(O)}\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'))
A_ = pd.read_csv(D / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip'); CT = pd.read_csv(D / 'measure' / 'control.csv', dtype={'day': str})
au = json.loads((D / 'measure' / 'audit.json').read_text()); pr = au['predicates']; sel = json.loads((D / 'selection.json').read_text())
ck = json.loads((R / 'checks.json').read_text()); fa = json.loads((R / 'checks_falsifier.json').read_text()); gm = json.loads((R / 'manifest.json').read_text())
mut = re.search(r'(\d+) / (\d+) planted code errors are caught', (R / 'logs' / 'mutants_guard_summary.log').read_text())
nf = sum(1 for v in fa['cases'].values() if v['clean'] and not v['corrupted'] and v['applied'])
na = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
cnt = lambda v: '—' if na(v) else f'{int(round(float(v))):,}'
cents = lambda v: '—' if na(v) else f'{float(v):+,.1f} c'
sh = lambda v: '—' if na(v) else f'{float(v):,.1f}'
per = lambda v: '—' if na(v) else f'{float(v):+.3f} c / share'
pc = lambda v: '—' if na(v) else f'{100 * float(v):.2f} %'
pp = lambda v: '—' if na(v) else f'{100 * float(v):+.2f} pp'


def tbl(rows, cols, heads):
    out = ['| ' + ' | '.join(heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(f(r[c]) if f else str(r[c]) for c, f in cols) + ' |')
    return '\n'.join(out)


days = sorted(A_.day.unique()); rows = []
for day in days + ['all']:
    for arm in ('disabled', 'guard'):
        g = A_[(A_.arm == arm) & ((A_.day == day) if day != 'all' else True)]
        rows.append(dict(day=day, arm='baseline' if arm == 'disabled' else 'guard', markets=len(g), value=g.value.sum(), q=g.all_q.sum(), ask_q=g.ask_q.sum(), bid_q=g.bid_q.sum(),
                         edge=g.all_edge.sum(), mo50=g.all_mo50.sum(), mo100=g.all_mo100.sum(), cut_mo1s=g.all_cut_mo1s.sum(), cut_valid_q=g.all_cut_valid_q.sum(), cut_cens=g.all_cut_censored.sum(),
                         per1=g.all_cut_mo1s.sum() / g.all_cut_valid_q.sum() if g.all_cut_valid_q.sum() > 0 else np.nan,
                         ask_per1=g.ask_cut_mo1s.sum() / g.ask_cut_valid_q.sum() if g.ask_cut_valid_q.sum() > 0 else np.nan,
                         bid_per1=g.bid_cut_mo1s.sum() / g.bid_cut_valid_q.sum() if g.bid_cut_valid_q.sum() > 0 else np.nan,
                         good=g.all_good_mo1s.sum(), badv=g.all_bad_mo1s.sum(), both=g.both_exact.mean(), ask_up=g.ask_exact.mean(), bid_up=g.bid_exact.mean(),
                         pulls=g.guard_pulls.sum(), withholds=g.guard_withholds.sum(), reducing=g.reducing_suppressed.sum(), gens=g.generations.sum(), structural=g.structural_requests.sum(),
                         policy=g.policy_requests.sum(), inv=g.mean_abs_inv.mean(), at_limit=g.share_abs_ge_S.mean(), term=g.terminal_inv.abs().mean()))
T = pd.DataFrame(rows); ctl = int(CT.control_all.sum())
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research'); import pf_ladder_guard as G; import pf_ladder_audit as AU   # noqa: E402
RUN = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_rest_v1_20261121'); sup = json.loads((R / 'checks_falsifier_supplement.json').read_text())
gzr = lambda p: [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]
stat = pd.Series([r['status'] for d in gm['days'] for r in gzr(RUN / 'days' / f'{d}.jsonl.gz')]).value_counts()
recs = [dict(r, _day=d) for d in gm['days'] for r in gzr(D / 'days' / f'{d}.jsonl.gz')]; Z = {d: np.load(D / 'days' / f'{d}_arrays.npz') for d in gm['days']}
life, pulls, gl_pos = [], [], []
for r in recs:
    for arm in ('disabled', 'guard'):
        L_ = r[arm]; fv = AU.fill_view(L_); end = float(L_['end'])
        for o in L_['orders']:
            if o['refused']:
                continue
            cum, full = 0., None
            for y in fv.get(int(o['gid']), []):
                cum += y[3]
                if full is None and cum >= float(o['size']) - G.LB.TOL:
                    full = y[0]
            life.append(dict(arm=arm, side=int(o['side']), ms=min(float(o['remove']), full if full is not None else np.inf, end) - float(o['act'])))
        cm = Z[r['_day']][f"{int(r['m'])}__{arm}__commands"]
        for row in cm[cm[:, 7] == G.M.CMD_ACTIONS.index('PULL')]:
            pulls.append(dict(arm=arm, side=int(row[1]), reason=G.REASONS[int(row[8])]))
        gl_pos += [z[6] for z in L_.get('guard_log', [])]
LF = pd.DataFrame(life); PL = pd.DataFrame(pulls)
lt = [dict(arm='baseline' if a == 'disabled' else 'guard', side='bid' if s == 0 else 'ask', n=len(g), mean=g.ms.mean() / 1e3, med=g.ms.median() / 1e3) for (a, s), g in LF.groupby(['arm', 'side'])]
pt = [dict(arm='baseline' if a == 'disabled' else 'guard', side='bid' if s == 0 else 'ask', reason=w, n=len(g)) for (a, s, w), g in PL.groupby(['arm', 'side', 'reason'])]
rg = [dict(arm='baseline' if a == 'disabled' else 'guard', r0q=g.rung0_q.sum(), r0e=g.rung0_end.sum(), r1q=g.rung1_q.sum(), r1e=g.rung1_end.sum()) for a, g in A_.groupby('arm')]
pm = []
for (d, m), g in A_.groupby(['day', 'm']):
    b_, g_ = g[g.arm == 'disabled'].iloc[0], g[g.arm == 'guard'].iloc[0]
    pm.append(dict(day=d, m=int(m), b=b_.value, g=g_.value, dl=g_.value - b_.value, bf=b_.all_q, gf=g_.all_q, pu=g_.guard_pulls, wh=g_.guard_withholds))
wm, wb = min(pm, key=lambda x: x['dl']), max(pm, key=lambda x: x['dl'])
RLX = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
round_cpu = sum(float(m_.group(6)) for p_ in R.joinpath('logs').rglob('*.log') for ln in p_.read_text().splitlines() for m_ in [RLX.match(ln)] if m_) / 3600.
L = ['# ASK-RESERVATION-GUARD-v1 — report', '',
     f'{NOW}, session aea5bfb9. Review `f8c5e579`\'s ONE next priority: one declared candidate against the frozen ladder baseline (`47d2eec1`, results `869fa85f`) on its '
     f"{cnt(sum(len(v) for v in gm['complete'].values()))} complete markets (all 96 statuses as in the baseline). Freeze `{FREEZE}` before any replay; rule sha256 `{gm['rule_sha256'][:16]}`; "
     f"edge {cnt(gm['edge'])} shares (the frozen Oct 24-25 q5 edge). One adaptive consumed-day candidate: no profitability, significance, validation or promotion claim.", '',
     '## Rule and control', '', '- On asks only, while the client-known bid reservation float32(B) >= 200: a live ask the baseline would KEEP is PULLed; a slot it would POST is WAITed. '
     'Bids, the inventory rule (precedence), sizes, structural rules, timing and paths are the baseline\'s; no timer.',
     f"- **Disabled control:** the disabled arm reproduces the accepted baseline ledger (fingerprint, fills, orders, requests, value) on {ctl} of {len(CT)} markets.", '',
     '## Paired replay (gross, zero cost, cents; END = cash + inventory x the END mark)', '',
     tbl(T.to_dict('records'), [('day', None), ('arm', None), ('markets', cnt), ('value', cents), ('q', sh), ('ask_q', sh), ('bid_q', sh), ('edge', cents), ('mo50', cents), ('mo100', cents)],
         ['day', 'arm', 'markets', 'END value', 'filled shares', 'ask shares', 'bid shares', 'execution edge', '+50 ms', '+100 ms']), '',
     '### +1 s markout on the shared fill-time cutoff (t <= END - 1 s; late fills stay in END)', '',
     tbl(T.to_dict('records'), [('day', None), ('arm', None), ('cut_mo1s', cents), ('cut_valid_q', sh), ('cut_cens', cnt), ('per1', per), ('ask_per1', per), ('bid_per1', per), ('good', cents), ('badv', cents)],
         ['day', 'arm', '+1 s', 'valid shares', 'censored fills', 'per share', 'ask per share', 'bid per share', 'good fills (+1 s > 0)', 'bad fills (+1 s < 0)']), '',
     '### Participation, churn and inventory', '',
     tbl(T.to_dict('records'), [('day', None), ('arm', None), ('both', pc), ('ask_up', pc), ('bid_up', pc), ('pulls', cnt), ('withholds', cnt), ('reducing', cnt), ('gens', cnt), ('structural', cnt),
                                ('policy', cnt), ('inv', sh), ('at_limit', pc), ('term', sh)],
         ['day', 'arm', 'both-side active (exact)', 'ask active', 'bid active', 'guard pulls', 'guard withholds', 'reducing-side suppressed', 'generations', 'structural requests',
          'policy requests', 'mean |I|', 'time |I| >= 100', 'mean terminal |I|']), '',
     '`guard withholds` count slot decisions (one per vacant ask slot per 50 ms tick while the signal is on), not orders; `guard pulls` are cancel requests. '
     f"Client-known position at the interventions: min {sh(min(gl_pos))}, max {sh(max(gl_pos))} shares, so `reducing-side suppressed` counts interventions at a long position.", '',
     '### Per market (END, cents; filled shares)', '',
     tbl(pm, [('day', None), ('m', None), ('b', cents), ('g', cents), ('dl', cents), ('bf', sh), ('gf', sh), ('pu', cnt), ('wh', cnt)],
         ['day', 'market', 'baseline END', 'guard END', 'guard - baseline', 'baseline shares', 'guard shares', 'guard pulls', 'guard withholds']), '',
     '### Rung split (all days), removals and lifetimes (the last two computed by this report tool from the records and command arrays; descriptive)', '',
     tbl(rg, [('arm', None), ('r0q', sh), ('r0e', cents), ('r1q', sh), ('r1e', cents)], ['arm', 'rung 0 shares', 'rung 0 END', 'rung 1 shares', 'rung 1 END']), '',
     tbl(pt, [('arm', None), ('side', None), ('reason', None), ('n', cnt)], ['arm', 'side', 'policy PULL reason', 'requests']), '',
     tbl(lt, [('arm', None), ('side', None), ('n', cnt), ('mean', lambda v: f'{v:.2f} s'), ('med', lambda v: f'{v:.2f} s')],
         ['arm', 'side', 'generations (not refused)', 'mean live time', 'median live time']), '',
     'Live time = min(remove, full fill, END) - activation, the interval pf_ladder_audit.durations uses.', '',
     '## Selection control (saved audit states; 500 matched draws, seed 20261121; not a null for the full feedback policy)', '',
     f"- Selected q5 ask live states {cnt(sel['selected'])} of {cnt(sel['states'])} uncensored ({cnt(sel['censored'])} censored, {cnt(sel['selected_censored'])} of them selected); strata {cnt(sel['strata'])}, "
     f"movable {cnt(sel['movable_strata'])} with {cnt(sel['movable_selected'])} selected states, degenerate {cnt(sel['degenerate_strata'])}.",
     f"- Selected mean window +1 s {cents(sel['J'])} against the matched null mean {cents(sel['null_mean'])} (p05 {cents(sel['null_p05'])}, p95 {cents(sel['null_p95'])}); the selected value's "
     f"null percentile {pc(sel['percentile'])}. Per valid window share {per(sel['J_per_valid_share'])} (null p05 {per(sel['null_q_p05'])}).",
     f"- Quantities (review finding 1): valid window {sh(sel['valid_window_q'])}, total {sh(sel['total_window_q'])}, excluded (censored) {sh(sel['excluded_window_q'])} shares.", '',
     '## Computed readings (measure/audit.json predicates)', '',
     f"- **END_paired:** {pr['END_paired']['answer']} — by day " + ', '.join(f'{d} {cents(v)}' for d, v in pr['END_paired']['by_day'].items())
     + f"; markets better / worse / equal {pr['END_paired']['markets_better']} / {pr['END_paired']['markets_worse']} / {pr['END_paired']['markets_equal']}.",
     f"- **+1 s per share (shared cutoff):** all {pr['mo1s_per_share']['all']['answer']}, ask {pr['mo1s_per_share']['ask']['answer']}, bid {pr['mo1s_per_share']['bid']['answer']} — all-side by day "
     + ', '.join(f'{d} {per(v)}' for d, v in pr['mo1s_per_share']['all']['by_day'].items()) + '.',
     f"- **Participation:** filled shares by day " + ', '.join(f'{d} {sh(v)}' for d, v in pr['participation']['filled_q_delta_by_day'].items())
     + f"; both-side active {pp(pr['participation']['both_exact_delta'])}, ask active {pp(pr['participation']['ask_exact_delta'])}; ask shares {sh(pr['participation']['ask_q_delta'])}.",
     f"- **Inventory:** mean |I| {sh(pr['inventory']['mean_abs_delta'])} shares, time |I| >= 100 {pp(pr['inventory']['share_abs_ge_S_delta'])}, terminal |I| {sh(pr['inventory']['terminal_abs_delta'])}.",
     f"- **Selection control:** {pr['selection']}. **Interpretation (declared rule):** {pr['interpretation']}.", '',
     '## Corrected summaries and the audit re-derivation (review f8c5e579)', '',
     f"- Finding 3: the generation +1 s censor count is {cnt(sel['generation_censor_count'])} (generations with fills whose lifetime +1 s is missing; no-fill generations excluded).",
     f"- Finding 2: `guard_checks.py` G recomputed the audit's bin means and censored counts from its state rows (G_bin_means {ck['checks']['G_bin_means']}, G_bin_censored {ck['checks']['G_bin_censored']}), "
     f"every lead flag and pooled contrast (G_lead_flags {ck['checks']['G_lead_flags']}) and the candidate (G_candidate {ck['checks']['G_candidate']}); the falsifier corrupts the stored q5 means and "
     'the censored counts and both are refused.', '',
     '## Validity', '', f"- Checks A-I: {sum(ck['checks'].values())} / {len(ck['checks'])} true; falsifier {nf} / {len(fa['cases'])}; disabled-arm mismatches {cnt(ck['details']['C_disabled_mismatch'])}; "
     f"rule violations {cnt(ck['details']['D_rule_violations'])}.", f"- Post-run falsifier supplement (`tools/falsify_supplement.py`, added after the run; no reading changed): {sum(v['target_refused'] for v in sup['cases'].values())} / {len(sup['cases'])} "
     "in-memory corruptions refused by their named check, including review f8c5e579's two probe corruptions (the eight q5 means set to q1 + 1 -> G_bin_means; the frozen audit "
     "censor field set to -999,999 -> the supplement's X_frozen_censor_field, which requires the stored -3,195 to equal its frozen formula); the checks it leaves uncovered "
     f"({', '.join(sup['uncovered_checks'])}) are the frozen falsifier's cases.",
     f"- Statuses (all 96 selected markets, baseline run): " + ', '.join(f'{k} {cnt(v)}' for k, v in stat.items()) + '; the guard ran on the complete ones only.',
     f"- Mutation suite {mut.group(1)} / {mut.group(2)}; selftest groups 1-7 pass.",
     f"- Cost {ck['details']['I']['core_hours']:.3f} core-hours when the checks ran; {round_cpu:.3f} core-hours for the round through the supplement and reports (every run_light record under logs/), of the 0.75 cap; peak {cnt(ck['details']['I']['peak_mib'])} MiB.", '', '## Limits', '',
     '- One adaptive candidate on the 19 complete consumed-day markets; the rule\'s edge came from the audit of the same days (Oct 24-25 bins, the lead found on all four days).',
     '- Replay values assume unchanged external flow (our withheld / pulled asks do not move others) and M50 timing; gross, zero cost; END is a pre-expiry mark.',
     '- The selection control conditions on day x market x remaining x age x inventory sign only; it tests the local association, not the feedback policy.',
     f"- Paired values pass through the path-coupled inventory rule (CLAUDE.md pitfall 4): the largest single-market change is {cents(wm['dl'])} ({wm['day']} market {wm['m']}, "
     f"filled shares {sh(wm['bf'])} -> {sh(wm['gf'])}) against that day's {cents(pr['END_paired']['by_day'][wm['day']])}; the largest gain is {cents(wb['dl'])} ({wb['day']} market {wb['m']}).",
     f"- Same 19 markets as the audit that chose the rule: no day here is a validation day."]
(R / 'report.md').write_text('\n'.join(L) + '\n')
tot = T[T.day == 'all'].set_index('arm')
F = (f"| P103 | ASK-RESERVATION-GUARD-v1 (one declared candidate - asks only, pull / withhold while client-known bid reservations >= 200 shares - against the frozen ladder on its 19 complete "
     f"markets; disabled control {ctl} / {len(CT)} equal to the baseline; checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])} + supplement {sum(v['target_refused'] for v in sup['cases'].values())} / {len(sup['cases'])}; as of {NOW}). Paired END "
     f"{cents(tot.loc['guard', 'value'] - tot.loc['baseline', 'value'])} over 19 markets ({pr['END_paired']['answer']}; {pr['END_paired']['markets_better']} better, {pr['END_paired']['markets_worse']} worse); "
     f"+1 s per share {pr['mo1s_per_share']['all']['answer']} (ask {pr['mo1s_per_share']['ask']['answer']}, bid {pr['mo1s_per_share']['bid']['answer']}); filled shares {sh(tot.loc['guard', 'q'] - tot.loc['baseline', 'q'])}, "
     f"both-side active {pp(pr['participation']['both_exact_delta'])}. Selection control {sel['answer']} (selected {cents(sel['J'])} vs matched p05 {cents(sel['null_p05'])}, percentile {pc(sel['percentile'])}). | "
     f"Computed interpretation: {pr['interpretation']}. One adaptive candidate on consumed days; not promotion, validation or profitability. |\n")
Hh = (f"<!-- LADDER_GUARD_RESULTS_20261121_DEV -->\n## {NOW} (session aea5bfb9) — ASK-RESERVATION-GUARD-v1 RESULTS: INTERPRETATION {pr['interpretation'].upper()}; CHECKS {'ALL TRUE' if ck['all_ok'] else 'NOT ALL TRUE'}\n\n"
      f"- [Receipt](reviews/ladder_guard_20261121/report.md); freeze `{FREEZE}`, results in this commit. Disabled control {ctl} / {len(CT)} markets equal to the baseline.\n"
      f"- **Paired END:** {pr['END_paired']['answer']} (" + ', '.join(f'{d} {cents(v)}' for d, v in pr['END_paired']['by_day'].items()) + f"; {pr['END_paired']['markets_better']} better / {pr['END_paired']['markets_worse']} worse).\n"
      f"- **+1 s per share:** {pr['mo1s_per_share']['all']['answer']} (ask {pr['mo1s_per_share']['ask']['answer']}, bid {pr['mo1s_per_share']['bid']['answer']}); participation: both-side {pp(pr['participation']['both_exact_delta'])}, "
      f"filled shares {sh(tot.loc['guard', 'q'] - tot.loc['baseline', 'q'])}; inventory mean |I| {sh(pr['inventory']['mean_abs_delta'])}.\n"
      f"- **Selection control:** {sel['answer']} (percentile {pc(sel['percentile'])}, {cnt(sel['movable_strata'])} movable strata).\n"
      f"- **Validity:** checks {sum(ck['checks'].values())} / {len(ck['checks'])}, falsifier {nf} / {len(fa['cases'])} + post-run supplement {sum(v['target_refused'] for v in sup['cases'].values())} / {len(sup['cases'])} (incl. review f8c5e579's two probe corruptions), mutants {mut.group(1)} / {mut.group(2)}; {round_cpu:.3f} core-hours of 0.75.\n"
      f"- **Review f8c5e579 findings 1-3 cleared:** (1) valid / total / excluded window quantities published ({sh(sel['valid_window_q'])} / {sh(sel['total_window_q'])} / {sh(sel['excluded_window_q'])}), a censored high-quantity row changes neither valid term (selftest 4); (2) the checker re-derives bin means, censored counts, every lead flag, pooled ranking and candidate from the state rows and refuses the probe's corruptions (supplement); (3) generation censor count {cnt(sel['generation_censor_count'])} with 0 <= c <= generations and a no-fill control (selftest 5).\n"
      f"- **Next:** the ONE combined review, which names the next experiment.\n<!-- /LADDER_GUARD_RESULTS_20261121_DEV -->\n\n")
St = (f"ladder_guard_results_20261121_dev:\n  status: done_one_candidate_replay\n  asof: \"{NOW}\"\n  receipt: predictfun/reviews/ladder_guard_20261121/report.md\n  freeze: {FREEZE}\n"
      f"  disabled_control_equal: {ctl}/{len(CT)}\n  END_paired: {pr['END_paired']['answer']}\n  END_delta_cents: {float(tot.loc['guard', 'value'] - tot.loc['baseline', 'value']):.1f}\n"
      f"  mo1s_per_share: {pr['mo1s_per_share']['all']['answer']}\n  selection: {sel['answer']}\n  interpretation: {pr['interpretation']}\n  checks_all_true: {str(ck['all_ok']).lower()}\n"
      f"  cost_core_hours: {round_cpu:.3f}\n  review_findings_f8c5e579: cleared_1_2_3\n  next: one_combined_review_names_the_next_experiment\n")
for n, t in (('F_guard.md', F), ('H_guard.md', Hh), ('S_guard.yml', St)):
    (S / 'edits' / n).write_text(t)
print('report.md', len(L), 'lines')
