"""The four-part review's receipt tables, computed from review_<day>.json (pf_pipeline_paired_review.py) and the committed evaluation v2 - no
number in tables.md is typed by hand. Usage: python3 tables.py > tables.md"""
import json
import sys
from pathlib import Path

D = Path('/home/yuqing/ctaNew/data/predict_fun/research/pipeline_paired_review_20261119')
DAYS = ('20261026', '20261027')
J = {d: json.loads((D / f'review_{d}.json').read_text()) for d in DAYS}
B = 'M50|A58|L37|N100'


def f(x, nd=1, sign=True):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x
    return f'{x:+,.{nd}f}' if sign else f'{x:,.{nd}f}'


def pc(x):
    return 'n/a' if x is None else f'{100 * x:.1f}%'


def row(*c):
    return '| ' + ' | '.join(str(x) for x in c) + ' |'


def head(*c, align=None):
    return [row(*c), '|' + '|'.join(align or ['---'] + ['---:'] * (len(c) - 1)) + '|']


out = []
V = {d: J[d]['readings'] for d in DAYS}
# ---- VALIDITY
out += ['### Validity (each predicate computed on every row unless named; a FAIL is an open defect)', '']
out += head('Predicate', *[f'Oct {d[-2:]}' for d in DAYS], align=['---', '---', '---'])
k1, k2, k3, k4, k5 = [next(k for k in V[DAYS[0]] if k.startswith(f'V{i}')) for i in range(1, 6)]
out.append(row('V1 labels = a per-row loop over the engine fills (sampled orders, oid % 25 = 0)',
               *[f"{V[d][k1]['verdict']}: {V[d][k1]['mismatches']:,} of {V[d][k1]['rows_checked']:,} rows" for d in DAYS]))
out.append(row('V2 value identity on every label row (NaN fails)', *[f"{V[d][k2]['verdict']}: {V[d][k2]['failures']:,} of {V[d][k2]['rows']:,}" for d in DAYS]))
out.append(row('V3 timing: export rows off / pi0 steps off / kept fills outside [activation, END]',
               *[f"{V[d][k3]['verdict']}: {V[d][k3]['export_rows_off']:,} of {V[d][k3]['export_rows']:,} / {V[d][k3]['pi0_steps_off']:,} / {V[d][k3]['kept_fills_outside']:,}" for d in DAYS]))
out.append(row('V4 populations and statuses', *[V[d][k4]['verdict'] for d in DAYS]))
out.append(row('V5 accounting: ledger = evaluation at B (1e-6 c), labels = PI0-ALL, null mean, first-review shares, input hashes',
               *[f"{V[d][k5]['verdict']}: {V[d][k5]['mismatches']} of {V[d][k5]['compared']} differ" for d in DAYS]))
out.append(row('Open defects', *[str(V[d]['OPEN DEFECTS (validity)'] or 'none') for d in DAYS]))
out += ['', '**Populations (V4)**', '']
out += head('Count', *[f'Oct {d[-2:]}' for d in DAYS])
P4 = {d: V[d][k4] for d in DAYS}
for lab, key in (('Original opportunities (keep / pull run)', 'original'), ('Export POST rows (one per original)', 'export_post_rows'),
                 ('With a path (eligible + post-only refused)', 'with_path'), ('No path (statuses)', 'no_path'), ('Labelled POST rows', 'labelled_post'),
                 ('Labelled RETAIN rows', 'labelled_retain'), ('Eligible RETAIN export rows', 'eligible_retain'),
                 ('RETAIN steps without a forecast (kept: fallback)', 'fallback_retain_steps'), ('... on A\'s reached path', 'fallback_steps_on_A_reached_path'),
                 ("... at A's first review", 'fallback_steps_at_A_first_review'), ("A's reached rows", 'A_reached_rows'),
                 ('A reached orders not admitted', 'A_reached_orders_not_admitted'), ('Refused POST rows with a nonzero label', 'refused_nonzero_labels')):
    out.append(row(lab, *[f'{P4[d][key]:,}' for d in DAYS]))
out.append(row('Two sources agree: original / with a path', *[f"{P4[d]['original_two_sources']} / {P4[d]['with_path_two_sources']}" for d in DAYS]))
out.append(row('Admitted (A / AF / AM / PI0 / NEU / E1)', *[' / '.join(f"{P4[d]['admitted'][r]:,}" for r in ('A', 'AF', 'AM', 'PI0', 'NEU', 'E1')) for d in DAYS]))
out.append(row('Nonfinite predictions (A / AF / AM / pi0 / NEU)', *[' / '.join(str(P4[d]['fallbacks'][h]) for h in ('A', 'AF', 'AM', 'pi0', 'NEU')) for d in DAYS]))
out.append(row('Keep / pull statuses', *[', '.join(f'{k} {v:,}' for k, v in sorted(P4[d]['statuses'].items())) for d in DAYS]))
out.append(row('Export RETAIN statuses', *[', '.join(f'{k} {v:,}' for k, v in sorted(P4[d]['export_retain_statuses'].items())) for d in DAYS]))
# ---- PREDICTION
out += ['', '### Prediction (A; c per row; eb = episode-balanced; the old gate is the verdict, the plan v2 screen a diagnostic)', '']
GR = ('POST (pred > 0)', 'NO_ENTRY (pred < 0)', 'KEEP reached (pred > 0)', 'PULL reached (pred < 0)', 'KEEP neutral (pred > 0)', 'PULL neutral (pred < 0)')
out += head('Day, side, group', 'Rows', 'Episodes (nonzero)', 'Mean pred / label', 'eb pred / label', 'Sum pred / label', 'Old gate', 'Plan v2 screen',
            align=['---', '---:', '---:', '---:', '---:', '---:', '---', '---'])
for d in DAYS:
    for sd in ('bid', 'ask'):
        g = J[d]['prediction'][f'A {sd}']
        for gn in GR:
            x = g[gn]
            if not x['rows']:
                out.append(row(f'Oct {d[-2:]} {sd} {gn}', 0, '', '', '', '', '', '')); continue
            out.append(row(f'Oct {d[-2:]} {sd} {gn}', f"{x['rows']:,}", f"{x['episodes']:,} ({x['nonzero_label_episodes']:,})", f"{f(x['mean_pred'], 3)} / {f(x['mean_label'], 3)}",
                           f"{f(x['eb_pred'], 3)} / {f(x['eb_label'], 3)}", f"{f(x['sum_pred'], 0)} / {f(x['sum_label'], 0)}", x['old_gate'], x['plan_v2_screen']))
out += ['', '**AF and AM (old gate / plan v2 screen per group)**', '']
out += head('Head, day, side', *GR, align=['---'] + ['---'] * len(GR))
for h in ('AF', 'AM'):
    for d in DAYS:
        for sd in ('bid', 'ask'):
            g = J[d]['prediction'][f'{h} {sd}']
            out.append(row(f'{h} Oct {d[-2:]} {sd}', *[(f"{g[gn].get('old_gate')} / {g[gn].get('plan_v2_screen')}" if g[gn]['rows'] else 'empty') for gn in GR]))
# ---- CONTROLLER
out += ['', '### Controller (the committed evaluation v2 at B; c at END marks, not P&L)', '']
out += head('Policy', *[f'Oct {d[-2:]} {sd}' for d in DAYS for sd in ('both', 'bid', 'ask')])
for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1', 'PA'):
    out.append(row(p_, *[f(J[d]['controller'][f'{p_} {sd}']['value'], 0) for d in DAYS for sd in ('both', 'bid', 'ask')]))
CK = 'CONTROLLER: A above N0 / PI0 / NEU / E1 / A-PA at B (both sides)'
out += ['', head('Reading', *[f'Oct {d[-2:]}' for d in DAYS], align=['---', '---', '---'])[0], '|---|---|---|']
out.append(row('A above N0 / PI0 / NEU / E1 / A-PA at B (both sides)', *[' / '.join(str(x) for x in V[d][CK]) for d in DAYS]))
sk = 'CONTROLLER: scenarios with A above 0 / above PI0'
out.append(row('Scenarios (of 19) with A above 0 / above PI0', *[f"{V[d][sk][0]} / {V[d][sk][1]} of {V[d][sk][2]}" for d in DAYS]))
for k in ('CONTROLLER: the practical predicates vs A-PA', 'CONTROLLER: MATCHED CONTROL WITH ITS SPREAD (draws, one-sided p <= 0.05)'):
    out.append(row(k.replace('CONTROLLER: ', ''), *[str(V[d][k]) for d in DAYS]))
ev = {d: V[d]['CONTROLLER: evaluation readings (carried)'] for d in DAYS}
for k in ev[DAYS[0]]:
    out.append(row(f'(evaluation) {k}', *[str(ev[d].get(k)) for d in DAYS]))
out += ['', '**The matched null of A with its spread (exact moments; 1,000 seeded draws)**', '']
out += head('Day, side', 'A (c)', 'Null mean', 'Null sd', 'z', 'p (normal, approx.)', 'p (draws)', 'Draws agree')
for d in DAYS:
    for sd in ('both', 'bid', 'ask'):
        x = J[d]['matched_null'][sd]
        out.append(row(f'Oct {d[-2:]} {sd}', f(x['value'], 0), f(x['null_mean'], 0), f(x['null_sd'], 0, False), f(x['z'], 2), f(x['p_normal_one_sided'], 4, False),
                       f(x['p_draws_one_sided'], 4, False), x['moments_agree']))
out += ['', '**Quoting (both sides; A vs PI0 / NEU / E1 is not nested)**', '']
out += head('Day, policy', 'Resting', 'At best', 'At best both sides', 'Median / p90 life (ms)', 'Pulled at first review', 'Cancels / 100 filled sh', 'Attempts', 'Filled')
for d in DAYS:
    for p_, x in J[d]['quoting'].items():
        lt = J[d]['controller'][f'{p_} both']['lifetime'] or [None] * 3
        out.append(row(f'Oct {d[-2:]} {p_}', pc(x['resting']), pc(x['at_best']), pc(x['at_best_both']), f"{f(lt[1], 0, False)} / {f(lt[2], 0, False)}", pc(x['first_review_of_live']),
                       f(x['cancels_per_100_filled_shares'], 2, False), f"{x['attempts']:,}", f"{x['filled']:,}"))
qk = next(k for k in V[DAYS[0]] if k.startswith('CONTROLLER: quoting'))
out += ['', row('A vs', *[f'Oct {d[-2:]}: more resting / more both-sides at best / longer median life / fewer first-review pulls / fewer cancels per filled share' for d in DAYS]),
        '|---|---|---|']
for x in ('PI0', 'NEU', 'E1'):
    out.append(row(x, *[' / '.join(str(y) for y in V[d][qk][x]) for d in DAYS]))
fk = 'CONTROLLER: pulled at the first review (share of the orders live then)'
for k in V[DAYS[0]][fk]:
    out.append(row(f'Pulled at the first review: {k}', *[pc(V[d][fk][k]) for d in DAYS]))
# ---- RISK
out += ['', '### Risk (the ledger at B; episodes = opportunities with a path; loss = max(-V, 0); severe = V < -30 c)', '']
out += head('Day, policy', 'Mean loss (c)', 'P(loss)', 'P(severe)', 'Severe episodes', 'p95 / p99 loss, filled', 'p95 / p99 loss, loss episodes', 'Worst 1% share of loss', 'Largest loss')
for d in DAYS:
    for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1', 'PA', 'PI0-ALL'):
        x = J[d]['risk'][p_]['both']
        out.append(row(f'Oct {d[-2:]} {p_}', f(x['mean_loss'], 3, False), pc(x['p_loss']), pc(x['p_severe']), f"{x['severe_episodes']:,}",
                       f"{f(x['p95_loss_filled'], 1, False)} / {f(x['p99_loss_filled'], 1, False)}", f"{f(x['p95_loss_loss'], 1, False)} / {f(x['p99_loss_loss'], 1, False)}",
                       pc(x['worst_1pct_share_of_loss']), f(x['largest_loss'], 1, False)))
out += ['', '**Where A\'s loss and severe episodes arise (both sides)**', '']
out += head('Day, stratum', 'Episodes', 'Total loss (c)', 'Severe episodes', 'Severe excess (c)', 'Value (c)')
for d in DAYS:
    w = J[d]['risk']['A']['where']
    for st in ('side', 'phase', 'spread', 'sigma tercile'):
        for u, x in w[st].items():
            out.append(row(f'Oct {d[-2:]} {st} {u}', f"{x['episodes']:,}", f(x['total_loss'], 0, False), f"{x['severe_episodes']:,}", f(x['severe_excess'], 0, False), f(x['value'], 0)))
out += ['', head('Day, policy', 'Window: before the first review', 'earliest-effect delay', 'after the earliest effect', 'Severe: before', 'Severe: delay', 'Severe: after',
                 'Spread / movement component', 'Time-integrated inventory (share-s)', 'Movement per filled share')[0],
        '|---|' + '---:|' * 9]
for d in DAYS:
    for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1'):
        w = J[d]['risk'][p_]['where']; a_, s_ = list(w['window (all episodes)'].values()), list(w['window (severe episodes)'].values())
        out.append(row(f'Oct {d[-2:]} {p_}', *[f(x, 0) for x in a_], *[f(x, 0) for x in s_], f"{f(w['components']['spread'], 0)} / {f(w['components']['movement'], 0)}",
                       f(w['inventory']['time_integrated_share_s'], 0, False), f(w['inventory']['movement_per_filled_share'], 3)))
out += ['', '**FINDINGS C1 / C3 / C12 (both sides; c)**', '']
out += head('Day, policy', 'Age < 100 ms', '100-500 ms', '>= 500 ms', 'Without young fills', 'B1a-weighted (M0 / M50)', 'B1a sign', 'Cancel-pending fills / value',
            'Pending share of gross', 'WS / E2 / imputed / mixed')
for d in DAYS:
    for p_, y in J[d]['findings_c1_c3_c12'].items():
        x = y['both']; c1 = x['C1']
        out.append(row(f'Oct {d[-2:]} {p_}', *[f(v, 0) for v in c1['by_age'].values()], f(c1['without_young'], 0),
                       ' / '.join(f(v, 0) for v in c1['b1a_weighted'].values()), c1['b1a_sign'], f"{x['C3']['fills']:,} / {f(x['C3']['value'], 0)}", pc(x['C3']['share_of_gross']),
                       ' / '.join(f(x['C12'][k], 0) for k in ('WS', 'E2', 'imputed', 'mixed'))))
out.append('')
for d in DAYS:
    out.append(f"- Oct {d[-2:]}: A minus A-PA by fill age: " + ', '.join(f'{k} {f(v, 0)} c' for k, v in V[d]['FINDINGS C1: A minus A-PA by fill age (both)'].items()))
# ---- computed statements: every sentence of the receipt's verdicts is one of these predicates (descriptive; computed after the run from its outputs)
ST = {}
ag = [J[d]['prediction'][f'A {sd}'][g].get('old_gate') for d in DAYS for sd in ('bid', 'ask') for g in GR if J[d]['prediction'][f'A {sd}'][g]['rows']]
ST["A's old-gate groups failing (of those with rows)"] = f"{sum(x == 'FAIL' for x in ag)} of {len(ag)}"
ST['A POST: mean prediction above 0 and realized label at most +0.02 c per row, every day and side'] = all(
    J[d]['prediction'][f'A {sd}']['POST (pred > 0)']['mean_pred'] > 0 and J[d]['prediction'][f'A {sd}']['POST (pred > 0)']['mean_label'] <= .02 for d in DAYS for sd in ('bid', 'ask'))
ST['Validity PASS (V1-V5) with no open defect, both days'] = all(V[d]['VALIDITY (all five)'] == 'PASS' and not V[d]['OPEN DEFECTS (validity)'] for d in DAYS)
ST['A below N0, PI0, NEU, E1 and A-PA at B (both sides), both days'] = all(not any(V[d][CK]) for d in DAYS)
ST['Matched control with its spread (both sides): Oct 26 / Oct 27'] = ' / '.join(V[d]['CONTROLLER: MATCHED CONTROL WITH ITS SPREAD (draws, one-sided p <= 0.05)']['both'] for d in DAYS)
ST['Evaluation matched control vs the null mean only: Oct 26 / Oct 27'] = ' / '.join(str(ev[d]['MATCHED CONTROL AT B: A above the matched null of its admissions']) for d in DAYS)
ST['A rests longer and is pulled at the first review less often than PI0, NEU and E1, but cancels more per filled share, both days'] = all(
    V[d][qk][x][0] and V[d][qk][x][3] and not V[d][qk][x][4] for d in DAYS for x in ('PI0', 'NEU', 'E1'))
ST['Every policy pulls more than half of its live orders at the first review (median life = the first-review landing), both days'] = all(
    J[d]['quoting'][p_]['first_review_of_live'] > .5 for d in DAYS for p_ in J[d]['quoting'])
ST['Fewer than 1% of A\'s episodes lose (the worst-1% share is then 100% by construction), both days'] = all(J[d]['risk']['A']['both']['p_loss'] < .01 for d in DAYS)
sw = {d: J[d]['risk']['A']['where']['window (severe episodes)'] for d in DAYS}
ST["A's severe-episode loss is largest after the earliest effect (the continuation window), both days"] = all(min(sw[d], key=sw[d].get) == 'after the earliest effect' for d in DAYS)
ph = {d: {u: x['total_loss'] / x['episodes'] for u, x in J[d]['risk']['A']['where']['phase'].items()} for d in DAYS}
ST["A's loss per episode is highest in the market window's last minute (phase 5), both days"] = all(max(ph[d], key=ph[d].get) == '5' for d in DAYS)
ST["A's loss per episode in phase 5 / the other phases' mean (c)"] = ' ; '.join(f"Oct {d[-2:]} {ph[d]['5']:.2f} / {sum(v for u, v in ph[d].items() if u != '5') / 4:.2f}" for d in DAYS)
sp = {d: J[d]['risk']['A']['where']['spread'] for d in DAYS}
ST["Share of A's loss at a spread of at most 1 c"] = ' ; '.join(f"Oct {d[-2:]} {pc(sp[d]['(0, 1]']['total_loss'] / sum(x['total_loss'] for x in sp[d].values()))}" for d in DAYS)
cm = {d: J[d]['risk']['A']['where']['components'] for d in DAYS}
ST["A's spread component positive and movement component negative, both days"] = all(cm[d]['spread'] > 0 and cm[d]['movement'] < 0 for d in DAYS)
tg = {d: {u: x['total_loss'] for u, x in J[d]['risk']['A']['where']['sigma tercile'].items()} for d in DAYS}
ST["The sigma tercile carrying A's largest loss: Oct 26 / Oct 27"] = ' / '.join(max(tg[d], key=tg[d].get) for d in DAYS)
ST["A's time-integrated inventory above PI0's, NEU's and E1's, both days"] = all(
    J[d]['risk']['A']['where']['inventory']['time_integrated_share_s'] > J[d]['risk'][x]['where']['inventory']['time_integrated_share_s'] for d in DAYS for x in ('PI0', 'NEU', 'E1'))
fc_ = {d: J[d]['findings_c1_c3_c12'] for d in DAYS}
ST["Share of A's value from fills under 100 ms old"] = ' ; '.join(f"Oct {d[-2:]} {pc(fc_[d]['A']['both']['C1']['by_age']['lt100'] / sum(fc_[d]['A']['both']['C1']['by_age'].values()))}" for d in DAYS)
ST['A below PI0, NEU and E1 under each B1a age weighting (FINDINGS C1), both days'] = all(
    fc_[d]['A']['both']['C1']['b1a_weighted'][k] < fc_[d][x]['both']['C1']['b1a_weighted'][k] for d in DAYS for x in ('PI0', 'NEU', 'E1') for k in fc_[d]['A']['both']['C1']['b1a_weighted'])
ST['A without fills under 100 ms old below 0, both days'] = all(fc_[d]['A']['both']['C1']['without_young'] < 0 for d in DAYS)
ST["A's E2-timed trade value below its WS-timed, Oct 27 (FINDINGS C12)"] = fc_['20261027']['A']['both']['C12']['E2'] < fc_['20261027']['A']['both']['C12']['WS']
out += ['', '### Computed statements (descriptive; each sentence of the verdicts below is one of these)', '']
out += head('Statement', 'Value', align=['---', '---'])
for k, v in ST.items():
    out.append(row(k, v))
print('\n'.join(out))
sys.exit(0)
