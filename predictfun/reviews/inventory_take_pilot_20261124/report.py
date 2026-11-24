import collections,gzip,json,resource,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];A=R/'predictfun/reviews/maker3_size_inventory_20261124';D=R/'data/predict_fun/research/inventory_take_pilot_20261124'
resource.setrlimit(resource.RLIMIT_CPU,(8,9))
S=json.loads((H/'summary.json').read_text());C=json.loads((H/'checks.json').read_text());E=json.loads((H/'execution_checks.json').read_text());assert C['pass_all'] and E['pass_all']
rows={(r['m'],r['arm']):r['metrics'] for r in S['rows']};arms=['PASSIVE_100','TAKE_100','PASSIVE_220','TAKE_220'];markets=[3702960,3789484,3877472]
pooled={}
for arm in arms:
    rs=[rows[m,arm] for m in markets]
    pooled[arm]={k:sum(r[k] for r in rs) for k in ['net_c','fee2_net_c','total_c','completed_c','residual_c','fees_c','maker_shares','taker_shares','take_requests','takes_filled','risk_reducing_takes','passive_generations','passive_pulls']}
    pooled[arm]['mean_abs_inventory']=sum(r['mean_abs_inventory'] for r in rs)/3
comparisons={str(q):dict(delta_net_usd=(pooled[f'TAKE_{q}']['net_c']-pooled[f'PASSIVE_{q}']['net_c'])/100,
    delta_completed_usd=(pooled[f'TAKE_{q}']['completed_c']-pooled[f'PASSIVE_{q}']['completed_c'])/100,
    delta_residual_usd=(pooled[f'TAKE_{q}']['residual_c']-pooled[f'PASSIVE_{q}']['residual_c'])/100,
    lower_mean_inventory=pooled[f'TAKE_{q}']['mean_abs_inventory']<pooled[f'PASSIVE_{q}']['mean_abs_inventory'],
    markets_better=sum(rows[m,f'TAKE_{q}']['net_c']>rows[m,f'PASSIVE_{q}']['net_c'] for m in markets)) for q in [100,220]}
result=dict(pooled=pooled,contrasts=comparisons,all_pooled_negative=all(p['net_c']<0 for p in pooled.values()),
    taking_worse_pooled_both_sizes=all(v['delta_net_usd']<0 for v in comparisons.values()),asof='2026-11-24',validation=False)
result['taking_worse_before_fees_both_sizes']=all(pooled[f'TAKE_{q}']['total_c']<pooled[f'PASSIVE_{q}']['total_c'] for q in [100,220])
assert result['taking_worse_before_fees_both_sizes']
(H/'readings.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Quote, accumulate, actively reduce — bounded pilot','',
'Frozen before replay: `7626e320f`. USER-directed follow-up to maker3 sizing/inventory audit. Same three consumed dashboard markets; opening-only first-batch rule retained throughout. Four complete feedback arms, no threshold tuning after the outputs. Twelve ledgers; the three PASSIVE_100 controls exactly reproduce the accepted OPENING_ONLY fills, orders, commands, requests, cash, position and modes.','',
'100shares/level uses cap200, soft100/release50. 220shares/level uses cap440, soft220/release110. The220 sensitivity scales risk limits too; it is not a pure order-size effect. Both use two levels and the same pressure/structural/entry rules. Maker3 private cap is unknown; these are our experimental risk limits.','',
'Active arm: when abs(client-known inventory)>=soft, submit an IOC toward the release level, capped at one order size and reservation room. Preserve reducing-side resting orders; do not bypass cancel-pending/own-cross protections. Re-evaluate changed state, no fixed cooldown. IOC fills pay saved registry200bps * min(p,1-p), collateral-equivalent fee; passive fee0 as before, no rebates.','',
'| Market | Passive100 | Take100 | Passive220 | Take220 |','|---|---:|---:|---:|---:|']
for m in markets:lines.append('| '+str(m)+' | '+' | '.join(f"${rows[m,a]['net_c']/100:+.2f}" for a in arms)+' |')
lines.append('| **Pooled** | '+' | '.join(f"**${pooled[a]['net_c']/100:+.2f}**" for a in arms)+' |')
lines += ['','Values are cash + remaining inventory at the common298s END mark, minus modeled fees. They are NOT binary settlement P&L, live achievable returns or untouched validation.','',
'| Arm | Completed FIFO turnover, gross USD | Residual, gross USD | Taker fees USD | Mean absolute inventory | Taker shares / filled IOC events |','|---|---:|---:|---:|---:|---:|']
for a in arms:
    p=pooled[a];lines.append(f"| {a} | {p['completed_c']/100:+.2f} | {p['residual_c']/100:+.2f} | {p['fees_c']/100:.2f} | {p['mean_abs_inventory']:.1f} | {p['taker_shares']:.1f} / {p['takes_filled']} |")
lines += ['', '**Computed reading:** '+json.dumps({k:result[k] for k in ['all_pooled_negative','taking_worse_pooled_both_sizes']})+'.', '']
for q,v in comparisons.items():
    lines.append(f"At size{q}, takes change net value by${v['delta_net_usd']:+.2f}: gross completed-turnover change${v['delta_completed_usd']:+.2f}, marked-residual change${v['delta_residual_usd']:+.2f}, less taker fees. Mean inventory falls: {v['lower_mean_inventory']}; {v['markets_better']}/3 markets improve.")
lines += ['', 'Both active variants remain worse before fees too; fees alone do not explain the gap. The exposure reduction changes the subsequent passive fills and closing opportunities. This experiment rejects this simple automatic soft-limit-to-release rule on these selected examples; it does not show that selective maker3-style takes cannot work. The 220 passive result also mixes a higher cap/soft target with size and is still pooled negative. No larger-size or active-take promotion follows.','',
'**Execution reliability and limits.** All124 requests independently reconstructed from hash-bound venue frames/trades. Same10ms arrival,100ms fill/result notice; limit from received best, no worse-price sweep, execution at venue best only if fresh/valid and available. Net-depth budget prevents reusing unchanged displayed liquidity; intervening public trades reduce availability. Positive net depth changes can replenish. This can miss gross replenishment and does not model reactions/market impact: historical external flow remains exogenous. It is a conservative liquidity convention, not a mathematical profit bound or measured exchange execution. Native token fee balances are not modeled; fees are charged as cash-equivalent cost.','',
'Fifteen synthetic control groups cover both directions, latency, partial/empty/refused/self-cross fills, replenishment/public-volume depletion, delayed notices, reserve bounds, hidden-fill risk, unchanged-state retry and FIFO/fees. Independent checking covers all12 ledgers and6 corruptions; separate raw-tape execution checker covers124requests and6 availability corruptions. The first independent checker counted post-only-refused orders as self-cross liquidity; corrected checker and raw refused-order diagnostic are retained (no strategy or replay changed). All actual filled takes reduce absolute venue inventory in these twelve runs; the synthetic hidden-fill case demonstrates this is not guaranteed by a decision based on known I.','',
'**Next priority:** do not tune another inventory threshold from these results. Explain the costly active closures against the passive continuation on the same saved paths, and compare the known-state situations with maker3’s small/selective trims. The missing rule is the economic choice between paying to reduce now and continuing to seek passive offsetting fills. The larger maker3 exposure range alone is not a reason to expand our cap. Keep whole-book turnover/residual and costs separate, and preserve the opening-only scope.','']
lines += ['Double-fee stress, same fills: '+', '.join(f"{a} ${pooled[a]['fee2_net_c']/100:+.2f}" for a in arms if a.startswith('TAKE'))+'. This is an accounting stress, not another feedback run.','']
(H/'report.md').write_text('\n'.join(lines))
# Publish the separate factual audit, including all contextual proxy warnings.
Q=json.loads((A/'summary.json').read_text());ctx=json.loads((A/'context.json').read_text());assert json.loads((A/'checks.json').read_text())['pass_all']
out=['# Final-minute quoting, maker3 sizes and inventory — 2026-11-24','',
'Declared audit freeze `d052f8c1c`; saved data only. Our policy has no final-minute stop: the supported simulation ends298s after open and only disallows arrivals at/after END. The original dashboard has an old inventory-mode bug; corrected default and opening-only traces are the active comparison.','',
'| Market / arm | Live either side in [240,298] | Both sides | New posts | Filled shares |','|---|---:|---:|---:|---:|']
for row in Q['own']:
    if row['window']=='last_minute':out.append(f"| {row['mid']} / {row['arm']} | {row['either_s']:.3f}s | {row['both_s']:.3f}s | {row['posts']} | {row['filled_shares']:.1f} |")
out += ['', 'In3789484, the latest opening-only replay quotes at least one side53.25/58s, both15.058s, submits199orders and fills1,635.349shares in the final interval. Old3702960/3877472 inactivity must not be described as a scheduled final-minute cutoff. Reasons and full-path numbers are in summary.json; WAIT counts are proposal/tick counts, not distinct orders.','',
'**What size can be established:** among61,373 observed filled maker3 hashes over934markets, signed original size is available for only33 selected hashes:16 at10shares and17 at220. The selected sample is not representative of all submitted orders. In3789484 there are115filled hashes, only one signed original size (220); maximum recorded filled quantity per hash220. A filled quantity is a lower bound on original size, not a level-size observation. Simultaneous unfilled orders and total per-level commitments remain unidentified.','',
'**Inventory:** zero-start accepted recorded net flow in3789484 ranges−562.49 to+934.16shares and ends+62.59. Across934markets the median peak absolute flow is132.71, p90 1,556.90, maximum~3,000. Only one market reaches>=2,990; this is insufficient to assert a hard3,000 cap. The daily median peak varies96.18 /107.22 /764.10 /154.52shares. Unknown starting balances, transfers/merges, token fees, invalid/excluded source records and record-time ordering prevent treating this proxy as private wallet inventory or a known cap.','',
'**At active takes in3789484:**','',
'| Upper time | Maker3 recorded net before | Native action / shares | Our default known I | Our opening-only known I |','|---|---:|---|---:|---:|']
for r in Q['takes_vs_us']:out.append(f"| {r['t']:.3f}s | {r['recorded_net_before']:+.1f} | {r['native_action']} / {r['shares']:.3f} | {r['FIXED_DEFAULT_known']:+.1f} | {r['OPENING_ONLY_known']:+.1f} |")
out += ['', 'At82.838s maker3 trims positive exposure while our exposure is negative: blindly copying its sell-equivalent take would add to our short. At202.668–202.705s and248.630s both have positive exposure, but our100-share/200-cap book is much smaller. These are alignments at public record upper times, not a reconstruction of the maker’s private decision inputs.','',
'**Completed follow-on:** [quote-accumulate-reduce pilot](../inventory_take_pilot_20261124/report.md) tests active reduction against matched passive controls at100 and220share sizes. No private cap is imputed. Preserve the experiment’s adverse result; do not read “reduce risk” as “increase economic value”.','',
'Checks: exact interval union independently reconstructed from prior verified chart orders for6arms/12windows, a1second uptime corruption refused; example cumulative-flow extrema independently summed; signed order sizes reconcile to the accepted source and exceed recorded fills. Original9-arm ledger histories remain intact. Day context in context.json is descriptive, not a fitted cap/regime model.','']
(A/'report.md').write_text('\n'.join(out))
print(json.dumps(result,indent=2))
