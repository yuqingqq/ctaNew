"""VENUE-MATCHED COMPARISON stage 2: merge the day files, run the declared checks (K1-K9, falsifiers included), compute the readings as
predicates and write report.md (declaration.md). Every number comes from the day files, the ledgers or stage1_readings.json.
Usage: PF_SLOT=claude predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_exit_venue_20261125/vm_report.py merge|check|report
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import vm_replay as VR  # noqa: E402

V, F, OD, MM, TR, C = VR.V, VR.F, VR.OD, VR.MM, VR.TR, VR.C
OUT, CONFIGS, PRIMARY, EPS = VR.OUT, VR.CONFIGS, VR.PRIMARY, VR.EPS


def merge():
    fn = OUT / 'summary.json'
    C.require(not fn.exists(), 'summary.json exists (write-once)')
    days = [json.loads((OUT / f'day_{d}.json').read_text()) for d in OD.DAYS3]
    TR.dump(fn, dict(asof='2026-11-25', census=[c for d in days for c in d['census']], rows=[r for d in days for r in d['rows']],
                     parity=[p for d in days for p in d['parity']], path_parity={k: sum(d['path_parity'].get(k, 0) for d in days) for k in ('complete', 'identical')},
                     cpu_s=sum(d['cpu_s'] for d in days), cpu_s_by_day={d['day']: d['cpu_s'] for d in days}, peak_rss_mib=max(d['peak_rss_mib'] for d in days)))
    print('merged', len(days), 'days')


def sent_below_min(rec):
    """Sent takes whose opening (buy) part is worth < 1 USDT, from the ledger's own signal log (known position, side, size, limit)."""
    n = 0
    for r in rec['signal_log']:
        if r['status'] != 'sent':
            continue
        s, q, I, lim = r['side'], r['q'], r['known_I'], r['limit']
        red = min(q, max(0., -I if s == 0 else I)); op = q - red; pn = lim if s == 0 else 100. - lim
        n += int(op > EPS and op * pn < VR.MIN_BUY_C - 1e-9)
    return n


def check():
    sm = json.loads((OUT / 'summary.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)

    v3c = {(c['day'], int(c['mid'])) for c in json.loads((OD.OUT3 / 'summary.json').read_text())['census']}
    mine = collections.Counter((c['day'], int(c['mid'])) for c in sm['census'])
    ok('K9_census', set(mine) == v3c and max(mine.values()) == 1, f"{len(mine)} scheduled markets, statuses {dict(collections.Counter(c['status'] for c in sm['census']))}")
    pp = sm['path_parity']
    ok('K1_path_parity', pp['complete'] == 408 and pp['identical'] == pp['complete'], f"{pp['identical']} / {pp['complete']} complete markets identical")
    ok('K2_off_parity', len(sm['parity']) >= 10 and all(all(x['same'].values()) for x in sm['parity']), f"{len(sm['parity'])} market-arms")
    b = collections.Counter()
    fal_v1 = 0
    rg = {d: pd.read_parquet(MM.S1 / d / 'registry.parquet').set_index('mid') for d in OD.DAYS3}
    for r in sm['rows']:
        b['ties'] += int(r['payout'] == .5)
        b['payout_mismatch'] += int(V.payout_of(rg[r['day']].loc[r['mid']]) != r['payout'])
        for key, a in r['arms'].items():
            name, arm = key.split('|')
            raw = (OUT / f"{r['mid']}_{name}_{arm}.json.gz").read_bytes()
            C.require(hashlib.sha256(raw).hexdigest() == a['sha256'], 'a ledger changed')
            rec = json.loads(gzip.decompress(raw))
            a2 = VR.native_account(rec, r['payout'])
            b['account_mismatch'] += int(any(abs(a2[k] - a[k]) > 1e-6 for k in a2))
            b['below_min_sent'] += sent_below_min(rec)
            b['bound_order'] += int(not (a['unresolved_lo_c'] - 1e-6 <= a['total_c'] <= a['unresolved_hi_c'] + 1e-6))
            d = CONFIGS[name]['delay']
            b['delay'] += sum(abs(v['act'] - v['submit'] - d) > 1e-9 for v in rec['iocs'].values())
            b['exit_activation'] += sum(abs(o['act'] - o['submit'] - VR.VM_PROFILE['place']) > 1e-9 for o in rec['orders'] if o['why'] != 'IOC')
            b['ledgers'] += 1
        if r['status'] == 'complete' and b['v1_checked'] < 40:            # K4 falsifier: the frozen v1 FLOOR ledgers have no minimum rule
            v1 = json.loads(gzip.decompress((F.OUT / f"{r['mid']}_FLOOR.json.gz").read_bytes()))
            fal_v1 += sent_below_min(v1); b['v1_checked'] += 1
    ok('K3_accounting_rederived', b['ledgers'] == 6 * len(sm['rows']) and b['account_mismatch'] == 0, f"{b['ledgers']} ledgers re-accounted, {b['account_mismatch']} mismatches")
    ok('K4_min_buy', b['below_min_sent'] == 0, f"{b['below_min_sent']} sent takes with a buy part below 1 USDT")
    ok('K4_falsifier_v1_flagged', fal_v1 > 0, f"{fal_v1} below-minimum takes in the first {b['v1_checked']} complete v1 FLOOR ledgers")
    try:
        VR.selftest(); st = True
    except AssertionError:
        st = False
    ok('K5_queue_and_native_selftests', st, 'queue L vs M, native book, minimum, accounting bounds')
    ok('K6_ties', b['ties'] > 0 and b['payout_mismatch'] == 0, f"{b['ties']} tie markets paid 50, payout mismatches {b['payout_mismatch']}")
    ok('K7_bounds', b['bound_order'] == 0, f"{b['bound_order']} rows outside their bounds")
    ok('K8_delays', b['delay'] == 0 and b['exit_activation'] == 0, f"take delay mismatches {b['delay']}, exit activation mismatches {b['exit_activation']}")
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    fn = H / 'stage2_checks.json'
    C.require(not fn.exists(), 'stage2_checks.json exists (write-once)')
    TR.dump(fn, res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


def arm_block(rows, key, n_sched):
    A = [r['arms'][key] for r in rows]
    s = lambda k: float(sum(a[k] for a in A)) / 100
    day = collections.defaultdict(float)
    for r in rows:
        day[r['day']] += r['arms'][key]['total_c'] / 100
    sig, ioc = collections.Counter(), collections.Counter()
    for a in A:
        sig.update(a['sig_status']); ioc.update(a['ioc_status'])
    comp = [r['arms'][key] for r in rows if r['status'] == 'complete']
    exit_sh = float(sum(a['maker_shares'] for a in A))
    return dict(
        total_usd=s('total_c'), per_scheduled_market_usd=s('total_c') / n_sched, lower_usd=s('unresolved_lo_c'), upper_usd=s('unresolved_hi_c'),
        by_day_usd=dict(sorted(day.items())), days_positive=int(sum(v > 0 for v in day.values())),
        round_trips_usd=s('rt_net_c'), round_trips_exit_usd=s('rt_exit_net_c'), round_trips_take_usd=s('rt_take_net_c'), held_to_settlement_usd=s('held_settle_net_c'),
        held_tokens=float(sum(a['held_yes'] + a['held_no'] for a in A)), held_cost_usd=s('held_cost_c'),
        markets_holding_1_or_more=int(sum(a['held_yes'] + a['held_no'] >= 1 for a in A)),
        fees_cash_usd=s('fee_cash_c'), fees_in_shares_usd_at_purchase=s('fee_token_cost_c'),
        taker_shares=float(sum(a['taker_shares'] for a in A)), exit_shares=exit_sh, oversell_shares=float(sum(a['oversell_shares'] for a in A)),
        c1_total_without_young_usd=s('total_without_young_c'), c1_young_exit_usd=s('young_exit_c'),
        c1_young_share_of_exit_shares=float(sum(a['young_exit_shares'] for a in A)) / exit_sh if exit_sh > EPS else None,
        c3_cancel_pending_exit_usd=s('cancel_pending_exit_c'),
        c3_cancel_pending_share_of_exit_shares=float(sum(a['cancel_pending_exit_shares'] for a in A)) / exit_sh if exit_sh > EPS else None,
        signal_status=dict(sig), min_capped=int(sum(a['min_capped'] for a in A)), ioc_status=dict(ioc), unresolved_orders=int(sum(a['unresolved_orders'] for a in A)),
        complete_markets=len(comp), complete_total_usd=float(sum(a['total_c'] for a in comp)) / 100)


def readings():
    sm = json.loads((OUT / 'summary.json').read_text())
    ck = json.loads((H / 'stage2_checks.json').read_text())
    C.require(ck['all_pass'], 'stage 2 checks did not pass')
    rows, n_sched = sm['rows'], len(sm['census'])
    out = dict(asof='2026-11-25', scheduled_markets=n_sched, replayed_markets=len(rows), path_status=dict(collections.Counter(r['status'] for r in rows)),
               census=dict(collections.Counter(c['status'] for c in sm['census'])), configs={})
    for name in CONFIGS:
        blk = {arm: arm_block(rows, f'{name}|{arm}', n_sched) for arm in V.ARMS}
        pdd = collections.defaultdict(float)
        for r in rows:
            pdd[r['day']] += (r['arms'][f'{name}|FLOOR']['total_c'] - r['arms'][f'{name}|BASE']['total_c']) / 100
        blk['floor_minus_base'] = dict(total_usd=sum(pdd.values()), by_day_usd=dict(sorted(pdd.items())), days_positive=int(sum(v > 0 for v in pdd.values())))
        out['configs'][name] = blk

    def survives(name, arm='FLOOR'):
        a = out['configs'][name][arm]
        return bool(a['lower_usd'] > 0 and a['days_positive'] >= 4)
    out['readings'] = dict(survives_venue_match=survives(PRIMARY), floor_beats_base_venue=out['configs'][PRIMARY]['floor_minus_base']['days_positive'] >= 4,
                           holds_at_258=survives('VM258-M50'), holds_at_m250=survives('VM208-M250'),
                           base_survives_venue_match=survives(PRIMARY, 'BASE'))
    fn = H / 'stage2_readings.json'
    C.require(not fn.exists(), 'stage2_readings.json exists (write-once)')
    md = render_md(out, ck)                                             # rendered before any write: a rendering failure leaves nothing behind
    TR.dump(fn, out)
    (H / 'report.md').write_text(md)
    print(json.dumps(out['readings']))


def usd(x):
    return '—' if x is None else f'{x:+,.2f}'


def render_md(out, ck):
    s1 = json.loads((H / 'stage1_readings.json').read_text())
    L = ['# TAKE-EXIT VENUE-MATCHED COMPARISON v1 — BASE vs FLOOR under venue rules (2026-11-25)', '',
         '[declaration.md](declaration.md) (declared before any run; resource amendment R1 before stage 2). Review 4e67e0856. Five consumed v3 days; '
         'development evidence only.', '', '## Stage 1 — the speed bump alone (408 complete markets, settlement, ties 50 / 50)', '',
         '| | 58 ms (v1 harness) | 208 ms (+150 ms speed bump) | speed-bump cost |', '|---|---:|---:|---:|']
    for arm in V.ARMS:
        a = s1['arms'][arm]
        L.append(f"| {arm} | {usd(a['at_58']['settle_net_usd'])} | {usd(a['at_208']['settle_net_usd'])} | {usd(a['at_208']['settle_net_usd'] - a['at_58']['settle_net_usd'])} |")
    L += ['', f"Stage 1 checks: ties = 8 {s1['checks']['ties_8']}; the 58 ms values reproduce the reviewer's independent recount "
          f"{s1['checks']['reviewer_recount_reproduced']}.", '',
          f"## Stage 2 — venue-matched ({out['replayed_markets']} of {out['scheduled_markets']} scheduled markets replayed)", '',
          f"Path status of replayed markets: {out['path_status']}.", '']
    for name in CONFIGS:
        blk = out['configs'][name]
        L += [f'### {name}' + (' (primary)' if name == PRIMARY else ''), '', '| | BASE | FLOOR |', '|---|---:|---:|']
        rowsdef = [('Total at settlement', 'total_usd'), ('… per scheduled market', 'per_scheduled_market_usd'), ('… lower bound (unresolved orders)', 'lower_usd'),
                   ('… upper bound', 'upper_usd'), ('Round trips after fees', 'round_trips_usd'), ('… closed by the maker exit', 'round_trips_exit_usd'),
                   ('… closed by an opposite take', 'round_trips_take_usd'), ('Held to settlement', 'held_to_settlement_usd'), ('Cost of tokens held at the end', 'held_cost_usd'),
                   ('C1: total without exit fills < 100 ms old', 'c1_total_without_young_usd'), ('C3: cancel-pending exit fills ($)', 'c3_cancel_pending_exit_usd'),
                   ('Complete markets only (408)', 'complete_total_usd')]
        for lab, k in rowsdef:
            L.append(f"| {lab} | {usd(blk['BASE'][k])} | {usd(blk['FLOOR'][k])} |")
        for lab, k in (('Days positive', 'days_positive'), ('Taker shares', 'taker_shares'), ('Tokens held at the end', 'held_tokens'),
                       ('Markets holding ≥ 1 token', 'markets_holding_1_or_more'), ('Takes cut by the 1 USDT minimum (min_capped)', 'min_capped')):
            L.append(f"| {lab} | {blk['BASE'][k]:,.0f} | {blk['FLOOR'][k]:,.0f} |" if isinstance(blk['BASE'][k], (int, float)) else f"| {lab} | {blk['BASE'][k]} | {blk['FLOOR'][k]} |")
        fm = blk['floor_minus_base']
        L += ['', f"By day (FLOOR / BASE / difference): " + '; '.join(f"{d} {usd(blk['FLOOR']['by_day_usd'][d])} / {usd(blk['BASE']['by_day_usd'][d])} / {usd(fm['by_day_usd'][d])}"
                                                               for d in sorted(fm['by_day_usd'])), '',
              f"Take outcomes (FLOOR): {blk['FLOOR']['ioc_status']}; signal statuses: {blk['FLOOR']['signal_status']}.", '']
    L += ['## Readings (computed)', ''] + [f'- `{k}`: {v}' for k, v in out['readings'].items()]
    L += ['', '## Checks', ''] + [f"- {'PASS' if v['ok'] else 'FAIL'} `{k}`: {v['detail']}" for k, v in ck.items() if isinstance(v, dict)]
    L += ['', '## Limits', '',
          '- Consumed days; the speed bump\'s historical presence is unknown (the archived market objects lack the field), so the 208 / 258 ms arrival '
          'is applied to a book recorded under whatever regime held then.',
          '- Our order is not in the recorded book: competitors\' reactions to it are not modelled. The strict two-frame IOC rule and queue L are '
          'conservative; the 1 s staleness rule truncates paths at recorder gaps.',
          '- A resting exit that would oversell after a take is dropped in the token accounting but not in the engine path (counted as oversell).']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    dict(merge=merge, check=check, report=readings)[sys.argv[1]]()
