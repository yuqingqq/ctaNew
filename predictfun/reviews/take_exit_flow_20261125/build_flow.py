"""TAKE-EXIT FLOW (2026-11-25; USER: "it looks promising. can you show me the whole flow of the execution"). Builds flow.html: the
execution loop of the frozen take + maker-exit rule with counts over the 408 v3 STRICT58 markets, and three real markets tick by tick.
Read-only on the v3 ledgers (take_exit_replay_20261125/other_days.py; freeze 99ac28e53, results 53c03cb56). No new result, nothing tuned.
- Examples (rule fixed here, applied by code, not chosen by eye): among STRICT58 markets with >= 3 filled takes, the market whose net is
  closest to the 50th / 10th / 90th percentile of that set (ties: lower mid).
- Each example's world is reloaded through the unchanged loader and the frozen replay (other_days.one, STRICT58) is re-run on it; it must
  reproduce the saved ledger exactly (fills, orders, IOC results, signal / guard / structural logs, value, tape id, fingerprint), so every
  plotted order and fill is the ledger's own.
- Computed predicates (flow_data.json['checks']): per example the fills rebuild the ledger's net at the END mark; over all 408 ledgers the
  take legs at END plus the exits vs holding to END equal the net.
Usage: PF_SLOT=claude predictfun/research/run_light.sh txflow -- python3 predictfun/reviews/take_exit_flow_20261125/build_flow.py data|html
"""
import collections
import gzip
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H.parent / 'take_exit_replay_20261125'))
import other_days as OD  # noqa: E402

MM, TR, P = OD.MM, OD.TR, OD.P
LED = OD.OUT3
EPS = 1e-9
PICKS = (('typical', 0.5), ('weak', 0.1), ('strong', 0.9))
SAME = ('fills', 'orders', 'ioc_results', 'signal_log', 'guard_log', 'struct_log', 'iocs', 'value', 'cash', 'position', 'mark', 'tape_id',
        'fingerprint', 'net_value_c', 'take_marks')


def fin(x, n=2):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if math.isfinite(x) else None


def ledger(mid):
    return json.loads(gzip.decompress((LED / f'{mid}_STRICT58.json.gz').read_bytes()))


def pick():
    d = pd.read_csv(OD.H / 'v3_per_market.csv', dtype={'day': str})
    s = d[(d.setting == 'STRICT58') & (d.takes_filled >= 3)]
    out = []
    for name, q in PICKS:
        tgt = float(s.net_c.quantile(q))
        r = s.assign(dist=(s.net_c - tgt).abs()).sort_values(['dist', 'mid']).iloc[0]
        out.append(dict(name=name, pct=q, target_net_c=tgt, pool=int(len(s)), day=str(r.day), mid=int(r.mid), hour=int(r.hour)))
    assert len({x['mid'] for x in out}) == len(PICKS)
    return out


def reload(day, mid, edges):
    src = MM.KP.Tape()
    reg = MM.registry(day).set_index('mid')
    got = list(MM.worlds(day, [mid], src, src.offset(day), reg))
    assert len(got) == 1 and got[0][0] == mid and got[0][1] == 'complete', [g[:2] for g in got]
    _, _, p, info = got[0]
    assert float(reg.loc[mid, 'fee_bps']) == 200.
    ws = float(reg.loc[mid, 'ws']) * 1000.
    assert ws + 300000. < OD.FWD_MS
    cache = OD.make_cache3(float(info['t_start']), float(info['end_rel']), OD.SEG_OF[day])
    old = dict(end=float(info['end_rel']), ident=MM.ident_of(day, mid, ws, info, p))
    rec = json.loads(json.dumps(OD.one(p, cache, old, edges, 200., 'STRICT58'), allow_nan=False))
    return p, cache, rec


def sides(f):
    bid_ok = (f[:, 2] >= 0) & (f[:, 2] <= 100) & (f[:, 7] > 0)
    ask_ok = (f[:, 12] >= 0) & (f[:, 12] <= 100) & (f[:, 17] > 0)
    return bid_ok, ask_ok


def book_series(f, end):
    bid_ok, ask_ok = sides(f)
    out, last = [], None
    for j in np.flatnonzero(f[:, 0] <= end + EPS):
        b = (float(f[j, 2]), fin(f[j, 7])) if bid_ok[j] else (None, None)
        a = (float(f[j, 12]), fin(f[j, 17])) if ask_ok[j] else (None, None)
        if (b, a) != last:
            out.append([fin(f[j, 0], 1), b[0], b[1], a[0], a[1]])
            last = (b, a)
    return out


def shown(f, t, side, price):
    """The displayed size at (side, price) in the last frame with venue time <= t (the engine's queue-ahead rule at activation)."""
    j = int(np.searchsorted(f[:, 0], t, side='right')) - 1
    if j < 0:
        return None
    pr, sz = f[j, 2 + side * 10: 7 + side * 10], f[j, 7 + side * 10: 12 + side * 10]
    return float(sz[np.abs(pr - price) < EPS].sum())


def sg(s):
    return 1. if int(s) == 0 else -1.


def example(ex, edges):
    p, cache, rec = reload(ex['day'], ex['mid'], edges)
    saved = ledger(ex['mid'])
    same = {k: rec[k] == saved[k] for k in SAME}
    assert all(same.values()), same
    f = np.asarray(p['frames'], float)
    tr = np.asarray(p['trades'], float).reshape(-1, 4)
    end, mark = float(rec['end']), float(rec['mark'])
    fills = sorted(rec['fills'], key=lambda x: (x[0], x[1]))
    fees = sorted((float(x['act']), float(x['fee_c'])) for x in rec['ioc_results'] if x['filled'] > EPS)

    # rebuild the ledger's net from the fills at the END mark (a computed predicate, not an assumption)
    cash = -sum(sg(x[2]) * x[3] * x[4] for x in fills)
    pos = sum(sg(x[2]) * x[3] for x in fills)
    net_rebuilt = cash + pos * mark - sum(v for _, v in fees)
    ok_net = abs(net_rebuilt - rec['net_value_c']) < 1e-6

    def pos_at(t):
        return sum(sg(x[2]) * x[3] for x in fills if x[0] <= t + EPS)

    # series
    n = len(cache['x'][1])
    bn = [fin(v, 3) for v in cache['x'][1]]
    bid_ok, ask_ok = sides(f)
    two = bid_ok & ask_ok & (f[:, 2] < f[:, 12])
    mid_f = np.where(two, (f[:, 2] + f[:, 12]) / 2, np.nan)
    mid_ff = pd.Series(mid_f).ffill().to_numpy()
    grid = np.asarray(rec['grid'], float)
    value, last = [], None
    fee_t = np.array([t for t, _ in fees]) if fees else np.zeros(0)
    fee_v = np.cumsum([v for _, v in fees]) if fees else np.zeros(0)
    for g, I, RB, RA, ps, cs in grid:
        j = int(np.searchsorted(f[:, 0], g, side='right')) - 1
        m = mid_ff[j] if j >= 0 else np.nan
        k = int(np.searchsorted(fee_t, g, side='right')) - 1
        v = cs + ps * m - (fee_v[k] if k >= 0 else 0.) if np.isfinite(m) else None
        v = fin(v, 2)
        if v != last:
            value.append([fin(g, 0), v])
            last = v
    value.append([fin(end, 0), fin(rec['net_value_c'], 2)])
    posn, last = [], None
    for g, I, RB, RA, ps, cs in grid:
        row = (fin(ps, 2), fin(I, 2))
        if row != last:
            posn.append([fin(g, 0), row[0], row[1]])
            last = row
    posn.append([fin(end, 0), fin(rec['position'], 2), posn[-1][2]])
    trades = [[fin(t, 1), int(s), float(pz), fin(q)] for t, s, pz, q in tr if t <= end + EPS]

    # events
    ev = []
    for s in rec['signal_log']:
        k = int(s['k'])
        ev.append(dict(t=s['t'], kind='trigger', dir='UP' if s['guard_side'] == 1 else 'DOWN', side=s['side'], r250=fin(cache['x'][1][k], 3),
                       status=s['status'], q=s['q'], room=fin(s['room']), known_I=fin(s['known_I']), limit=s.get('limit'), gid=s['gid']))
    for x in rec['ioc_results']:
        ev.append(dict(t=x['act'], kind='take', gid=x['gid'], side=x['side'], q=x['q'], limit=x['limit'], submit=x['submit'],
                       filled=fin(x['filled']), price=x['price'], fee_c=fin(x['fee_c'], 3), reason=x['reason'],
                       lenient_q=fin(x.get('lenient_q')), strict_cap=fin(x.get('strict_cap')), frame_ms=fin(x.get('venue_frame_ms'), 1)))
        if x['filled'] > EPS:
            ev.append(dict(t=x['ack'], kind='notice', gid=x['gid'], side=x['side'], q=fin(x['filled'])))
    struct = {int(s['gid']): s for s in rec['struct_log']}
    pulls = {int(g[3]): g for g in rec['guard_log'] if g[2] == 'live' and g[10] == 'pressure_pull'}
    pull_why = {int(c[3]): c[8] for c in rec['commands'] if c[7] == 'PULL'}
    filled_q, last_fill = collections.Counter(), {}
    for x in fills:
        if x[5] >= 0:
            filled_q[int(x[1])] += x[3]
            last_fill[int(x[1])] = max(last_fill.get(int(x[1]), -1.), x[0])
    orders = []
    for o in rec['orders']:
        if o['why'] == 'IOC':
            continue
        gid = int(o['gid'])
        rem_t = None if o['remove'] > 1e29 else o['remove']
        ahead = None if o['refused'] else fin(shown(f, o['act'], o['side'], o['price']))
        # how long the order rests: until its cancel takes effect, its last fill if it filled completely, or END
        until = o['act'] if o['refused'] else min(rem_t if rem_t is not None else end, last_fill[gid] if o['rem'] <= EPS and gid in last_fill else end, end)
        assert o['refused'] or o['req'] is not None or o['rem'] > EPS or gid in last_fill, ('filled order without a fill', gid)
        orders.append(dict(gid=gid, side=o['side'], price=o['price'], q=o['size'], submit=o['submit'], act=o['act'], refused=o['refused'],
                           remove=rem_t, until=fin(until, 1), why=o['why'], filled=fin(filled_q[gid]), ahead=ahead,
                           open_at_end=bool(not o['refused'] and o['req'] is None and o['rem'] > EPS)))
        ev.append(dict(t=o['submit'], kind='post', gid=gid, side=o['side'], price=o['price'], q=o['size'], act=o['act'], refused=o['refused'],
                       ahead=ahead))
        if o['req'] is not None:
            row = dict(t=o['req'], kind='cancel', gid=gid, side=o['side'], price=o['price'], why=o['why'], effect=rem_t)
            if o['why'] == 'structural':
                st = struct[gid]
                row.update(reason='structural_' + st['primary'], best=st['best'], opp=st['opp'])
            else:
                assert o['why'] == 'policy' and pull_why[gid] in ('pressure_pull', 'exit_not_needed'), (gid, o['why'], pull_why.get(gid))
                row.update(reason=pull_why[gid])
                if gid in pulls:
                    row.update(r250=fin(cache['x'][1][int(round(pulls[gid][0] / 50.))], 3))
            ev.append(row)
    wh = [g for g in rec['guard_log'] if g[2] == 'slot' and g[10] == 'pressure_withhold']
    run = []
    for g in wh + [None]:
        if run and (g is None or g[1] != run[-1][1] or g[0] - run[-1][0] > 50. + EPS):
            ev.append(dict(t=run[0][0], kind='withhold', side=run[0][1], t_end=run[-1][0], ticks=len(run), q=fin(run[0][7]),
                           r250=fin(cache['x'][1][int(round(run[0][0] / 50.))], 3), x_max=fin(max(r[5] for r in run), 3)))
            run = []
        if g is not None:
            run.append(g)
    for x in fills:
        if x[5] < 0:
            continue
        tp = float(tr[int(x[5]), 2])
        ev.append(dict(t=x[0], kind='exit_fill', gid=int(x[1]), side=x[2], q=fin(x[3]), price=x[4], trade_price=tp,
                       through=bool(tp > x[4] + EPS if x[2] == 1 else tp < x[4] - EPS)))
    ev.append(dict(t=end, kind='end', position=fin(rec['position']), mark=mark, net_c=fin(rec['net_value_c'])))
    order = {'trigger': 0, 'take': 1, 'notice': 2, 'withhold': 3, 'cancel': 4, 'post': 5, 'exit_fill': 6, 'end': 7}
    ev.sort(key=lambda e: (e['t'], order[e['kind']], e.get('gid') or 0))
    for e in ev:
        e['pos_after'] = fin(pos_at(e['t']))
        e['t'] = fin(e['t'], 1)

    # zoom windows: one per filled take, from 2 s before its trigger to 1.5 s after the position is next flat (at most 20 s);
    # flat = under one share open (exits are whole shares, so a fractional remainder from a partial take cannot be exited)
    eps_, ti = [], 0
    for x in sorted(rec['ioc_results'], key=lambda r: r['act']):
        if x['filled'] <= EPS:
            continue
        ti += 1
        flat = next((y[0] for y in fills if y[0] > x['act'] + EPS and abs(pos_at(y[0])) < 1.), None)
        t1 = min(end, x['act'] + 20000., (flat + 1500.) if flat is not None else 1e30)
        # clean = from flat, back to flat through exit fills only (no other take fill in between)
        between = [y for y in fills if x['act'] + EPS < y[0] <= (flat if flat is not None else -1.) + EPS]
        clean = bool(abs(pos_at(x['act'] - 1e-6)) < 1. and flat is not None and between and all(y[5] >= 0 for y in between))
        rt = None
        if clean:                                      # the exited shares' result against the take price, after the take's fee
            d_ = 1. if x['side'] == 0 else -1.
            rt = sum(d_ * y[3] * (y[4] - x['price']) for y in between) - x['fee_c']
        eps_.append(dict(n=ti, t0=max(0., x['submit'] - 2000.), t1=t1, side=x['side'], q=fin(x['filled']), price=x['price'], t=x['act'],
                         flat_t=fin(flat, 1), clean=clean, rt_c=fin(rt), rt_shares=fin(sum(y[3] for y in between)) if clean else None))

    met = rec['metrics']
    return dict(
        name=ex['name'], pct=ex['pct'], pool=ex['pool'], target_net_c=fin(ex['target_net_c']), day=ex['day'], mid=ex['mid'], hour=ex['hour'],
        start_abs_ms=rec['ident']['start_abs_ms'], end=end, mark=mark, net_c=fin(rec['net_value_c']), fees_c=fin(rec['fees_c']),
        metrics={k: fin(met[k]) for k in ('takes_sent', 'takes_filled', 'taker_shares', 'exit_fills', 'exit_shares', 'triggers', 'exit_orders',
                                          'peak_inventory', 'time_in_position', 'exit_value_vs_hold_c', 'young_exit_value_c')},
        series=dict(bn=bn, bn_step=50., book=book_series(f, end), trades=trades, pos=posn, value=value),
        orders=orders, events=ev, episodes=eps_, rederived=same, net_rebuilt_c=fin(net_rebuilt, 6), net_rebuilt_ok=bool(ok_net),
        n_ticks=n)


def aggregates():
    d = pd.read_csv(OD.H / 'v3_per_market.csv', dtype={'day': str})
    mids = sorted(d[d.setting == 'STRICT58'].mid.astype(int).tolist())
    assert len(mids) == 408
    trig, trig_dir, ioc, ioc_sh = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    mk = {h: [0., 0., 0] for h in ('m1s_c', 'm3s_c', 'm10s_c', 'end_c')}
    o_n, o_ref, o_end = 0, 0, collections.Counter()
    ex_n, ex_sh, ages = 0, 0., []
    guard = collections.Counter()
    tot = collections.Counter()
    pull_reason, struct_primary, take_effect = collections.Counter(), collections.Counter(), collections.Counter()
    bad_identity = 0
    for m in mids:
        rec = ledger(m)
        assert rec['setting'] == 'STRICT58'
        for s in rec['signal_log']:
            trig[s['status']] += 1
            trig_dir[('UP' if s['guard_side'] == 1 else 'DOWN', s['status'])] += 1
        for x in rec['ioc_results']:
            ioc[x['reason']] += 1
            ioc_sh['requested'] += x['q']
            ioc_sh['filled'] += x['filled']
            if x.get('lenient_q') is not None:
                ioc_sh['lenient_would_fill'] += x['lenient_q']
        te = 0.
        for x in rec['take_marks']:
            for h in mk:
                if x[h] is None:
                    mk[h][2] += 1
                else:
                    mk[h][0] += x[h] * x['q']
                    mk[h][1] += x['q']
            te += x['end_c'] * x['q']
        fq = collections.Counter()
        for x in rec['fills']:
            if x[5] >= 0:
                fq[int(x[1])] += x[3]
                ex_n += 1
                ex_sh += x[3]
        for o in rec['orders']:
            if o['why'] == 'IOC':
                continue
            o_n += 1
            if o['refused']:
                o_ref += 1
                o_end['refused_post_only'] += 1
            elif o['req'] is not None:
                o_end[f"cancel_{o['why']}"] += 1
            elif o['rem'] <= EPS:
                o_end['filled_completely'] += 1
            else:
                o_end['open_at_end'] += 1
        ages += rec['passive_ages_ms']
        for g in rec['guard_log']:
            guard[g[10]] += 1
        for c in rec['commands']:
            if c[7] == 'PULL':
                pull_reason[c[8]] += 1
        for st in rec['struct_log']:
            struct_primary[st['primary']] += 1
        for x in rec['ioc_results']:
            if x['filled'] <= EPS:
                continue
            b, a = float(x['venue_before']), float(x['venue_after'])
            k = ('from_flat' if abs(b) < 0.5 else 'adds' if abs(a) > abs(b) + EPS and b * a > 0 else
                 'flips' if b * a < 0 and abs(a) >= 0.5 else 'reduces')
            take_effect[k] += 1
        met = rec['metrics']
        tot['matched_shares'] += met['matched_shares']
        tot['completed_c'] += met['completed_c']
        tot['residual_c'] += met['residual_c']
        xv = sum(rec['passive_exit_value_c'])
        bad_identity += int(abs(te + xv - rec['net_value_c']) > 1e-6 * max(1., abs(rec['net_value_c'])))
        tot['net_c'] += rec['net_value_c']
        tot['fees_c'] += rec['fees_c']
        tot['take_end_c'] += te
        tot['exit_vs_hold_c'] += xv
        tot['taker_shares'] += sum(x[3] for x in rec['fills'] if x[5] < 0)
        tot['abs_end_position'] += abs(rec['position'])
        tot['markets_with_fill'] += int(any(x['filled'] > EPS for x in rec['ioc_results']))
    ag = np.asarray(ages, float)
    return dict(
        markets=len(mids), triggers=dict(trig), triggers_by_dir={f'{a}|{b}': v for (a, b), v in sorted(trig_dir.items())},
        ioc_reasons=dict(ioc), ioc_shares={k: round(v, 2) for k, v in ioc_sh.items()},
        take_markouts={h: dict(cents_per_share=round(v[0] / v[1], 3) if v[1] else None, shares=round(v[1], 2), censored_fills=v[2]) for h, v in mk.items()},
        exit_orders=dict(posted=o_n, refused=o_ref, outcome=dict(o_end)), exit_fills=dict(fills=ex_n, shares=round(ex_sh, 2),
                                                                                         age_ms_p50=float(np.median(ag)), age_ms_p90=float(np.quantile(ag, .9)),
                                                                                         young_share=float(np.mean(ag < 100.))),
        guard=dict(guard), pulls_by_reason=dict(pull_reason), structural_by_reason=dict(struct_primary), take_position_effect=dict(take_effect),
        totals={k: round(v, 4) for k, v in tot.items()}, identity_mismatches=bad_identity,
        fifo_identity_ok=bool(abs(tot['completed_c'] + tot['residual_c'] - tot['fees_c'] - tot['net_c']) < 1e-4))


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (300, 301))
    edges = TR.T.P.frozen_edges()
    assert edges[1][-1] == P.LAST_EDGES[1] and edges[0][-1] == P.LAST_EDGES[0]
    exs = [example(e, edges) for e in pick()]
    ag = aggregates()
    st = ledger(exs[0]['mid'])['settings']
    data = dict(
        asof='2026-11-25', source=dict(freeze='99ac28e53', results='53c03cb56', ledgers='take_exit_replay_v3_20261125 STRICT58',
                                       report='predictfun/reviews/take_exit_replay_20261125/v3_report.md'),
        rule=dict(edge_up_bps=P.LAST_EDGES[1], edge_down_bps=-P.LAST_EDGES[0], quiet_ms=TR.QUIET_TICKS * 50., take_q=TR.TAKE_Q, cap=OD.ER.CAP_POS,
                  take_delay_ms=st['take_delay_ms'], step_ms=st['step_ms'], post_activation_ms=st['activation_ms'], cancel_effect_ms=st['cancel_effect_ms'],
                  fill_notice_ms=st['fill_notice_ms'], cancel_ack_ms=st['cancel_ack_ms'], q0=st['q0'], taker_fee_bps=st['taker_fee_bps'],
                  end_ms=st['end_ms'], bn_in_ms=MM.B_IN),
        example_rule='among STRICT58 markets with >= 3 filled takes, the market whose net is closest to the 50th / 10th / 90th percentile '
                     'of that set (ties: lower mid)',
        examples=exs, agg=ag, v3=json.loads((OD.H / 'v3_readings.json').read_text())['STRICT58'],
        census=json.loads((OD.H / 'v3_readings.json').read_text())['census'],
        measured=dict(order_live_p50_ms=58., order_live_lower_bound_ms=46., cancel_effect_ms=37.,
                      source='predictfun/reviews/take_exit_review_20261125/reviewer_report.md items 2 and 5 (pf_clock_rl.py; Tokyo 58 / 37 ms)'),
        checks=dict(examples_rederived=all(all(e['rederived'].values()) for e in exs), examples_net_rebuilt=all(e['net_rebuilt_ok'] for e in exs),
                    take_plus_exit_equals_net_all=ag['identity_mismatches'] == 0, fifo_round_trips_plus_residual_minus_fees_equals_net=ag['fifo_identity_ok'],
                    net_matches_v3_readings=abs(ag['totals']['net_c'] / 100 - json.loads((OD.H / 'v3_readings.json').read_text())['STRICT58']['net_usd']) < 1e-6))
    assert all(data['checks'].values()), data['checks']
    (H / 'flow_data.json').write_text(json.dumps(data, allow_nan=False, separators=(',', ':')) + '\n')
    print(json.dumps(dict(examples=[(e['name'], e['day'], e['mid'], e['net_c'], len(e['events'])) for e in exs], checks=data['checks'],
                          totals=ag['totals'], markouts=ag['take_markouts'], ioc=ag['ioc_reasons'], trig=ag['triggers'], exits=ag['exit_orders'],
                          exit_fills=ag['exit_fills'], guard=ag['guard'], pulls=ag['pulls_by_reason'], struct=ag['structural_by_reason'],
                          take_effect=ag['take_position_effect'], pool=exs[0]['pool']), indent=1))


def html():
    data = json.loads((H / 'flow_data.json').read_text())
    assert all(data['checks'].values()), data['checks']
    tpl = (H / 'flow_template.html').read_text()
    assert tpl.count('/*__FLOW_DATA__*/null') == 1
    (H / 'flow.html').write_text(tpl.replace('/*__FLOW_DATA__*/null', json.dumps(data, allow_nan=False, separators=(',', ':'))))
    print('flow.html', (H / 'flow.html').stat().st_size)


if __name__ == '__main__':
    {'data': main, 'html': html}[sys.argv[1]]()
