"""When do the winning takers fire, to the ms? (USER 2026-11-26: "we first need to get the time when the takers fire, then we check what
happened, to figure out how fast we need to be"). The send times so far are the first fill's websocket time - 50 ms - the 150 ms hold
(M50, about +-30 ms). Every REST match record carries a settlementId, a UUIDv7 whose first 48 bits are the venue's ms creation time of the
settlement: an upper bound on the match under the premises of FINDINGS C23 (created at / after the match, venue clocks aligned). A taker
order waits the 150 ms hold and matches on release, so its venue receipt is at or before t_set - 150 ms (the fire time used here); the send
is earlier by its network path.
Stage `scan <day>`: the day's REST match records (takers.stems: the day's 24 hours + the permitted next day's 00-01Z, guarded) whose taker
signer is one of the top-10 takers by 1 s value (winners_trigger.json, matched by prefix / suffix, unique), prefiltered on the signer
string before parsing; one row per (taker order hash, settlementId): t_set, executedAt second, transaction, market, amount, first receipt.
Output (write-once): OUT/fire_times_<day>.parquet + .json (counts).
Stage `join`: their orders exactly as winners_trigger.analyze builds them (winners_legs, the same filters), each joined to its settlement
by order hash (an order filled by several makers has several settlements, normally in one ms: the earliest is its release, the span is
reported). Checks and readings: orders with a settlement / none / several; the settlements' market equals the order's; t_set before the end of
its executedAt second (C23); fire time t_set - 150 against the M50 send time (quantiles of the difference per group); bursts (one
taker, one market, fire times within 2 ms of the previous). Output (write-once): fire_times.json + OUT/fire_times_orders.parquet.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_times.py \
       scan <day> [<rehearsal dir> <stems>] | join [<rehearsal dir>]
"""
import gzip
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / 'take_exit_venue_20261125'))
import winners_trigger as WT  # noqa: E402
import takers as TK  # noqa: E402

IO, C, TR, OUT = TK.IO, WT.C, WT.TR, WT.OUT
BURST_MS = 2.
QS = (.05, .25, .5, .75, .95)


def top10():
    d = json.loads((H / 'winners_trigger.json').read_text())
    t = pd.concat([pd.read_parquet(OUT / f'winners_legs_{x}.parquet', columns=['taker']) for x in WT.DAYS]).taker.dropna().unique()
    out = []
    for r in d['top10_by_value']:
        pre, suf = r['taker'].split('…')
        m = [x for x in t if x.startswith(pre) and x.endswith(suf)]
        C.require(len(m) == 1, f"{r['taker']}: {len(m)} addresses match")
        out.append(m[0])
    return out


def uuid7_ms(u):
    h = u.replace('-', '')
    C.require(len(h) == 32 and h[12] == '7', f'not a UUIDv7: {u}')
    return int(h[:12], 16)


def scan(day, out_dir=None, n_stems=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    od = Path(out_dir) if out_dir else OUT
    fn, fj = od / f'fire_times_{day}.parquet', od / f'fire_times_{day}.json'
    C.require(not fn.exists() and not fj.exists(), f'{fn.name} exists (write-once)')
    addrs = top10()
    rank = {a: i + 1 for i, a in enumerate(addrs)}
    rows, cnt = {}, dict(match_lines=0, lines_with_a_signer=0, records=0, no_settlement_id=0, no_market=0)
    stems = TK.stems(day)[:n_stems] if n_stems else TK.stems(day)
    for h in stems:
        p = Path(f'{IO.PF_RUN}/rest/{h}.jsonl.gz')
        if not p.exists():
            continue
        with gzip.open(p, 'rt') as f:
            for line in f:
                j = line.find('"path":"', 0, 400)
                if j < 0 or not line.startswith('/v1/orders/matches', j + 8):
                    continue
                cnt['match_lines'] += 1
                if not any(a in line for a in addrs):
                    continue
                r = json.loads(line)
                if not (r.get('path') or '').startswith('/v1/orders/matches') or r.get('status') != 200:
                    continue
                try:
                    pl = json.loads(r['raw'])
                except (TypeError, ValueError):
                    continue
                cnt['lines_with_a_signer'] += 1
                for x in pl.get('data') or []:
                    t = x.get('taker') or {}
                    if t.get('signer') not in rank:
                        continue
                    cnt['records'] += 1
                    sid, mk = x.get('settlementId'), (x.get('market') or {}).get('id')
                    if not sid:
                        cnt['no_settlement_id'] += 1
                        continue
                    if mk is None:
                        cnt['no_market'] += 1
                        continue
                    k = (t['hash'], sid)
                    if k in rows:
                        rows[k]['n_seen'] += 1
                        rows[k]['recv_ns'] = min(rows[k]['recv_ns'], int(r['recv_ns']))
                        continue
                    rows[k] = dict(rank=rank[t['signer']], taker=t['signer'], taker_hash=t['hash'], sid=sid, t_set=uuid7_ms(sid), exec_s=int(IO.iso_s(x['executedAt'])),
                                   tx=x.get('transactionHash'), mid=int(mk), amount=int(x['amountFilled']) / IO.E18,
                                   recv_ns=int(r['recv_ns']), n_seen=1)
    S = pd.DataFrame(list(rows.values()))
    C.require(len(S) > 0, f'{day}: no records')
    ns = S.groupby('taker_hash').sid.nunique()
    S.to_parquet(fn, index=False)
    TR.dump(fj, dict(asof='2026-11-26', day=day, stems=stems, **cnt, rows=int(len(S)), orders=int(S.taker_hash.nunique()),
                     orders_with_several_settlements=int((ns > 1).sum()), by_rank={int(k): int(v) for k, v in S.groupby('rank').size().items()}))
    print(day, cnt, len(S), flush=True)


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def join(out_dir=None):
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    od = Path(out_dir) if out_dir else None
    fj, fp = (od or H) / 'fire_times.json', (od or OUT) / 'fire_times_orders.parquet'
    C.require(not fj.exists() and not fp.exists(), 'an output exists (write-once)')
    vs = json.loads((WT.VR.OUT / 'summary.json').read_text())
    addrs = top10()
    rank = {a: i + 1 for i, a in enumerate(addrs)}
    parts = []
    for day in WT.DAYS:
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
        S = pd.read_parquet(OUT / f'fire_times_{day}.parquet')
        J = pd.read_parquet(OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'shares', 't_ms2', 'how2', 'taker', 'taker_hash', 'ambiguous', 'exec_s'])
        J = J[J.taker.isin(addrs) & ~J.ambiguous & J.mid.isin(comp) & J.how2.isin([1, 3])].copy()
        J['t_v'] = J.t_ms2.astype(float) - WT.M50
        J['d'] = np.where(J.side == 1, 1, -1)
        first_how = J.sort_values('t_v', kind='stable').drop_duplicates('taker_hash').set_index('taker_hash')['how2']
        O = J.groupby('taker_hash').agg(taker=('taker', 'first'), mid=('mid', 'first'), t=('t_v', 'min'), d=('d', 'first'), dn=('d', 'nunique'),
                                        shares=('shares', 'sum'), exec_s=('exec_s', 'min')).reset_index()
        O = O[O.dn == 1].drop(columns='dn')
        O['how2_first'] = O.taker_hash.map(first_how).astype(int)        # how the first leg was timed: 1 websocket, 3 book drop
        agg = S.groupby('taker_hash').agg(t_set=('t_set', 'min'), t_last=('t_set', 'max'), n_settlements=('sid', 'nunique'), mid_rest=('mid', 'first'),
                                          n_mid=('mid', 'nunique'), exec_s_rest=('exec_s', 'min')).reset_index()
        O = O.merge(agg, on='taker_hash', how='left', validate='one_to_one')       # several makers -> several settlements; the first is the release
        O['n_settlements'] = O.n_settlements.fillna(0).astype(int)
        O['day'], O['rank'] = day, O.taker.map(rank)
        parts.append(O)
    A = pd.concat(parts, ignore_index=True)
    got = A.t_set.notna()
    C.require((A.mid[got] == A.mid_rest[got]).all() and (A.n_mid[got] == 1).all(), 'a settlement market differs from its order market')
    A['span_ms'] = A.t_last - A.t_set
    A['fire'] = A.t_set - WT.HOLD
    A['s_m50'] = A.t - WT.HOLD
    A['diff_ms'] = A.fire - A.s_m50
    A['before_exec_end'] = A.t_set < A.exec_s_rest * 1000. + 1000.          # C23: every settlement precedes its REST second's end
    A = A.sort_values(['day', 'taker', 'mid', 'fire'], kind='stable').reset_index(drop=True)
    gap = A.groupby(['day', 'taker', 'mid']).fire.diff()
    A['burst'] = (~(gap <= BURST_MS)).cumsum()
    out = dict(asof='2026-11-26', burst_ms=BURST_MS, groups={})
    for g, m in (('TOP1', A['rank'] == 1), ('TOP2_10', A['rank'] > 1)):
        G = A[m]
        Gg = G[G.t_set.notna()]
        out['groups'][g] = dict(orders=int(len(G)), with_settlement=int((G.n_settlements > 0).sum()), none=int((G.n_settlements == 0).sum()),
                                several=int((G.n_settlements > 1).sum()), several_span_gt0=int((Gg.span_ms > 0).sum()), several_span_gt10ms=int((Gg.span_ms > 10).sum()),
                                t_set_before_exec_second_end=float(Gg.before_exec_end.mean()),
                                fire_minus_m50_ms=qs(Gg.diff_ms), fire_minus_m50_ms_by_first_leg_timing={str(h): qs(Gg.diff_ms[Gg.how2_first == h]) for h in (1, 3)},
                                first_leg_timing={str(h): int((Gg.how2_first == h).sum()) for h in (1, 3)}, within_10ms=float((Gg.diff_ms.abs() <= 10).mean()), within_30ms=float((Gg.diff_ms.abs() <= 30).mean()),
                                bursts=int(Gg.burst.nunique()), orders_per_burst={str(k): int(v) for k, v in Gg.groupby('burst').size().value_counts().sort_index().items()},
                                by_day={dy: dict(orders=int(len(D)), with_fire=int(D.t_set.notna().sum()), fire_minus_m50_p50=float(D.diff_ms.median()) if D.t_set.notna().any() else None)
                                        for dy, D in G.groupby('day')})
    C.require(out['groups']['TOP1']['with_settlement'] > 0, 'no top taker order has a settlement')
    C.require(bool(A.before_exec_end[got].all()), 'a settlement after its executedAt second (C23 premise broken)')
    A.drop(columns=['mid_rest', 'n_mid']).to_parquet(fp, index=False)
    TR.dump(fj, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'scan':
        scan(a[1], *(a[2:3] + [int(a[3])] if len(a) > 3 else []))
    elif a[0] == 'join':
        join(*a[1:2])
