"""When and how does the top taker exit its Predict positions? (USER 2026-11-26: "Can we check their exiting time at predict as well"; "Ok.
Then we check their exit times"). Readout on the four consumed days; no strategy change.
scan <day>: every REST match record (/v1/orders/matches, fire_times' hour stems) in which the top taker is the taker or a maker; one row per
leg (a maker order matched in the record), de-duplicated by transaction + maker order hash (earliest receipt); the settlement id time
(UUIDv7 ms; the match is ~34 ms earlier, FIRE_TIMING), the market, the top taker's role (taker / maker), its side in UP space (+1 = buys UP
or sells DOWN, -1 = sells UP or buys DOWN; as taker the opposite of the maker leg's side), the UP-space price (c) and the shares.
analyze (declared before the run): per market its net UP-space position through its legs in settlement order; a leg against the open
position closes it first-in first-out (an exit), the rest opens (an entry); round trips = entry piece and exit piece: holding time, entry
and exit role, net per share = side x (exit - entry) minus fees (2% x min(p, 100 - p) c per share on taker legs, 0 on maker legs); shares
still open at the market end are held to settlement (holding = window end - entry, value at the payout: 100 / 0, a 50 / 50 tie at 50).
Readings: legs and shares by role; entry volume by role and by ending (taker exit / maker exit / settlement); holding-time quantiles and
net per share by ending; the share of entry volume closed within 1 / 5 / 30 / 60 s; per taker order (its fires): the time to the first
exit of its shares and that exit's role.
Outputs (write-once): OUT/exit_legs_<day>.parquet + .json (scan), exit_times.json and OUT/exit_trips.parquet (analyze).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/exit_times.py \
       scan <day> [<rehearsal dir> <n stems>] | analyze [<rehearsal dir> <day> ...]
"""
import collections
import gzip
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import fire_times as FT  # noqa: E402

WT, TK, IO, C, TR = FT.WT, FT.TK, FT.IO, FT.C, FT.TR
OUT = WT.OUT
TOP = '0x7cab1c46a5B0557008198b92697F36df855ac0AD'
LAG = 34.
QS = (.1, .25, .5, .75, .9)


def qs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS} if len(x) else None


def wq(x, w):
    """Share-weighted quantiles."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return None
    o = np.argsort(x[ok]); xs, cw = x[ok][o], np.cumsum(w[ok][o])
    return {f'p{int(q * 100)}': float(xs[np.searchsorted(cw, q * cw[-1])]) for q in QS}


def yes(leg):
    side, py = IO.yes_side(leg)
    return (1 if side == 'bid' else -1), 100. * py


def scan(day, out_dir=None, n_stems=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in WT.DAYS, f'{day} is not a declared day')
    od = Path(out_dir) if out_dir else OUT
    fn, fj = od / f'exit_legs_{day}.parquet', od / f'exit_legs_{day}.json'
    C.require(not fn.exists() and not fj.exists(), f'{fn.name} exists (write-once)')
    top = TOP.lower()
    rows, cnt = {}, collections.Counter()
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
                if top not in line.lower():
                    continue
                r = json.loads(line)
                if r.get('status') != 200:
                    continue
                try:
                    pl = json.loads(r['raw'])
                except (TypeError, ValueError):
                    cnt['bad_raw'] += 1
                    continue
                for x in pl.get('data') or []:
                    t = x.get('taker') or {}
                    is_t = (t.get('signer') or '').lower() == top
                    mk = [m for m in x.get('makers') or [] if is_t or (m.get('signer') or '').lower() == top]
                    if not mk:
                        continue
                    sid, mid, tx = x.get('settlementId'), (x.get('market') or {}).get('id'), x.get('transactionHash') or ''
                    if not sid or mid is None:
                        cnt['no_settlement_or_market'] += 1
                        continue
                    for m in mk:
                        k = tx[2:18] + (m.get('hash') or '')[2:18]
                        if k in rows:
                            rows[k]['n_seen'] += 1
                            rows[k]['recv_ns'] = min(rows[k]['recv_ns'], int(r['recv_ns']))
                            continue
                        s, pc = yes(m)
                        role = 'taker' if is_t else 'maker'
                        cnt[f'{role}_legs'] += 1
                        cnt['index_set_types_' + type(m['outcome']['indexSet']).__name__] += 1
                        rows[k] = dict(day=day, mid=int(mid), t_set=FT.uuid7_ms(sid), sid=sid, role=role, side=-s if is_t else s, price_up_c=pc,
                                       shares=int(m['amount']) / IO.E18, taker_hash=t.get('hash'), maker_hash=m.get('hash'), tx=tx, recv_ns=int(r['recv_ns']), n_seen=1)
    S = pd.DataFrame(list(rows.values()))
    C.require(len(S) > 0, f'{day}: no legs')
    S.to_parquet(fn, index=False)
    TR.dump(fj, dict(asof='2026-11-26', day=day, stems=stems, **{k: int(v) for k, v in cnt.items()}, legs=int(len(S)), by_role={k: int(v) for k, v in S.role.value_counts().items()},
                     shares_by_role={k: float(v) for k, v in S.groupby('role').shares.sum().items()}, markets=int(S.mid.nunique())))
    print(day, dict(cnt), len(S), flush=True)


def fee(p):
    return .02 * np.minimum(p, 100. - p)


def trips_of_market(L, end, payout):
    """FIFO round trips of one market's legs (sorted by settlement time)."""
    lots = collections.deque()          # [t, price, role, taker_hash, shares, side]
    out = []
    for r in L.itertuples():
        q, s = float(r.shares), int(r.side)
        while q > 1e-9 and lots and lots[0][5] == -s:
            lot = lots[0]
            x = min(q, lot[4])
            net = lot[5] * (r.price_up_c - lot[1]) - (fee(lot[1]) if lot[2] == 'taker' else 0.) - (fee(r.price_up_c) if r.role == 'taker' else 0.)
            out.append(dict(entry_t=lot[0], exit_t=float(r.t_set), hold_ms=float(r.t_set) - lot[0], entry_role=lot[2], exit_role=r.role, ending=f'{r.role}_exit',
                            entry_price=lot[1], exit_price=float(r.price_up_c), side=lot[5], shares=x, net_c=net, taker_hash=lot[3]))
            lot[4] -= x; q -= x
            if lot[4] <= 1e-9:
                lots.popleft()
        if q > 1e-9:
            lots.append([float(r.t_set), float(r.price_up_c), r.role, r.taker_hash if r.role == 'taker' else None, q, s])
    for lot in lots:
        pay = 100. * payout
        out.append(dict(entry_t=lot[0], exit_t=end, hold_ms=end - lot[0], entry_role=lot[2], exit_role='settlement', ending='settlement', entry_price=lot[1],
                        exit_price=pay, side=lot[5], shares=lot[4], net_c=lot[5] * (pay - lot[1]) - (fee(lot[1]) if lot[2] == 'taker' else 0.), taker_hash=lot[3]))
    return out


def analyze(out_dir=None, days=None):
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    od = Path(out_dir) if out_dir else None
    fn, fp = (od or H) / 'exit_times.json', (od or OUT) / 'exit_trips.parquet'
    C.require(not fn.exists() and not fp.exists(), 'an output exists (write-once)')
    days = days or WT.DAYS
    L = pd.concat([pd.read_parquet(OUT / f'exit_legs_{d}.parquet') for d in days], ignore_index=True)
    n_raw = len(L)
    L = L.sort_values('recv_ns', kind='stable').drop_duplicates(['tx', 'maker_hash'], keep='first')   # a next-day 00-01Z hour is scanned under two days
    vs = json.loads((WT.VR.OUT / 'summary.json').read_text())
    pay = {int(r['mid']): r['payout'] for r in vs['rows']}
    reg = pd.concat([WT.MM.registry(d) for d in days])
    end = dict(zip(reg.mid.astype(int), reg.ws.astype(float) * 1000. + 300000.))
    L = L.sort_values(['mid', 't_set', 'recv_ns'], kind='stable')
    trips, skipped, used = [], collections.Counter(), 0.
    for m, g in L.groupby('mid'):
        if m not in end or pay.get(m) not in (0., .5, 1.):
            skipped['legs_outside_the_days_markets'] += len(g)
            continue
        trips += trips_of_market(g, end[m], pay[m])
        used += float(g.shares.sum())
    T = pd.DataFrame(trips)
    entered = T.shares.sum()
    closed = T[T.ending != 'settlement']
    C.require(abs(closed.shares.sum() * 2 + T[T.ending == 'settlement'].shares.sum() - used) < 1e-6 * max(1., used), 'shares do not add up: entries + exits != legs')
    w = lambda g: float(g.shares.sum())
    out = dict(asof='2026-11-26', days=list(days), legs_raw=int(n_raw), legs=int(len(L)), legs_by_role={k: int(v) for k, v in L.role.value_counts().items()},
               shares_by_role={k: float(v) for k, v in L.groupby('role').shares.sum().items()}, markets=int(L.mid.nunique()), skipped=dict(skipped),
               entries_by_role={k: w(g) / entered for k, g in T.groupby('entry_role')},
               endings={k: dict(share=w(g) / entered, hold_ms=wq(g.hold_ms, g.shares), net_c_per_share=float((g.net_c * g.shares).sum() / g.shares.sum()),
                                 n=int(len(g))) for k, g in T.groupby('ending')},
               endings_by_entry_role={er: {k: w(g) / w(ge) for k, g in ge.groupby('ending')} for er, ge in T.groupby('entry_role')},
               closed_within={f'{int(s)}s': float(T.shares[T.hold_ms <= s * 1000.].sum() / entered) for s in (1, 5, 30, 60)},
               hold_ms_weighted=dict(all=wq(T.hold_ms, T.shares), taker_entries=wq(T.hold_ms[T.entry_role == 'taker'], T.shares[T.entry_role == 'taker'])),
               net_c_per_share_by_entry_role={k: float((g.net_c * g.shares).sum() / g.shares.sum()) for k, g in T.groupby('entry_role')})
    E = T[(T.entry_role == 'taker') & T.taker_hash.notna()]
    first = E.sort_values('exit_t').groupby('taker_hash').agg(first_exit_ms=('hold_ms', 'min'), first_ending=('ending', 'first'), shares=('shares', 'sum'))
    out['per_taker_order'] = dict(orders=int(len(first)), first_exit_ms=qs(first.first_exit_ms), first_ending={k: float(v) for k, v in first.first_ending.value_counts(normalize=True).items()},
                                  fully_held_to_settlement=float((first.first_ending == 'settlement').mean()))
    T.to_parquet(fp, index=False)
    TR.dump(fn, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'scan':
        scan(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == 'analyze':
        analyze(sys.argv[2] if len(sys.argv) > 2 else None, tuple(sys.argv[3:]) or None)
