"""TAKE-EXIT review sensitivity S1 (declared in review_plan.md before running): the v2 PASSIVE_EXIT arm re-run on the same 281 complete
markets with a STRICT IOC availability rule. An IOC may only take the liquidity at its fill price p that is still displayed in the first
venue frame after its effect, plus public trades at p in (effect, that frame] (later takers: we came first). A quote that vanished
without trades inside the frame interval is treated as cancelled before our arrival. This is a lower bound on take fills; v2's lenient
rule (the last frame before the effect only) is the upper bound. Everything else unchanged. Parity: with the cap off the engine must
reproduce the v2 ledger exactly (market 3697902).
Usage: run_light.sh txrv_strict -- python3 predictfun/reviews/take_exit_review_20261125/strict_rerun.py parity|run|summary
"""
import gzip
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'take_exit_replay_20261125'))
import more_markets as MM  # noqa: E402

ER, TR = MM.ER, MM.TR
T, X, LB, BE = TR.T, TR.X, TR.LB, TR.BE
OUT3 = X.DATA / 'take_exit_review_20261125'
EPS = 1e-9


class StrictIOCEngine(TR.SignalIOCEngine):
    """The pilot's execute_ioc, unchanged except for the strict cap on the filled quantity (strict=False: exact parent behaviour)."""
    strict = True

    def execute_ioc(self, gid):
        a = self.iocs[gid]; t = a['act']; s = a['side']; j = int(np.searchsorted(self.f[:, 0], t, side='right')) - 1
        row = dict(a, filled=0., price=None, fee_c=0., frame=j, reason='empty', available_before=None, public_debit=None)
        pos = sum((1 - 2 * f[2]) * f[3] for f in self.fills); row['venue_before'] = pos
        if j < 0:
            row['reason'] = 'no_venue_book'
        else:
            f = self.f[j]; opp = 1 - s; p = float(f[2 + opp * 10]); row['venue_frame_ms'] = float(f[0]); row['venue_received_ms'] = float(f[1])
            if not (0 <= f[2] < f[12] <= 100) or t - f[0] > 1000:
                row['reason'] = 'invalid_or_stale'
            elif not 1 <= p <= 99:
                row['reason'] = 'price_boundary'
            elif (p > a['limit'] if s == 0 else p < a['limit']):
                row['reason'] = 'limit_moved'
            elif any(o['gid'] not in self.iocs and o['side'] == opp and o['rem'] > EPS and o['act'] <= t < o['remove'] and
                     (o['price'] <= a['limit'] if s == 0 else o['price'] >= a['limit']) for o in self.o):
                row['reason'] = 'own_cross'
            else:
                available, j, debit = self.liquidity(t, opp, p); q = min(a['q'], available)
                row.update(available_before=available, public_debit=debit, frame=j, price=p, lenient_q=q)
                if self.strict and q > EPS:
                    j2 = int(np.searchsorted(self.f[:, 0], t, side='right'))
                    if j2 >= len(self.f):
                        q2 = 0.
                    else:
                        f2 = self.f[j2]; pr = f2[2 + opp * 10: 7 + opp * 10]; sz = f2[7 + opp * 10: 12 + opp * 10]
                        left = float(sz[np.abs(pr - p) < EPS].sum())
                        tr = self.tr; m_ = (tr[:, 0] > t) & (tr[:, 0] <= f2[0]) & (tr[:, 1] == opp) & (np.abs(tr[:, 2] - p) < EPS)
                        q2 = left + float(tr[m_, 3].sum())
                    row['strict_cap'] = q2
                    if q2 < q - EPS:
                        q = max(0., q2); row['strict_cut'] = True
                if q > EPS:
                    self.available[(opp, p)] -= q
                    self.fills.append((t, gid, s, q, p, -1)); fee = T.fee_c(p, q, self.fee_bps); self.fees.append((t, gid, fee))
                    row.update(filled=q, fee_c=fee, reason='filled' if q >= a['q'] - EPS else 'partial')
                elif row.get('strict_cut'):
                    row['reason'] = 'strict_cancelled'
        row['venue_after'] = pos + (1 - 2 * s) * row['filled']; row['absolute_risk_reduced'] = abs(row['venue_after']) < abs(pos) - EPS
        self.exec_log.append(row); a['done'] = True; self.pending.remove(gid)


class StrictBranch(TR.SignalBranch):
    def __init__(self, path, end, ident, bps, host, take, strict=True):
        super().__init__(path, end, ident, bps, host, take)
        assert not self.E.o
        self.E = StrictIOCEngine(path, end, bps)
        self.E.strict = bool(strict)


def run_one(path, cache, old, edges, bps, strict=True):
    """exit_replay.run_one for PASSIVE_EXIT with the strict branch (same steps, same metrics)."""
    orig = TR.SignalBranch
    TR.SignalBranch = lambda *a, **k: StrictBranch(*a, strict=strict, **k)    # ER.run_one constructs TR.SignalBranch
    try:
        return ER.run_one(path, cache, old, edges, bps, 'PASSIVE_EXIT')
    finally:
        TR.SignalBranch = orig


def worlds_selected():
    cen = json.loads((MM.OUT2 / 'census.json').read_text())
    sel = {(d, int(m)) for d, m in cen['selected']}
    src = MM.KP.Tape()
    for day in MM.DAYS:
        reg = MM.registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            want = [int(x) for x in g.index if (day, int(x)) in sel]
            if want:
                for m, st, p, info in MM.worlds(day, want, src, off, reg):
                    yield day, h, m, p, info, reg


def inputs_for(day, m, p, info, reg):
    ws = float(reg.loc[m, 'ws']) * 1000.
    return MM.make_cache(float(info['t_start']), float(info['end_rel'])), dict(end=float(info['end_rel']), ident=MM.ident_of(day, m, ws, info, p))


def parity():
    edges = TR.T.P.frozen_edges()
    for day, h, m, p, info, reg in worlds_selected():
        if m != 3697902:
            continue
        cache, old = inputs_for(day, m, p, info, reg)
        rec = json.loads(json.dumps(run_one(p, cache, old, edges, 200., strict=False), allow_nan=False))
        prev = json.loads(gzip.decompress((MM.OUT2 / f'{m}_PASSIVE_EXIT.json.gz').read_bytes()))
        same = {k: rec[k] == prev[k] for k in ['fills', 'orders', 'commands', 'value', 'cash', 'position', 'signal_log']}
        print('parity strict=False vs v2', same)
        assert all(same.values())
        return


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (600, 601))
    OUT3.mkdir(parents=True, exist_ok=True)
    assert not (OUT3 / 'strict_summary.json').exists()
    edges = TR.T.P.frozen_edges()
    rows, t0 = [], time.process_time()
    for day, h, m, p, info, reg in worlds_selected():
        cache, old = inputs_for(day, m, p, info, reg)
        rec = run_one(p, cache, old, edges, 200., strict=True)
        fn = OUT3 / f'{m}_PASSIVE_EXIT_STRICT.json.gz'
        assert not fn.exists()
        fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
        cut = sum(1 for x in rec['ioc_results'] if x.get('strict_cut'))
        rows.append(dict(day=day, mid=m, hour=int(h), metrics=rec['metrics'], strict_cuts=cut, sha256=TR.sha(fn)))
    TR.dump(OUT3 / 'strict_summary.json', dict(rows=rows, cpu_s=time.process_time() - t0, status='COMPLETE'))
    print('done', len(rows), round(time.process_time() - t0, 1))


def summary():
    s = json.loads((OUT3 / 'strict_summary.json').read_text())
    v2 = pd.read_csv(H.parent / 'take_exit_replay_20261125/v2_per_market.csv')
    v2 = v2[v2.arm == 'PASSIVE_EXIT'].set_index(['day', 'mid'])
    df = pd.DataFrame([dict(day=r['day'], mid=r['mid'], strict_cuts=r['strict_cuts'], **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)}) for r in s['rows']])
    df = df.set_index(['day', 'mid'])
    dd = df.groupby('day').net_c.sum()
    out = dict(markets=int(len(df)), net_usd=float(df.net_c.sum() / 100), per_market_usd=float(df.net_c.mean() / 100), days_positive=int((dd > 0).sum()),
               day_net_usd={k: float(v / 100) for k, v in dd.items()}, markets_positive=int((df.net_c > 0).sum()), completed_usd=float(df.completed_c.sum() / 100),
               residual_usd=float(df.residual_c.sum() / 100), fees_usd=float(df.fees_c.sum() / 100), takes_filled=int(df.takes_filled.sum()),
               taker_shares=float(df.taker_shares.sum()), strict_cuts=int(df.strict_cuts.sum()),
               net_without_young_exit_value_usd=float((df.net_c.sum() - df.young_exit_value_c.sum()) / 100),
               v2_lenient_net_usd=float(v2.net_c.sum() / 100), v2_lenient_takes_filled=int(v2.takes_filled.sum()), v2_lenient_taker_shares=float(v2.taker_shares.sum()),
               markets_better_strict=int((df.net_c > v2.net_c.reindex(df.index)).sum()), cpu_s=s['cpu_s'])
    (H / 'strict_readings.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    {'parity': parity, 'run': run, 'summary': summary}[sys.argv[1]]()
