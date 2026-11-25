"""TAKE-EXIT review sensitivity S2 (review_plan.md): S1's strict IOC rule with the measured 58 ms order-live delay; PASSIVE_EXIT, 281 markets.
Usage: run_light.sh txrv_s2 -- python3 predictfun/reviews/take_exit_review_20261125/strict58.py run|summary"""
import gzip, json, resource, sys, time
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import strict_rerun as S  # noqa: E402
S.TR.TAKE_DELAY = 58.                                   # read at call time by SignalIOCEngine.submit_ioc and the near_end check
assert S.TR.TAKE_DELAY == 58.


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (450, 451))
    assert not (S.OUT3 / 's2_summary.json').exists()
    edges = S.TR.T.P.frozen_edges(); rows = []; t0 = time.process_time()
    for day, h, m, p, info, reg in S.worlds_selected():
        cache, old = S.inputs_for(day, m, p, info, reg)
        rec = S.run_one(p, cache, old, edges, 200., strict=True)
        assert all(abs(v['act'] - v['submit'] - 58.) < 1e-9 for v in rec['iocs'].values())
        fn = S.OUT3 / f'{m}_PASSIVE_EXIT_STRICT58.json.gz'; assert not fn.exists()
        fn.write_bytes(gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0))
        rows.append(dict(day=day, mid=m, metrics=rec['metrics'], sha256=S.TR.sha(fn)))
    S.TR.dump(S.OUT3 / 's2_summary.json', dict(rows=rows, cpu_s=time.process_time() - t0, status='COMPLETE'))
    print('done', len(rows), round(time.process_time() - t0, 1))


def summary():
    s = json.loads((S.OUT3 / 's2_summary.json').read_text())
    df = pd.DataFrame([dict(day=r['day'], mid=r['mid'], **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)}) for r in s['rows']])
    dd = df.groupby('day').net_c.sum()
    out = dict(markets=len(df), net_usd=float(df.net_c.sum() / 100), per_market_usd=float(df.net_c.mean() / 100), days_positive=int((dd > 0).sum()),
               day_net_usd={k: float(v / 100) for k, v in dd.items()}, markets_positive=int((df.net_c > 0).sum()), takes_filled=int(df.takes_filled.sum()),
               taker_shares=float(df.taker_shares.sum()), fees_usd=float(df.fees_c.sum() / 100), completed_usd=float(df.completed_c.sum() / 100),
               net_without_young_exit_value_usd=float((df.net_c.sum() - df.young_exit_value_c.sum()) / 100), cpu_s=s['cpu_s'])
    (H / 's2_readings.json').write_text(json.dumps(out, indent=1, default=float) + '\n'); print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    {'run': run, 'summary': summary}[sys.argv[1]]()
