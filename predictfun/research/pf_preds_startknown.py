"""Start-price withhold (review check 1, declared in STATUS.yml 2026-11-04 13:41Z; the exec port's F1: features use each window's
start price from tau = ws, live learns it later). Known time of a window's start price = the first predictMarketChanged patch that
carries its startPrice (recorder raw predict_ws, this day's hours only), its venue tsMs put on the measured Tokyo clock
(pf_clock_rl.pf_known with the day's shift); no patch -> the registry's start_seen_ns (REST, later: a status). Rows with tau before
that time get v_bid = v_ask = -1 (the V0 guard withholds both sides); everything else unchanged. Writes step5/<day>/<src>k.parquet.
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_preds_startknown <src stem, e.g. preds_g60> <day> ...
"""
import sys, os, json, gzip
import numpy as np, pandas as pd
from . import pf_io as io
from . import pf_clock_rl as CK

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
SRC, DAYS = sys.argv[1], sys.argv[2:]
rep = {}
for d in DAYS:
    if d in ('20261028', '20261029', '20261030'):
        sys.exit(f'REFUSED: {d} is sealed')
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'start_price', 'start_seen_ns'])
    first, miss_h = {}, 0
    for h in range(24):
        p = f'{io.PF_RUN}/predict_ws/{d}_{h:02d}.jsonl.gz'
        if not os.path.exists(p):
            miss_h += 1; continue
        with gzip.open(p, 'rt') as fh:
            for line in fh:
                if 'predictMarketChanged' not in line or 'startPrice' not in line:
                    continue
                row = json.loads(line); data = json.loads(row['raw']).get('data') or {}
                sp = ((data.get('patch') or {}).get('variantData') or {}).get('startPrice')
                if sp is None or data.get('marketId') is None or data.get('tsMs') is None:
                    continue
                mid = int(data['marketId']); kn = float(CK.pf_known(float(data['tsMs']), row['recv_ns'], shift))
                if mid not in first or kn < first[mid][0]:
                    first[mid] = (kn, float(sp))
    known, n_ws, n_rest, n_mism = {}, 0, 0, 0
    for m, ws, sp, ss in zip(reg.mid.values, reg.ws.values, reg.start_price.values, reg.start_seen_ns.values):
        if int(m) in first:
            known[int(m)] = first[int(m)][0]; n_ws += 1; n_mism += int(np.isfinite(sp) and abs(first[int(m)][1] - sp) > 1e-6)
        else:
            known[int(m)] = float(ss) / 1e6; n_rest += 1
    lag = np.array([known[int(m)] - 1000.0 * ws for m, ws in zip(reg.mid.values, reg.ws.values)]) / 1000.0
    pr = pd.read_parquet(f'{io.OUT}/step5/{d}/{SRC}.parquet')
    tk = pr.mid.map(known).values.astype(float); no_reg = int(np.isnan(tk).sum())
    w = ~(pr.tau.values >= tk)                                  # before the start price is known (or window unknown): withhold
    for c in ('v_bid', 'v_ask'):
        pr[c] = np.where(w, np.float32(-1.0), pr[c].values).astype(np.float32)
    pr.to_parquet(f'{io.OUT}/step5/{d}/{SRC}k.parquet', index=False)
    rep[d] = dict(windows=len(reg), from_ws_patch=n_ws, from_rest_fallback=n_rest, price_mismatch=n_mism, missing_hours=miss_h,
                  rows_no_window=no_reg, rows_withheld=int(w.sum()), known_lag_s=dict(p10=float(np.quantile(lag, .1)), p50=float(np.median(lag)),
                  p90=float(np.quantile(lag, .9)), max=float(lag.max())))
    print(d, rep[d], flush=True)
io.write_json(f'{io.OUT}/step5/startknown_{SRC}_report.json', rep)
