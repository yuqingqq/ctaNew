"""TAKE-SIGNAL v1 stage 1: BTCUSDT futures bookTicker mid-change events with bn_r250 / bn_r1000 (contract.md), Oct 24-27.
Day rule: the hour stem must be on the allow-list (20261023_23 + every hour of 20261024..27) BEFORE any path is built; nothing else is
opened, listed or globbed. Mid = (bid + ask) / 2 of valid rows (bid > 0, ask > bid), deduplicated by update id (first kept), state at a
time = last row in (E, u) order. A 10 s raw carry across files handles the collector's cross-hour flush. Clock k = E + B_IN.
Output (write-once): data/predict_fun/research/take_signal_20261125/bn_changes.parquet (k_ms at B_IN = 0, mid) + receipt bn_events.json.
Usage: run_light.sh ts_bn -- python3 predictfun/reviews/take_signal_20261125/bn_events.py
"""
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
BN = R / 'data/mm_hf/raw/bookTicker/BTCUSDT'
OUT = R / 'data/predict_fun/research/take_signal_20261125'
DAYS = ('20261024', '20261025', '20261026', '20261027')
ALLOWED = ('20261023_23',) + tuple(f'{d}_{h:02d}' for d in DAYS for h in range(24))
CARRY_MS = 10_000


class Forbidden(Exception):
    pass


def path_of(stem):
    if stem not in ALLOWED:
        raise Forbidden(f'{stem!r} is not on the allow-list: nothing opened')
    return BN / f'{stem}.csv.gz'


def changes(k, u, m):
    """Change points of the state sequence ordered by (k, u): the last row per ms, kept where mid differs from the previous state."""
    o = np.lexsort((u, k))
    k, u, m = k[o], u[o], m[o]
    last = np.r_[k[1:] != k[:-1], True]
    k, m = k[last], m[last]
    ch = np.r_[True, m[1:] != m[:-1]]
    return k[ch], m[ch]


def main():
    t0 = time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'bn_changes.parquet'
    assert not out.exists(), 'write-once'
    rec = dict(files={}, rows=0, dup_u=0, invalid=0)
    buf = None
    emitted_k, emitted_m = [], []
    emit_lo = -np.inf
    last_mid = None
    for stem in ALLOWED:
        p = path_of(stem)
        raw = p.read_bytes()
        rec['files'][stem] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        cols = dict(header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'u', 'bid', 'ask'], on_bad_lines='skip')
        try:
            df = pd.read_csv(p, dtype={'E': 'float64', 'u': 'float64', 'bid': 'float64', 'ask': 'float64'}, **cols)
        except ValueError:                                         # a non-numeric field: the builder's coercing parse, counted
            df = pd.read_csv(p, dtype=str, **cols).apply(pd.to_numeric, errors='coerce')
            rec['coerced_files'] = rec.get('coerced_files', 0) + 1
        n0 = len(df)
        df = df.dropna()
        ok = (df.bid > 0) & (df.ask > df.bid)
        rec['invalid'] += int(n0 - ok.sum())
        df = df[ok]
        nd = len(df)
        df = df.drop_duplicates('u', keep='first')
        rec['dup_u'] += int(nd - len(df))
        rec['rows'] += len(df)
        rec['files'][stem]['rows'] = len(df)
        k = df.E.to_numpy(np.int64)
        u = df.u.to_numpy(np.int64)
        m = ((df.bid.to_numpy() + df.ask.to_numpy()) / 2)
        if buf is not None:
            k, u, m = np.r_[buf[0], k], np.r_[buf[1], u], np.r_[buf[2], m]
            uniq, first = np.unique(u, return_index=True)          # a u already in the carry keeps its first copy
            rec['dup_u'] += int(len(u) - len(uniq))
            k, u, m = k[first], u[first], m[first]
        ck, cm = changes(k, u, m)
        hi = k.max() - CARRY_MS if stem != ALLOWED[-1] else np.inf
        sel = (ck > emit_lo) & (ck <= hi)
        # the state at emit_lo must continue: drop a first emitted point that repeats the last emitted mid
        if last_mid is not None and sel.any():
            first_i = int(np.argmax(sel))
            if cm[first_i] == last_mid:
                sel[first_i] = False
                rec['boundary_repeats'] = rec.get('boundary_repeats', 0) + 1
        emitted_k.append(ck[sel])
        emitted_m.append(cm[sel])
        if sel.any():
            last_mid = cm[sel][-1]
        if np.isfinite(hi):
            emit_lo = hi
        keep = k > k.max() - 2 * CARRY_MS
        buf = (k[keep], u[keep], m[keep])
        mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        assert mib < 2500, f'VmHWM guard {mib:.0f} MiB at {stem}'
    K = np.concatenate(emitted_k)
    M = np.concatenate(emitted_m)
    assert np.all(np.diff(K) > 0), 'change points must be strictly increasing in time'
    assert np.all(M[1:] != M[:-1]), 'consecutive change points must differ'
    pd.DataFrame(dict(k_ms=K, mid=M)).to_parquet(out, index=False)
    rec.update(changes=int(len(K)), first_k=int(K[0]), last_k=int(K[-1]), cpu_s=round(time.process_time() - t0, 2),
               vmhwm_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               out_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (H / 'bn_events.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'files'}))


if __name__ == '__main__':
    main()
