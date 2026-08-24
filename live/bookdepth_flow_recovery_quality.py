"""Full structural and exact-flow audit for recovered v3 reaction data."""
from __future__ import annotations

import argparse, json, math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from live.bookdepth_flow_dynamics import _load_trades
from live.bookdepth_flow_recovery_build import DEFAULT_OUT, _all_local_symbols, _local_days

CORE = ["return_5min", "buy_to_ask_5min", "sell_to_bid_5min",
        "ask_depth_residual_5min", "bid_depth_residual_5min"]


def close(a, b, rtol=1e-9, atol=1e-10):
    return np.isclose(np.asarray(a, float), np.asarray(b, float),
                      rtol=rtol, atol=atol, equal_nan=True)


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)): return [jsonable(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): x = float(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, (Path, pd.Timestamp)): return str(x)
    return x


def scan_all(files_by_symbol):
    bad, infs, by_size, summaries = Counter(), Counter(), Counter(), []
    rows = files = 0
    for n, (symbol, paths) in enumerate(sorted(files_by_symbol.items()), 1):
        dataset = ds.dataset([str(p) for p in paths], format="parquet")
        table = dataset.to_table(columns=["__filename", *dataset.schema.names])
        d = table.to_pandas(split_blocks=True, self_destruct=True); del table
        rows += len(d)
        for col in d.select_dtypes(include=[np.number]):
            infs[col] += int(np.isinf(d[col].to_numpy(float, na_value=np.nan)).sum())
        counts = d.groupby("__filename", observed=True).size()
        files += len(counts); by_size.update(counts.astype(int).tolist())
        bad["file_count"] += abs(len(counts) - len(paths))
        bad["symbol"] += int(d.symbol.ne(symbol).sum())
        bad["duplicate_key"] += int(d.bar_time.duplicated().sum())
        bad["bar_alignment"] += int((d.bar_time.dt.floor("5min") != d.bar_time).sum())
        bad["snapshot_bounds"] += int(((d.snapshot_time < d.bar_time) |
            (d.snapshot_time >= d.bar_time + pd.Timedelta("5min"))).sum())
        path_day = pd.to_datetime(d.__filename.str.extract(
            r"/(\d{4}-\d{2}-\d{2})\.parquet$", expand=False), utc=True)
        bad["partition_day"] += int((d.bar_time.dt.floor("1D") != path_day).sum())
        bad["rows_above_288"] += int((counts > 288).sum())
        bad["negative_depth"] += int((d.bid1.lt(0) | d.ask1.lt(0)).sum())
        bad["nonpositive_price"] += int(d.price.le(0).sum())
        bad["imb_bounds"] += int(d.imb1.abs().gt(1 + 1e-12).sum())
        bad["gap_flag"] += int((d.gap_interval != d.interval_seconds.gt(90)).sum())

        identities = {
            "imb": (d.imb1, (d.bid1-d.ask1)/(d.bid1+d.ask1)),
            "ratio": (d.ask_bid_ratio, d.ask1/d.bid1.where(d.bid1.gt(0))),
            "pressure": (d.signed_pressure_5min, d.buy_to_ask_5min-d.sell_to_bid_5min),
            "ret": (d.return_5min, d.price/d.price_start_5min-1),
            "bidchg": (d.bid_change_5min, d.bid1/d.bid1_start_5min-1),
            "askchg": (d.ask_change_5min, d.ask1/d.ask1_start_5min-1),
            "imbchg": (d.imb_change_5min, d.imb1-d.imb1_start_5min),
            "buynorm": (d.buy_to_ask_5min, d.buy_quote_5min/d.ask1_start_5min),
            "sellnorm": (d.sell_to_bid_5min, d.sell_quote_5min/d.bid1_start_5min),
            "askres": (d.ask_depth_residual_5min,
                       d.ask_change_5min+d.buy_to_ask_5min),
            "bidres": (d.bid_depth_residual_5min,
                       d.bid_change_5min+d.sell_to_bid_5min),
        }
        for name, (a, b) in identities.items(): bad[f"identity_{name}"] += int((~close(a,b)).sum())

        expected_rows = d.groupby("__filename", sort=False).bar_time.transform("size")
        bad["day_count"] += int(d.source_day_bar_count.ne(expected_rows).sum())
        bad["source_complete"] += int((d.source_day_complete != d.source_day_bar_count.ge(280)).sum())
        bad["raw_gap_window"] += int((d.any_raw_gap_5min != d.gap_count_5min.gt(0)).sum())
        start = d.window_start_snapshot_time_5min
        start_stale = (d.snapshot_time-pd.Timedelta("5min")-start).dt.total_seconds()
        elapsed = (d.snapshot_time-start).dt.total_seconds()
        end_stale = (d.bar_time+pd.Timedelta("5min")-d.snapshot_time).dt.total_seconds()
        bad["start_stale"] += int((~close(d.window_start_staleness_seconds_5min,start_stale)).sum())
        bad["elapsed"] += int((~close(d.window_elapsed_seconds_5min,elapsed)).sum())
        bad["end_stale"] += int((~close(d.bar_end_staleness_seconds_5min,end_stale)).sum())
        sf = start.notna() & start_stale.between(0,90); ef = end_stale.between(0,90)
        endpoint = sf & ef
        bad["start_fresh"] += int((d.start_endpoint_fresh_5min != sf).sum())
        bad["end_fresh"] += int((d.end_endpoint_fresh_5min != ef).sum())
        bad["endpoint"] += int((d.endpoint_time_valid_5min != endpoint).sum())
        bad["flow_exact"] += int((d.flow_exact_5min != start.notna()).sum())
        extreme = d.imb1.abs().gt(.999) | d.imb1_start_5min.abs().gt(.999)
        valid = d[CORE].notna().all(axis=1) & endpoint & ~extreme
        bad["extreme"] += int((d.extreme_imbalance_5min != extreme).sum())
        bad["window_valid"] += int((d.window_data_valid_5min != valid).sum())
        bad["quality"] += int((d.quality_valid_5min != valid).sum())
        bad["recovered_gap"] += int((d.recovered_internal_gap_5min !=
                                      (valid & d.any_raw_gap_5min)).sum())
        cross = valid & start.notna() & (start.dt.floor("1D") < d.snapshot_time.dt.floor("1D"))
        bad["recovered_crossday"] += int((d.recovered_cross_day_5min != cross).sum())
        ask = valid & d.signed_pressure_5min.ge(.25) & d.ask_depth_residual_5min.gt(0) & d.return_5min.le(0)
        bid = valid & d.signed_pressure_5min.le(-.25) & d.bid_depth_residual_5min.gt(0) & d.return_5min.ge(0)
        bad["ask_candidate"] += int((d.ask_absorption_candidate_5min != ask).sum())
        bad["bid_candidate"] += int((d.bid_absorption_candidate_5min != bid).sum())
        summaries.append(dict(symbol=symbol, partitions=len(counts), rows=len(d),
            first_day=d.bar_time.min(), last_day=d.bar_time.max(),
            quality_valid=int(valid.sum()), quality_valid_rate=float(valid.mean()),
            raw_gap_windows=int(d.any_raw_gap_5min.sum()),
            recovered_internal_gap=int(d.recovered_internal_gap_5min.sum()),
            recovered_cross_day=int(d.recovered_cross_day_5min.sum()),
            stale_start=int((~d.start_endpoint_fresh_5min).sum()),
            stale_end=int((~d.end_endpoint_fresh_5min).sum()),
            extreme_windows=int(d.extreme_imbalance_5min.sum()),
            ask_candidates=int(d.ask_absorption_candidate_5min.sum()),
            bid_candidates=int(d.bid_absorption_candidate_5min.sum())))
        print(f"scan {n:03d}/{len(files_by_symbol)} {symbol}: {len(d):,} rows | "
              f"valid {valid.mean():.3%} | recovered {int(d.recovered_internal_gap_5min.sum()):,}", flush=True)
    return dict(rows=rows, files_seen=files, rows_per_file_distribution=dict(by_size),
                infinite_counts=dict(infs), violations=dict(bad)), pd.DataFrame(summaries)


def raw_check(symbol, path):
    day = pd.Timestamp(path.stem, tz="UTC"); d = pd.read_parquet(path)
    t = _load_trades(symbol, pd.DatetimeIndex([day-pd.Timedelta(days=1), day]))
    base = dict(symbol=symbol, day=path.stem, path=str(path), rows=len(d))
    if t.empty: return {**base, "status":"failed", "error":"no aggTrades context"}
    s = pd.DatetimeIndex(d.window_start_snapshot_time_5min).as_unit("ns")
    e = pd.DatetimeIndex(d.snapshot_time).as_unit("ns")
    valid = d.window_start_snapshot_time_5min.notna().to_numpy(); ii=np.flatnonzero(valid)
    tn = pd.DatetimeIndex(t.transact_time).as_unit("ns").asi8
    left=np.searchsorted(tn,s.asi8[ii],side="right"); right=np.searchsorted(tn,e.asi8[ii],side="right")
    quote=t.price.to_numpy(float)*t.quantity.to_numpy(float); maker=t.is_buyer_maker.to_numpy(bool)
    raw={"buy_quote_5min":np.where(~maker,quote,0.),"sell_quote_5min":np.where(maker,quote,0.),
         "buy_count_5min":(~maker).astype(float),"sell_count_5min":maker.astype(float)}
    mismatch=0; maxerr=0.
    for col,x in raw.items():
        prefix=np.r_[0.,np.cumsum(x)]; expected=prefix[right]-prefix[left]
        observed=d.loc[valid,col].to_numpy(float)
        mismatch += int((~np.isclose(observed,expected,rtol=1e-10,atol=1e-5)).sum())
        if len(expected): maxerr=max(maxerr,float(np.max(np.abs(observed-expected))))
    price=t.price.to_numpy(float); ep=np.searchsorted(tn,e.asi8,side="right")-1
    sp=np.searchsorted(tn,s.asi8[ii],side="right")-1
    mismatch += int((price[ep] != d.price.to_numpy(float)).sum())
    mismatch += int((price[sp] != d.loc[valid,"price_start_5min"].to_numpy(float)).sum())
    old=Path("/home/yuqing/ctaNew/data/ml/cache/research/bookdepth_flow_all_5min_v2")/symbol/path.name
    endpoint=0
    if old.exists():
        o=pd.read_parquet(old,columns=["snapshot_time","bid1","ask1","imb1"])
        m=d[["snapshot_time","bid1","ask1","imb1"]].merge(o,on="snapshot_time",suffixes=("_n","_o"),how="outer",indicator=True)
        endpoint += int(m._merge.ne("both").sum()); both=m._merge.eq("both")
        for col in ["bid1","ask1","imb1"]:
            endpoint += int((~close(m.loc[both,col+"_n"],m.loc[both,col+"_o"],1e-11,1e-8)).sum())
    return {**base,"status":"ok","error":"","trade_price_mismatches":mismatch,
            "v2_endpoint_mismatches":endpoint,"max_flow_abs_error":maxerr}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=str(DEFAULT_OUT)); ap.add_argument("--workers",type=int,default=4); a=ap.parse_args()
    root=Path(a.root); symbols=_all_local_symbols()
    expected={(s,d.strftime("%Y-%m-%d")) for s in symbols for d in _local_days(s,None,None)}
    files_by={s:sorted((root/s).glob("*.parquet")) for s in symbols if (root/s).exists()}
    actual={(p.parent.name,p.stem) for paths in files_by.values() for p in paths}
    manifest=pd.read_parquet(root/"_manifest.parquet")
    empty=set(map(tuple,manifest.loc[manifest.status.eq("empty"),["symbol","day"]].to_numpy()))
    missing=expected-actual; unexpected=missing-empty; extra=actual-expected
    print(f"full scan: {len(actual):,} partitions, {len(files_by)} symbols",flush=True)
    scan,symbol_report=scan_all(files_by)
    fractions=[.10,.35,.65,.90]; tasks=[]
    for i,(s,paths) in enumerate(sorted(files_by.items())):
        tasks.append((s,paths[min(len(paths)-1,int((len(paths)-1)*fractions[i%4]))]))
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(raw_check,s,p):s for s,p in tasks}
        for i,f in enumerate(as_completed(futures),1):
            try: rows.append(f.result())
            except Exception as exc: rows.append(dict(symbol=futures[f],status="failed",error=str(exc)))
            if i%20==0 or i==len(tasks): print(f"raw exact-flow {i}/{len(tasks)}",flush=True)
    raw=pd.DataFrame(rows).sort_values("symbol"); ok=raw.status.eq("ok")
    mismatches=int(raw.loc[ok,["trade_price_mismatches","v2_endpoint_mismatches"]].fillna(0).to_numpy().sum())
    violations={k:int(v) for k,v in scan["violations"].items() if int(v)}
    infinite={k:int(v) for k,v in scan["infinite_counts"].items() if int(v)}
    passed=not unexpected and not extra and not violations and not infinite and ok.all() and mismatches==0
    old=json.loads((Path("/home/yuqing/ctaNew/data/ml/cache/research/bookdepth_flow_all_5min_v2")/"_quality_report.json").read_text())["validity"]["quality_valid_rows"]
    quality=int(symbol_report.quality_valid.sum())
    report=dict(audit_version=3,root=str(root),verdict="pass" if passed else "fail",
      coverage=dict(symbols_expected=len(symbols),symbols_actual=len(files_by),expected_symbol_days=len(expected),
        actual_partitions=len(actual),missing_partitions=len(missing),missing_matching_manifest_empty=len(missing&empty),
        unexpected_missing_partitions=len(unexpected),extra_partitions=len(extra),unexpected_missing_examples=sorted(unexpected)[:20]),
      full_scan=scan,validity=dict(quality_valid_rows=quality,quality_valid_rate=quality/scan["rows"],
        v2_quality_valid_rows=old,quality_valid_lift_rows=quality-old,
        recovered_internal_gap_rows=int(symbol_report.recovered_internal_gap.sum()),
        recovered_cross_day_rows=int(symbol_report.recovered_cross_day.sum()),
        stale_start_rows=int(symbol_report.stale_start.sum()),stale_end_rows=int(symbol_report.stale_end.sum()),
        extreme_window_rows=int(symbol_report.extreme_windows.sum())),hard_violations=violations,infinite_nonzero=infinite,
      exact_flow_recompute=dict(requested=len(raw),successful=int(ok.sum()),failures=int((~ok).sum()),
        total_mismatches=mismatches,max_flow_abs_error=float(raw.loc[ok,"max_flow_abs_error"].max())),
      required_filters=["Use only quality_valid_5min.","Use snapshot_time as availability time.",
        "Construct the tradable universe point-in-time.","Recovered gaps provide net endpoint reaction, not intrawindow path data."])
    symbol_report.to_parquet(root/"_quality_symbol.parquet",index=False); raw.to_parquet(root/"_quality_recompute.parquet",index=False)
    (root/"_quality_report.json").write_text(json.dumps(jsonable(report),indent=2,sort_keys=True))
    print(f"verdict {report['verdict']} | rows {scan['rows']:,} | valid {quality:,} | recovered {report['validity']['recovered_internal_gap_rows']:,} | mismatches {mismatches}",flush=True)
    print("RECOVERYQUALITYDONE",flush=True)

if __name__ == "__main__": main()
