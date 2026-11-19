"""Bounded, review-only saved-artifact audit; no fitting, raw tape or validation reads.

Run with run_light.sh, PF_SLOT=codex. Imports only the detached target checkout.
"""
import gc
import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "edfeda22dd72aacaee0c758e96e6e29fe2667422"
PIN = Path("/tmp/pf-review-edfeda22")
ROOT = Path("/home/yuqing/ctaNew")
OUT = ROOT / "predictfun/reviews/automatic/20261119_edfeda22"
DATA = ROOT / "data/predict_fun/research"
PAIR = DATA / "pipeline_paired_20261119"
EXP = DATA / "pipeline_export_v2_20261119"
KP_DIR = DATA / "keep_pull_20261117"
EV = DATA / "pipeline_paired_eval_20261119"
BENCH = DATA / "pipeline_bench_v3_20261119"
sys.path.insert(0, str(PIN / "predictfun/research"))
import pf_pipeline_paired as PQ
import pf_keep_pull as KP


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


verified = {}


def checkhash(p, want):
    got = sha(p)
    assert got == want, (str(p), got, want)
    key = "pinned_source/" + str(p.relative_to(PIN)) if p.is_relative_to(PIN) else str(p.relative_to(ROOT))
    verified[key] = got


def save(name, obj):
    obj["review_asof"] = datetime.now(timezone.utc).isoformat()
    obj["target_commit"] = TARGET
    obj["review_script_sha256"] = sha(Path(__file__))
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def synthetic():
    # Same fill price and terminal mark, two decision mids: a physically coherent case.
    r = pd.DataFrame(dict(oid=[1, 1], j=[2, 3], value_m50=[10., 5.],
                          q_m50=[10., 5.], qm_m50=[10., -45.],
                          pi0_V=[1., -1.], pi0_ok=[True, True]))
    lab, ts, first = PQ.label_retain(r)
    post = pd.DataFrame(dict(oid=[1], status=["eligible"], value_m50=[10.],
                             q_m50=[10.], qm_m50=[10.]))
    plab, _ = PQ.label_post(post, first, r)
    assert lab["pair_v"][0] == plab["pair_v"][0] == 5.
    assert lab["pair_q"][0] == plab["pair_q"][0] == 5.
    assert lab["pair_qm"][0] == plab["pair_qm"][0] == 55.
    # Positive control: the old subtraction works when the mids are identical.
    same = r.copy(); same["qm_m50"] = same.value_m50
    sl, _, sf = PQ.label_retain(same); sp, _ = PQ.label_post(post, sf, same)
    assert sl["pair_qm"][0] == sp["pair_qm"][0] == 5.
    # A reached state must also belong to an admitted order.
    post_pred = pd.DataFrame(dict(oid=[1, 2], value=[-1., 1.]))
    ids, js, kv = np.array([1, 2]), np.array([2, 2]), np.array([-1., -1.])
    reported = PQ.reached(ids, js, kv)
    correct = reported & np.isin(ids, post_pred.oid[post_pred.value > 0])
    assert reported.tolist() == [True, True] and correct.tolist() == [False, True]
    assert (reported & np.isin(ids, post_pred.oid)).all()  # all admitted positive control
    return dict(movement=dict(p=50, mid_now=50, mid_future=60, END_mark=51,
                              current_q=10, future_q=5, pair_v=5, pair_q=5,
                              observed_pair_qm=55, expected_pair_qm=5,
                              same_mid_control=True, POST_and_RETAIN_reproduced=True),
                reach=dict(reported=reported.tolist(), actual=correct.tolist(),
                           all_admitted_control=True))


def manifests(pj):
    checkhash(PIN / "predictfun/research/pf_pipeline_paired.py", pj["source_sha256"])
    for name, h in pj["models"].items():
        checkhash(PAIR / "models" / name, h)
    for name, h in pj["outputs"].items():
        checkhash(PAIR / name, h)
    for day, h in pj["exports"].items():
        checkhash(EXP / day / "receipt.json", h)
    for day, h in pj["keep_pull"].items():
        checkhash(KP_DIR / day / "receipt.json", h)
    for day in PQ.EVAL:
        ej = read(EV / f"eval_{day}.json")
        for name, h in ej["outputs"].items():
            checkhash(EV / name, h)
        for key, p in (("paired", PAIR / "paired.json"),
                       ("predictions", PAIR / f"pred_{day}.parquet"),
                       ("bench", BENCH / "bench.json"),
                       ("bench_predictions", BENCH / f"pred_{day}.parquet"),
                       ("export_receipt", EXP / day / "receipt.json")):
            checkhash(p, ej["inputs"][key])
        checkhash(PIN / "predictfun/research/pf_pipeline_paired_eval.py", ej["source_sha256"])


def label_day(day):
    r = read(EXP / day / "receipt.json")
    parts = []
    for name, h in r["outputs"].items():
        checkhash(EXP / day / name, h)
        parts.append(pd.read_parquet(EXP / day / name, columns=["oid", "j", "p", "pf_mid"]))
    x = pd.concat(parts, ignore_index=True); del parts
    lab = pd.read_parquet(PAIR / f"labels_{day}.parquet")
    z = lab.merge(x, on=["oid", "j"], validate="one_to_one"); del x, lab
    edge = np.where(z.side == 0, 1., -1.) * (z.pf_mid - z.p)
    assert np.allclose(z.value_m50, edge * z.q_m50 + z.qm_m50, rtol=0, atol=1e-6)
    error = edge * z.pair_q + z.pair_qm - z.pair_v
    out = dict(rows=len(z), export_asof=r["asof"], statuses=z.status.value_counts().to_dict(), kinds={})
    for kind in ("post", "retain"):
        m = (z.kind == kind).values; bad = m & (np.abs(error.values) > 1e-6)
        ghost = bad & (np.abs(z.pair_q.values) < 1e-9)
        cols = ["oid", "j", "side", "p", "pf_mid", "tstar", "pair_v", "pair_q", "pair_qm"]
        examples = z.loc[bad, cols].copy(); examples["mapping_error_cents"] = error[bad]
        examples = examples.reindex(examples.mapping_error_cents.abs().sort_values(ascending=False).index).head(3)
        out["kinds"][kind] = dict(rows=int(m.sum()), mismatches=int(bad.sum()),
                                  zero_quantity_nonzero_movement=int(ghost.sum()),
                                  max_abs_error_cents=float(np.abs(error[m]).max()), examples=examples.to_dict("records"))
    # Independent window accounting of ALL saved paired value AND quantity labels.
    # Sorted observed fills and per-order prefix sums; no label arithmetic reused.
    kr = read(KP_DIR / day / "receipt.json")
    for name in ("orders.parquet", "fills.parquet"):
        checkhash(KP_DIR / day / name, kr["outputs"][name])
    o = pd.read_parquet(KP_DIR / day / "orders.parquet", columns=["oid", "side", "p", "mark"])
    f = pd.read_parquet(KP_DIR / day / "fills.parquet", columns=["oid", "clock", "t", "q"])
    f = f[f.clock == 50.].merge(o, on="oid", validate="many_to_one").sort_values(["oid", "t"])
    fg = {int(oid): a for oid, a in f.groupby("oid", sort=False)}
    maxv = maxq = 0.
    for oid, a in z.groupby("oid", sort=False):
        b = fg.get(int(oid)); vv = np.zeros(len(a)); qq = np.zeros(len(a))
        if b is not None:
            # Existing task's baseline: post effect +58; pull effect 50*j+37.
            lo = np.where(a.kind.values == "post", 58., a.j.values * 50. + 37.)
            hi = np.where(np.isfinite(a.tstar.values), a.tstar.values * 50. + 37., np.inf)
            ia = np.searchsorted(b.t.values, lo, side="left")
            ib = np.searchsorted(b.t.values, hi, side="left")
            cs = np.r_[0., np.cumsum(b.q.values)]
            qq = cs[ib] - cs[ia]
            vv = qq * (1. if b.side.iloc[0] == 0 else -1.) * (b.mark.iloc[0] - b.p.iloc[0])
        maxv = max(maxv, float(np.abs(vv - a.pair_v.values).max()))
        maxq = max(maxq, float(np.abs(qq - a.pair_q.values).max()))
    assert maxv < 1e-6 and maxq < 1e-6, (day, maxv, maxq)
    out["fill_reconciliation"] = dict(rows=len(z), max_value_error=maxv, max_quantity_error=maxq)
    print(day, "labels", json.dumps({k: {s: v for s, v in a.items() if s != "examples"} for k, a in out["kinds"].items()}), flush=True)
    return out


def evaluate(day, pj):
    cols = ["oid", "j", "kind", "side", "A_V", "AF_V", "AM_V", "pi0_V", "pi0_ok", "NEU_V", "NEU_ok", "pair_v"]
    p = pd.read_parquet(PAIR / f"pred_{day}.parquet", columns=cols)
    post, ret = p[p.kind == "post"].copy(), p[p.kind == "retain"].copy(); del p
    ej = read(EV / f"eval_{day}.json")
    out = dict(eval_asof=ej["asof"], original_shift=ej["shift"], groups={}, corrected_shift={})
    for side in (0, 1):
        pp = post[post.side == side]; rr = ret[ret.side == side]
        for head in ("A", "AF", "AM"):
            values = rr[f"{head}_V"].values
            old = PQ.reached(rr.oid.values, rr.j.values, values)
            sel = pp.oid[pp[f"{head}_V"] > 0]
            new = old & rr.oid.isin(sel).values
            d = dict(admissions=len(sel), reported_reached=int(old.sum()), actual_reached=int(new.sum()),
                     excluded_never_posted=int((old & ~new).sum()), groups={})
            for name, sign in (("KEEP", values > 0), ("PULL", values < 0)):
                m = new & sign
                d["groups"][name] = dict(n=int(m.sum()), predicted=float(values[m].sum()),
                                         realized=float(rr.pair_v.values[m].sum()),
                                         gate=PQ.gate_group(values[m], rr.pair_v.values[m], name == "KEEP"))
            oldrec = pj["diagnostics"][day][str(side)][head]
            assert int(old.sum()) == oldrec["reached_rows"]
            for name, sign in (("KEEP", values > 0), ("PULL", values < 0)):
                m = old & sign; saved = oldrec["groups"][f"{name} selected"]
                assert saved["n"] == int(m.sum())
                assert abs(saved["realized"] - float(rr.pair_v.values[m].sum())) < 1e-6
            out["groups"][f"{head}_{side}"] = d
    old = PQ.reached(ret.oid.values, ret.j.values, ret.A_V.values)
    reached = old & ret.oid.isin(post.oid[post.A_V > 0]).values
    out["corrected_shift"] = dict(reached_rows=int(reached.sum()), excluded_never_posted=int((old & ~reached).sum()),
                                   mean_pred=float(ret.A_V.values[reached].mean()), mean_label=float(ret.pair_v.values[reached].mean()))
    assert ej["shift"]["reached_rows"] == int(old.sum())
    # Independent one-fill-one-contribution replay at B; no production outcome function.
    o = pd.read_parquet(KP_DIR / day / "orders.parquet")
    f = pd.read_parquet(KP_DIR / day / "fills.parquet", columns=["oid", "clock", "t", "q"])
    f = f[f.clock == 50.].merge(o[["oid", "side", "p", "mark"]], on="oid", validate="many_to_one")
    fv = f.q.values * np.where(f.side.values == 0, 1., -1.) * (f.mark.values - f.p.values)
    bp = pd.read_parquet(BENCH / f"pred_{day}.parquet", columns=["oid", "j", "kind", "J_V", "J_ok"])
    post = post.merge(bp[bp.kind == "post"][["oid", "J_V", "J_ok"]], on="oid", validate="one_to_one")
    ret = ret.merge(bp[bp.kind == "retain"][["oid", "j", "J_V", "J_ok"]], on=["oid", "j"], validate="one_to_one"); del bp
    results = {}
    for name, col, ok in (("A", "A_V", None), ("AF", "AF_V", None), ("AM", "AM_V", None),
                          ("PI0", "pi0_V", "pi0_ok"), ("NEU", "NEU_V", "NEU_ok"), ("E1", "J_V", "J_ok")):
        ids = post.oid[(post[col] > 0) & (post[ok].astype(bool) if ok else True)]
        pull = ret[(ret[col] < 0) & (ret[ok].astype(bool) if ok else True)].groupby("oid").j.min()
        land = f.oid.map(pull).fillna(np.inf).values * 50. + 37.
        admission = f.oid.isin(ids).values
        for pol, kept in ((name, admission & (f.t.values < land)), (name + "-PA", admission & (f.t.values < 137.))):
            vals = {}
            for sd, s in (("both", None), ("bid", 0), ("ask", 1)):
                m = kept & ((f.side.values == s) if s is not None else True)
                v = float(fv[m].sum()); want = ej["values"][pol][sd]["M50|A58|L37|N100"]
                assert abs(v - want) < 1e-6, (day, pol, sd, v, want)
                vals[sd] = v
            results[pol] = dict(admissions=len(ids), values=vals)
    pa = float(fv[f.t.values < 137.].sum())
    assert abs(pa - ej["values"]["PA"]["both"]["M50|A58|L37|N100"]) < 1e-6
    results["PA"] = dict(admissions=len(post), values=dict(both=pa))
    out["baseline_fill_replay"] = dict(original_orders=len(o), path_orders=len(post), m50_fills=len(f), policies=results)
    out["reported_metrics_A"] = ej["metrics"]["A"]
    print(day, "values", {k: v["values"]["both"] for k, v in results.items()}, flush=True)
    print(day, "reached", out["corrected_shift"], flush=True)
    return out


def main():
    mode = sys.argv[1]
    if mode == "synthetic":
        s = synthetic()
        try:
            runpy.run_path(str(PIN / "predictfun/reviews/pipeline_one_day_v3_20261119/g0_coherence_cases.py"), run_name="__main__")
        except SystemExit as e:
            assert e.code == 0
        s["G0_boundary_cases_passed"] = 12
        save("synthetic.json", s)
    elif mode == "artifacts":
        pj = read(PAIR / "paired.json"); manifests(pj)
        result = dict(paired_asof=pj["asof"], labels={}, evaluation={})
        for day in (*PQ.LABEL, *PQ.EVAL):
            result["labels"][day] = label_day(day); gc.collect()
            if day in PQ.EVAL:
                result["evaluation"][day] = evaluate(day, pj); gc.collect()
        result["verified_hashes"] = verified
        save("receipt.json", result)
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    main()
