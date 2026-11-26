"""Share-weighted FIFO inventory age from matched fill timestamps, with residuals at payout."""
import collections
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from runpy import run_path

HERE = Path(__file__).resolve().parent
inventory_pairs = run_path(str(HERE.parent / "urgency_size100_20261126/size_test.py"))["U"].C.inventory_pairs
RAW = HERE.parents[2] / "data/predict_fun/research/urgency100_holds_20261126/all.jsonl.gz"


def measure(fills, expiry, payout):
    # Rows are (seconds carried, shares, P&L cents).
    matched = []
    used = collections.Counter()
    for a, b, q in inventory_pairs(fills):
        age = (b[0] - a[0]) / 1000.
        assert age >= 0 and a[2] != b[2] and q > 0
        matched.append((age, q, q*(1-2*a[2])*(b[4]-a[4])))
        used[id(a)] += q
        used[id(b)] += q
    residual = []
    last = net = exposure = 0.
    for f in fills:
        t, _, side, q, price, _ = f
        assert last <= t <= expiry and q > 0 and side in (0, 1)
        exposure += abs(net)*(t-last)/1000.
        net += (1-2*side)*q
        last = t
        left = q-used[id(f)]
        assert left >= -1e-8
        if left > 1e-9:
            residual.append(((expiry-t)/1000., left, left*(1-2*side)*(payout-price)))
    exposure += abs(net)*(expiry-last)/1000.
    assert abs(sum(q for _,q,_ in residual)-abs(net)) < 1e-6
    assert abs(exposure - sum(t*q for pop in (matched, residual) for t,q,_ in pop)) < 1e-4
    return matched, residual


def stats(rows):
    n = sum(q for _,q,_ in rows)
    assert n > 0
    def quantile(p):
        cum = 0.
        for t,q,_ in sorted(rows):
            cum += q
            if cum >= p*n:
                return t
        raise AssertionError("empty quantile")
    return dict(shares=n, mean_s=sum(t*q for t,q,_ in rows)/n,
                median_s=quantile(.5), p90_s=quantile(.9),
                pct_over30=100*sum(q for t,q,_ in rows if t>30)/n,
                usd=sum(c for _,_,c in rows)/100,
                usd_over30=sum(c for t,_,c in rows if t>30)/100)


def controls():
    fills = [(0,1,0,4,55,0),(20000,1,0,3,58,1),(30000,2,1,5,60,2),
             (60000,3,1,5,54,3),(61000,4,0,1,50,4)]
    closed, open_lots = measure(fills, 90000, 100)
    assert closed == [(30.,4,20),(10.,1,2),(40.,2,-8),(1.,1,4)]
    assert open_lots == [(30.,2,-92)]
    sample = stats(closed)
    assert sample["median_s"] == 30 and sample["pct_over30"] == 25
    try:
        measure([fills[2], fills[0]], 90000, 100)
    except AssertionError:
        return
    raise AssertionError("reversed match times accepted")


def close_enough(a,b):
    return math.isclose(a, float(b), abs_tol=1e-6, rel_tol=0)


def analysis(raw):
    controls()
    reference = {}
    for path in sorted((HERE.parent/"urgency_size50_20261126").glob("markets_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["size"] == "100":
                    reference[(r["day"],int(r["mid"]))] = r
    seen = set()
    days = collections.defaultdict(lambda: ([],[]))
    with gzip.open(raw, "rt") as f:
        for line in f:
            r = json.loads(line)
            key = r["day"], r["mid"]
            assert key not in seen
            seen.add(key)
            ref = reference[key]
            assert len(r["fills"]) == int(ref["fill_events"])
            m,s = measure(r["fills"], r["settle_ms"], r["payout_c"])
            assert close_enough(sum(q for _,q,_ in m), ref["closed"])
            assert close_enough(sum(q for _,q,_ in s), ref["abs_end"])
            assert close_enough(sum(c for _,_,c in m)/100, ref["turnover"])
            assert close_enough(sum(c for _,_,c in s)/100, ref["residual"])
            days[key[0]][0].extend(m)
            days[key[0]][1].extend(s)
    assert seen == set(reference) and len(seen)==689
    groups = [(d, *g) for d,g in sorted(days.items())]
    groups.append(("all", [v for m,_ in days.values() for v in m],
                          [v for _,s in days.values() for v in s]))
    for day,m,s in groups:
        for name,rows in (("completed",m),("residual_at_payout",s),("all_opened",m+s)):
            yield dict(day=day,population=name,**stats(rows))


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        controls()
    else:
        result = list(analysis(RAW))
        with (HERE/"holding.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(result)
        print("fill journal sha256",hashlib.sha256(RAW.read_bytes()).hexdigest())
        for row in result[-3:]:
            print(row)
