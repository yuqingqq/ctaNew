"""Apply the historical recount's horizon rules to the published Tokyo estimates.

Arguments are the root of the extracted pmmmrust code receipt and the Tokyo
shadow run directory. The original estimator still owns all inferred fills.
"""
import bisect
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from subprocess import check_output
from unittest.mock import patch

root, run = (Path(v).resolve() for v in sys.argv[1:])
repo = Path(__file__).resolve().parents[3] / "pmmmrust"
source = "29741ec4182bd278b44d1af1c8dbd3d20033db59"
for name in ("scripts/pf_urgency_estimate.py", "config/predict/passive_urgency_100_v1/fit.json"):
    assert (root / name).read_bytes() == check_output(["git", "show", f"{source}:{name}"], cwd=repo)
data = "391f89b96f3d3a21d273355f5cc8873ea9ea58a2"
folder = "tokyo-urgency100-shadow-20261126"
manifest = check_output(["git", "show", f"{data}:{folder}/MANIFEST.sha256"], cwd=repo).decode()
for name in ("inputs.jsonl.gz", "outputs.jsonl.gz"):
    expected, = [l.split()[0] for l in manifest.splitlines() if l.split()[-1] == f"./{run.name}/{name}"]
    assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected
spec = importlib.util.spec_from_file_location("est", root / "scripts/pf_urgency_estimate.py")
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)
capture = {}
simulate = est.simulate


def capture_fills(orders, frames, accepted, removed):
    result = simulate(orders, frames, accepted, removed)
    market, = {o["market"] for o in orders}
    assert market not in capture
    capture[market] = frames, result[0]
    return result


with patch.object(est, "simulate", capture_fills), patch.object(sys, "argv", ["estimate", str(run)]):
    with contextlib.redirect_stdout(io.StringIO()):
        est.main()
receipt_path = "docs/receipts/20261126_urgency100_shadow_review/shadow_estimate.json"
receipt = json.loads(check_output(["git", "show", f"{source}:{receipt_path}"], cwd=repo))
assert set(capture) == {r["market"] for r in receipt["markets"]}
writer = None
for market in receipt["markets"]:
    frames, modes = capture[market["market"]]
    clock = [f[0] for f in frames]
    assert clock == sorted(clock)
    for mode in ("low_capped", "high_capped"):
        legs = modes[mode]
        published = market[mode]
        total = sum(q for _, q, _ in legs)
        assert abs(total - published["up_shares"] - published["down_shares"]) < 1e-6
        for horizon in (0, 1, 5, 30):
            eligible = missing = censored = cents = 0.
            for t, qty, o in legs:
                target = t + horizon * 1000.
                if target > 1000. * market["ws_s"] + 298000.:
                    censored += qty
                    continue
                i = bisect.bisect_right(clock, target) - 1
                if i < 0 or not frames[i][1] or not frames[i][2]:
                    missing += qty
                    continue
                bid, ask = max(frames[i][1]), min(frames[i][2])
                if not 0 <= bid < ask <= 100:
                    missing += qty
                    continue
                eligible += qty
                cents += qty * (1 - 2 * o["side"]) * ((bid + ask) / 2 - o["tick"])
            assert abs(eligible + missing + censored - total) < 1e-6
            row = dict(mid=market["market"], mode=mode, horizon_s=horizon, shares=total,
                       valid_shares=eligible, cents=cents, missing_book_shares=missing,
                       after_end_shares=censored)
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
