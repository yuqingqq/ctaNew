"""Recount only the 100-share arm of the frozen size sensitivity and save fill tapes.

FIFO matched holding time runs from the opening fill to each actual offsetting
fill, by share. Include unmatched opening shares in a second view at the
five-minute payout, two seconds after supported quoting ends. Report weighted
mean/median/p90, share over 30 s and matched P&L either side of 30 s. All 689
original complete markets, no new dates or strategy changes.
"""
import gzip
import json
import sys
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
original = run_path(str(HERE.parent / "urgency_size50_20261126/replay.py"))
run = original["run"]
state = run.__globals__
saved_replay = state["replay"]
saved_ledger = state["ledger"]


def capture(output):
    def replay(size, path, info, ident, flags, fit, payout):
        assert size == 100
        def ledger(fills, mark):
            fees_free_row = saved_ledger(fills, mark)
            settle_t = ident["opening_ms"] + 300000 - ident["start_abs_ms"]
            assert abs(settle_t - ident["end_rel"] - 2000) < 1e-6
            assert all(0 <= f[0] <= ident["end_rel"] for f in fills)
            output.write(json.dumps(dict(day=ident["day"], mid=ident["instrument_id"],
                                         settle_ms=settle_t, payout_c=mark,
                                         fills=fills), allow_nan=False, separators=(",", ":"))+"\n")
            return fees_free_row

        with patch.dict(state, ledger=ledger):
            return saved_replay(size, path, info, ident, flags, fit, payout)
    return replay


if __name__ == "__main__":
    root = HERE.parents[2] / "data/predict_fun/research/urgency100_holds_20261126"
    root.mkdir(exist_ok=True)
    prefix = "-".join(sys.argv[2:]) or "all"
    dest = root / f"{prefix}.jsonl.gz"
    if dest.exists():
        raise FileExistsError(dest)
    with TemporaryDirectory(dir=root) as temp:
        partial = Path(temp) / "capture.gz"
        with gzip.open(partial, "wt") as out, patch.dict(state, SIZES=(100,), HERE=Path(temp), replay=capture(out)):
            run()
        partial.rename(dest)
