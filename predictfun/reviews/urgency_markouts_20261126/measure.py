"""Descriptive recount on the committed size-100 population. No strategy changes.

Value each engine fill against the last received two-sided midpoint at match
time plus 0/1/5/30 s. Exclude horizons beyond the supported endpoint and
report excluded shares. Positive cents are favorable to the resting side.
"""
import csv
import sys
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "urgency_size100_20261126"
S = run_path(str(PARENT / "size_test.py"))
R, U, OD = (S[key] for key in ("R", "U", "OD"))
HORIZONS = (0, 1, 5, 30)


def measure(fills, frames, end):
    frames = np.asarray(frames)
    clock = frames[:, 1]
    assert np.all(np.diff(clock) >= 0)
    rows = []
    for horizon in HORIZONS:
        row = dict(horizon_s=horizon, valid_shares=0., cents=0.,
                   after_end_shares=0., missing_book_shares=0.)
        for t, _, side, qty, price, _ in fills:
            target = t + horizon * 1000
            if target > end:
                row["after_end_shares"] += qty
                continue
            i = np.searchsorted(clock, target, side="right") - 1
            bid, ask = frames[i, 2], frames[i, 12]
            if i < 0 or not 0 <= bid < ask <= 100:
                row["missing_book_shares"] += qty
                continue
            row["valid_shares"] += qty
            row["cents"] += qty * (1 - 2 * side) * ((bid + ask) / 2 - price)
        rows.append(row)
    return rows


def controls():
    frames = np.zeros((3, 22))
    frames[:, 1] = [0, 1000, 1200]
    frames[:, 2] = [49, 51, -1]
    frames[:, 12] = [51, 53, 55]
    fills = [(0, 0, 0, 3, 49, 0), (0, 1, 1, 2, 53, 1),
             (200, 2, 1, 1, 51, 2)]
    rows = measure(fills, frames, 5000)
    assert rows[0]["cents"] == 10 and rows[1]["cents"] == 11
    assert rows[1]["missing_book_shares"] == 1 and rows[2]["after_end_shares"] == 1
    assert rows[3]["after_end_shares"] == 6
    try:
        measure(fills, frames[::-1], 5000)
    except AssertionError:
        return
    raise AssertionError("unordered receipt times admitted")


def run():
    controls()
    fit = U.S.read(R.HERE / "fit_v2.json")["cells"]
    src = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    census = U.S.read(R.CENSUS)["rows"]
    writer = None
    for day in S["DAYS"]:
        reference = pd.read_csv(PARENT / f"markets_{day}.csv", float_precision="round_trip")
        reference = reference.query("size == 100").set_index("mid", verify_integrity=True)
        original = S["archived"](day)
        assert set(original.index) == set(reference.index)
        reg = R.MM.registry(day).set_index("mid")
        rg = pd.read_parquet(R.MM.S1 / day / "registry.parquet").set_index("mid")
        off = src.offset(day)
        seg = OD.SEG_OF.get(day)
        tape_ids = original.tape_id.to_dict() if seg else {
            r["mid"]: r["tape_id"] for r in census if r["day"] == day and r["status"] == "complete"
        }
        assert set(tape_ids) == set(reference.index)
        seen = set()
        for _, group in reg.groupby("hour"):
            ids = sorted(set(group.index) & set(reference.index))
            if not ids:
                continue
            for mid, status, path, info in R.MM.worlds(day, ids, src, off, reg):
                assert status == "complete" and mid in ids and mid not in seen
                seen.add(mid)
                ident = R.MM.ident_of(day, mid, 1000. * reg.at[mid, "ws"], info, path)
                assert ident["tape_id"] == tape_ids[mid]
                start, end = info["t_start"], info["end_rel"]
                cache = OD.make_cache3(start, end, seg) if seg else R.MM.make_cache(start, end)
                marks = []
                finish = R.LB.finish

                def record(branch):
                    finish(branch)
                    marks.extend(measure(branch.E.fills, path["frames"], end))

                with patch.object(R.LB, "finish", record):
                    result = S["simulate"](path, info, ident, R.triggers(cache, edges), fit,
                                           R.VM.V.payout_of(rg.loc[mid]), 100)
                assert all(abs(result[k] - reference.at[mid, k]) < 1e-6 for k in S["FIELDS"])
                assert len(marks) == len(HORIZONS)
                for mark in marks:
                    assert abs(sum(v for k, v in mark.items() if k.endswith("_shares")) - result["shares"]) < 1e-6
                    row = dict(day=day, mid=mid, shares=result["shares"], **mark)
                    if writer is None:
                        writer = csv.DictWriter(sys.stdout, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
        assert seen == set(reference.index)
        sys.stdout.flush()
        print(day, len(seen), file=sys.stderr, flush=True)


if __name__ == "__main__":
    run()
