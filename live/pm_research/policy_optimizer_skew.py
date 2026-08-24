"""Policy-optimizer Stage B: incentive-free pessimistic skew replay.

This is a stateful offline replay, not a live decision or execution module. It
runs the six skew cells already declared in POLICY_OPTIMIZER_PROTOCOL.md and
uses `SKEW_LB`: front only on genuine level formation and rejoin behind
displayed depth after a full lift.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import flow_intensity as fi
import policy_bounds_v1 as pb
import policy_optimizer as opt
import warning_window as ww

OUT = fi.PM / "derived/policy_optimizer_stageB_skew_v1.json"
STAGE_A_RECEIPT = fi.PM / "derived/policy_optimizer_stageA.json"
PLAN = Path(__file__).with_name("POLICY_OPTIMIZER_PROTOCOL.md")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _fill_net(wf: Any) -> float:
    bought = sum(f.size for f in wf.fills if f.maker_side == "BUY_UP")
    sold = sum(f.size for f in wf.fills if f.maker_side == "SELL_UP")
    return bought - sold


def _cash_at_risk(wf: Any) -> float:
    net = _fill_net(wf)
    mid = wf.mid_v[-1]
    return net * mid if net > 0 else -net * (1.0 - mid)


def _cell_metrics(rows: Sequence[Sequence[Any]],
                  windows: Sequence[Any]) -> dict[str, float | int | None]:
    flat = [row for group in rows for row in group]
    nets = np.asarray([abs(_fill_net(wf)) for wf in windows], dtype=float)
    cash = np.asarray([_cash_at_risk(wf) for wf in windows], dtype=float)
    return {
        "n_windows": len(rows),
        "pnl_per_window_cents": (lambda value: None if value is None
                                 else round(value, 2))(
            opt.total_pnl_per_window(rows)),
        "shares_per_window": round(
            sum(row.size for row in flat) / max(1, len(rows)), 1),
        "swm_cents": (lambda value: None if value is None
                      else round(value, 4))(pb.swm(flat)),
        "terminal_abs_net_p95_shares":
            round(float(np.quantile(nets, 0.95)), 2) if len(nets) else None,
        "terminal_cash_at_risk_p95_usd":
            round(float(np.quantile(cash, 0.95)), 2) if len(cash) else None,
    }


def _select() -> tuple[list[Any], list[str], list[str], list[str], dict[str, int]]:
    by_day = ww.select_by_day(30)
    days = sorted(by_day)
    selected = [
        item for day in days for item in by_day[day]
        if item[0].split("-")[0] in opt.VERDICT_COINS
    ]
    counts = collections.Counter(fi.slug_day(item[0]) for item in selected)
    complete = [day for day in days if counts[day] == 60]
    partial = [day for day in days if 0 < counts[day] < 60]
    holdout = [day for day in complete if day not in opt.TRAIN_DAYS]
    return selected, days, holdout, partial, dict(counts)


def _controls(selected: Sequence[Any]) -> dict[str, Any]:
    sample = list(selected[:4])
    controls: dict[str, Any] = {}
    for item in sample:
        slug, path, up, down, gaps = item
        specs = [
            {"cell": "JOIN_CONTROL", "placement": "JOIN", "r_cut": 0,
             "size": 5.0},
            {"cell": "SKEW_INF_CONTROL", "placement": "SKEW_LB",
             "r_cut": 0, "size": 5.0, "skew": True,
             "skew_band_shares": float("inf"), "front_on_repost": False},
        ]
        got = opt.replay_cells(path, up, down, gaps, specs=specs)
        if (got is None or not pb.conformant(
                got["JOIN_CONTROL"], got["SKEW_INF_CONTROL"])):
            raise SystemExit(f"[opt-skew] infinite-band JOIN parity break {slug}")
    controls["infinite_band_join_parity"] = f"exact on {len(sample)} windows"

    item = sample[0]
    first = opt.replay_cells(item[1], item[2], item[3], item[4], opt.STAGE_B)
    second = opt.replay_cells(item[1], item[2], item[3], item[4], opt.STAGE_B)
    deterministic = bool(first is not None and second is not None and all(
        pb.fill_key(first[cell]) == pb.fill_key(second[cell]) for cell in first))
    if not deterministic:
        raise SystemExit("[opt-skew] determinism control failed")
    controls["determinism"] = True
    controls["pessimistic_repost"] = all(
        cell["skew"] and not cell["front_on_repost"] for cell in opt.STAGE_B)
    if not controls["pessimistic_repost"]:
        raise SystemExit("[opt-skew] Stage B is not uniformly SKEW_LB")
    return controls


def run() -> dict[str, Any]:
    selected, days, holdout, partial, counts = _select()
    if not STAGE_A_RECEIPT.exists():
        raise RuntimeError("Stage-A receipt missing; comparison anchor unavailable")
    stage_a = json.loads(STAGE_A_RECEIPT.read_text())
    controls = _controls(selected)
    print(f"[opt-skew] controls PASS {controls}", flush=True)

    rows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    windows: dict[tuple[str, str, str], list[Any]] = collections.defaultdict(list)
    sampled: list[Path] = []
    for index, (slug, path, up, down, gaps) in enumerate(selected, 1):
        coin, day = slug.split("-")[0], fi.slug_day(slug)
        got = opt.replay_cells(path, up, down, gaps, specs=opt.STAGE_B)
        if got is None:
            continue
        sampled.append(path)
        for cell, wf in got.items():
            markout_rows, _ = pb.rows_h(wf, opt.H)
            rows[(cell, coin, day)].append(markout_rows)
            windows[(cell, coin, day)].append(wf)
        if index % 60 == 0:
            print(f"[opt-skew] {index}/{len(selected)} windows", flush=True)

    table: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    for spec in opt.STAGE_B:
        cell = spec["cell"]
        table[cell] = {}
        verdicts[cell] = {}
        r_cut, size = spec["r_cut"], int(spec["size"])
        join_cell = f"JOIN:r{r_cut}:s{size}"
        front_cell = f"FRONT:r{r_cut}:s{size}"
        for coin in opt.VERDICT_COINS:
            table[cell][coin] = {}
            day_pass: list[bool] = []
            for day in days:
                metrics = _cell_metrics(
                    rows.get((cell, coin, day), []),
                    windows.get((cell, coin, day), []))
                join_value = stage_a["cells"][join_cell][coin][day][
                    "pnl_per_window_cents"]
                front_value = stage_a["cells"][front_cell][coin][day][
                    "pnl_per_window_cents"]
                value = metrics["pnl_per_window_cents"]
                metrics["stage_a_join_cents"] = join_value
                metrics["stage_a_front_cents"] = front_value
                metrics["delta_vs_join_cents"] = (
                    None if value is None or join_value is None
                    else round(float(value) - float(join_value), 2))
                metrics["delta_vs_front_cents"] = (
                    None if value is None or front_value is None
                    else round(float(value) - float(front_value), 2))
                table[cell][coin][day] = metrics
                if day in holdout:
                    day_pass.append(value is not None and value > 0
                                    and value > join_value and value > front_value)
            verdicts[cell][coin] = bool(day_pass and all(day_pass))
        verdicts[cell]["PROMOTED"] = all(
            verdicts[cell][coin] for coin in opt.VERDICT_COINS)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "POLICY_OPTIMIZER_PROTOCOL_STAGE_B_SKEW_LB",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "DEVELOPMENT",
        "decision_eligible": False,
        "incentives": "EXCLUDED_BY_USER_DIRECTION",
        "population": {
            "windows": len(selected), "days": counts,
            "train": list(opt.TRAIN_DAYS), "holdout_complete": holdout,
            "partial_beside": partial,
        },
        "semantics": {
            "placement": "SKEW_LB_FRONT_REDUCING_SIDE_ON_FORMATION_ONLY",
            "front_on_repost": False,
            "skew_band_shares": 5.0,
            "cancellation": False,
            "objective": "TOTAL_M5_PNL_PER_WINDOW_WITH_INVENTORY_DIAGNOSTICS",
        },
        "controls": controls,
        "cells": table,
        "promotion": verdicts,
        "provenance": {
            "polymarket": fi.provenance(sampled=sampled),
            "code_sha256": _file_sha(Path(__file__)),
            "engine_sha256": _file_sha(Path(opt.__file__)),
            "protocol_sha256": _file_sha(PLAN),
            "stage_a_receipt_sha256": _file_sha(STAGE_A_RECEIPT),
        },
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    receipt["artifact_id"] = hashlib.sha256(payload).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(receipt, indent=1, allow_nan=False))
    print(f"[opt-skew] receipt -> {OUT}", flush=True)
    promoted = [cell for cell, value in verdicts.items() if value["PROMOTED"]]
    print(f"[opt-skew] promoted {promoted or 'NONE'}", flush=True)
    return receipt


def selftest() -> int:
    checks = 0

    def ok(condition: bool, name: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(name)
        checks += 1

    ok(len(opt.STAGE_B) == 6, "six frozen Stage-B cells")
    ok(all(spec["placement"] == "SKEW_LB" for spec in opt.STAGE_B),
       "all cells use SKEW_LB")
    class Fill:
        def __init__(self, side: str, size: float):
            self.maker_side, self.size = side, size
    class Window:
        fills = [Fill("BUY_UP", 8.0), Fill("SELL_UP", 3.0)]
        mid_v = [0.6]
    ok(_fill_net(Window()) == 5.0, "inventory folds maker sides")
    ok(abs(_cash_at_risk(Window()) - 3.0) < 1e-12,
       "long inventory cash risk is side aware")
    print(f"[opt-skew] selftest OK — {checks} checks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.command == "run":
        selftest()
        run()
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
