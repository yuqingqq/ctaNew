"""Noise cost + coverage of a "calm coin wakes up" trigger path, added (OR) to the live
strict rules.

Motivation: QNT 2026-09-24 (news day) was +8% in 1h at z 8.8 with 20x volume but never
passed the strict size floors (1h 30%). The wake path trades a smaller size floor for
extreme z and volume. Its parameters were chosen by LOOKING AT QNT (a consumed window),
so QNT's catch is in-sample by construction; the numbers that matter here are the extra
alerts it adds on other coins in August (calibration) and 09-01..09-19.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.wake_path
Output: data/replay/wake_path/summary.json
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from movewatch.alerts import AlertCfg  # noqa: E402
from movewatch.signals import DEFAULT_RULES, HorizonRule, Rules  # noqa: E402
from research.replay import (OUT_DIR, big_move_events, candidate_table, good, load_panel, mem,  # noqa: E402
                             run_alerts, score_events, summarize, triggers)

STRICT = DEFAULT_RULES.scaled(3.0, 1.25, 1.0)          # the live cal_v4 pick
STRICT.liq_min_usd = 5e6
VARIANTS = {
    "none (live strict)": None,
    "1h >=5% z>=6 v>=8": {3600: HorizonRule(5.0, 6.0, 8.0)},
    "1h >=5% z>=7 v>=10": {3600: HorizonRule(5.0, 7.0, 10.0)},
    "1h >=6% z>=6 v>=10": {3600: HorizonRule(6.0, 6.0, 10.0)},
}


def union(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    tr = pd.concat([a, b], ignore_index=True)
    return tr.sort_values(["i", "t", "absr"], ascending=[True, True, False]).drop_duplicates(["i", "t"])


def first_alert(al: pd.DataFrame, sym: str, after: str) -> dict | None:
    a = al[(al["asset"] == sym) & (al["direction"] > 0) & (al["time"] >= pd.Timestamp(after, tz="UTC"))]
    if not len(a):
        return None
    r = a.sort_values("t").iloc[0]
    return {"time": str(r["time"]), "kind": r["kind"], "price": round(float(r["price"]), 2),
            "horizon": r["horizon"] if isinstance(r["horizon"], str) else None}


def main() -> None:
    out = OUT_DIR / "wake_path"
    out.mkdir(parents=True, exist_ok=True)
    syms, idx, C, V = load_panel("2026-08-01", "2026-09-28")
    z = np.load(OUT_DIR / "blocks_2026-08-01_2026-09-28_r15.npz")
    SIG, V1M, LIQ = z["SIG"], z["V1M"], z["LIQ"]
    mem("loaded")
    # lossless superset for strict (floor 3.0 / z 1.25 / v 1.0) and every wake variant
    # (1h floor >= 5% = 0.5 x base 10%, z >= 6 = 1.2 x base 5, v >= 8 > base 3)
    cand = candidate_table(C, V, SIG, V1M, LIQ, DEFAULT_RULES, 15, pre_floor=0.5, pre_z=1.2, pre_v=1.0)
    print(f"candidates {len(cand)}", flush=True)
    mem("candidates")
    ev0 = big_move_events(C, syms)
    mem("events")
    split_t = int(idx.get_indexer([pd.Timestamp("2026-09-01", tz="UTC")])[0])
    cons_t = int(idx.get_indexer([pd.Timestamp("2026-09-20", tz="UTC")])[0])
    tr_strict = triggers(cand, STRICT)
    res = {}
    for name, wake in VARIANTS.items():
        tr = tr_strict
        if wake is not None:
            wr = Rules(horizons=wake, liq_min_usd=5e6, market_liq_usd=5e6)
            tr = union(tr_strict, triggers(cand[cand["h"].isin(list(wake))], wr))
        al = run_alerts(tr, C, syms, idx, AlertCfg())
        ev = score_events(ev0, al)
        cal = summarize(al, ev, idx, 1440, split_t)
        val = summarize(al, ev, idx, split_t, cons_t)
        res[name] = {
            "calibration": {k: cal[k] for k in ("new_per_day", "messages_per_day", "max_messages_one_day")}
            | {"good": good(cal)},
            "validation_0901_0919": {k: val[k] for k in ("new_per_day", "messages_per_day", "max_messages_one_day")}
            | {"good": good(val)},
            "QNT_first_alert_after_0924_1200": first_alert(al, "QNTUSDT", "2026-09-24 12:00"),
            "neg_control_new": {s: int(((al["asset"] == s) & (al["kind"] == "NEW")).sum()) for s in ("BTCUSDT", "ETHUSDT")},
        }
        print(name, json.dumps(res[name]), flush=True)
        mem(name)
    (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
