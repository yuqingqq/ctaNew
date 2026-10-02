"""Evaluate the forward test declared in docs/FORWARD_TEST.md.

  python3 -m research.forward_eval            # counts + data quality only (allowed any time)
  python3 -m research.forward_eval --final    # the one look; refuses below the minimum or a second time

Reads data/alerts/followups-*.jsonl written by the live engine. Admits only records written by
production runs (not dry runs) whose code fingerprint equals the freeze commit's (computed with
the fingerprint function AS OF that commit); the test starts at the first such run and ends at
the first later production run with other code. Every row that does not count carries a status,
per hypothesis. --final writes a receipt with the commit hashes to data/forward_test/.
Declaration: docs/FORWARD_TEST_v2.md (v1, docs/FORWARD_TEST.md, was superseded unread).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from movewatch.followup import GAP_TOL_MIN  # noqa: E402

DECLARATION = "docs/FORWARD_TEST_v2.md"
DUP_S, DUP_LOG_PX = 300, 0.01      # same move under two tickers: same direction, <= 5 min, price within 1%
MIN = {"H1": 200, "H2": 300, "B1": 200, "weeks": 5}
LATE = {"FU_4H": 600, "FU_72H": 3600}
MIN_COINS = 10
DRAWS = 2000
SEED = 20260928
NAN = float("nan")


# ---------------------------------------------------------------- freeze identity
def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=root).stdout


def freeze_commit(root: Path = ROOT) -> str:
    """The commit that added the declaration (CLAUDE.md rule 12: a freeze is a commit)."""
    out = _git(root, "log", "--diff-filter=A", "--format=%H", "--", DECLARATION).decode().split()
    if not out:
        raise SystemExit(f"refused: {DECLARATION} is not committed, so there is no freeze")
    return out[-1]


def fingerprint_at(commit: str, root: Path = ROOT) -> str:
    """movewatch.followup.code_fingerprint recomputed from the commit's tree, with the
    fingerprint() function of that commit (so a later edit to it cannot move the identity)."""
    names = _git(root, "ls-tree", "-r", "--name-only", commit, "--", "movewatch").decode().split("\n")
    files = [(n[len("movewatch/"):], _git(root, "show", f"{commit}:./{n}")) for n in names if n.endswith(".py")]
    cfg = yaml.safe_load(_git(root, "show", f"{commit}:./config/movewatch.yaml")) or {}
    ns: dict = {"__name__": "frozen_followup"}
    exec(compile(dict(files)["followup.py"], "frozen_followup.py", "exec"), ns)  # noqa: S102 -- our own committed file
    return ns["fingerprint"](files, cfg)


# ---------------------------------------------------------------- records
def load(root: Path = ROOT) -> list[dict]:
    out = []
    for f in sorted((root / "data" / "alerts").glob("followups-*.jsonl")):
        out += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return out


def admission(recs: list[dict], freeze_sha: str) -> tuple[set, float | None, float | None]:
    """(admitted run ids, test start, test end or None while it runs)."""
    starts = sorted((r for r in recs if r.get("kind") == "FU_START" and not r.get("dry_run")), key=lambda r: r["t"])
    good = [r for r in starts if r.get("code_sha") == freeze_sha]
    if not good:
        return set(), None, None
    start = good[0]["t"]
    end = next((r["t"] for r in starts if r["t"] > start and r.get("code_sha") != freeze_sha), None)
    runs = {r["run"] for r in good if end is None or r["t"] < end}
    return runs, start, end


def first_by_id(recs: list[dict], kind: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in sorted((r for r in recs if r.get("kind") == kind), key=lambda r: r.get("t", 0.0)):
        out.setdefault(r["id"], r)            # a re-written record after a crash keeps the first
    return out


def duplicates(opens: dict[str, dict], runs: set) -> set[str]:
    """Rows = actions (CLAUDE.md rule 2): one move alerted under two tickers (e.g. 龙虾 / LONGXIA
    before the alias map) is one row. A later admitted FU_OPEN with another asset name, the same
    direction, <= 5 min after and an alert price within 1% of an earlier one is a DUPLICATE."""
    rows = sorted((o for o in opens.values() if o.get("run") in runs), key=lambda o: (o["t"], o["id"]))
    out: set[str] = set()
    for k, o in enumerate(rows):
        for p in rows[:k]:
            if (p["id"] not in out and p["asset"] != o["asset"] and p["dir"] == o["dir"]
                    and o["t"] - p["t"] <= DUP_S and p["alert_px"] > 0 and o["alert_px"] > 0
                    and abs(math.log(o["alert_px"] / p["alert_px"])) <= DUP_LOG_PX):
                out.add(o["id"])
                break
    return out


def complete_weeks(start: float, now: float) -> int:
    """Whole ISO weeks (Monday 00:00 UTC to Monday 00:00 UTC) inside [start, now]."""
    s = pd.Timestamp(start, unit="s")
    monday = s.normalize() - pd.Timedelta(days=s.weekday())
    first = monday if s == monday else monday + pd.Timedelta(days=7)
    return max(0, int((pd.Timestamp(now, unit="s") - first) // pd.Timedelta(days=7)))


def mkt_ret(p0, p1, eligible) -> float:
    """Median log return of the eligible universe between two snapshots (>= 10 coins)."""
    if not isinstance(p0, dict) or not isinstance(p1, dict):
        return NAN
    r = [math.log(p1[k] / p0[k]) for k in eligible if p0.get(k, 0) > 0 and p1.get(k, 0) > 0]
    return float(np.median(r)) if len(r) >= MIN_COINS else NAN


def _lr(p0, p1) -> float:
    return math.log(p1 / p0) if (p0 or 0) > 0 and (p1 or 0) > 0 else NAN


def _fin(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def check(rec: dict | None, kind: str, ref_key, runs: set, end: float | None) -> str:
    if rec is None:
        return "MISSING"
    if rec.get("run") not in runs or (end is not None and rec["t"] >= end):
        return "OTHER_RUN"
    if rec["late_s"] > LATE[kind]:
        return "LATE"
    if not _fin(rec.get("px")) or rec["px"] <= 0:
        return "NO_PRICE"
    if rec.get("px_key") != ref_key:
        return "PX_SOURCE_CHANGED"
    return "OK"


def _first_fail(conds: list[tuple[str, bool]]) -> str:
    return next((name for name, bad in conds if bad), "OK")


def outcomes(recs: list[dict], runs: set, start: float | None, end: float | None) -> pd.DataFrame:
    opens, f4, f72 = (first_by_id(recs, k) for k in ("FU_OPEN", "FU_4H", "FU_72H"))
    dup = duplicates(opens, runs)
    rows = []
    for i, o in opens.items():
        a4, a72 = f4.get(i) or {}, f72.get(i) or {}
        in_test = (o.get("run") in runs and start is not None and o["t"] >= start
                   and (end is None or o["t"] < end))
        s4 = check(f4.get(i), "FU_4H", o.get("ref_key"), runs, end)
        s72 = check(f72.get(i), "FU_72H", o.get("ref_key"), runs, end)
        d = o["dir"]
        univ0 = list((o.get("mkt_px") or {}).keys())
        m04 = mkt_ret(o.get("mkt_px"), a4.get("mkt_px"), univ0)
        m472 = mkt_ret(a4.get("mkt_px"), a72.get("mkt_px"), a4.get("mkt_liq") or [])
        m072 = mkt_ret(o.get("mkt_px"), a72.get("mkt_px"), univ0)
        r04, r472, r072 = _lr(o["alert_px"], a4.get("px")), _lr(a4.get("px"), a72.get("px")), _lr(o["alert_px"], a72.get("px"))
        vr = a4.get("vol_ratio")
        gaps = max(a4.get("gap_min_before", math.inf), a4.get("gap_min_after", math.inf))
        vol_bad = not (_fin(vr) and vr > 0 and gaps <= GAP_TOL_MIN)
        st_h2 = _first_fail([("OTHER_RUN", not in_test), ("DUPLICATE", i in dup), ("NO_BREADTH", not _fin(o.get("breadth"))),
                             (f"4H_{s4}", s4 != "OK"), ("NO_MARKET", not _fin(m04))])
        st_h1 = _first_fail([("OTHER_RUN", not in_test), ("DUPLICATE", i in dup), (f"4H_{s4}", s4 != "OK"), (f"72H_{s72}", s72 != "OK"),
                             ("VOL_GAP", vol_bad), ("NO_MARKET", not _fin(m472))])
        st_b1 = _first_fail([("OTHER_RUN", not in_test), ("DUPLICATE", i in dup), (f"72H_{s72}", s72 != "OK"), ("NO_MARKET", not _fin(m072))])
        vbn = a4.get("vol_ratio_binance_usdm")
        rows.append({
            "id": i, "asset": o["asset"], "dir": d, "alert_ts": o["t"], "breadth": o.get("breadth"),
            "vol_ratio": vr if st_h1 == "OK" else NAN,
            "st_H1": st_h1, "st_H2": st_h2, "st_B1": st_b1,
            "x0_4": d * (r04 - m04) if st_h2 == "OK" else NAN,
            "x4_72": d * (r472 - m472) if st_h1 == "OK" else NAN,
            "raw4_72": d * r472 if st_h1 == "OK" else NAN,
            "x0_72": d * (r072 - m072) if st_b1 == "OK" else NAN,
            "lvr": math.log10(vr) if st_h1 == "OK" else NAN,
            "lvr_bn": math.log10(vbn) if st_h1 == "OK" and _fin(vbn) and vbn > 0 else NAN,
        })
    d = pd.DataFrame(rows)
    if len(d):
        d["week"] = pd.to_datetime(d["alert_ts"], unit="s").dt.to_period("W").dt.start_time
    return d


# ---------------------------------------------------------------- statistics
def spearman(x, y) -> float:
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 10 and x.std() > 0 and y.std() > 0 else NAN


def _week_draws(wk: np.ndarray):
    rng = np.random.default_rng(SEED)
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    for _ in range(DRAWS):
        yield np.concatenate([pos[k] for k in rng.integers(0, len(uw), len(uw))])


def ic_ci(df: pd.DataFrame, f: str, out: str) -> dict:
    """Spearman IC + 95% CI from an ISO-week block bootstrap. Pass = IC < 0 and CI below 0."""
    x = df[["week", f, out]].replace([np.inf, -np.inf], np.nan).dropna()
    ic = spearman(x[f], x[out])
    fv, ov = x[f].to_numpy(), x[out].to_numpy()
    bs = [spearman(fv[ii], ov[ii]) for ii in _week_draws(x["week"].to_numpy())]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"ic": ic, "ci": [float(lo), float(hi)], "n": int(len(x)), "weeks": int(x["week"].nunique()),
            "pass": bool(ic < 0 and hi < 0)}


def median_ci(df: pd.DataFrame, col: str) -> dict:
    x = df[["week", col]].dropna()
    v = x[col].to_numpy()
    bs = [float(np.median(v[ii])) for ii in _week_draws(x["week"].to_numpy())]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pct = lambda z: (math.exp(z) - 1) * 100  # noqa: E731
    return {"median_pct": pct(float(np.median(v))), "ci_pct": [pct(lo), pct(hi)],
            "arith_pct": float((np.exp(v) - 1).mean() * 100), "n": int(len(v)), "weeks": int(x["week"].nunique())}


def counts(d: pd.DataFrame, start: float | None, end: float | None, now: float) -> dict:
    pumps = d[d["dir"] > 0] if len(d) else d
    c = {"test_start": str(pd.Timestamp(start, unit="s")) if start else None,
         "test_end": str(pd.Timestamp(end, unit="s")) if end else None,
         "complete_weeks": complete_weeks(start, end or now) if start else 0,
         "alerts": int(len(d)), "pumps": int(len(pumps))}
    for h, pop in (("H1", pumps), ("H2", d), ("B1", pumps)):
        c[f"{h}_status"] = pop[f"st_{h}"].value_counts().to_dict() if len(pop) else {}
        c[f"{h}_complete"] = int((pop[f"st_{h}"] == "OK").sum()) if len(pop) else 0
    return c


def ready(c: dict) -> bool:
    return (c["H1_complete"] >= MIN["H1"] and c["H2_complete"] >= MIN["H2"]
            and c["B1_complete"] >= MIN["B1"] and c["complete_weeks"] >= MIN["weeks"])


def final_stats(d: pd.DataFrame) -> dict:
    pumps = d[d["dir"] > 0]
    h1 = pumps[pumps["st_H1"] == "OK"]
    res = {"H1": ic_ci(h1, "lvr", "x4_72"), "H1_raw": ic_ci(h1, "lvr", "raw4_72"),
           "H1_binance_usdm_ratio": ic_ci(h1, "lvr_bn", "x4_72"),
           "H2": ic_ci(d[d["st_H2"] == "OK"], "breadth", "x0_4"),
           "B1": median_ci(pumps[pumps["st_B1"] == "OK"], "x0_72")}
    grp = {}
    for lab, sel in (("quiet", h1["vol_ratio"] < 0.94), ("heavy", h1["vol_ratio"] > 2.26)):
        v = h1.loc[sel, "x4_72"].to_numpy()
        grp[lab] = {"n": int(len(v)), "arith_pct": float((np.exp(v) - 1).mean() * 100) if len(v) else None,
                    "median_pct": float((math.exp(np.median(v)) - 1) * 100) if len(v) else None}
    if grp["quiet"]["n"] and grp["heavy"]["n"]:
        grp["heavy_minus_quiet_arith_pp"] = grp["heavy"]["arith_pct"] - grp["quiet"]["arith_pct"]
        grp["heavy_minus_quiet_median_pp"] = grp["heavy"]["median_pct"] - grp["quiet"]["median_pct"]
    res["H1_terciles"] = grp
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    commit = freeze_commit()
    sha = fingerprint_at(commit)
    recs = load()
    runs, start, end = admission(recs, sha)
    now = time.time()
    d = outcomes(recs, runs, start, end)
    c = counts(d, start, end, now) if len(d) else {"alerts": 0, "complete_weeks": 0, "H1_complete": 0,
                                                    "H2_complete": 0, "B1_complete": 0}
    c.update(freeze_commit=commit, freeze_sha=sha, admitted_runs=len(runs),
             runs_seen=len({r["run"] for r in recs if r.get("kind") == "FU_START"}))
    print(json.dumps(c, indent=1, default=str))
    ok = ready(c)
    if not args.final:
        print("counts only (reading rule). minimum met." if ok else "counts only (reading rule). minimum NOT met yet.")
        return
    if not ok:
        raise SystemExit(f"refused: minimum sample not met ({MIN}); the test is read once, at the minimum")
    out = ROOT / "data" / "forward_test"
    if list(out.glob("receipt_*.json")):
        raise SystemExit(f"refused: a receipt already exists in {out}; the test is read once")
    res = {"freeze_commit": commit, "freeze_sha": sha,
           "evaluator_head": _git(ROOT, "rev-parse", "HEAD").decode().strip(),
           "counts": c, **final_stats(d)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"receipt_{pd.Timestamp.utcnow():%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print("receipt:", path)


if __name__ == "__main__":
    main()
