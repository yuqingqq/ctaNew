"""layer2_v1 — carry-to-resolution under the R-14-FROZEN bar.

LAYER2_PROTOCOL.md: ONE population (the simulated two-sided JOIN_BBO 5-share
maker, day-series selection, 4 era days x 30 windows/coin), BOTH marks —
M_h (Layer 1's mid markout, h=5 primary) and M_T = s·(payoff − level) from
the E-M6-verified settlement winners — with the bridge(h) closing to an
exact identity per fill:

    M_T = spread + drift_h + bridge(h)          (selftest control 1)

Bar (§3, FROZEN with R-14's amendments): per (coin, day) cell on the
share-weighted all-fills arm, POSITIVE/NEGATIVE by within-day window-
clustered CI, VOID under 500 fills; coin roll-up ≥75 % of era days, zero
contrary, minimum 4 era days. Power declaration in the protocol: at
census-scale effects the expectation is UNDETERMINED everywhere; the bar
fires only at carry-amplification scale (|M_T| ≳ 2.4 ¢/share per cell).
An arm sign disagreement on a verdict cell is a FIRST-CLASS FINDING
(amendment 3). Reading receipts is authorized: the bar froze first.

Selftest: python3 live/pm_research/layer2_v1.py --selftest
Run:      python3 live/pm_research/layer2_v1.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from typing import Any, Sequence

import flow_intensity as fi
import edge_layer1 as el
from warning_window import select_by_day

OUT = fi.PM / "derived/layer2_v1.json"

H_PRIMARY = 5.0
MIN_FILLS = 500          # frozen VOID floor
PROPORTION = 0.75        # R-14 amendment 1
MIN_DAYS = 4
N_BOOT = 2000
SEED = 20260823
VERDICT_COINS = ("btc", "eth")


# --------------------------------------------------------------------------
# per-fill marks — pure, controlled by selftest fixtures
# --------------------------------------------------------------------------

def m_t(maker_side: str, level: float, payoff: float) -> float:
    """Hold-to-resolution maker markout: s·(payoff − ℓ)."""
    return el.maker_sign(maker_side) * (payoff - level)


def fill_rows(wf: el.WindowFills, payoff: float,
              h: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Per fill: M_T always; spread/drift/bridge only where M_h is valid
    (Layer-1 truncation/touch conventions, reported not dropped)."""
    rows: list[dict[str, Any]] = []
    excl = {"n_mh_truncated": 0, "n_mh_gap_or_tick": 0, "n_mh_no_mid": 0}
    for f in wf.fills:
        r: dict[str, Any] = {
            "w": f.size, "micro": f.aggressor_micro,
            "mt": m_t(f.maker_side, f.level, payoff), "mh": None,
            "spread": None, "drift": None, "bridge": None,
        }
        if f.t + h > fi.WINDOW_S + 1e-12:
            excl["n_mh_truncated"] += 1
        elif wf.touched(f.t, f.t + h):
            excl["n_mh_gap_or_tick"] += 1
        else:
            later = wf.mid_at(f.t + h)
            if later is None:
                excl["n_mh_no_mid"] += 1
            else:
                mk, sp, dr = el.decompose(f.maker_side, f.level,
                                          f.mid_at_fill, later)
                r.update(mh=mk, spread=sp, drift=dr, bridge=r["mt"] - mk)
        rows.append(r)
    return rows, excl


# --------------------------------------------------------------------------
# cell aggregation — both weightings, window-clustered bootstrap
# --------------------------------------------------------------------------

def _wmean(win_rows: Sequence[Sequence[dict]], weighted: bool,
           ex_micro: bool, key: str = "mt") -> float | None:
    num = den = 0.0
    for rows in win_rows:
        for r in rows:
            if ex_micro and r["micro"]:
                continue
            if r[key] is None:
                continue
            w = r["w"] if weighted else 1.0
            num += w * r[key]
            den += w
    return num / den if den > 0 else None


def _ci(win_rows: Sequence[Sequence[dict]], weighted: bool, ex_micro: bool,
        n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float | None, float | None]:
    pw = [w for w in win_rows if w]
    if len(pw) < 2:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [pw[rng.randrange(len(pw))] for _ in range(len(pw))]
        m = _wmean(sample, weighted, ex_micro)
        if m is not None:
            means.append(m)
    if not means:
        return (None, None)
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means))])


def cell_verdict(n_fills: int, lo: float | None, hi: float | None) -> str:
    """Frozen §3 cell rule."""
    if n_fills < MIN_FILLS or lo is None or hi is None:
        return "VOID"
    if lo > 0:
        return "POSITIVE"
    if hi < 0:
        return "NEGATIVE"
    return "UNDETERMINED"


def coin_verdict(day_verdicts: Sequence[str]) -> str:
    """R-14 amendment 1: ≥75 % of era days, ZERO contrary, minimum 4 days."""
    days = [v for v in day_verdicts if v != "VOID"]
    if len(days) < MIN_DAYS:
        return "UNDETERMINED"
    pos = sum(1 for v in days if v == "POSITIVE")
    neg = sum(1 for v in days if v == "NEGATIVE")
    if pos / len(days) >= PROPORTION and neg == 0:
        return "CARRY_RESCUES"
    if neg / len(days) >= PROPORTION and pos == 0:
        return "CARRY_FAILS"
    return "UNDETERMINED"


# --------------------------------------------------------------------------

def load_winners() -> dict[str, float]:
    """slug -> payoff in Up terms, FINAL resolutions only. Missing slugs are
    NAMED exclusions at the caller (control 3)."""
    # tier1_pipeline imports by package path (live.pm_research.*): the repo
    # root must be importable — same sys.path arrangement as
    # cross_window_correlation.py, which consumes it the same way.
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import tier1_pipeline as tp
    _, resolutions, _ = tp.load_market_metadata()
    out: dict[str, float] = {}
    for slug, res in resolutions.items():
        wu = getattr(res, "winner_up", None)
        if wu is not None:
            out[slug] = 1.0 if wu else 0.0
    return out


def run(per_coin: int) -> None:
    winners = load_winners()
    by_day = select_by_day(per_coin)
    days_out: dict[str, Any] = {}
    sampled = []
    for day, selected in by_day.items():
        cells: dict[str, Any] = {}
        per_coin_rows: dict[str, list] = collections.defaultdict(list)
        per_coin_excl: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        for i, (slug, path, up, down, gaps) in enumerate(selected, 1):
            if i % 50 == 0 or i == 1:
                print(f"[l2] {day} {i}/{len(selected)} {slug}", flush=True)
            sampled.append(path)
            coin = slug.split("-")[0]
            payoff = winners.get(slug)
            if payoff is None:
                per_coin_excl[coin]["n_windows_unresolved"] += 1   # control 3
                continue
            wf = el.replay_window(path, up, down, gaps, front=False)
            if wf is None:
                per_coin_excl[coin]["n_windows_no_state"] += 1
                continue
            rows, excl = fill_rows(wf, payoff, H_PRIMARY)
            for k, v in excl.items():
                per_coin_excl[coin][k] += v
            per_coin_rows[coin].append(rows)

        for coin, win_rows in sorted(per_coin_rows.items()):
            cell: dict[str, Any] = {"n_windows": len(win_rows)}
            for arm, exm in (("all", False), ("ex_micro", True)):
                a: dict[str, Any] = {}
                n = sum(1 for w in win_rows for r in w
                        if not (exm and r["micro"]))
                a["n_fills"] = n
                for wname, wtd in (("share", True), ("per_fill", False)):
                    m = _wmean(win_rows, wtd, exm)
                    lo, hi = _ci(win_rows, wtd, exm)
                    a[wname] = {"mt_cents": None if m is None else m * 100,
                                "ci95_cents": [None if lo is None else lo * 100,
                                               None if hi is None else hi * 100]}
                # decomposition on the M_h-valid subpopulation (identity check)
                for key in ("mh", "spread", "drift", "bridge"):
                    v = _wmean(win_rows, True, exm, key)
                    a[f"{key}_share_cents"] = None if v is None else v * 100
                mt_sub = _wmean(
                    [[r for r in w if r["mh"] is not None] for w in win_rows],
                    True, exm)
                a["mt_on_mh_subpop_cents"] = None if mt_sub is None else mt_sub * 100
                cell[arm] = a
            sh = cell["all"]["share"]
            v = cell_verdict(cell["all"]["n_fills"],
                             None if sh["ci95_cents"][0] is None
                             else sh["ci95_cents"][0] / 100,
                             None if sh["ci95_cents"][1] is None
                             else sh["ci95_cents"][1] / 100)
            cell["verdict"] = v
            pf = cell["all"]["per_fill"]["mt_cents"]
            shp = sh["mt_cents"]
            cell["arm_sign_disagreement"] = (          # amendment 3
                shp is not None and pf is not None
                and (shp > 0) != (pf > 0) and abs(shp) > 1e-9 and abs(pf) > 1e-9)
            cell["exclusions"] = dict(per_coin_excl[coin])
            cells[coin] = cell
        days_out[day] = cells

    coins: dict[str, Any] = {}
    for coin in sorted({c for d in days_out.values() for c in d}):
        dvs = [days_out[d][coin]["verdict"] for d in sorted(days_out)
               if coin in days_out[d]]
        coins[coin] = {"day_verdicts": dvs, "verdict": coin_verdict(dvs),
                       "is_verdict_coin": coin in VERDICT_COINS}

    res = {
        "protocol": "layer2_v1",
        "bar": "LAYER2_PROTOCOL.md §3 FROZEN per R-14 (proportion 0.75, zero "
               "contrary, min 4 era days; VOID 500; h=5 primary; "
               "share-weighted primary with per-fill beside; arm sign "
               "disagreement = first-class finding; power declaration: "
               "UNDETERMINED expected at census-scale effects, MDE ~2.4c/cell)",
        "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
        "h_primary_s": H_PRIMARY,
        "days": days_out,
        "coins": coins,
    }
    res["provenance"] = fi.provenance(sampled=sampled)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    print("\n[l2] per-cell reading (share-weighted, all fills, h=5, cents/share):")
    for day in sorted(days_out):
        for coin in VERDICT_COINS:
            c = days_out[day].get(coin)
            if not c:
                continue
            sh = c["all"]["share"]
            lo, hi = sh["ci95_cents"]
            pf = c["all"]["per_fill"]["mt_cents"]
            fmt = lambda x: "-" if x is None else f"{x:+.3f}"
            flag = "  <-- ARM SIGN DISAGREEMENT (first-class finding)" \
                if c["arm_sign_disagreement"] else ""
            print(f"  {day} {coin}: n={c['all']['n_fills']:>6} "
                  f"M_T={fmt(sh['mt_cents'])} CI[{fmt(lo)},{fmt(hi)}] "
                  f"per-fill={fmt(pf)} -> {c['verdict']}{flag}")
    for coin, cv in coins.items():
        if cv["is_verdict_coin"]:
            print(f"[l2] {coin} COIN VERDICT: {cv['verdict']} "
                  f"(days: {cv['day_verdicts']})")
    print(f"[l2] receipt -> {OUT}")


# --------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # control 2 — known-winner fixture, signs pinned BOTH ways, both sides
    ok(abs(m_t("BUY_UP", 0.40, 1.0) - 0.60) < 1e-12, "BUY win = +0.60")
    ok(abs(m_t("BUY_UP", 0.40, 0.0) + 0.40) < 1e-12, "BUY lose = -0.40")
    ok(abs(m_t("SELL_UP", 0.40, 1.0) + 0.60) < 1e-12, "SELL win(-Up) = -0.60")
    ok(abs(m_t("SELL_UP", 0.40, 0.0) - 0.40) < 1e-12, "SELL lose = +0.40")

    # control 1 — identity closure on a synthetic fill with known mid path
    wf = el.WindowFills("btc-updown-5m-0", "btc",
                        [el.Fill(10.0, "BUY_UP", 0.49, 5.0, 0.50, False)],
                        [0.0, 15.0], [0.50, 0.52], [], {})
    rows, excl = fill_rows(wf, 1.0, H_PRIMARY)
    r = rows[0]
    ok(abs(r["spread"] - 0.01) < 1e-12 and abs(r["drift"] - 0.02) < 1e-12,
       "Layer-1 legs reproduce")
    ok(abs(r["mt"] - 0.51) < 1e-12, "M_T = payoff - level")
    ok(abs(r["spread"] + r["drift"] + r["bridge"] - r["mt"]) < 1e-12,
       "IDENTITY: spread + drift + bridge == M_T exactly")
    ok(sum(excl.values()) == 0, "no exclusions on the clean fixture")

    # truncated fill: M_T valid, M_h legs absent, exclusion named
    wf2 = el.WindowFills("btc-updown-5m-0", "btc",
                         [el.Fill(298.0, "BUY_UP", 0.49, 5.0, 0.50, False)],
                         [0.0], [0.50], [], {})
    rows2, excl2 = fill_rows(wf2, 0.0, H_PRIMARY)
    ok(rows2[0]["mt"] is not None and rows2[0]["mh"] is None
       and excl2["n_mh_truncated"] == 1, "terminal fill: M_T only, named")

    # control 4 — winner shuffle moves M_T (join non-vacuous)
    a, _ = fill_rows(wf, 1.0, H_PRIMARY)
    b, _ = fill_rows(wf, 0.0, H_PRIMARY)
    ok(abs(a[0]["mt"] - b[0]["mt"]) > 0.5, "flipping the winner moves M_T")

    # frozen cell rule branches
    ok(cell_verdict(499, 0.01, 0.02) == "VOID", "VOID floor")
    ok(cell_verdict(600, 0.01, 0.02) == "POSITIVE", "POSITIVE")
    ok(cell_verdict(600, -0.02, -0.01) == "NEGATIVE", "NEGATIVE")
    ok(cell_verdict(600, -0.01, 0.02) == "UNDETERMINED", "UNDETERMINED")

    # R-14 amendment-1 roll-up: proportion + zero-contrary + min-days
    ok(coin_verdict(["POSITIVE"] * 3 + ["UNDETERMINED"]) == "CARRY_RESCUES",
       "3/4 positive, zero contrary -> RESCUES")
    ok(coin_verdict(["NEGATIVE"] * 3 + ["UNDETERMINED"]) == "CARRY_FAILS",
       "3/4 negative, zero contrary -> FAILS")
    ok(coin_verdict(["POSITIVE"] * 3 + ["NEGATIVE"]) == "UNDETERMINED",
       "zero-contrary violated -> UNDETERMINED")
    ok(coin_verdict(["POSITIVE"] * 3) == "UNDETERMINED",
       "below minimum era days -> UNDETERMINED")
    ok(coin_verdict(["POSITIVE"] * 6 + ["UNDETERMINED", "VOID"])
       == "CARRY_RESCUES", "6/7 non-void positive at 7 days -> RESCUES "
       "(the proportion rule keeps meaning as the tape grows)")

    # weighting arms genuinely differ (weighted vs unweighted)
    wr = [[{"w": 1.0, "micro": False, "mt": 0.10, "mh": None,
            "spread": None, "drift": None, "bridge": None},
           {"w": 9.0, "micro": False, "mt": -0.10, "mh": None,
            "spread": None, "drift": None, "bridge": None}]]
    ok(abs(_wmean(wr, True, False) - (-0.08)) < 1e-12
       and abs(_wmean(wr, False, False)) < 1e-12,
       "share vs per-fill arms diverge on the fixture")

    print(f"[l2] selftest OK — {checks} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a.per_coin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
