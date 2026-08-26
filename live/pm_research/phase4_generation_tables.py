"""Phase-4 wiring: generation-level tranche tables, per R-165(2) item 5.

AUTHORISATION (R-126, in-file): R-165(2) item 5 / R-177, Phase-4 queue.

WHAT PHASE 4 MUST CONSUME. R-165(2) item 5: Phase 4 is fed GENERATION-LEVEL
tranche tables, as `harmful_exposure_rows.generation_table` produces -- not
per-row latency labels. The reason is the one rule 2 states: a generation is
the cancellable unit, and several rows can share one outcome, so a per-row feed
lets one generation's outcome be counted many times.

THE 1-SECOND CAP IS PART OF THE ESTIMAND, NOT A FOOTNOTE. The per-row latency
labels are capped at FILL_HORIZON_S = 1.0s: a fill later than one second after
the decision row is not attributed to it. Any Phase-4 cell that consumes those
labels is therefore answering "value preventable WITHIN ONE SECOND", not
"value preventable". R-165(2) item 5 requires that cap be DECLARED as part of
the estimand, so `tranche_table` refuses to emit without carrying it.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

FILL_HORIZON_S = 1.0        # mirrors harmful_exposure_rows.FILL_HORIZON_S
MARKOUT_S = 5.0


class UndeclaredEstimand(RuntimeError):
    """A tranche table was requested without declaring the horizon cap."""


def tranche_table(rows, latency_ms: int, *, declare_cap: bool = False) -> dict:
    """Generation-level tranches for one population.

    `declare_cap` must be passed EXPLICITLY. It is not defaulted True, because
    the whole point of R-165(2) item 5 is that a consumer cannot inherit the
    cap silently -- it has to say it knows."""
    if not declare_cap:
        raise UndeclaredEstimand(
            "refusing to emit: the per-row latency labels are capped at "
            f"{FILL_HORIZON_S}s, so any cell built on them estimates 'value "
            f"preventable WITHIN {FILL_HORIZON_S}s', not 'value preventable'. "
            "Pass declare_cap=True to acknowledge the cap is part of the "
            "estimand (R-165(2) item 5).")
    L = str(latency_ms)
    gens: dict = {}
    for r in rows:
        k = (r.get("slug"), r.get("side"), r.get("gen"))
        lat = (r.get("latency") or {}).get(L) or {}
        g = gens.setdefault(k, {"n_rows": 0, "preventable_value_cents": 0.0,
                                "preventable_shares": 0.0, "stale_shares": 0.0,
                                "t_start_min": None, "t_start_max": None,
                                "coin": r.get("coin"), "day": r.get("day")})
        g["n_rows"] += 1
        ts = float(r.get("t_start", 0.0))
        g["t_start_min"] = ts if g["t_start_min"] is None else min(g["t_start_min"], ts)
        g["t_start_max"] = ts if g["t_start_max"] is None else max(g["t_start_max"], ts)
        # The generation's value is the value at its FIRST crossing, which the
        # policy layer selects -- NOT the sum over rows. Summing would count one
        # outcome once per row (rule 2, measured 1.99 rows/fill, max 23).
        if g["n_rows"] == 1 or ts <= g["t_start_min"]:
            g["preventable_value_cents"] = lat.get("preventable_value_cents", 0.0)
            g["preventable_shares"] = lat.get("preventable_shares", 0.0)
            g["stale_shares"] = lat.get("stale_shares", 0.0)
    return {
        "unit": "GENERATION",
        "latency_ms": latency_ms,
        "estimand_horizon_s": FILL_HORIZON_S,
        "estimand_note": (f"values are 'preventable within {FILL_HORIZON_S}s of the "
                          f"decision row'; a later fill is not attributed here"),
        "markout_s": MARKOUT_S,
        "n_generations": len(gens),
        "n_rows_consumed": sum(g["n_rows"] for g in gens.values()),
        "rows_per_generation": (sum(g["n_rows"] for g in gens.values()) / len(gens)) if gens else None,
        "generations": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in gens.items()},
    }


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    def _r(slug, side, gen, ts, v):
        return {"slug": slug, "side": side, "gen": gen, "t_start": ts,
                "coin": "btc", "day": "d",
                "latency": {"50": {"preventable_value_cents": v,
                                   "preventable_shares": 1.0, "stale_shares": 0.0}}}

    rows = [_r("s", "BUY_UP", 1, 0.0, 10.0), _r("s", "BUY_UP", 1, 0.5, 10.0),
            _r("s", "BUY_UP", 1, 1.0, 10.0), _r("s", "SELL_UP", 2, 0.0, -4.0)]
    try:
        tranche_table(rows, 50)
        ok(False, "must refuse without declare_cap")
    except UndeclaredEstimand as e:
        ok("part of the estimand" in str(e),
           "POSITIVE CONTROL: emitting without declaring the 1s cap is REFUSED, "
           "naming what the cell would actually be estimating")
    t = tranche_table(rows, 50, declare_cap=True)
    ok(t["unit"] == "GENERATION" and t["n_generations"] == 2,
       "KNOWN-GOOD: 4 rows collapse to 2 GENERATIONS -- the cancellable unit")
    ok(t["n_rows_consumed"] == 4 and abs(t["rows_per_generation"] - 2.0) < 1e-12,
       "and rows-per-generation is carried, so a reader sees the collapse ratio")
    g = t["generations"]["s|BUY_UP|1"]
    ok(g["preventable_value_cents"] == 10.0,
       "THE ANTI-DOUBLE-COUNT ARM: a 3-row generation worth 10c reports 10c, "
       "NOT 30c -- summing rows would count one outcome once per row (rule 2)")
    ok(t["estimand_horizon_s"] == FILL_HORIZON_S and "within" in t["estimand_note"],
       "the emitted table CARRIES its horizon cap, so a downstream cell cannot "
       "lose the estimand it inherited")
    print(f"phase4_generation_tables selftest: {checks} checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest() if "--selftest" in sys.argv else 0)
