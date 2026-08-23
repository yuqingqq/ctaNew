"""EV-Replay v1 — the replay environment (plans/EV_REPLAY_PLAN.md, B3).

v1 DESIGN CHOICE, stated so nobody mistakes it: the engine IS the reference
dialect (`edge_layer1.replay_window`) behind the environment interface, so
golden-window parity holds by construction today. The parity gate exists to
guard every FUTURE engine change — the acceptance is measurable from day one
(plan §4.1) and any divergence is loud.

The EV boundary, enforced structurally (plan §0): `ReplayEnv` emits RAW
records only. Markout/evaluation lives in `evaluate_markout()`, a separate
pass OUTSIDE the environment class; the env namespace carries no evaluation
symbol, and the selftest asserts that.

Receipts (Ruling R-6): every run stamps the SP parameter set it ran under.
v1 stamps the OPERATIVE set for provenance and does NOT claim to enforce it —
Constraints binding lands with B2, and a stamped-but-unenforced cap must never
read as a gate (`applied: false`).

Selftest: python3 live/pm_research/ev_replay.py --selftest
Smoke:    python3 live/pm_research/ev_replay.py smoke
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi
import flow_fill_development as fd
import inventory_walk as iw
import edge_layer1 as el
from warning_window import conformant

OUT_DIR = fi.PM / "derived"

# SP_PLANE_PLAN.md §5 — OPERATIVE per Ruling R-6 (no live orders => no second
# configuration exists behind these). Stamped in every receipt; Class A/B
# movement is the coordinator's per R-6.
SP_OPERATIVE = {
    "set_name": "SP_PLANE_PLAN_s5_operative_R6",
    "capital_budget_usd": 1000.0,
    "max_quote_size_shares": 5.0,
    "kappa_usd_per_market": 50.0,
    "scenario_loss_limit_usd": 200.0,
    "refuse_k": 1.0,
    "gamma_ladder": [0.0, 1e-3, 1e-2, 1e-1],
    "applied": False,   # stamped for provenance; enforcement lands with B2
}

ARMS = ("JOIN", "FRONT")   # scripted arms in v1; plugin path lands later


@dataclass
class RunRecord:
    """RAW events of one window replay. No evaluation fields, by design."""
    slug: str
    coin: str
    arm: str
    fills: list[tuple[float, str, float, float, float, bool]]
    mid_t: list[float]
    mid_v: list[float]
    unavailable_iv: list[tuple[float, float]]
    diagnostics: dict[str, int]


@dataclass
class ReplayEnv:
    """Explicit-window replay environment. Selection is NOT its job (R-ADMISS):
    it takes the window list and stamps it, never chooses."""
    windows: Sequence[tuple[str, Path, str, str, list[tuple[float, float]]]]
    seed: int = 20260823
    sp_set: dict[str, Any] = field(default_factory=lambda: dict(SP_OPERATIVE))

    def run(self, arm: str) -> list[RunRecord]:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; v1 arms are {ARMS}")
        front = arm == "FRONT"
        out: list[RunRecord] = []
        for slug, path, up, down, gaps in self.windows:
            wf = el.replay_window(path, up, down, gaps, front=front)
            if wf is None:
                continue
            out.append(RunRecord(
                wf.slug, wf.coin, arm,
                [(f.t, f.maker_side, f.level, f.size, f.mid_at_fill,
                  f.aggressor_micro) for f in wf.fills],
                list(wf.mid_t), list(wf.mid_v),
                [(a, b) for a, b in wf.bad_iv],
                dict(wf.diagnostics)))
        return out

    def receipt(self, records: Sequence[RunRecord], label: str,
                gates: dict[str, Any] | None = None) -> dict[str, Any]:
        body = {
            "protocol": "ev_replay_v1",
            "status": "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE",
            "engine": "edge_layer1.replay_window (reference dialect; parity "
                      "by construction in v1, gate guards evolution)",
            "engine_hash": engine_hash(),
            "label": label,
            "state_lag_s": fd.STATE_LAG_S,
            "quote_size_shares": el.QUOTE_SIZE,
            "seed": self.seed,   # reserved for stochastic arms; v1 consumes none
            # gate outcomes persisted IN the artifact (iteration 8: they were
            # stdout-only, leaving the plan's PASS cells checkable only by
            # entailment from fail-loud ordering)
            "gates": gates or {},
            "sp_parameter_set": self.sp_set,
            "windows": [w[0] for w in self.windows],
            "collector_era": fi.ERA,   # iteration 7: era was promised, not stamped
            "records": [{
                "slug": r.slug, "coin": r.coin, "arm": r.arm,
                # queue bound stamped so the receipt CONSUMER never infers it
                # (iteration 7/8; the 1:1 arm->bound map below is v1-internal
                # and KeyError-loud; it becomes a run parameter when both
                # bounds share a run, per plan §3.4)
                "queue_bound": {"JOIN": "BACK_DISPLAYED",
                                "FRONT": "FRONT"}[r.arm],
                "n_fills": len(r.fills), "fills": r.fills,
                "n_unavailable_iv": len(r.unavailable_iv),
                "diagnostics": r.diagnostics,
                # record_hash covers EVERYTHING the record carries, including
                # the mid path and interval endpoints the receipt body does not
                # serialize -- two runs differing only there must not collide.
                "record_hash": record_hash(r),
            } for r in records],
        }
        body["provenance"] = fi.provenance(
            sampled=[w[1] for w in self.windows])
        body["run_hash"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()
        return body


# --------------------------------------------------------------------------
# evaluation — a SEPARATE pass, outside the environment (plan §0 boundary)
# --------------------------------------------------------------------------

def record_hash(r: RunRecord) -> str:
    """Content hash over the FULL record: fills, mid path, unavailable
    intervals, diagnostics. The receipt's run_hash covers these via this
    field, so determinism is checked on everything `evaluate_markout`
    consumes, not only on what the receipt serializes."""
    return hashlib.sha256(json.dumps(
        [r.slug, r.coin, r.arm, r.fills, r.mid_t, r.mid_v,
         r.unavailable_iv, sorted(r.diagnostics.items())],
        sort_keys=True, default=str).encode()).hexdigest()


def engine_hash() -> str:
    """SHA-256 over the engine's TRANSITIVE load-bearing sources, not only
    `replay_window`'s own body (iteration 7: queue accounting lives in
    `RestingSide`, state in `BookState`, folding in `fold_*` — an edit to any
    of these changes fills while the old hash stayed put). A change-detector,
    NOT a conformance checker."""
    import inspect
    parts = [inspect.getsource(f) for f in (
        el.replay_window, iw.RestingSide, fd.BookState, fd._parse_book,
        fi.fold_price, fi.fold_side, fi._gz_lines)]
    parts.append(repr((fd.STATE_LAG_S, fi.WINDOW_S, fi.MICRO_SIZE,
                       el.QUOTE_SIZE, el.HORIZONS,
                       fi.TRADE_MARK, fi.QUOTE_MARK,
                       fd.BOOK_MARK, fd.TICK_MARK)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def evaluate_markout(rec: RunRecord, h: float) -> list[tuple[float, float, float]]:
    """(markout, spread, drift) per fill at horizon h. Runs on a completed
    RunRecord; the environment never sees these numbers."""
    wf = el.WindowFills(
        rec.slug, rec.coin,
        [el.Fill(*f) for f in rec.fills],
        rec.mid_t, rec.mid_v, list(rec.unavailable_iv), {})
    rows, _ = el.horizon_rows(wf, h)
    return [(r.markout, r.spread, r.drift) for r in rows]


# --------------------------------------------------------------------------
# gates — the harness is measured before it measures anything (plan §4)
# --------------------------------------------------------------------------

def parity_gate(env: ReplayEnv, arm: str) -> tuple[int, int]:
    """Golden-window fill parity vs the reference loop. (pass, fail)."""
    front = arm == "FRONT"
    p = f = 0
    for slug, path, up, down, gaps in env.windows:
        ref = el.replay_window(path, up, down, gaps, front=front)
        got = el.replay_window(path, up, down, gaps, front=front)
        # v1: engine == reference, so parity compares two invocations
        # (determinism of the engine itself); future engines replace `got`.
        if ref is None and got is None:
            continue
        if ref is not None and got is not None and conformant(got, ref):
            p += 1
        else:
            f += 1
    return p, f


def determinism_gate(env: ReplayEnv, arm: str, label: str) -> bool:
    a = env.receipt(env.run(arm), label)
    b = env.receipt(env.run(arm), label)
    # run_hash covers the full records via record_hash (fills, mid path,
    # unavailable intervals, diagnostics); provenance is path-stable
    return a["run_hash"] == b["run_hash"]


# --------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # boundary: neither the environment nor the record type exposes any
    # evaluation symbol -- METHODS included (iteration 6) AND annotation-only
    # dataclass FIELDS included (iteration 7: fields without defaults do not
    # appear in vars(cls), so `markout_h5: list` on RunRecord passed the scan)
    for cls in (ReplayEnv, RunRecord):
        names = set(vars(cls)) | set(getattr(cls, "__dataclass_fields__", {}))
        for name in ("markout", "evaluate", "calibration", "gate", "verdict"):
            ok(not any(name in attr.lower() for attr in names),
               f"{cls.__name__} namespace clean of '{name}'")
    ok("evaluate_markout" in globals(), "evaluation exists OUTSIDE the env")

    # RunRecord carries raw fields only
    raw_fields = set(RunRecord.__dataclass_fields__)
    ok(not raw_fields & {"markout", "spread", "drift", "pnl"},
       "RunRecord is raw-only")

    # hash sensitivity MUST-FAIL controls (iteration 6): the hash must move
    # when anything the record carries moves -- including the parts the
    # receipt does not serialize
    base = RunRecord("s", "btc", "JOIN",
                     [(1.0, "BUY_UP", 0.5, 5.0, 0.505, False)],
                     [0.0, 2.0], [0.50, 0.51], [(9.0, 9.5)], {"d": 1})
    import copy
    for mutate, label in (
        (lambda r: r.mid_v.__setitem__(1, 0.52), "mid path"),
        (lambda r: r.unavailable_iv.__setitem__(0, (9.0, 9.6)), "gap interval"),
        (lambda r: r.fills.__setitem__(0, (1.0, "BUY_UP", 0.5, 4.0, 0.505,
                                           False)), "fill size"),
    ):
        m = copy.deepcopy(base)
        mutate(m)
        ok(record_hash(m) != record_hash(base),
           f"record_hash sensitive to {label}")
    ok(len(engine_hash()) == 64, "engine hash from source, not a label")

    # evaluation pass reproduces the reference decomposition on a synthetic
    rec = RunRecord("btc-updown-5m-0", "btc", "JOIN",
                    [(10.0, "BUY_UP", 0.49, 5.0, 0.50, False)],
                    [0.0, 12.0], [0.50, 0.52], [], {})
    rows = evaluate_markout(rec, 5.0)
    ok(len(rows) == 1 and abs(rows[0][0] - 0.03) < 1e-12
       and abs(rows[0][1] - 0.01) < 1e-12 and abs(rows[0][2] - 0.02) < 1e-12,
       f"two-pass evaluation matches hand-check, got {rows}")

    # SP stamp: operative set present, honest about non-enforcement
    env = ReplayEnv(windows=[])
    r = env.receipt([], "selftest")
    ok(r["sp_parameter_set"]["set_name"] == "SP_PLANE_PLAN_s5_operative_R6",
       "operative SP set stamped")
    ok(r["sp_parameter_set"]["applied"] is False,
       "stamped-not-enforced is explicit (a cap that cannot fire must not "
       "read as a gate)")
    ok("run_hash" in r and len(r["run_hash"]) == 64, "run hash stamped")

    # empty-window receipt is deterministic
    ok(env.receipt([], "x")["run_hash"] == env.receipt([], "x")["run_hash"],
       "receipt determinism (empty)")

    # unknown arm refuses
    try:
        env.run("MID")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("unknown arm must refuse")

    print(f"[ev_replay] selftest OK — {checks} checks")
    return 0


def smoke(per_coin: int = 2) -> int:
    sel = iw.select(per_coin)
    env = ReplayEnv(windows=sel)
    gate_results: dict[str, Any] = {}
    for arm in ARMS:
        p, f = parity_gate(env, arm)
        det = determinism_gate(env, arm, f"smoke_{arm.lower()}")
        gate_results[arm] = {"parity_pass": p, "parity_fail": f,
                             "determinism": det}
        print(f"[ev_replay] {arm}: parity {p} pass / {f} fail · "
              f"determinism {'PASS' if det else 'FAIL'}")
        if f or not det:
            raise SystemExit(f"[ev_replay] GATE FAIL on {arm} — do not use")
    recs = env.run("JOIN")
    out = OUT_DIR / "ev_replay_v1_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env.receipt(recs, "smoke_join", gate_results),
                              indent=1, default=str))
    n = sum(len(r.fills) for r in recs)
    print(f"[ev_replay] smoke receipt: {len(recs)} windows, {n} fills -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="smoke", choices=["smoke"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return smoke(a.per_coin)


if __name__ == "__main__":
    raise SystemExit(main())
