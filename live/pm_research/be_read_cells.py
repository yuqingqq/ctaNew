"""THE CONSUMER: the declared cells, computed from a TWO-ARM feed.

Round 25. The producer now emits both arms (`score` and `score_incumbent`)
in one replay pass; this reads that feed and computes the cells the read
declaration froze BEFORE the read -- the estimand, the cells, the nulls, the
sidedness and the cluster disclosure were all fixed at `4330b79`, so running
this is EXECUTING a declaration, not selecting on seen data.

IT SELECTS NOTHING AND CHOOSES NOTHING. Every parameter it uses is read from
a committed declaration or a frozen module constant, and a cell it cannot
compute is REPORTED AS NOT COMPUTED with its reason -- never dropped, because
an unreported cell is invisible and the Holm denominator was declared at 18
before any of this existed.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import be_forward_metric as FM          # noqa: E402
import be_operating_point as OP         # noqa: E402
import phase2_declaration as PD         # noqa: E402
import phase2_iter011 as I11            # noqa: E402

DECL_DIR = HERE / "declarations"
FAMILY = json.loads((DECL_DIR / "be_forward_family_declaration_v1.json"
                     ).read_text())
READ_DECL = json.loads((DECL_DIR / "be_read_declaration_v1.json").read_text())

#: The sensitivity arm has no operating point ANYWHERE in this repo -- it
#: appears only in the family's cell list. Its six cells are therefore
#: reported NOT COMPUTED rather than silently omitted, and the Holm
#: denominator stays 18, exactly as the family's own `holm_note` requires:
#: "unreported cells are invisible".
UNCOMPUTABLE_ARM = "FROZEN_FROM_A_CONSUMED_DAY"


class ReadCellsRefused(RuntimeError):
    """A named refusal."""


def load_two_arm_feed(path: Path, latency_ms: int) -> dict:
    """Per-coin (rows, cand_scores, inc_scores), streamed.

    REFUSES a one-arm feed BY NAME. The failure this prevents is the quiet
    one: with the incumbent column missing, a caller could pass the candidate
    twice and get an increment of exactly zero that looks like a result."""
    per: dict = collections.defaultdict(
        lambda: {"rows": [], "cand": [], "inc": []})
    n, missing = 0, 0
    with path.open() as fh:
        for line in fh:
            fr = json.loads(line)
            n += 1
            if "score_incumbent" not in fr:
                raise ReadCellsRefused(
                    f"REFUSED: feed row {n} carries no `score_incumbent`. "
                    f"This is a ONE-ARM feed and the declared estimand is an "
                    f"increment OVER the incumbent; computing it from one arm "
                    f"would compare the candidate with itself and return a "
                    f"zero that looks like a measurement.")
            si = fr["score_incumbent"]
            if si is None:
                missing += 1
                continue
            coin = fr["slug"].split("-", 1)[0]
            b = per[coin]
            b["rows"].append(FM.feed_row_to_eval_row(fr, latency_ms))
            b["cand"].append(float(fr["score"]))
            b["inc"].append(float(si))
    if not per:
        raise ReadCellsRefused(
            f"REFUSED: no scored rows in {path}. An empty read is a FAILURE, "
            f"not an empty result (R-141).")
    return {"per_coin": dict(per), "n_feed_rows": n,
            "n_rows_without_an_incumbent_score": missing}


def compute(feed_path: Path, latency_ms: int = None) -> dict:
    L = PD.TARGET_LATENCY_MS if latency_ms is None else latency_ms
    loaded = load_two_arm_feed(feed_path, L)
    per = loaded["per_coin"]
    budgets = list(FAMILY["factors"]["budgets"])
    coins = [c for c in FAMILY["factors"]["coins"] if c in per]

    cells, not_computed = {}, {}
    for coin in coins:
        b = per[coin]
        rows, cs, isc = b["rows"], b["cand"], b["inc"]
        # THROUGH THE FENCE, never around it: `require_fenced_op` refuses a
        # raw declaration by name because it carries no operating-point
        # token. Verified: the raw form IS refused, the fenced form returns
        # theta with token_recomputed True.
        op = FM.require_operating_point(OP.op_declaration_for(coin))
        for bud in budgets:
            frac = int(bud.rstrip("%")) / 100.0
            # PRIMARY: causal, theta declared before the read.
            key = f"BY_THRESHOLD/FROZEN_FROM_TRAIN_QUANTILE/{coin}/{bud}"
            cell = FM.increment(rows, cs, isc, op=op, latency_ms=L,
                                convention="BY_THRESHOLD", budget=frac,
                                budget_key=bud)
            cell["null"] = FM.paired_null(cell["increment_by_window"])
            cells[key] = cell
            # BRIDGE: non-causal by construction, acknowledged explicitly.
            key2 = f"BY_COUNT/NO_OPERATING_POINT/{coin}/{bud}"
            cell2 = FM.increment(rows, cs, isc, op=None, latency_ms=L,
                                 convention="BY_COUNT", budget=frac,
                                 bridge_to_development_ack=True)
            cell2["null"] = FM.paired_null(cell2["increment_by_window"])
            cells[key2] = cell2
        for bud in budgets:
            k = f"BY_THRESHOLD/{UNCOMPUTABLE_ARM}/{coin}/{bud}"
            not_computed[k] = (
                "NOT COMPUTED: no operating point of form "
                f"{UNCOMPUTABLE_ARM!r} exists in this repo -- the arm appears "
                "only in the family's cell list. It is reported here rather "
                "than omitted, and the Holm denominator stays at the declared "
                "18, because an unreported cell is invisible to a reader.")

    declared = set(FAMILY["cells"])
    covered = set(cells) | set(not_computed)
    if covered != declared:
        raise ReadCellsRefused(
            f"REFUSED: the computed+not-computed set is not the declared "
            f"family. Missing {sorted(declared - covered)}; extra "
            f"{sorted(covered - declared)}. A cell outside the declared "
            f"family is a cell chosen after the fact.")

    # Holm over the DECLARED denominator, on the PRIMARY convention only.
    prim = {k: v for k, v in cells.items() if k.startswith("BY_THRESHOLD/")}
    ps = sorted(((v["null"]["p_value"], k) for k, v in prim.items()))
    m = FAMILY["holm_denominator"]
    holm, prev = {}, 0.0
    for i, (p, k) in enumerate(ps):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        holm[k] = {"p_raw": p, "p_holm": adj,
                   "rank": i + 1, "denominator": m}
    return {
        "protocol": "BE_READ_CELLS_V1",
        "day": READ_DECL["read_day"],
        "governed_by": "the read declaration frozen before the read",
        "latency_ms": L,
        "feed": {"path": str(feed_path), **{
            k: loaded[k] for k in ("n_feed_rows",
                                   "n_rows_without_an_incumbent_score")}},
        "cells": cells,
        "cells_not_computed": not_computed,
        "holm": holm,
        "n_declared": len(declared), "n_computed": len(cells),
        "selects_nothing": True,
    }
