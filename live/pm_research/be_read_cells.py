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


def _select_by_exact_count(gens, scores, k_target: int):
    """Top-k by this arm's OWN ranking, at a count SET FROM OUTSIDE.

    This is "lower the incumbent's theta until it cancels k actions",
    expressed the way the module already selects: the incumbent's ranking is
    untouched and only its cutoff moves. `_select_by_count` takes a FRACTION
    of the arm's own action count, which is not the same thing when the
    target count comes from the other arm."""
    gmax = {kk: max(scores[i] for i in gens[kk]) for kk in gens}
    order = sorted(gens, key=lambda kk: (-gmax[kk], kk))   # R-234 tie-break
    k = max(0, min(int(k_target), len(order)))
    chosen = order[:k]
    cut = gmax[chosen[-1]] if chosen else float("inf")
    return chosen, {kk: cut for kk in chosen}, cut


def _toxicity(rows, gens, scores, chosen, theta, latency_ms) -> dict:
    """harm avoided / good flow sacrificed / rho, for ONE arm's selection.

    THE CORRECTED OBJECTIVE (USER, round 27). `net_cents` is
    harm_avoided MINUS sacrifice, so a model that cancels indiscriminately
    raises BOTH terms and its net can climb while the SEPARATION gets worse.
    These are the two terms kept apart.

    WHAT THE QUANTITY IS, read from the builder rather than assumed:
    `preventable_value_cents` is `sum(-markout_cents_per_share * shares)` over
    the tranches a cancel at t+L would have prevented, and
    `markout_cents_per_share` is `sgn * (mid(t_fill + MARKOUT_S) - fill_level)
    * 100` -- the fill's OWN P&L over the 5 s after it, signed so positive
    means the fill was good. So a NEGATIVE markout is adverse post-fill drift
    and enters here as HARM AVOIDED; a positive one is good flow the cancel
    FORFEITED. rho = adverse drift dodged per cent of good flow given up."""
    L = str(latency_ms)
    harm = sac = 0.0
    n_pos = n_neg = n_zero = 0
    for gk in chosen:
        i = next((j for j in gens[gk] if scores[j] >= theta[gk]), None)
        if i is None:
            continue
        r = rows[i]
        v = (r["latency"][L]["preventable_value_cents"]
             if r.get("any_fill_ahead") and "latency" in r else 0.0)
        if v > 0:
            harm += v; n_pos += 1
        elif v < 0:
            sac += -v; n_neg += 1
        else:
            n_zero += 1
    return {"harm_avoided_cents": harm, "sacrifice_cents": sac,
            "net_cents": harm - sac,
            "rho_captured_over_sacrificed": (harm / sac) if sac > 0 else None,
            "n_cancels_avoiding_harm": n_pos,
            "n_cancels_forfeiting_good_flow": n_neg,
            "n_cancels_worth_nothing": n_zero,
            "markout_horizon_s": 5.0,
            "fill_horizon_s": 1.0}


def matched_volume(rows, cand_scores, inc_scores, theta: float,
                   latency_ms: int) -> dict:
    """THE PRIMARY STATISTIC (interim declaration, USER round 26).

    The candidate acts at its FROZEN theta. The incumbent is then given the
    candidate's REALISED count by lowering its own cutoff. Both arms spend the
    same cancellation budget, so what is left is ranking quality.

    It also returns the EXACT decomposition the declaration fixed before the
    read: BY_THRESHOLD increment == VOLUME + QUALITY. That identity telescopes
    algebraically and is CHECKED here rather than narrated."""
    gens = FM._gen_index(rows)
    c_sel, c_th, c_cut = FM._select_by_threshold(gens, cand_scores, theta)
    i_sel, i_th, i_cut = FM._select_by_threshold(gens, inc_scores, theta)
    m_sel, m_th, m_cut = _select_by_exact_count(gens, inc_scores, len(c_sel))

    cb = FM._cancel_value(rows, gens, cand_scores, c_sel, c_th, latency_ms)
    ib = FM._cancel_value(rows, gens, inc_scores, i_sel, i_th, latency_ms)
    mb = FM._cancel_value(rows, gens, inc_scores, m_sel, m_th, latency_ms)

    wins = sorted(set(cb) | set(mb))
    by_window = {w: cb.get(w, 0.0) - mb.get(w, 0.0) for w in wins}
    cand_net = math.fsum(cb.values())
    inc_net_at_theta = math.fsum(ib.values())
    inc_net_matched = math.fsum(mb.values())

    tox_c = _toxicity(rows, gens, cand_scores, c_sel, c_th, latency_ms)
    tox_i = _toxicity(rows, gens, inc_scores, i_sel, i_th, latency_ms)
    tox_m = _toxicity(rows, gens, inc_scores, m_sel, m_th, latency_ms)

    quality = cand_net - inc_net_matched          # == the matched-volume stat
    volume = inc_net_matched - inc_net_at_theta   # what volume alone bought
    by_threshold = cand_net - inc_net_at_theta
    resid = abs((volume + quality) - by_threshold)

    n_act = len(gens)
    return {
        "statistic": "MATCHED_VOLUME",
        "theta_declared": theta,
        "candidate_n_cancelled": len(c_sel),
        "incumbent_n_cancelled_at_theta": len(i_sel),
        "incumbent_n_cancelled_matched": len(m_sel),
        "counts_matched": len(m_sel) == len(c_sel),
        "incumbent_cutoff_at_theta": i_cut,
        "incumbent_cutoff_matched": m_cut,
        "incumbent_cutoff_was_LOWERED": (m_cut <= i_cut),
        "n_actions": n_act,
        "candidate_delivered_rate": len(c_sel) / n_act if n_act else None,
        "incumbent_delivered_rate_at_theta": (len(i_sel) / n_act if n_act
                                              else None),
        "candidate_net_cents": cand_net,
        "incumbent_net_cents_at_theta": inc_net_at_theta,
        "incumbent_net_cents_matched": inc_net_matched,
        "candidate_cents_per_cancellation": (cand_net / len(c_sel)
                                             if c_sel else None),
        "incumbent_cents_per_cancellation_at_theta": (
            inc_net_at_theta / len(i_sel) if i_sel else None),
        "incumbent_cents_per_cancellation_matched": (
            inc_net_matched / len(m_sel) if m_sel else None),
        "MATCHED_VOLUME_increment_cents": quality,
        "toxicity_candidate": tox_c,
        "toxicity_incumbent_at_theta": tox_i,
        "toxicity_incumbent_matched": tox_m,
        "rho_candidate": tox_c["rho_captured_over_sacrificed"],
        "rho_incumbent_matched": tox_m["rho_captured_over_sacrificed"],
        "rho_advantage_at_matched_volume": (
            None if (tox_c["rho_captured_over_sacrificed"] is None
                     or tox_m["rho_captured_over_sacrificed"] is None)
            else tox_c["rho_captured_over_sacrificed"]
            - tox_m["rho_captured_over_sacrificed"]),
        "decomposition": {
            "by_threshold_increment_cents": by_threshold,
            "volume_term_cents": volume,
            "quality_term_cents": quality,
            "identity_residual_cents": resid,
            "identity_holds": resid < 1e-6,
            "identity": "BY_THRESHOLD == VOLUME + QUALITY",
        },
        "increment_by_window": by_window,
        "n_windows": len(wins),
        "latency_ms": latency_ms,
        "unit": "ACTION",
        "baseline": "INCUMBENT AT THE CANDIDATE'S OWN CANCELLATION COUNT",
    }


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
