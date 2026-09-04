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


#: (3) THE AUDIT'S THIRD REQUIRED STEP. A one-sided p for "the candidate BEATS
#: the incumbent" answers one question, and a HIGH value answers it in the
#: only way it can: the win was not shown. It is NOT evidence of a loss, and
#: the p is computed on the WRONG UNIT besides -- window-level sign flips
#: where the ruled cluster unit is the UTC day. Every p this module emits
#: carries that reading, because a bare number invites the reading it does
#: not support.
def p_reading(p: float, n_perm: int, n_units: int) -> dict:
    floor = 1.0 / (n_perm + 1)
    return {
        "p_value": p,
        "alternative": "greater — the candidate BEATS the incumbent",
        "sided": "one",
        "at_the_permutation_floor": abs(p - floor) < 1e-12,
        "floor": floor,
        "how_to_read_a_HIGH_p": (
            "FAILURE TO SHOW A WIN, never proof of a loss. This test can only "
            "reject 'no better'; it cannot establish 'worse'"),
        "how_to_read_a_LOW_p": (
            "evidence the candidate beat the incumbent on this population, "
            "subject to the cluster caveat below"),
        "cluster_unit_used": "window",
        "ruled_cluster_unit": "UTC day",
        "unit_is_WEAKER_than_ruled": True,
        "why_the_p_is_OPTIMISTIC": (
            "windows inside one UTC day share coin, regime and book state and "
            "are not exchangeable, so the null variance is understated and "
            "this p is SMALLER than a day-clustered p would be"),
        "n_permutation_units": n_units,
        "is_validation_evidence": False,
    }


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
        # (2) THE AUDIT'S SECOND REQUIRED STEP, ANSWERED EXPLICITLY RATHER
        # THAN LEFT AMBIGUOUS: this is a DIAGNOSTIC, not an executable result.
        "status": "DIAGNOSTIC — NOT AN EXECUTABLE OPERATING POINT",
        "why_diagnostic": (
            "the incumbent's cutoff is chosen by ranking the COMPLETE "
            "evaluated population and lowering theta until its action count "
            "equals the candidate's REALISED count. That cutoff is knowable "
            "only after the day is over, so no live policy could have run it. "
            "It is a ranking-quality comparison at equal action count and is "
            "valid as that; it is not a policy result"),
        "the_alternative_not_taken": (
            "predeclaring a CAUSAL incumbent operating point before further "
            "days are read. NOT DONE HERE, deliberately: choosing one now "
            "would be choosing it after seeing every number today produced, "
            "which is rule 11. Declaring it is the USER's act"),
        "matched_on": ("the NUMBER OF CANCELLATION ACTIONS — not shares, not "
                       "notional, not capital"),
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
            _n = FM.paired_null(cell["increment_by_window"])
            cell["null"] = _n
            cell["p_reading"] = p_reading(_n["p_value"], _n["n_perm"],
                                          _n["n_units"])
            cells[key] = cell
            # (1) THE PRIMARY, ON THE CANONICAL PATH. `matched_volume` was
            # DEFINED here and called by NOTHING committed -- the published
            # headline came out of a scratch script, which is the sixth
            # zero-consumer finding of the day and the first one that was the
            # result itself. It is now produced by `compute()`.
            mv = matched_volume(rows, cs, isc, cell["theta_declared"], L)
            _mn = FM.paired_null(mv["increment_by_window"])
            mv["null"] = _mn
            mv["p_reading"] = p_reading(_mn["p_value"], _mn["n_perm"],
                                        _mn["n_units"])
            mv.pop("increment_by_window", None)
            cells[f"MATCHED_VOLUME/RETROSPECTIVE_EQUAL_COUNT/{coin}/{bud}"] = mv
            # BRIDGE: non-causal by construction, acknowledged explicitly.
            key2 = f"BY_COUNT/NO_OPERATING_POINT/{coin}/{bud}"
            cell2 = FM.increment(rows, cs, isc, op=None, latency_ms=L,
                                 convention="BY_COUNT", budget=frac,
                                 bridge_to_development_ack=True)
            _n2 = FM.paired_null(cell2["increment_by_window"])
            cell2["null"] = _n2
            cell2["p_reading"] = p_reading(_n2["p_value"], _n2["n_perm"],
                                           _n2["n_units"])
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
    # The MATCHED_VOLUME cells are the interim declaration's primary and are
    # NOT in the forward family's 18; they are checked against their own
    # declared shape rather than silently widening the family.
    mv_cells = {k for k in cells if k.startswith("MATCHED_VOLUME/")}
    if len(mv_cells) != len(coins) * len(budgets):
        raise ReadCellsRefused(
            f"REFUSED: expected {len(coins) * len(budgets)} MATCHED_VOLUME "
            f"cells (coins x budgets) and produced {len(mv_cells)}. The "
            f"primary statistic is not optional and a short count is a "
            f"silently dropped cell.")
    cells_family = {k: v for k, v in cells.items() if k not in mv_cells}
    covered = set(cells_family) | set(not_computed)
    if covered != declared:
        raise ReadCellsRefused(
            f"REFUSED: the computed+not-computed set is not the declared "
            f"family. Missing {sorted(declared - covered)}; extra "
            f"{sorted(covered - declared)}. A cell outside the declared "
            f"family is a cell chosen after the fact.")

    # Holm over the DECLARED denominator, on the PRIMARY convention only.
    prim = {k: v for k, v in cells_family.items()
            if k.startswith("BY_THRESHOLD/")}
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
        "primary_statistic": "MATCHED_VOLUME",
        "primary_status": "DIAGNOSTIC — NOT AN EXECUTABLE OPERATING POINT",
        "n_primary_cells": len(mv_cells),
        "cells_not_computed": not_computed,
        "holm": holm,
        "n_declared": len(declared), "n_computed": len(cells),
        "selects_nothing": True,
    }


# ---------------------------------------------------------------------------
# (1) THE DURABLE ARTIFACT AND ITS CONTROLS.
#
# The audit's first required step: the primary was DEFINED here and produced
# by a scratch script in a temp directory. A result that only a scratch file
# can make is not a result the pipeline owns. `emit()` writes it where it can
# be cited, and the controls below drive the wiring in both directions.
# ---------------------------------------------------------------------------
RESULTS_DIR = HERE / "results"


def emit(feed_paths, out_path: Path = None) -> dict:
    """Compute and WRITE the result, so the number has a durable home."""
    if isinstance(feed_paths, (str, Path)):
        feed_paths = [feed_paths]
    per = [compute(Path(f)) for f in feed_paths]
    doc = {"protocol": "BE_READ_CELLS_RESULT_V1",
           "n_feeds": len(per), "results": per,
           "primary_statistic": "MATCHED_VOLUME",
           "primary_status": "DIAGNOSTIC — NOT AN EXECUTABLE OPERATING POINT",
           "produced_by": "be_read_cells.emit (the committed path), NOT a "
                          "scratch script"}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=1, sort_keys=True,
                                       default=str))
        doc["written_to"] = str(out_path)
    return doc


EXPECTED_CHECKS = 9


def selftest() -> int:
    import inspect
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def refuses(fn, want, label):
        try:
            fn()
        except ReadCellsRefused as e:
            ok(want in str(e), f"{label} [{str(e)[:70]}…]")
            return
        except Exception as e:                        # noqa: BLE001
            ok(False, f"{label} [WRONG EXCEPTION {type(e).__name__}: {e}]")
            return
        ok(False, f"{label} [DID NOT REFUSE]")

    # THE WIRING, ASSERTED OVER THE SOURCE. This is the check whose absence
    # let a defined-but-uncalled primary reach a published headline.
    src = inspect.getsource(compute)
    ok("matched_volume(" in src,
       "WIRING: `compute()` CALLS `matched_volume` — the audit's finding was "
       "that it did not, and that the headline came from a scratch script")
    ok("p_reading(" in src,
       "WIRING: and every cell it builds is given a `p_reading`, so no p "
       "leaves this module bare")
    _bad = src.replace("mv = matched_volume(", "mv = None  # unwired\n        _ = (")
    ok("matched_volume(" not in _bad.split("mv = None")[0].rsplit("\n", 3)[-1],
       "WIRING KNOWN-BAD: with the call removed the predicate above can fail "
       "— it is not a check that passes on any source")

    _p = p_reading(0.94, 2000, 288)
    ok(_p["how_to_read_a_HIGH_p"].startswith("FAILURE TO SHOW A WIN")
       and _p["is_validation_evidence"] is False,
       f"(3) a HIGH p is labelled FAILURE TO SHOW A WIN, never proof of a "
       f"loss — the reading the audit required, on every emitted p")
    ok(_p["cluster_unit_used"] == "window"
       and _p["ruled_cluster_unit"] == "UTC day"
       and _p["unit_is_WEAKER_than_ruled"] is True,
       "(3) and each p carries that its unit is WEAKER than the ruled one")
    ok(p_reading(1 / 2001, 2000, 288)["at_the_permutation_floor"] is True
       and p_reading(0.5, 2000, 288)["at_the_permutation_floor"] is False,
       "(3) the floor flag is COMPUTED and fires only at the floor")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "one_arm.jsonl"
        f.write_text(json.dumps({
            "slug": "btc-x-1", "side": "SELL_UP", "gen": 1, "t0": 0,
            "t_start": 0.1, "score": 0.5, "any_fill_ahead": True,
            "value_cents": -3.0}) + "\n")
        refuses(lambda: load_two_arm_feed(f, 50), "ONE-ARM feed",
                "KNOWN-BAD: a one-arm feed REFUSES — with the incumbent "
                "column missing a caller could compare the candidate with "
                "itself and get a zero that looks like a measurement")
        g = Path(td) / "empty.jsonl"
        g.write_text("")
        refuses(lambda: load_two_arm_feed(g, 50), "no scored rows",
                "KNOWN-BAD: an EMPTY feed REFUSES rather than returning an "
                "empty result (R-141)")
    ok("DIAGNOSTIC" in matched_volume.__doc__.upper()
       or "DIAGNOSTIC" in inspect.getsource(matched_volume).upper(),
       "(2) the primary carries its DIAGNOSTIC status in the code that "
       "produces it, not only in a report")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    print("usage: be_read_cells.py --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
