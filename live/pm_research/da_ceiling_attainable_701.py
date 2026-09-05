"""DA: the ATTAINABLE companion to the 701% ceiling, on the 701%'s OWN surface.

WHY THIS EXISTS AND WHY THE ROUND-49 NUMBER COULD NOT BE REUSED. Q-DA-249
established by construction that `V_oracle` has r's disease -- two books
identical in everything the statistic can see, different in attainable value --
and measured a 10.0% overstatement.  That measurement was on the GATE-0 book:
355 actions of `static_cancel_value_cents`.  BE checked it at the artifact and
said, correctly, that the surfaces differ and the 10.0% does NOT transfer.  The
701% lives on a different book: the measured hour's 4,315 fills of
markout x shares.  So the 701% still had no companion on its own surface.
This computes it.

WHAT V_ORACLE IS HERE, read at `de_phase4_diag_runner.value_ceiling`:

    V_oracle = SUM over fills with v < 0 of |v|

One filter and one sum over FILLS.  It is blind to the fact that an overlay
cannot decline a fill: it cancels an ORDER.  BE names the same limit in its own
artifact -- "a real overlay cancels ORDERS and loses whatever fills follow by
cascade" -- and leaves it unquantified.  This quantifies it.

THE MECHANISM, AND WHY THE NON-OVERLAP IS NOT AN ASSUMPTION. A cancel issued on
a lineage at its generation's decision time takes effect after
`cancel_effective_latency_ms` and the order is then absent for
`repost_dwell_s`.  Every fill the lineage would have taken inside that window
is lost -- the losing ones (which is the point) and the winning ones (which is
the cost).  While the order is absent there is nothing to cancel, so a second
cancel on that lineage cannot start before the first window ends.  That makes
the choice per lineage an exact weighted-interval schedule with a fixed
blocking length, not a restriction imposed for tractability.

BUDGET. `k` counts FILLS REMOVED, the same unit BE's cells use, so
`attainable(k)` and BE's `ORACLE_at_this_k` are read off the same axis.

    python3 live/pm_research/da_ceiling_attainable_701.py --selftest
    python3 live/pm_research/da_ceiling_attainable_701.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

PROTOCOL = "P003_DA_CEILING_ATTAINABLE_701_V1"
REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "data/pm_5min/derived/de_section81_cache_12.pkl"
BE_NULL = REPO / "data/pm_5min/derived/be_ceiling_null_v1.json"
#: The V2 policy's own declared numbers, carried so the sweep always contains
#: the values the programme actually declared.
DECLARED_REPOST_DWELL_S = 2.0
DECLARED_CANCEL_LATENCY_S = 0.250
#: BE's five cells, in fills.
BE_BUDGETS = (107, 216, 432, 647, 1440)
#: The reproduction gate: this book or nothing.
GATE = {"n_fills": 4315, "n_negative": 2072, "n_positive": 2234,
        "n_zero": 9, "net_cents": 8598.758849499998,
        "V_oracle_cents": 60303.760723}


class AttainableRefused(RuntimeError):
    """The book cannot support the question asked of it."""


def load_book(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else CACHE
    if not p.is_file():
        raise AttainableRefused(f"REFUSED: no cached book at {p}")
    d = pickle.loads(p.read_bytes())
    ref = (d.get("fr") or {}).get("reference")
    if not ref:
        raise AttainableRefused("REFUSED: cache carries no reference book")
    lineages, fills = {}, []
    n_gen = 0
    for slug in sorted(ref):
        for side in sorted(ref[slug]):
            key = (slug, side)
            gens = []
            for g in ref[slug][side]:
                n_gen += 1
                tr = g.get("tranches") or []
                if not tr:
                    continue
                # v is markout_cents_per_share * shares. The markout already
                # carries the side sign -- verified by the reproduction gate,
                # which fails on the side-multiplied variant.
                vs = [(float(t["t"]),
                       float(t["markout_cents_per_share"]) * float(t["shares"]))
                      for t in tr]
                gens.append({"gen": g["gen"], "t0": float(g["t0"]),
                             "fills": vs})
                for t, v in vs:
                    fills.append(v)
            if gens:
                gens.sort(key=lambda x: (x["t0"], x["gen"]))
                lineages[key] = gens
    if not fills:
        raise AttainableRefused("REFUSED: book has zero fills")
    return {"lineages": lineages, "fills": fills, "n_generations": n_gen,
            "n_generations_with_fills": sum(len(v) for v in lineages.values()),
            "source": str(p),
            "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def book_stats(fills: list) -> dict:
    a = np.asarray(fills, dtype=float)
    return {"n_fills": int(a.size),
            "n_negative": int((a < 0).sum()),
            "n_positive": int((a > 0).sum()),
            "n_zero": int((a == 0).sum()),
            "net_cents": float(a.sum()),
            "V_oracle_cents": float(-a[a < 0].sum()),
            "mean_cents_per_fill": float(a.mean())}


def reproduction_gate(stats: dict, tol: float = 1e-6) -> dict:
    fields = {}
    for k, want in GATE.items():
        got = stats[k]
        fields[k] = {"want": want, "got": got,
                     "match": (abs(got - want) <= tol
                               if isinstance(want, float) else got == want)}
    return {"fields": fields, "n_fields_checked": len(fields),
            "status": "PASS" if all(f["match"] for f in fields.values())
                      else "FAIL"}


def oracle_curve(fills: list, kmax: int) -> np.ndarray:
    """Free-choice oracle: decline the k most negative fills, any k."""
    a = np.asarray(fills, dtype=float)
    neg = -np.sort(a[a < 0])                      # magnitudes, descending
    cum = np.concatenate([[0.0], np.cumsum(neg)])
    out = np.zeros(kmax + 1)
    n = min(kmax, neg.size)
    out[:n + 1] = cum[:n + 1]
    out[n + 1:] = cum[n]                          # budget beyond the negatives
    return out


def lineage_curve(gens: list, kmax: int, *, dwell_s: float,
                  latency_s: float, cascade: bool) -> np.ndarray:
    """Best value from cancels on ONE lineage, by fills removed.

    cascade=False is the POSITIVE CONTROL: each fill is independently
    declinable, which is exactly what V_oracle assumes."""
    if not cascade:
        vals = sorted((v for g in gens for _, v in g["fills"] if v < 0),
                      reverse=False)
        cur = np.zeros(kmax + 1)
        run = 0.0
        for i, v in enumerate(vals[:kmax], start=1):
            run += -v
            cur[i] = run
        if vals:
            cur[min(len(vals), kmax) + 1:] = run
        return cur

    L = latency_s + dwell_s
    # Every fill on the lineage, in time order.
    ft = sorted((t, v) for g in gens for t, v in g["fills"])
    times = np.array([t for t, _ in ft])
    vals = np.array([v for _, v in ft])
    csum = np.concatenate([[0.0], np.cumsum(vals)])

    # Candidate cancels: one per generation that has fills, decided at t0.
    cands = []
    for g in gens:
        start = g["t0"] + latency_s
        end = start + dwell_s
        lo = int(np.searchsorted(times, start, "left"))
        hi = int(np.searchsorted(times, end, "left"))
        if hi <= lo:
            continue
        w = hi - lo
        val = -(csum[hi] - csum[lo])              # value ADDED by removing
        cands.append((g["t0"], end, w, val, lo, hi))
    if not cands:
        return np.zeros(kmax + 1)
    cands.sort(key=lambda c: c[0])
    starts = np.array([c[0] for c in cands])
    # prev[i]: last candidate whose blocking window ended at or before this
    # candidate's decision time -- the order must be back before it can be
    # cancelled again.
    prev = []
    for c in cands:
        j = int(np.searchsorted(starts, c[0], "left")) - 1
        while j >= 0 and cands[j][1] > c[0]:
            j -= 1
        prev.append(j)

    NEG = -np.inf
    best = np.zeros((len(cands) + 1, kmax + 1))
    for i, c in enumerate(cands, start=1):
        _, _, w, val, _, _ = c
        row = best[i - 1].copy()
        if val > 0 and w <= kmax:
            base = best[prev[i - 1] + 1]
            cand = np.full(kmax + 1, NEG)
            cand[w:] = base[:kmax + 1 - w] + val
            row = np.maximum(row, cand)
        best[i] = row
    return best[len(cands)]


def merge_curves(curves: list, kmax: int) -> np.ndarray:
    """Exact max-plus convolution across lineages under one shared budget."""
    out = np.zeros(kmax + 1)
    for c in curves:
        nxt = out.copy()
        nz = np.nonzero(c > 0)[0]
        for j in nz:
            cand = np.full(kmax + 1, -np.inf)
            cand[j:] = out[:kmax + 1 - j] + c[j]
            nxt = np.maximum(nxt, cand)
        out = nxt
    return np.maximum.accumulate(out)


def attainable(book: dict, budgets=BE_BUDGETS, *,
               dwell_s: float = DECLARED_REPOST_DWELL_S,
               latency_s: float = DECLARED_CANCEL_LATENCY_S,
               cascade: bool = True) -> dict:
    if not book.get("fills") or not book.get("lineages"):
        raise AttainableRefused(
            "REFUSED: an empty book cannot be priced -- a zero attainable "
            "from a book with no fills is not a measurement (rule 15)")
    kmax = max(budgets)
    curves = [lineage_curve(g, kmax, dwell_s=dwell_s, latency_s=latency_s,
                            cascade=cascade)
              for g in book["lineages"].values()]
    merged = merge_curves(curves, kmax)
    orc = oracle_curve(book["fills"], kmax)
    V = book_stats(book["fills"])["V_oracle_cents"]
    rows = {}
    for k in budgets:
        att, o = float(merged[k]), float(orc[k])
        rows[str(k)] = {
            "k_fills": k,
            "oracle_cents": o,
            "oracle_capture_of_V": o / V if V else None,
            "attainable_cents": att,
            "attainable_capture_of_V": att / V if V else None,
            "attainable_share_of_oracle_at_k": (att / o) if o else None,
            "overstatement_pct_of_oracle_at_k": (100.0 * (o - att) / o
                                                 if o else None),
        }
    return {"dwell_s": dwell_s, "latency_s": latency_s, "cascade": cascade,
            "V_oracle_cents": V, "by_budget": rows}


def run_real() -> dict:
    book = load_book()
    stats = book_stats(book["fills"])
    gate = reproduction_gate(stats)
    if gate["status"] != "PASS":
        raise AttainableRefused(
            f"REFUSED: reproduction gate FAILED -- this is not the book the "
            f"701% was computed over: {gate['fields']}")
    # BE's five cells PLUS the unbudgeted case, which is the direct analogue
    # of the filed headline: the 701% is V_oracle/net with no budget at all,
    # so its companion must be attainable/net with no budget either.
    nf = stats["n_fills"]
    main = attainable(book, budgets=tuple(sorted(BE_BUDGETS + (nf,))))
    out = {
        "protocol": PROTOCOL,
        "book": {**stats, "n_generations": book["n_generations"],
                 "n_generations_with_fills": book["n_generations_with_fills"],
                 "fills_per_generation_with_fills":
                     stats["n_fills"] / book["n_generations_with_fills"],
                 "n_lineages": len(book["lineages"]),
                 "source": book["source"],
                 "source_sha256": book["source_sha256"]},
        "reproduction_gate": gate,
        "declared": {"repost_dwell_s": DECLARED_REPOST_DWELL_S,
                     "cancel_effective_latency_s": DECLARED_CANCEL_LATENCY_S},
        "attainable_at_declared_policy": main,
        "headline_unbudgeted": {
            "k_fills": stats["n_fills"],
            "net_cents": stats["net_cents"],
            "V_oracle_cents": stats["V_oracle_cents"],
            "V_oracle_pct_of_net":
                100.0 * stats["V_oracle_cents"] / stats["net_cents"],
            "attainable_cents":
                main["by_budget"][str(stats["n_fills"])]["attainable_cents"],
            "attainable_pct_of_net":
                100.0 * main["by_budget"][str(stats["n_fills"])][
                    "attainable_cents"] / stats["net_cents"],
            "attainable_share_of_oracle":
                main["by_budget"][str(stats["n_fills"])][
                    "attainable_share_of_oracle_at_k"],
            "ceiling_overstatement_pct":
                main["by_budget"][str(stats["n_fills"])][
                    "overstatement_pct_of_oracle_at_k"],
            "reading": ("the filed ceiling is 701.31% of net; the most any "
                        "overlay could attain once a cancel silences its "
                        "lineage for the declared latency+dwell is 516.11% "
                        "of net. Both are ORACLE bounds on a realised book "
                        "and neither is a forecast"),
        },
        "sensitivity_dwell_s": {
            str(d): attainable(book, dwell_s=d)["by_budget"]
            for d in (0.25, 1.0, 5.0)},
        "positive_control_cascade_off": attainable(book, cascade=False),
        "role": "REPORTED, NOT ENFORCED (rule 14). This is the attainable "
                "companion R-535(E) requires beside a ceiling figure. It "
                "promotes nothing and clears no gate.",
        "limits": [
            "one hour, 12 windows, G=0 complete UTC days, cluster n=1 -- a "
            "structural companion, never a population estimate",
            "the blocking model is the declared latency plus dwell; the real "
            "cascade also changes WHICH generations exist and what they would "
            "have filled, which can only widen the gap, never narrow it",
            "budget k counts FILLS REMOVED so the axis matches BE's cells; an "
            "overlay's real budget is in CANCELS and is not the same axis -- "
            "and because of that unit, attainable RISES with dwell here (one "
            "cancel sweeps more losers) rather than falling as it does when "
            "the budget is in cancels. The cascade's COST is isolated by the "
            "cascade-off control, not by the dwell sweep",
            "values are the book's own markout x shares, unchanged; no fee, "
            "queue reset or terminal inventory enters here",
        ],
    }
    if BE_NULL.is_file():
        be = json.loads(BE_NULL.read_text())
        cells = be.get("cells") or {}
        cmp_ = {}
        for name, c in cells.items():
            k = str(c["k"])
            if k in main["by_budget"]:
                mine = main["by_budget"][k]["oracle_capture_of_V"]
                cmp_[name] = {
                    "k": c["k"], "BE_ORACLE_at_this_k": c["ORACLE_at_this_k"],
                    "DA_oracle_capture_of_V": mine,
                    "agree_to_1e_9": abs(mine - c["ORACLE_at_this_k"]) < 1e-9,
                    "DA_attainable_capture_of_V":
                        main["by_budget"][k]["attainable_capture_of_V"],
                }
        out["cross_check_against_BE"] = cmp_
        out["oracle_curve_agrees_with_BE_at_every_cell"] = all(
            v["agree_to_1e_9"] for v in cmp_.values())
    return out


def _toy():
    """Two lineages. A: three losers 1 s apart (a 2 s dwell blocks the 2nd
    and 3rd). B: one loser alone. Hand-computable."""
    return {"lineages": {
        ("s", "A"): [{"gen": 1, "t0": 0.0, "fills": [(0.0, -10.0)]},
                     {"gen": 2, "t0": 1.0, "fills": [(1.0, -10.0)]},
                     {"gen": 3, "t0": 2.0, "fills": [(2.0, -10.0)]}],
        ("s", "B"): [{"gen": 1, "t0": 100.0, "fills": [(100.0, -5.0)]}]},
        "fills": [-10.0, -10.0, -10.0, -5.0]}


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    toy = _toy()
    # cascade OFF: every fill independently declinable == the oracle.
    off = attainable(toy, budgets=(1, 2, 3, 4), cascade=False,
                     dwell_s=2.0, latency_s=0.0)
    orc = oracle_curve(toy["fills"], 4)
    ok(all(abs(off["by_budget"][str(k)]["attainable_cents"] - orc[k]) < 1e-9
           for k in (1, 2, 3, 4)),
       "POSITIVE CONTROL: with the cascade OFF, attainable == the oracle at "
       "every budget -- so a gap this reports is the cascade and not the "
       "estimator")
    # cascade ON, latency 0, dwell 2: on lineage A one cancel at t=0 removes
    # the fills at 0 and 1 (window [0,2)); the fill at t=2 needs a second
    # cancel, which cannot start before t=2 -- and it can. So all three are
    # reachable, but only in pairs the windows allow.
    on = attainable(toy, budgets=(1, 2, 3, 4), cascade=True,
                    dwell_s=2.0, latency_s=0.0)
    b = on["by_budget"]
    # MY FIRST EXPECTATION HERE WAS WRONG AND THE SUITE CAUGHT IT. I wrote
    # 5.0c believing only lineage B offered a single-fill cancel. It does not:
    # A's THIRD generation opens its window at t=2.0, by which time the
    # earlier fills are past, so that window holds exactly one fill worth 10c.
    # The corrected expectation is 10.0c. Kept as a comment because the wrong
    # number was a claim about the fixture made without reading it.
    ok(abs(b["1"]["attainable_cents"] - 10.0) < 1e-9,
       f"CASCADE k=1: the best single-fill cancel is A's LAST generation, "
       f"whose window opens after the earlier fills are gone "
       f"({b['1']['attainable_cents']:.1f}c)")
    ok(abs(b["2"]["attainable_cents"] - 20.0) < 1e-9,
       f"CASCADE k=2: one cancel on A takes both its first two fills "
       f"({b['2']['attainable_cents']:.1f}c)")
    ok(abs(b["4"]["attainable_cents"] - 35.0) < 1e-9,
       f"CASCADE k=4: all four ({b['4']['attainable_cents']:.1f}c)")
    # THE GAP NEEDS A BOOK WHERE THE WINDOW CATCHES A WINNER -- which is the
    # mechanism, not a contrived case: it is what "loses whatever fills follow"
    # means. The toy above has no winners, so it shows no gap, and saying so is
    # the honest form.
    ok(abs(b["4"]["attainable_cents"] - orc[4]) < 1e-9,
       f"NO GAP WHERE NONE EXISTS: on an all-loser toy the cascade costs "
       f"nothing and attainable meets the oracle at full budget "
       f"({b['4']['attainable_cents']:.1f}c) -- the instrument does not "
       f"manufacture a gap")

    # A cancel that would remove a WINNER must not be taken for free.
    win = {"lineages": {("s", "A"): [
        {"gen": 1, "t0": 0.0, "fills": [(0.0, -1.0), (0.5, +100.0)]}]},
        "fills": [-1.0, 100.0]}
    w = attainable(win, budgets=(1, 2), cascade=True, dwell_s=2.0,
                   latency_s=0.0)
    worc = oracle_curve(win["fills"], 2)
    ok(w["by_budget"]["2"]["attainable_cents"] == 0.0,
       "COST IS CHARGED: a cancel whose window also removes a +100c winner is "
       "worth -99c and is therefore NOT taken -- attainable 0, while the "
       "oracle would bank the 1c loser")
    ok(w["by_budget"]["2"]["attainable_cents"] < worc[2],
       f"THE GAP IS REAL: oracle {worc[2]:.1f}c vs attainable "
       f"{w['by_budget']['2']['attainable_cents']:.1f}c -- the whole "
       f"overstatement, on two fills")

    # A DEFECT OF MINE THAT THIS SUITE CAUGHT, AND THE ASSERTION IS GONE.
    # I first wrote "MONOTONE in dwell: attainable is non-increasing", copying
    # the intuition from the Gate-0 surface where a longer dwell only silences
    # more. It PASSED on this toy only because both values were EQUAL at 35.0c
    # -- a comparison that never discriminated -- and it is FALSE on the real
    # book, where attainable RISES from 229.9% of net at 0.25 s to 612.7% at
    # 5 s. The reason is the budget unit: k counts FILLS REMOVED, so a longer
    # window is not a penalty, it is leverage -- one cancel sweeps more
    # losers. The cascade's cost shows up when the window also sweeps WINNERS,
    # which is what the cascade-off control isolates. So the direction is
    # MEASURED and reported, never asserted.
    dl = {d: attainable(toy, budgets=(4,), dwell_s=d, latency_s=0.0
                        )["by_budget"]["4"]["attainable_cents"]
          for d in (0.25, 1.0, 5.0)}
    ok(all(v >= 0 for v in dl.values()),
       f"DWELL IS SWEPT, NOT ASSERTED: {dl} -- the direction is a property of "
       f"the book and the budget unit, and this instrument reports it rather "
       f"than pinning a sign that is false on the real surface")
    ok(all(b[str(k)]["attainable_cents"] <= b[str(k + 1)]["attainable_cents"]
           + 1e-9 for k in (1, 2, 3)),
       "MONOTONE in budget: attainable is non-decreasing in k")

    # KNOWN-BAD, both directions.
    for bad, why in ((Path("/nonexistent.pkl"), "an absent cache"),):
        try:
            load_book(bad)
            ok(False, f"KNOWN-BAD: accepted {why} -- must refuse")
        except AttainableRefused:
            ok(True, f"KNOWN-BAD: refuses {why}")
    try:
        attainable({"lineages": {}, "fills": []}, budgets=(1,))
        ok(False, "KNOWN-BAD: priced an empty book -- must refuse")
    except AttainableRefused:
        ok(True, "KNOWN-BAD: an empty book REFUSES rather than reporting a "
                 "zero attainable")
    bad_stats = dict(GATE, n_fills=1)
    g = reproduction_gate({**GATE, "n_fills": 1, "mean_cents_per_fill": 0.0})
    ok(g["status"] == "FAIL",
       "GATE FALSIFIER: a book with the wrong fill count FAILS the "
       "reproduction gate -- the gate can fire, so a PASS means something")
    g2 = reproduction_gate({**GATE, "mean_cents_per_fill": 0.0})
    ok(g2["status"] == "PASS",
       "GATE POSITIVE CONTROL: the declared book PASSES")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = run_real()
        txt = json.dumps(out, indent=2, sort_keys=True)
        if a.output:
            a.output.write_text(txt)
        print(txt[:3000])
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
