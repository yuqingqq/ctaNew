"""THE PER-DAY DECISION LEDGER (R-765, the USER's ruling).

    "The sealing mechanism is stupid, make sure to store the numbers after
     each run, and record everything we can to avoid rerun"
                                  -- THE USER, 2026-09-07T07:47:01Z

WHAT THIS IS FOR. A day costs ~86 minutes on the heavy lock. Every
statistic anyone later wants that was not written down costs another one.
This persists what the runner HELD IN MEMORY at the moment it computed
the day, so a later question is answered by reading a file instead of
re-running the day.

WHAT A READER CAN RECOMPUTE FROM THIS ALONE, WITH THE BOOK ABSENT:
  D_E0                the observed excess, and the arm/baseline values it
                      is the difference of;
  Z, null_mean/sd     from the FULL null draw vector, not its summary;
  p one- AND two-sided  ditto -- the summary could give neither;
  the FILLS LEG       per fill: level, size, side, mid at fill, mid at
                      markout, so the maker P&L is a sum a reader forms;
  rho = adverse/spread  per fill: spread captured = level - mid_at_fill,
                      adverse = mid_at_markout - mid_at_fill, both signed.

WHAT IT CANNOT, AND THIS IS NAMED RATHER THAN GLOSSED: the INVENTORY LEG
and inventory before/after. No inventory state exists in the runner at
the point the day is computed -- the fill record carries no position and
`replay` returns none -- so it is not stored, because storing a field
nobody computed would be a fabricated number in a file whose whole
purpose is to be trusted later. Recording it needs a change in BE's
replay, which is not this seat's to make.

THE BYTES ARE GITIGNORED (`data/`); the RECEIPT carries the ledger's
path, sha256, row count and schema version, so the artifact that is
tracked names the artifact that is not, by digest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

#: v2 (DE 126): BE 96 (c707eb8) made the fill record carry the position at
#: the fill on the arm AND the 0-cancel baseline, so the inventory leg
#: stopped being ABSENT_UNTIL_BE_96 and became a number this file can
#: carry. v1 ledgers have no inventory fields and say so; v2 ledgers do.
SCHEMA_VERSION = 2
LEDGER_PREFIX = "p003_de_decision_ledger"


class LedgerRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def ledger_name(day: str, stamp: str) -> str:
    return f"{LEDGER_PREFIX}_{day.replace('-', '')}__{stamp}.jsonl.gz"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_ledger(path, day: str, per_arm: dict,
                 *, buy_side: str) -> dict:
    """One day's ledger. `per_arm[arm]` is what the runner held.

    ROWS ARE STORED VERBATIM as the producing code shaped them -- BE's
    fill records and decision rows are copied, not re-typed into a schema
    of mine. A re-typed copy is a second definition that drifts from the
    first, and the point of this file is to be the SAME numbers the
    receipt was computed from."""
    path = Path(path)
    n = 0
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({
            "row": "HEADER", "schema_version": SCHEMA_VERSION, "day": day,
            "ruling": "R-765, the USER's: store the numbers after each run",
            "what_cannot_be_recomputed_from_this": {
                "nothing_the_ruling_asked_for": "BE 96 (c707eb8) closed the "
                    "one gap. `inventory_before`, `inventory_after`, "
                    "`inventory_unit`, `inventory_mark_cents` and "
                    "`inventory_mark_source` are on every fill record here, "
                    "for the arm AND the 0-cancel baseline, so the "
                    "INVENTORY LEG is recomputable from this file. Schema "
                    "v1 ledgers carry none of it and are not comparable on "
                    "that leg.",
            },
            "inventory_fields_from_BE_96": [
                "inventory_before", "inventory_after", "inventory_unit",
                "inventory_mark_cents", "inventory_mark_source"],
            # R-801: whether THIS file carries the ruled endpoint's rows,
            # computed from what is about to be written -- never a
            # promise. A reader that finds `false` here knows the day was
            # not valued under the endpoint and does not go looking.
            "settlement_rows_present": sorted(
                arm for arm in per_arm if per_arm[arm].get("settlement")),
            "settlement_row_kinds": ["SETTLEMENT_SCALARS",
                                     "SETTLEMENT_SLUG"],
            # THE SIGN CONVENTION TRAVELS WITH THE FILE. `fill_value_cents`
            # signs by `HSP.SIDES[0]`; a reader that guessed "B" would
            # value every fill backwards the day that constant changed,
            # and would do it silently.
            "buy_side": buy_side,
            "arms": sorted(per_arm)}, sort_keys=True) + "\n")
        n += 1
        for arm in sorted(per_arm):
            a = per_arm[arm]
            fh.write(json.dumps({
                "row": "ARM_SCALARS", "arm": arm,
                # R-782: THE ABSOLUTES travel in the per-day summary too,
                # so a reader with the ledger and no receipt can answer
                # "what did 0-cancel make" and not only "how much better".
                "absolute": a.get("absolute"),
                "observed_D_E0": a["observed"],
                "arm_value_cents": a["arm_value"],
                "baseline_value_cents": a["base_value"],
                "head": a.get("head"), "theta": a.get("theta"),
                "seed": a.get("seed"),
                "n_decisions": a.get("n_decisions"),
                "n_cancels_issued": a.get("n_cancels_issued"),
                "n_fills_arm": a.get("n_fills_arm"),
                "n_fills_baseline": a.get("n_fills_baseline"),
            }, sort_keys=True) + "\n")
            n += 1
            for i, v in enumerate(a["null_values"]):
                fh.write(json.dumps({
                    "row": "NULL_DRAW", "arm": arm, "i": i, "value": v,
                    "cancels": (a["null_cancels"][i]
                                if a.get("null_cancels") else None)},
                    sort_keys=True) + "\n")
                n += 1
            for which, fills in (("ARM", a["arm_fills"]),
                                 ("BASELINE", a["baseline_fills"])):
                for f in fills:
                    fh.write(json.dumps(
                        {"row": "FILL", "arm": arm, "book": which, **f},
                        sort_keys=True) + "\n")
                    n += 1
            for d in a.get("decisions") or []:
                fh.write(json.dumps(
                    {"row": "DECISION", "arm": arm, **d},
                    sort_keys=True) + "\n")
                n += 1
            # R-801, THE USER: "the pnls are from trades and remaining
            # position's settlement p&l". The per-slug decomposition of
            # the RULED quantity, for the arm and the 0-cancel baseline,
            # so a reader holding this file and no receipt can re-form it
            # slug by slug. One SETTLEMENT_SCALARS row per arm and one
            # SETTLEMENT_SLUG row per slug per book; absent (not null,
            # not zero) when the day was not valued under the endpoint.
            _st = a.get("settlement")
            if _st:
                fh.write(json.dumps({
                    "row": "SETTLEMENT_SCALARS", "arm": arm,
                    "ruling": _st.get("ruling"),
                    "unit": _st.get("unit"),
                    "D_E_settle": _st.get("D_E_settle"),
                    "arm_total_cents": _st.get("arm_total_cents"),
                    "baseline_total_cents": _st.get("baseline_total_cents"),
                    "winner_source": _st.get("winner_source"),
                }, sort_keys=True) + "\n")
                n += 1
                for which, per in (("ARM", _st.get("arm_per_slug") or {}),
                                   ("BASELINE",
                                    _st.get("baseline_per_slug") or {})):
                    for slug in sorted(per):
                        fh.write(json.dumps(
                            {"row": "SETTLEMENT_SLUG", "arm": arm,
                             "book": which, **per[slug]},
                            sort_keys=True) + "\n")
                        n += 1
    return {"path": str(path), "sha256": _sha(path), "n_rows": n,
            "schema_version": SCHEMA_VERSION,
            "bytes": path.stat().st_size,
            "the_bytes_are_gitignored": True,
            "named_here_by_digest": "the receipt is tracked and names this "
                                    "file by sha256; the file is not"}


def read_ledger(path, *, expect_sha256: str | None = None) -> dict:
    """READ IT BACK, WITH THE BOOK ABSENT. Digest checked BY NAME."""
    path = Path(path)
    if not path.is_file():
        raise LedgerRefused(
            f"DECISION_LEDGER_ABSENT: {path} is not there. The receipt "
            f"names a ledger; without it the day would have to be re-run, "
            f"which is what the ruling exists to prevent.")
    got = _sha(path)
    if expect_sha256 is not None and got != expect_sha256:
        raise LedgerRefused(
            f"DECISION_LEDGER_DIGEST_MISMATCH: the receipt names "
            f"{expect_sha256} and the file on disk is {got}. A ledger that "
            f"is not the one the receipt was written beside cannot be read "
            f"as the numbers that receipt reports.")
    out: dict = {"header": None, "arms": {}}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            k = r["row"]
            if k == "HEADER":
                out["header"] = r
                continue
            a = out["arms"].setdefault(r["arm"], {
                "scalars": None, "null_values": [], "null_cancels": [],
                "fills": {"ARM": [], "BASELINE": []}, "decisions": []})
            if k == "ARM_SCALARS":
                a["scalars"] = r
                a["absolute"] = r.get("absolute")
            elif k == "NULL_DRAW":
                a["null_values"].append(r["value"])
                a["null_cancels"].append(r["cancels"])
            elif k == "FILL":
                a["fills"][r["book"]].append(r)
            elif k == "DECISION":
                a["decisions"].append(r)
    return out


def recompute(led: dict, arm: str) -> dict:
    """EVERY STATISTIC THE RECEIPT REPORTS, FROM THE LEDGER ALONE.

    Nothing here reads a book, a model or the receipt. If this and the
    receipt disagree, one of them is wrong and the acceptance cell says
    which."""
    a = led["arms"][arm]
    vals = list(a["null_values"])
    obs = a["scalars"]["observed_D_E0"]
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    n = len(vals)
    ge = sum(1 for v in vals if v >= obs)
    le = sum(1 for v in vals if v <= obs)
    # THE SAME +1/+1 CONVENTION THE DESIGN USES for a permutation p.
    p_one = (ge + 1) / (n + 1)
    p_two = min(1.0, 2.0 * min((ge + 1) / (n + 1), (le + 1) / (n + 1)))
    fills = a["fills"]["ARM"]
    legs = {"n_fills_valued": 0, "fills_leg_cents": 0.0,
            "spread_captured_cents": 0.0, "adverse_cents": 0.0,
            "excluded_no_mid_at_fill": 0, "excluded_unvalued": 0}
    for f in fills:
        lvl, mkt, sz = f.get("px_cents"), f.get("mid_cents_at_markout"), f.get("size")
        if lvl is None or mkt is None or not sz:
            legs["excluded_unvalued"] += 1
            continue
        buy = led["header"].get("buy_side")
        if buy is None:
            raise LedgerRefused(
                "DECISION_LEDGER_NO_SIGN_CONVENTION: the header carries no "
                "`buy_side`, so every fill's sign would be a guess.")
        sgn = 1.0 if f.get("side") == buy else -1.0
        legs["n_fills_valued"] += 1
        legs["fills_leg_cents"] += sgn * (float(mkt) - float(lvl)) * float(sz)
        mid = f.get("mid_cents_at_fill")
        if mid is None:
            legs["excluded_no_mid_at_fill"] += 1
            continue
        legs["spread_captured_cents"] += sgn * (float(lvl) - float(mid)) * float(sz)
        legs["adverse_cents"] += sgn * (float(mkt) - float(mid)) * float(sz)
    # ---- THE INVENTORY LEG (BE 96, DE 126) ---------------------------
    # The position at the fill, marked at the fill's own mark. A ledger
    # whose fill records lack the fields REFUSES BY NAME -- it is never
    # read as zero, because a zero inventory leg and an unrecorded one are
    # the same number and opposite facts.
    _inv_missing = [f for f in ("inventory_before", "inventory_after",
                                "inventory_mark_cents")
                    if fills and f not in fills[0]]
    if fills and _inv_missing:
        raise LedgerRefused(
            f"DECISION_LEDGER_NO_INVENTORY_FIELDS: the fill records lack "
            f"{_inv_missing} (schema v"
            f"{led['header'].get('schema_version')}). BE 96 put them on "
            f"every fill; a ledger without them cannot answer an inventory "
            f"question, and answering it with 0 would be a fact nobody "
            f"measured.")
    # R-803 / BE 99: THE NAME WAS WRONG AND THE MEASUREMENT SETTLED IT.
    # This accumulated sum((after - before) x mark), which BE showed is
    # sum(sgn x size x px) = BUYS - SELLS -- the exact NEGATIVE of the
    # trades cash flow, and not a residual mark-to-market at all (the
    # residual is priced at SETTLEMENT, in the R-801 legs). It is named
    # for what it is and SIGNED as the cash flow: SELLS - BUYS.
    inv = {"trades_cash_flow_cents": 0.0, "n_fills_with_inventory": 0,
           "inventory_unit": None, "mark_sources": {}}
    for f in fills:
        b, a_, mk = (f.get("inventory_before"), f.get("inventory_after"),
                     f.get("inventory_mark_cents"))
        if b is None or a_ is None or mk is None:
            continue
        inv["n_fills_with_inventory"] += 1
        inv["inventory_unit"] = f.get("inventory_unit") or inv["inventory_unit"]
        _src = f.get("inventory_mark_source")
        inv["mark_sources"][_src] = inv["mark_sources"].get(_src, 0) + 1
        # SELLS - BUYS: a BUY raises the position and pays cash out, so
        # the cash flow is the NEGATIVE of the position change valued at
        # the fill's own mark.
        inv["trades_cash_flow_cents"] += -(float(a_) - float(b)) * float(mk)
    rho = (legs["adverse_cents"] / legs["spread_captured_cents"]
           if legs["spread_captured_cents"] else None)
    return {
        "absolute": a.get("absolute"),
        "D_E0": obs, "n_null_draws": n,
        "null_mean": mean, "null_sd": sd,
        "Z": ((obs - mean) / sd) if sd else math.inf,
        "p_one_sided": p_one, "p_two_sided": p_two,
        "rho_adverse_over_spread": rho, **legs, **inv,
        "trades_cash_flow_cents": inv["trades_cash_flow_cents"],
        "trades_cash_flow_sign_convention": "SELLS - BUYS: positive means "
                                            "cash taken in. It is the "
                                            "NEGATIVE of the quantity this "
                                            "field was called "
                                            "`inventory_leg` until R-803",
        "trades_cash_flow_from": "BE 96's per-fill position change, valued "
                                 "at each fill's own mark, negated. BE 99 "
                                 "measured it equal to the R-801 trades "
                                 "leg on all four path-days",
    }


# ------------------------------------------------------- the battery

EXPECTED_CHECKS = 9


def selftest(quiet: bool = False) -> int:
    """RED FIRST, on a fixture. No book is opened by any path here."""
    import random
    import shutil
    import tempfile
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_decision_ledger] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    rng = random.Random(20260907)
    vals = [rng.gauss(0.0, 1.0) for _ in range(600)]
    obs = 2.5
    fills = [{"fill_ns": 1e18 + i, "gen_start_ns": 1e18 + i, "side": "B",
              "px_cents": 100.0 + i * 0.01, "size": 10.0,
              "slug": f"s{i}", "ref_gen": i,
              "mid_cents_at_fill": 99.5 + i * 0.01,
              "mid_cents_at_markout": 100.2 + i * 0.01,
              # BE 96 (c707eb8), on every fill for both books
              "inventory_before": float(i), "inventory_after": float(i) + 10.0,
              "inventory_unit": "shares",
              "inventory_mark_cents": 100.0 + i * 0.01,
              "inventory_mark_source": "mid_at_fill"} for i in range(40)]
    per_arm = {"A": {"observed": obs, "arm_value": 12.5, "base_value": 10.0,
                     "head": "h", "theta": 0.5, "seed": 7,
                     "n_decisions": 200, "n_cancels_issued": 11,
                     "n_fills_arm": len(fills), "n_fills_baseline": 55,
                     "null_values": vals, "null_cancels": None,
                     "arm_fills": fills, "baseline_fills": fills[:20],
                     "decisions": [{"t": 1.0, "slug": "s0", "side": "B",
                                    "gen": 0, "score": 0.9}]}}
    d = Path(tempfile.mkdtemp(prefix="ledger_"))
    lp = d / ledger_name("2026-09-07", "20260907T000000Z")
    w = write_ledger(lp, "2026-09-07", per_arm, buy_side="B")

    # ---- (1) RECOMPUTE FROM THE LEDGER, MATCH THE RECEIPT TO 1e-9 -----
    # The "receipt" here is computed the way the runner computes it, from
    # the SAME inputs; the cell is that the two paths agree, not that one
    # of them is echoed back.
    r = read_ledger(lp, expect_sha256=w["sha256"])
    rc = recompute(r, "A")
    _mean = statistics.fmean(vals)
    _sd = statistics.pstdev(vals)
    _z = (obs - _mean) / _sd
    ok(abs(rc["D_E0"] - obs) < 1e-9 and abs(rc["Z"] - _z) < 1e-9
       and abs(rc["null_mean"] - _mean) < 1e-9
       and abs(rc["null_sd"] - _sd) < 1e-9 and rc["n_null_draws"] == 600,
       f"R-765 (1): D_E0 and Z RECOMPUTED FROM THE LEDGER match the "
       f"receipt's own computation to 1e-9 -- D_E0 {rc['D_E0']:.6f}, Z "
       f"{rc['Z']:.9f} against {_z:.9f}, over {rc['n_null_draws']} stored "
       f"draws. The summary alone could not have produced either")
    ok(0.0 < rc["p_one_sided"] <= 1.0 and 0.0 < rc["p_two_sided"] <= 1.0
       and rc["p_two_sided"] >= rc["p_one_sided"]
       and rc["rho_adverse_over_spread"] is not None
       and rc["n_fills_valued"] == len(fills)
       and rc["trades_cash_flow_cents"] is not None,
       f"R-765 (1): and the statistics the SEALED receipts could not "
       f"carry -- p one-sided {rc['p_one_sided']:.4f}, p TWO-sided "
       f"{rc['p_two_sided']:.4f}, rho = adverse/spread "
       f"{rc['rho_adverse_over_spread']:.4f}, the fills leg over "
       f"{rc['n_fills_valued']} fills -- all come out of the stored rows. "
       f"And since BE 96 the TRADES CASH FLOW is among them "
       f"({rc['trades_cash_flow_cents']:.1f} cents, SELLS - BUYS), so "
       f"nothing the ruling asked for is missing from this file")

    # ---- (1c) THE TRADES CASH FLOW, RECOMPUTED FROM THE LEDGER ALONE --
    # R-803: this cell asserted a quantity called the INVENTORY LEG. BE 99
    # measured what it is -- sum((after - before) x mark) = BUYS - SELLS,
    # the exact NEGATIVE of the R-801 trades leg -- so the cell now
    # asserts the SIGNED CASH FLOW and the KNOWN-BAD below is the old
    # sign, which is what a reader of the landed ledgers must invert.
    _want_inv = -sum((f["inventory_after"] - f["inventory_before"])
                     * f["inventory_mark_cents"] for f in fills)
    ok(abs(rc["trades_cash_flow_cents"] - _want_inv) < 1e-9
       and abs(rc["trades_cash_flow_cents"]
               + sum((f["inventory_after"] - f["inventory_before"])
                     * f["inventory_mark_cents"] for f in fills)) < 1e-9
       and rc["n_fills_with_inventory"] == len(fills)
       and rc["inventory_unit"] == "shares"
       and rc["mark_sources"] == {"mid_at_fill": len(fills)},
       f"R-803 / BE 99: the TRADES CASH FLOW recomputes from the ledger "
       f"alone to {rc['trades_cash_flow_cents']:.6f} cents (SELLS - "
       f"BUYS) against {_want_inv:.6f} formed independently -- and it is "
       f"the exact NEGATIVE of the quantity this field was called "
       f"`inventory_leg`, which is what BE measured it to be. "
       f"{rc['n_fills_with_inventory']} fills, unit "
       f"{rc['inventory_unit']!r}, marks {rc['mark_sources']}; schema "
       f"v{SCHEMA_VERSION}")
    def _keys_named(o, name):
        """Every KEY by that name, at any depth. Keys, not substrings:
        the FIELD must be gone, while the prose beside it deliberately
        names the old one so a reader of the LANDED ledgers knows to
        invert the sign (rule 13 -- those bytes are not edited)."""
        n = 0
        if isinstance(o, dict):
            n += sum(1 for k in o if k == name)
            for v in o.values():
                n += _keys_named(v, name)
        elif isinstance(o, list):
            for e in o:
                n += _keys_named(e, name)
        return n
    ok(_keys_named(rc, "inventory_leg") == 0
       and _keys_named(rc, "trades_cash_flow_cents") == 1
       and "inventory_leg" in json.dumps(rc),
       f"R-803: NO FIELD is named `inventory_leg` in what `recompute` "
       f"returns ({_keys_named(rc, 'inventory_leg')} keys) and exactly "
       f"one is named `trades_cash_flow_cents` -- while the STRING "
       f"survives in the prose ON PURPOSE, telling a reader of the "
       f"landed ledgers that the old field carried the opposite sign. "
       f"The first draft of this cell tested the substring and failed on "
       f"its own sentence")
    # RED: a ledger whose fills lack the fields REFUSES -- never zero.
    _no_inv = {"A": {**per_arm["A"],
                     "arm_fills": [{k: v for k, v in f.items()
                                    if not k.startswith("inventory_")}
                                   for f in fills],
                     "baseline_fills": []}}
    _lp2 = d / ledger_name("2026-09-08", "20260908T000000Z")
    write_ledger(_lp2, "2026-09-08", _no_inv, buy_side="B")
    _code_inv = None
    try:
        recompute(read_ledger(_lp2), "A")
    except LedgerRefused as e:
        _code_inv = str(e).split(":")[0]
    ok(_code_inv == "DECISION_LEDGER_NO_INVENTORY_FIELDS",
       f"DE 126 RED: a ledger whose fill records lack BE 96's fields "
       f"REFUSES by name -- `{_code_inv}` -- and is NEVER read as an "
       f"inventory leg of zero. A zero leg and an unrecorded one are the "
       f"same number and opposite facts")

    # ---- (2) THE BOOK IS ABSENT ---------------------------------------
    # Nothing in the read path can reach a book: the reader takes a PATH
    # and the recompute takes its output. Driven by reading from a
    # directory that contains the ledger and nothing else.
    _iso = Path(tempfile.mkdtemp(prefix="ledger_iso_"))
    shutil.copy(lp, _iso / lp.name)
    r2 = read_ledger(_iso / lp.name)
    rc2 = recompute(r2, "A")
    ok(rc2 == rc and not any(p.suffix == ".pkl" for p in _iso.iterdir()),
       f"R-765 (2): the ledger reads and recomputes IDENTICALLY from a "
       f"directory holding it and nothing else -- no book, no model, no "
       f"receipt ({len(list(_iso.iterdir()))} file present). That is what "
       f"'avoid rerun' has to mean: the numbers survive without the "
       f"inputs that made them")

    # ---- (3) A LEDGER THAT IS NOT THE RECEIPT'S REFUSES BY NAME -------
    _tamper = _iso / lp.name
    with gzip.open(_tamper, "at") as fh:
        fh.write(json.dumps({"row": "DECISION", "arm": "A",
                             "planted": True}, sort_keys=True) + "\n")
    _code = None
    try:
        read_ledger(_tamper, expect_sha256=w["sha256"])
    except LedgerRefused as e:
        _code = str(e).split(":")[0]
    ok(_code == "DECISION_LEDGER_DIGEST_MISMATCH" and _sha(_tamper) != w["sha256"],
       f"R-765 (3) KNOWN-BAD: a ledger whose bytes differ from the digest "
       f"the receipt names REFUSES BY NAME -- `{_code}`. One appended row "
       f"is enough; the receipt is tracked and the ledger is not, so the "
       f"digest is the only thing binding them")
    _absent = None
    try:
        read_ledger(_iso / "no_such_ledger.jsonl.gz")
    except LedgerRefused as e:
        _absent = str(e).split(":")[0]
    ok(_absent == "DECISION_LEDGER_ABSENT",
       f"R-765 (3): and an ABSENT ledger refuses by its own name too "
       f"(`{_absent}`), rather than reading as a day with no rows")

    # ---- THE SIZE, MEASURED ------------------------------------------
    _per_row = w["bytes"] / w["n_rows"]
    ok(w["n_rows"] == 1 + 1 + 600 + 60 + 1 and w["bytes"] > 0
       and w["schema_version"] == SCHEMA_VERSION,
       f"R-765: the fixture ledger is {w['bytes']:,} bytes over "
       f"{w['n_rows']:,} rows ({_per_row:.0f} B/row gzipped), schema "
       f"v{w['schema_version']}. A REAL arm-day is ~600 draws + ~30-45k "
       f"fills and ~16-19k decisions per arm, so a two-arm day scales to "
       f"roughly {2 * (600 + 45000 + 19000) * _per_row / 1e6:.0f} MB "
       f"gzipped -- the estimate is stated here rather than discovered on "
       f"the first real day")

    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(_iso, ignore_errors=True)
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_decision_ledger] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_decision_ledger] PASS -- {n[0]} checks, "
              f"n_disarmed 0, n_skipped 0")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else "usage: --selftest")
