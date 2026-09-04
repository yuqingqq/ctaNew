"""THE ACTION-NATIVE FILL LEDGER — and what it is NOT.

It replaces the scratch `prof.py` whose numbers the USER withdrew. That script
kept the FIRST ROW of each `(slug, side, gen)` action and summed those rows as
though they were the action's total, so:

  * the denominator was not total filled notional;
  * the baseline was not the whole no-cancel book;
  * baseline and overlay need not even have used the same row of a multi-row
    action.

This module aggregates EVERY ROW OF EVERY ACTION, exactly once, and states its
population in its own emission rather than in a commit message.

AND IT SAYS PLAINLY WHAT IT CANNOT BE. Even correctly aggregated, what is here
is a FIVE-SECOND GROSS MARKOUT over a one-second preventable window. It is not
a net return, and `not_a_net_return_because` names every missing term as a
STATUS rather than treating it as zero.
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


class FillLedgerRefused(RuntimeError):
    """A named refusal."""


#: What `preventable_shares` actually covers, transcribed from the builder.
POPULATION_NOTE = {
    "unit": "one row per (slug, side, gen, t_start) — EVERY row, not the first",
    "shares_field": "preventable_shares",
    "what_it_counts": (
        "shares of tranches inside the row's ONE-SECOND action horizon "
        "(FILL_HORIZON_S = 1.0) that fill AT OR AFTER the latency cutoff "
        "t_start + L"),
    "what_it_EXCLUDES": (
        "fills before the latency cutoff, which the builder records "
        "separately as `stale_shares` and which this feed does not carry; and "
        "every fill outside the one-second horizon, which is outside this "
        "population entirely"),
    "therefore_this_is_NOT": (
        "total filled shares, total filled notional, or the whole no-cancel "
        "book. It is the PREVENTABLE WINDOW only"),
    "markout_horizon_s": 5.0,
    "source": "harmful_exposure_rows.label_rows, FILL_HORIZON_S / MARKOUT_S",
}

NOT_A_NET_RETURN = {
    "fees": "UNQUANTIFIED — no fee model exists in this repo; 911 of 1,957 "
            "real on-chain OrderFilled events carry a non-zero fee at no rate "
            "that could be pinned. Every figure here is GROSS",
    "realised_exit_and_settlement": "ABSENT — this is a 5 s mark-to-market "
                                    "after each fill, not the P&L of holding "
                                    "to settlement",
    "quote_size": "ABSENT — the row carries `level`, `resting` and `qahead` "
                  "but no quote size, so exposure cannot be formed",
    "inventory": "ABSENT — no inventory state is carried through this feed",
    "capital": "ABSENT — without quote size and inventory there is no capital "
               "base, so no return ON capital can be computed",
    "consequence": (
        "these are GROSS MARKOUT CENTS over a preventable window. They are "
        "not profitability and must not be quoted as a return"),
}


def ledger(feed_path: Path) -> dict:
    """Aggregate EVERY row of EVERY action. Nothing is taken first."""
    if not feed_path.exists():
        raise FillLedgerRefused(f"REFUSED: no feed at {feed_path}.")
    per = collections.defaultdict(
        lambda: {"n_rows": 0, "n_actions": 0, "shares": 0.0,
                 "notional": 0.0, "markout_cents": 0.0,
                 "rows_with_fill": 0, "rows_no_level": 0})
    actions = collections.defaultdict(set)
    n_lines = 0
    for line in feed_path.open():
        r = json.loads(line)
        n_lines += 1
        coin = r["slug"].split("-", 1)[0]
        b = per[coin]
        b["n_rows"] += 1
        actions[coin].add((r["slug"], r["side"], r["gen"]))
        if not r.get("any_fill_ahead"):
            continue
        v = r.get("value_cents")
        if v is None or not math.isfinite(float(v)):
            continue
        b["rows_with_fill"] += 1
        sh = float(r.get("preventable_shares") or 0.0)
        lv = r.get("level")
        b["shares"] += sh
        if lv is None:
            b["rows_no_level"] += 1
        else:
            b["notional"] += sh * float(lv)
        # book P&L of those tranches = +markout*shares = -preventable_value
        b["markout_cents"] += -float(v)
    if not n_lines:
        raise FillLedgerRefused(
            f"REFUSED: {feed_path} is empty. An empty ledger is a FAILURE, "
            f"not a zero (R-141).")
    out = {}
    for coin, b in sorted(per.items()):
        b["n_actions"] = len(actions[coin])
        b["rows_per_action"] = (b["n_rows"] / b["n_actions"]
                                if b["n_actions"] else None)
        b["gross_markout_dollars"] = b["markout_cents"] / 100.0
        b["preventable_notional_dollars"] = b["notional"]
        out[coin] = b
    return {
        "protocol": "BE_FILL_LEDGER_V1",
        "feed": str(feed_path),
        "n_feed_rows": n_lines,
        "aggregation": "EVERY ROW OF EVERY ACTION, exactly once",
        "explicitly_not_first_row_per_action": True,
        "population": POPULATION_NOTE,
        "not_a_net_return_because": NOT_A_NET_RETURN,
        "per_coin": out,
    }


EXPECTED_CHECKS = 12


def selftest() -> int:
    import tempfile
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
        except FillLedgerRefused as e:
            ok(want in str(e), f"{label} [{str(e)[:70]}…]")
            return
        except Exception as e:                        # noqa: BLE001
            ok(False, f"{label} [WRONG EXCEPTION {type(e).__name__}: {e}]")
            return
        ok(False, f"{label} [DID NOT REFUSE]")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # THE FIXTURE IS THE DEFECT: one action, THREE rows. First-row
        # selection would see 10 shares; the truth is 60.
        f = td / "feed.jsonl"
        rows = [
            {"slug": "btc-x-1", "side": "SELL_UP", "gen": 1, "t0": 0,
             "t_start": 0.1, "score": 0.5, "score_incumbent": 0.4,
             "any_fill_ahead": True, "value_cents": -3.0,
             "preventable_shares": 10.0, "level": 0.5},
            {"slug": "btc-x-1", "side": "SELL_UP", "gen": 1, "t0": 0,
             "t_start": 0.2, "score": 0.6, "score_incumbent": 0.4,
             "any_fill_ahead": True, "value_cents": -5.0,
             "preventable_shares": 20.0, "level": 0.5},
            {"slug": "btc-x-1", "side": "SELL_UP", "gen": 1, "t0": 0,
             "t_start": 0.3, "score": 0.7, "score_incumbent": 0.4,
             "any_fill_ahead": True, "value_cents": 2.0,
             "preventable_shares": 30.0, "level": 0.5},
        ]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        L = ledger(f)
        b = L["per_coin"]["btc"]
        ok(b["n_rows"] == 3 and b["n_actions"] == 1,
           f"POSITIVE CONTROL: three rows, ONE action "
           f"({b['rows_per_action']:.1f} rows/action) — the shape the audit "
           f"says first-row selection destroys")
        ok(abs(b["shares"] - 60.0) < 1e-9,
           f"THE DEFECT, DRIVEN: every row's shares are summed "
           f"({b['shares']}), not the first row's 10.0 — this check FAILS on "
           f"the code the USER withdrew")
        ok(abs(b["notional"] - 30.0) < 1e-9,
           f"notional is shares x level summed over ALL rows "
           f"({b['notional']}), not 5.0 from the first row alone")
        ok(abs(b["markout_cents"] - 6.0) < 1e-9,
           f"book markout is -(sum of preventable value) over ALL rows "
           f"({b['markout_cents']}), not 3.0 from the first")
        ok(b["rows_with_fill"] == 3 and b["rows_no_level"] == 0,
           "row statuses are counted, not inferred")
        # a no-fill row contributes nothing but is still counted as a row
        f2 = td / "feed2.jsonl"
        f2.write_text(json.dumps({**rows[0], "any_fill_ahead": False,
                                  "value_cents": 0.0}) + "\n")
        L2 = ledger(f2)
        b2 = L2["per_coin"]["btc"]
        ok(b2["n_rows"] == 1 and b2["rows_with_fill"] == 0
           and b2["shares"] == 0.0,
           "a row with NO fill ahead is COUNTED as a row and contributes no "
           "shares — quiet and empty are different (rule 11)")
        f3 = td / "feed3.jsonl"
        f3.write_text(json.dumps({**rows[0], "level": None}) + "\n")
        b3 = ledger(f3)["per_coin"]["btc"]
        ok(b3["rows_no_level"] == 1 and b3["shares"] == 10.0
           and b3["notional"] == 0.0,
           "a row with NO level contributes SHARES but not NOTIONAL, and the "
           "omission is COUNTED — a silently smaller denominator is the "
           "defect this module exists to end")
        refuses(lambda: ledger(td / "nope.jsonl"), "no feed at",
                "KNOWN-BAD: a missing feed REFUSES by name")
        (td / "empty.jsonl").write_text("")
        refuses(lambda: ledger(td / "empty.jsonl"), "is a FAILURE, not a zero",
                "KNOWN-BAD: an EMPTY feed REFUSES rather than reporting zeros")
        ok(L["explicitly_not_first_row_per_action"] is True
           and "EVERY ROW" in L["aggregation"],
           "the emission SAYS how it aggregates, in a field")
        ok("PREVENTABLE WINDOW" in L["population"]["therefore_this_is_NOT"]
           and "total filled notional" in
           L["population"]["therefore_this_is_NOT"]
           and L["population"]["shares_field"] == "preventable_shares",
           "and it SAYS in the VALUE, not just the key, that its population "
           "is the PREVENTABLE WINDOW and NOT total filled notional — the "
           "exact overstatement that was withdrawn")
        ok(set(L["not_a_net_return_because"]) >= {
               "fees", "realised_exit_and_settlement", "quote_size",
               "inventory", "capital"},
           f"and every missing term is a NAMED STATUS, not a zero: "
           f"{sorted(L['not_a_net_return_because'])}")
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
    if "--feed" in argv:
        i = argv.index("--feed")
        if i + 1 >= len(argv):
            print("REFUSED: --feed needs a path")
            return 2
        print(json.dumps(ledger(Path(argv[i + 1])), indent=1, sort_keys=True,
                         default=str))
        return 0
    print("usage: be_fill_ledger.py --selftest | --feed <feed.jsonl>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
