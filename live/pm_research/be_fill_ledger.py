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


#: WHY THERE IS NO "EXACTLY ONCE" NUMBER HERE, established at the builder.
#:
#: `prof.py` kept the FIRST ROW of an action and UNDER-counted. Replacing it
#: with "every row exactly once" OVER-counts, and by the same mechanism: a row
#: already sums every tranche inside its OWN 1s horizon, and consecutive rows
#: of an action sit far closer than that, so the same tranche is summed once
#: per covering row. "Every row exactly once" is NOT "every fill exactly once".
#:
#: AND THE DIFFERENCE IS NOT REPAIRABLE FROM WHAT WE HOLD. Tranches exist only
#: inside `harmful_exposure_rows` as `gens[k]["tranches"]` with their own `t`,
#: `shares` and `level`; the ROW receives only the SUMS
#: (`harmful_exposure_rows.py:369-370`), and the feed carries only those. There
#: is NO TRANCHE IDENTITY and no tranche timestamp anywhere downstream, so the
#: overlap cannot be subtracted and a de-duplicated total cannot be computed.
#:
#: So this module does NOT report a fill total. It reports the BRACKET that IS
#: computable, and names the quantity for what it measures:
#:   * LOWER BOUND  = sum over actions of the LARGEST single row's window
#:                    (the union of an action's windows contains any one of
#:                     them, so the distinct total is at least this)
#:   * UPPER BOUND  = sum over every row (each distinct tranche is counted at
#:                    least once, so the distinct total is at most this)
#: The true de-duplicated total lies between them and is NOT COMPUTABLE HERE.
#: An honest name for an uncomputable quantity beats a plausible number for it.

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
    "exactly_once_total": "NOT COMPUTABLE FROM HELD ARTIFACTS",
    "why_not_computable": (
        "the feed carries no tranche identity and no tranche timestamp -- "
        "only per-row SUMS over each row's own horizon "
        "(harmful_exposure_rows.py:369-370). Overlapping rows of one action "
        "therefore cannot be de-duplicated, so a distinct-fill total cannot "
        "be formed. A BRACKET is reported instead"),
    "what_would_make_it_computable": (
        "emitting tranche identity (t, shares, level) or a per-action "
        "de-duplicated total from the BUILDER, which is a producer change and "
        "a re-run, not a downstream repair"),
    "markout_horizon_s": 5.0,
    "fill_horizon_s": 1.0,
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


def ledger(feed_path: Path, latency_ms: int = None) -> dict:
    """The computable BRACKET, plus the overlap that makes it a bracket."""
    if not feed_path.exists():
        raise FillLedgerRefused(f"REFUSED: no feed at {feed_path}.")
    # R40: EVERY QUANTITY HERE DEPENDS ON L AND THE FEED DOES NOT CARRY IT.
    # The iteration-011 artifact was found carrying zero occurrences of
    # `latency` while every value in it was computed at TARGET_LATENCY_MS --
    # rule 7 satisfied in the estimand and invisible in the artifact. The same
    # gap was here: this module emitted a markout and a notional and recorded
    # the markout/fill HORIZONS but never the resolved LATENCY. It cannot be
    # read from the feed, so it must be SUPPLIED, and a number without it does
    # not leave.
    if latency_ms is None:
        raise FillLedgerRefused(
            "REFUSED: no `latency_ms`. Every quantity this module emits is "
            "computed from `preventable_shares` and `value_cents`, which are "
            "resolved at a LATENCY the feed does not record. Reporting them "
            "without naming L is the defect found in the iteration-011 "
            "artifact -- a value whose computed_over a reader cannot see. "
            "Supply it, from the producing receipt's `feed.latency_ms_resolved`.")
    acts = collections.defaultdict(list)     # action -> [(t_start, sh, lv, v)]
    per_meta = collections.defaultdict(
        lambda: {"n_rows": 0, "rows_with_fill": 0, "rows_no_level": 0,
                 "rows_no_shares_field": 0, "markout_cents": 0.0})
    n_lines = 0
    for line in feed_path.open():
        r = json.loads(line)
        n_lines += 1
        coin = r["slug"].split("-", 1)[0]
        m = per_meta[coin]
        m["n_rows"] += 1
        if not r.get("any_fill_ahead"):
            continue
        v = r.get("value_cents")
        if v is None or not math.isfinite(float(v)):
            continue
        m["rows_with_fill"] += 1
        m["markout_cents"] += -float(v)
        if "preventable_shares" not in r:
            m["rows_no_shares_field"] += 1
        sh = float(r.get("preventable_shares") or 0.0)
        lv = r.get("level")
        if lv is None:
            m["rows_no_level"] += 1
        acts[(coin, r["slug"], r["side"], r["gen"])].append(
            (float(r["t_start"]), sh, lv))
    if not n_lines:
        raise FillLedgerRefused(
            f"REFUSED: {feed_path} is empty. An empty ledger is a FAILURE, "
            f"not a zero (R-141).")

    # BE31-R3: REFUSE THE EMPTY FIELD, not only the empty FILE. The counters
    # below were computed and never consulted, and a feed predating the scale
    # fields printed 0.0 shares beside a non-zero markout as though the day
    # had no size.
    for coin, m in sorted(per_meta.items()):
        if m["rows_with_fill"] and m["rows_no_level"] == m["rows_with_fill"]:
            raise FillLedgerRefused(
                f"REFUSED: every one of {coin}'s {m['rows_with_fill']} "
                f"fill-bearing rows carries NO `level`, so no notional can be "
                f"formed and a zero here would mean 'field absent', not 'no "
                f"size'. This feed predates the scale fields; re-score it "
                f"before asking it for a denominator.")
        if m["rows_with_fill"] and \
                m["rows_no_shares_field"] == m["rows_with_fill"]:
            raise FillLedgerRefused(
                f"REFUSED: every one of {coin}'s {m['rows_with_fill']} "
                f"fill-bearing rows lacks a `preventable_shares` FIELD. "
                f"Absence is not zero (rule 11).")

    out = {}
    n_actions = collections.Counter()
    for (coin, *_), rows in acts.items():
        n_actions[coin] += 1
    for coin, m in sorted(per_meta.items()):
        up_sh = lo_sh = up_no = lo_no = 0.0
        pairs = inside = 0
        for (c, *_), rows in acts.items():
            if c != coin:
                continue
            rows.sort()
            up_sh += sum(sh for _, sh, _ in rows)
            lo_sh += max((sh for _, sh, _ in rows), default=0.0)
            up_no += sum(sh * lv for _, sh, lv in rows if lv is not None)
            lo_no += max((sh * lv for _, sh, lv in rows if lv is not None),
                         default=0.0)
            for i in range(len(rows) - 1):
                pairs += 1
                if rows[i + 1][0] - rows[i][0] < POPULATION_NOTE[
                        "fill_horizon_s"]:
                    inside += 1
        out[coin] = {
            "n_rows": m["n_rows"], "n_actions": n_actions[coin],
            "rows_per_action": (m["n_rows"] / n_actions[coin]
                                if n_actions[coin] else None),
            "rows_with_fill": m["rows_with_fill"],
            "rows_no_level": m["rows_no_level"],
            "shares_UPPER_BOUND_row_window_sum": up_sh,
            "shares_LOWER_BOUND_largest_single_window": lo_sh,
            "notional_UPPER_BOUND_dollars": up_no,
            "notional_LOWER_BOUND_dollars": lo_no,
            "bracket_width_ratio_shares": (up_sh / lo_sh) if lo_sh else None,
            # BE33-R1: this read as uncomputable IN PRINCIPLE, which would
            # make the producer change look pointless. The quantity IS
            # computable -- just not from what we KEPT. Scoped here, where a
            # reader actually sees it, not only in POPULATION_NOTE.
            "exactly_once_total":
                "NOT COMPUTABLE FROM HELD ARTIFACTS — the feed carries no "
                "tranche identity. It IS computable from the builder, which "
                "has it; recovering it is a PRODUCER change and a re-run",
            "gross_markout_cents_UPPER_BOUND": m["markout_cents"],
            "intra_action_row_pairs": pairs,
            "pairs_closer_than_the_horizon": inside,
            "pct_pairs_overlapping": (100.0 * inside / pairs) if pairs else None,
        }
    return {
        "protocol": "BE_FILL_LEDGER_V2",
        "feed": str(feed_path),
        "n_feed_rows": n_lines,
        "aggregation": ("A BRACKET, because exactly-once is not computable: "
                        "UPPER = every row summed, LOWER = the largest single "
                        "window per action"),
        "explicitly_not_first_row_per_action": True,
        "explicitly_not_claiming_exactly_once": True,
        "population": {**POPULATION_NOTE,
                       "latency_ms_RESOLVED": latency_ms,
                       "latency_is_not_in_the_feed": (
                           "the feed carries no latency field, so L is "
                           "SUPPLIED by the caller and must match the "
                           "producing receipt's feed.latency_ms_resolved. "
                           "Emitting it here is the computed_over half of "
                           "the contract"),
                       "every_quantity_below_depends_on_it": [
                           "shares_UPPER_BOUND_row_window_sum",
                           "shares_LOWER_BOUND_largest_single_window",
                           "notional_UPPER_BOUND_dollars",
                           "notional_LOWER_BOUND_dollars",
                           "gross_markout_cents_UPPER_BOUND"]},
        "not_a_net_return_because": NOT_A_NET_RETURN,
        "per_coin": out,
    }


EXPECTED_CHECKS = 15


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

        def row(t, sh, lv=0.5, v=-3.0, slug="btc-x-1", fill=True):
            return {"slug": slug, "side": "SELL_UP", "gen": 1, "t0": 0,
                    "t_start": t, "score": 0.5, "score_incumbent": 0.4,
                    "any_fill_ahead": fill, "value_cents": v,
                    "preventable_shares": sh, "level": lv}

        def write(name, rows):
            f = td / name
            f.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return f

        # FIXTURE A: rows spaced BEYOND the 1s horizon. Their windows cannot
        # overlap, so no tranche can be double-counted and the bracket must
        # COLLAPSE only in the sense that the upper bound is the honest total
        # -- the lower bound is still one window, because the module cannot
        # know the windows are disjoint without tranche identity.
        fa = write("beyond.jsonl", [row(0.0, 10.0), row(5.0, 20.0),
                                    row(10.0, 30.0)])
        A = ledger(fa, 50)["per_coin"]["btc"]
        ok(A["intra_action_row_pairs"] == 2
           and A["pairs_closer_than_the_horizon"] == 0
           and A["pct_pairs_overlapping"] == 0.0,
           "FIXTURE SPACED BEYOND THE HORIZON: zero overlapping pairs, so "
           "these rows genuinely cannot double-count — the old fixture put "
           "three rows at 0.1/0.2/0.3, DEEP inside the horizon, and asserted "
           "their sum was the truth, which enshrined the over-count as spec")
        ok(abs(A["shares_UPPER_BOUND_row_window_sum"] - 60.0) < 1e-9
           and abs(A["shares_LOWER_BOUND_largest_single_window"] - 30.0) < 1e-9,
           f"and the BRACKET is reported, not a point: upper "
           f"{A['shares_UPPER_BOUND_row_window_sum']} (every row) and lower "
           f"{A['shares_LOWER_BOUND_largest_single_window']} (largest single "
           f"window) — the true distinct total lies between and is NOT "
           f"computable here")

        # FIXTURE B: rows INSIDE the horizon -- the over-counting case.
        fb = write("inside.jsonl", [row(0.1, 10.0), row(0.2, 20.0),
                                    row(0.3, 30.0)])
        B = ledger(fb, 50)["per_coin"]["btc"]
        ok(B["pairs_closer_than_the_horizon"] == 2
           and B["pct_pairs_overlapping"] == 100.0,
           "FIXTURE INSIDE THE HORIZON: 100% of pairs overlap, so summing "
           "every row counts the same tranches repeatedly — the emission SAYS "
           "so rather than leaving a reader to assume exactness")
        ok(B["shares_UPPER_BOUND_row_window_sum"]
           > B["shares_LOWER_BOUND_largest_single_window"]
           and abs(B["bracket_width_ratio_shares"] - 2.0) < 1e-9,
           f"and the bracket is WIDE exactly where overlap is total "
           f"({B['bracket_width_ratio_shares']:.2f}x) — width is the honest "
           f"signal that the number is not knowable, not a defect")
        ok(B["exactly_once_total"].startswith("NOT COMPUTABLE"),
           "and every coin block says EXACTLY-ONCE IS NOT COMPUTABLE, in a "
           "field, because the feed carries no tranche identity")

        # statuses, still counted
        C = ledger(write("nofill.jsonl", [row(0.0, 10.0, fill=False, v=0.0)]), 50)
        ok(C["per_coin"]["btc"]["rows_with_fill"] == 0
           and C["per_coin"]["btc"]["n_rows"] == 1,
           "a row with NO fill ahead is COUNTED and contributes nothing — "
           "quiet and empty are different (rule 11)")

        # BE31-R3: the empty FIELD refuses, and the counter is consulted
        refuses(lambda: ledger(write("nolevel.jsonl",
                                     [row(0.0, 10.0, lv=None),
                                      row(5.0, 20.0, lv=None)]), 50),
                "carries NO `level`",
                "BE31-R3 KNOWN-BAD: a feed whose fill-bearing rows ALL lack "
                "`level` REFUSES BY NAME — the counter was computed and never "
                "consulted, and a zero there means 'field absent', not 'no "
                "size'")
        ok(ledger(write("mixed.jsonl", [row(0.0, 10.0, lv=None),
                                        row(5.0, 20.0, lv=0.5)]), 50
                  )["per_coin"]["btc"]["rows_no_level"] == 1,
           "BE31-R3 POSITIVE CONTROL: a feed where only SOME rows lack the "
           "field is ADMITTED with the omission counted — the refusal fires "
           "on absence of the field, not on any missing value")
        refuses(lambda: ledger(fa), "no `latency_ms`",
                "R40 KNOWN-BAD: a ledger asked for numbers WITHOUT its "
                "resolved L REFUSES — every quantity it emits is computed at "
                "a latency the feed does not record, and that is exactly the "
                "gap found in the iteration-011 artifact")
        ok(ledger(fa, 50)["population"]["latency_ms_RESOLVED"] == 50
           and ledger(fa, 50)["population"]["every_quantity_below_depends_on_it"],
           "R40 POSITIVE CONTROL: with L supplied it is RECORDED beside the "
           "quantities that depend on it, which are enumerated rather than "
           "left for a reader to infer")
        refuses(lambda: ledger(td / "nope.jsonl", 50), "no feed at",
                "KNOWN-BAD: a missing feed REFUSES by name")
        (td / "empty.jsonl").write_text("")
        refuses(lambda: ledger(td / "empty.jsonl", 50), "is a FAILURE, not a zero",
                "KNOWN-BAD: an EMPTY feed REFUSES rather than reporting zeros")
        L = ledger(fa, 50)
        ok(L["explicitly_not_claiming_exactly_once"] is True
           and "BRACKET" in L["aggregation"],
           "the emission SAYS it reports a bracket and does NOT claim "
           "exactly-once, in fields")
        ok("PREVENTABLE WINDOW" in L["population"]["therefore_this_is_NOT"]
           and L["population"]["exactly_once_total"].startswith("NOT COMPUTABLE")
           and "no tranche identity" in L["population"]["why_not_computable"],
           "and the population block names WHY exactly-once is uncomputable: "
           "no tranche identity survives the builder")
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
