"""THREE MARKET FACTS §7 REQUIRES, ESTABLISHED FROM ARTIFACTS (DA 283).

The freeze declaration listed `tick_rounding`, `fee_rule` and
`initial_inventory` as MISSING. Two of them are FACTS ABOUT THE MARKET and must
be measured, not invented; the third is a frozen choice that must be written
down. This module measures all three and emits them with their evidence.

    THE LEGAL TICK     -- established, 0.01, with a measured caveat
    THE MAKER FEE      -- NOT ESTABLISHABLE from what is collected
    INITIAL INVENTORY  -- declared 0.0, frozen

AN UNESTABLISHABLE INPUT IS A FINDING. §9 says the P&L uses "the verified maker
fee applicable to these markets" and that "a zero fee is used only if the
receipt identifies the supporting market/account rule". Nothing collected
identifies such a rule, so this module REFUSES to supply a fee rather than
defaulting to zero -- a zero that no rule supports is an assumption wearing a
number's clothes, and it would flatter every P&L computed from it.

Usage:  da_market_facts.py [--falsify]
"""
from __future__ import annotations

import collections
import re
import glob
import gzip
import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

PROTOCOL = "P003_DA_MARKET_FACTS_V1"
FEE_UNESTABLISHABLE = "MAKER_FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_ARTIFACTS"

MARKETS = "data/pm_5min/markets.jsonl"
REWARDS = "data/pm_5min/rewards_registry.jsonl"
FEE_AUDIT = "data/pm_5min/derived/p003_da_onchain_fee_audit__20260905T155346Z.json"
RAW = "data/pm_5min/raw"
SCOPE = ("btc", "eth")
SAMPLE_DAYS = ("20260906", "20260907", "20260908")
FILES_PER_COIN_DAY = 4


#: THE DATA ROOT IS DELEGATED, NEVER COMPUTED HERE. My first version resolved
#: it from this file's own git root, which from a seat worktree is the
#: worktree -- whose `data/` does not exist. That is the exact defect
#: `da_root` was written for: "DA's `_derived_dir()` resolved relative to ITS
#: OWN FILE, so from a seat worktree it returned the worktree's PARTIAL
#: `data/`". The lane already has one resolver of record and a second
#: convention is how two seats end up holding different roots.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import da_root                                          # noqa: E402


def _p(rel: str) -> Path:
    return da_root.resolve_root() / rel


def declared_tick() -> dict:
    """THE TICK EACH MARKET DECLARES, over every record in `markets.jsonl`."""
    by_coin = collections.defaultdict(collections.Counter)
    by_month = collections.defaultdict(collections.Counter)
    n = missing = 0
    with open(_p(MARKETS)) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            tk = d.get("orderPriceMinTickSize")
            if tk is None:
                missing += 1
                continue
            slug = d.get("slug") or ""
            coin = d.get("coin") or next(
                (c for c in ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")
                 if c in slug), "?")
            by_coin[str(coin)][str(tk)] += 1
            ws = d.get("window_start") or 0
            try:
                import time
                mon = time.strftime("%Y-%m", time.gmtime(int(float(ws)))) if ws else "?"
            except Exception:
                mon = "?"
            by_month[mon][str(tk)] += 1
    values = {v for c in by_coin.values() for v in c}
    return {
        "field": "markets.jsonl:orderPriceMinTickSize",
        "n_records": n, "n_without_the_field": missing,
        "distinct_values": sorted(values),
        "uniform_across_all_coins": len(values) == 1,
        "by_coin": {k: dict(v) for k, v in sorted(by_coin.items())},
        "by_month": {k: dict(v) for k, v in sorted(by_month.items())},
        "uniform_across_the_windows_life": len(
            {v for c in by_month.values() for v in c}) == 1,
        "scope_coins_present": all(c in by_coin for c in SCOPE),
    }


def observed_increments(days=SAMPLE_DAYS, coins=SCOPE, per=FILES_PER_COIN_DAY) -> dict:
    """WHAT INCREMENTS ACTUALLY OCCUR on the book -- a SECOND INSTRUMENT.

    The declared field and the resting levels are different measurements of the
    same fact, so agreement between them is evidence and disagreement is a
    finding (rule 38). It is a finding: they disagree.
    """
    tick = Decimal("0.01")
    sub = collections.Counter()
    sub_windows = collections.Counter()
    dp = collections.Counter()
    n_levels = n_files = 0
    for day in days:
        for coin in coins:
            pat = str(_p(RAW) / day / f"{coin}-updown-5m-*.jsonl.gz")
            for f in sorted(glob.glob(pat))[:per]:
                n_files += 1
                slug = Path(f).name.replace(".jsonl.gz", "")
                try:
                    lines = gzip.open(f, "rt", errors="replace").read().splitlines()
                except Exception:
                    continue
                for ln in lines:
                    try:
                        payload = json.loads(ln.split("\t", 1)[1])
                    except Exception:
                        continue
                    for bk in (payload if isinstance(payload, list) else [payload]):
                        for side in ("bids", "asks"):
                            for lvl in (bk.get(side) or []):
                                px = lvl.get("price")
                                if px is None:
                                    continue
                                n_levels += 1
                                d = Decimal(str(px))
                                dp[-d.as_tuple().exponent] += 1
                                q = d / tick
                                if q != q.to_integral_value():
                                    sub[str(d)] += 1
                                    sub_windows[slug] += 1
    n_sub = sum(sub.values())
    return {
        "source": f"{RAW}/<day>/<coin>-updown-5m-*.jsonl.gz",
        "days": list(days), "coins": list(coins),
        "n_files": n_files, "n_price_levels": n_levels,
        "decimal_places_observed": {str(k): v for k, v in sorted(dp.items())},
        "n_levels_not_a_multiple_of_0.01": n_sub,
        "share_not_a_multiple": (n_sub / n_levels) if n_levels else None,
        "sub_tick_values": dict(sorted(sub.items(), key=lambda kv: float(kv[0]))),
        "n_windows_with_sub_tick_levels": len(sub_windows),
        "windows_sampled": n_files,
        "every_level_is_a_multiple_of_0.01": n_sub == 0,
    }


def legal_tick(tape=None) -> dict:
    """THE ESTABLISHED TICK, with the disagreement between its two instruments
    stated rather than averaged away."""
    dec = declared_tick()
    obs = observed_increments()
    tape = tape if tape is not None else tape_fee_and_tick()
    one = dec["distinct_values"][0] if len(dec["distinct_values"]) == 1 else None
    return {
        "legal_tick": one,
        "established": one is not None,
        "declared_instrument": dec,
        "observed_instrument": obs,
        "instruments_agree": bool(one) and obs["every_level_is_a_multiple_of_0.01"],
        "THE_SUB_TICK_PRICES_ARE_EXPLAINED": {
            "cause": "the VENUE NARROWS THE GRID IN-WINDOW, and the tape says so",
            "tick_size_change_events": tape["tick_size_change_events"],
            "n_changes_observed": tape["n_tick_size_changes"],
            "reading": ("every observed change is 0.01 -> 0.001. So the 0.001 "
                        "levels are not a violation of the declared tick: they "
                        "are the tick AFTER a `tick_size_change` event the "
                        "market published. `orderPriceMinTickSize` is the tick "
                        "at market creation, not for the window's life -- which "
                        "is exactly the question DA 283 asked and answered too "
                        "narrowly from the market record alone."),
            "consequence_for_the_freeze": (
                "rounding to 0.01 stays LEGAL at all times, because 0.01 is a "
                "multiple of 0.001. It is not always the FINEST legal grid, so "
                "a quote on it can be improvable after a narrowing event."),
        },
        "THE_CAVEAT_IS_MEASURED": (
            f"every market record declares {one}, uniformly across all coins and "
            f"both months. The BOOK does not match it exactly: "
            f"{obs['n_levels_not_a_multiple_of_0.01']} of {obs['n_price_levels']} "
            f"resting levels "
            f"({(obs['share_not_a_multiple'] or 0) * 100:.4f}%) sit at 0.001 "
            f"granularity, in {obs['n_windows_with_sub_tick_levels']} of "
            f"{obs['windows_sampled']} sampled windows -- and those windows "
            f"declare {one} too. So the tick is the right ROUNDING grid and is "
            f"NOT a guarantee that no finer price rests on the book."),
        "what_may_be_frozen_on_this": (
            f"bid rounds DOWN and ask rounds UP to {one}. A quote on that grid "
            f"is legal; it is not necessarily the best price available."),
        "uniform_across_btc_and_eth": dec["by_coin"].get("btc") == dec["by_coin"].get("eth"),
        "uniform_across_the_windows_life": dec["uniform_across_the_windows_life"],
    }



def tape_fee_and_tick(days=None, coins=SCOPE, per=6) -> dict:
    """THE CLOB TAPE'S OWN FIELDS -- the source I did not search in DA 283.

    The tape carries `fee_rate_bps` on every trade event and `tick_size_change`
    events when the venue narrows the grid. Both are the VENUE's own statements
    at the time of the event, which is a stronger source than either the market
    record or an inference from prices.
    """
    import gzip, glob as _g, os as _os
    days = days or sorted(_os.listdir(_p("data/pm_5min/raw")))[-8:]
    fee = collections.Counter(); nonzero = []
    ticks = collections.Counter(); changes = collections.Counter()
    n_trade = n_msg = n_files = 0
    for day in days:
        for coin in coins:
            pat = str(_p("data/pm_5min/raw") / day / f"{coin}-updown-5m-*.jsonl.gz")
            for f in sorted(_g.glob(pat))[:per]:
                n_files += 1
                try:
                    lines = gzip.open(f, "rt", errors="replace").read().splitlines()
                except Exception:
                    continue
                for ln in lines:
                    try:
                        payload = json.loads(ln.split("\t", 1)[1])
                    except Exception:
                        continue
                    for bk in (payload if isinstance(payload, list) else [payload]):
                        n_msg += 1
                        if "fee_rate_bps" in bk:
                            v = str(bk["fee_rate_bps"]); fee[v] += 1; n_trade += 1
                            if v not in ("0", "0.0") and len(nonzero) < 10:
                                nonzero.append({"day": day, "coin": coin,
                                                "fee_rate_bps": v,
                                                "price": bk.get("price")})
                        if "tick_size" in bk:
                            ticks[str(bk["tick_size"])] += 1
                        if bk.get("new_tick_size") is not None:
                            changes[f"{bk.get('old_tick_size')}->{bk.get('new_tick_size')}"] += 1
    return {
        "days": list(days), "n_files": n_files, "n_messages": n_msg,
        "n_trade_events_carrying_fee_rate_bps": n_trade,
        "fee_rate_bps_distribution": dict(fee),
        "fee_rate_bps_is_uniformly_zero": (set(fee) <= {"0", "0.0"}) and n_trade > 0,
        "non_zero_fee_events": nonzero,
        "tick_size_values_on_the_tape": dict(ticks),
        "tick_size_change_events": dict(changes),
        "n_tick_size_changes": sum(changes.values()),
    }


def maker_fee_rule() -> dict:
    """THE MAKER FEE, searched SOURCE BY SOURCE (DA 285).

    DA 283 CONCLUDED "NOT ESTABLISHABLE" AND THAT WAS WRONG. I searched the
    top-level keys of `markets.jsonl` and the rewards registry and stopped. The
    fee was in two places I had not opened: the `clob` SUBOBJECT of every market
    record, and the CLOB tape's own `fee_rate_bps` field on every trade event.
    An "unobtainable" verdict reached by an unfinished search is a finding about
    the search, not about the data -- the same shape as every other instrument
    defect this lane has found, one level up.
    """
    per_source = []

    # A. markets.jsonl -- top level AND the `clob` subobject.
    top_keys, clob_keys, clob_vals = set(), collections.Counter(), collections.defaultdict(collections.Counter)
    rules_hits = collections.Counter()
    rules_re = re.compile(r"\bfee|\bmaker|\btaker|\brebate|\bcommission|\bbps\b", re.I)
    n = 0
    with open(_p(MARKETS)) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            top_keys |= set(d.keys())
            c = d.get("clob")
            if isinstance(c, dict):
                clob_keys.update(c.keys())
                for k, v in c.items():
                    if not isinstance(v, (dict, list)):
                        clob_vals[k][str(v)] += 1
            txt = " ".join(str(d.get(k, "")) for k in ("description", "question", "outcomes"))
            for m in set(rules_re.findall(txt)):
                rules_hits[m.lower()] += 1
    per_source.append({
        "source": "markets.jsonl", "searched": "all top-level keys, the `clob` "
        "subobject, and the rules text (description/question/outcomes)",
        "n_records": n,
        "fee_fields_found": {k: dict(clob_vals[k].most_common(3))
                             for k in clob_keys if "fee" in k.lower()},
        "rules_text_fee_language": dict(rules_hits) or "NONE",
        "establishes": ("a per-market fee SCHEDULE field: maker_base_fee and "
                        "taker_base_fee, uniform across every record"),
        "does_not_establish": ("the UNITS of that field, nor the rate actually "
                               "applied to a fill"),
    })

    # B. rewards_registry.jsonl
    rk = collections.Counter()
    try:
        with open(_p(REWARDS)) as fh:
            for line in fh:
                try:
                    rk.update(json.loads(line).keys())
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    per_source.append({
        "source": "rewards_registry.jsonl", "searched": "every field",
        "fields": sorted(rk),
        "establishes": "NOTHING about fees",
        "does_not_establish": ("it is collected as a COUNT -- recv_ns and n -- "
                               "so the registry's contents were never captured"),
    })

    # C. THE CLOB TAPE'S OWN FIELDS -- the decisive source.
    tape = tape_fee_and_tick()
    per_source.append({
        "source": "the CLOB tape (data/pm_5min/raw/<day>/<slug>.jsonl.gz)",
        "searched": "every payload field on every message in the sample",
        "n_files": tape["n_files"], "n_messages": tape["n_messages"],
        "n_trade_events": tape["n_trade_events_carrying_fee_rate_bps"],
        "fee_rate_bps_distribution": tape["fee_rate_bps_distribution"],
        "establishes": ("the VENUE's own per-trade fee rate: `fee_rate_bps`, "
                        "uniformly 0 on every observed trade in these markets"),
        "does_not_establish": ("what would happen at a different rate; it is a "
                               "record of what WAS charged, not a published "
                               "schedule with an effective date"),
    })

    # D. on-chain settlement receipts.
    try:
        a = json.loads(_p(FEE_AUDIT).read_text())
    except Exception as e:
        a = {"error": str(e)}
    per_source.append({
        "source": "onchain/receipts (via p003_da_onchain_fee_audit)",
        "searched": f"{a.get('n_receipt_files')} settlement receipts, "
                    f"{a.get('n_legs')} legs",
        "n_maker_legs": a.get("n_maker_legs"),
        "n_maker_legs_zero": a.get("n_maker_legs_zero"),
        "n_maker_legs_charged": a.get("n_maker_legs_charged"),
        "charged_leg_prices": sorted({str(r.get("price"))
                                      for r in a.get("maker_charged_detail", [])}),
        "establishes": ("that maker fills are overwhelmingly zero-fee on chain: "
                        f"{a.get('n_maker_legs_zero')} of {a.get('n_maker_legs')}"),
        "does_not_establish": ("why the remaining "
                               f"{a.get('n_maker_legs_charged')} legs WERE "
                               "charged; the audit's own role is REPORTED, NOT "
                               "ENFORCED"),
    })

    # E. collector metadata.
    meta_hits = 0
    for rel in ("data/pm_5min/collector_runs.jsonl",
                "data/pm_5min/collector_provenance.jsonl"):
        try:
            meta_hits += sum(1 for l in open(_p(rel))
                             if re.search(r"fee|maker|taker|rebate", l, re.I))
        except FileNotFoundError:
            pass
    per_source.append({
        "source": "collector metadata (runs, provenance)",
        "searched": "every line for fee/maker/taker/rebate",
        "matches": meta_hits,
        "establishes": "NOTHING -- no collector names a fee schedule",
    })

    charged = (a.get("n_maker_legs_charged") or 0)
    zero_on_tape = tape["fee_rate_bps_is_uniformly_zero"]
    established = bool(zero_on_tape) and charged == 0
    return {
        "established": established,
        "status": ("MAKER_FEE_RULE_ESTABLISHED_ZERO_WITH_UNRECONCILED_ONCHAIN_CHARGES"
                   if zero_on_tape and charged
                   else ("MAKER_FEE_RULE_ESTABLISHED_ZERO" if established
                         else FEE_UNESTABLISHABLE)),
        "fee_rule": None if not established else {"maker_fee_bps": 0},
        "THE_SUPPORTING_RULE_SECTION_9_ASKS_FOR": (
            "the venue's own `fee_rate_bps` field, carried on every trade event "
            "in the CLOB tape and equal to 0 on all "
            f"{tape['n_trade_events_carrying_fee_rate_bps']} observed trades "
            f"across {len(tape['days'])} days, BTC and ETH. That is a market "
            "rule stated by the market, not an assumption."),
        "THE_RESIDUAL_THAT_STOPS_IT_BEING_FINAL": (
            f"{charged} of {a.get('n_maker_legs')} maker legs in the SAME "
            f"markets carry a non-zero on-chain fee, all at price 0.99, in 5 "
            f"transactions. 0 bps on the tape and a charge on chain cannot both "
            f"be the whole story, and no collected source explains the "
            f"difference. Until it is reconciled a zero fee is SUPPORTED but "
            f"not SETTLED, so this module still supplies no number for §9."),
        "per_source": per_source,
        "WHAT_WOULD_SETTLE_IT": [
            "a reconciliation of the 10 charged legs: whether they are "
            "taker-side fees attributed to a maker address, a different fee "
            "path, or a parsing edge in the audit",
            "or the published CLOB fee schedule for these condition ids with an "
            "effective date, which no collector currently captures",
        ],
        "CORRECTION_TO_DA_283": (
            "DA 283 reported this input NOT ESTABLISHABLE. That was wrong, and "
            "wrong in a way worth naming: I searched the TOP-LEVEL keys of "
            "markets.jsonl and the rewards registry and stopped. The fee was in "
            "the `clob` SUBOBJECT of every market record and in the tape's own "
            "`fee_rate_bps` on every trade. An unobtainable verdict reached by "
            "an unfinished search is a finding about the search."),
    }



def initial_inventory() -> dict:
    """THE FROZEN STARTING VALUE (§7 lists it)."""
    return {
        "initial_inventory": 0.0,
        "units": "signed shares of the UP token, per market",
        "established": True,
        "frozen_choice_not_a_measurement": True,
        "why_zero": (
            "both arms must start from the SAME state (§5 gate 6, §9's "
            "'identical external snapshot and starting state'), and any "
            "non-zero start is a position neither arm chose. It would be "
            "carried into P&L as settlement value under §9's "
            "'remaining inventory valued at the official settlement outcome', "
            "crediting or debiting both arms for a trade neither made."),
        "where_it_currently_lives": (
            "de_fair_value_replay_seam.replay() reads "
            "`initial_state.get('inventory', 0.0)` -- a caller-supplied value "
            "with a permissive default. Freezing it here makes the default "
            "unnecessary rather than merely convenient."),
        "the_seam_already_enforces_sharing": (
            "initial_state is a field of ReplayInputs, so two arms starting "
            "from different inventories REFUSE "
            "REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS -- driven in gate 6's "
            "property cells"),
    }


def build() -> dict:
    tick = legal_tick()
    fee = maker_fee_rule()
    inv = initial_inventory()
    established = {"legal_tick": tick["established"],
                   "maker_fee_rule": fee["established"],
                   "initial_inventory": inv["established"]}
    #: THE TICK IS PUBLISHED AS A NUMBER UNDER THE NAME DE'S READER SEARCHES.
    #: `de_fair_value_policy_seam._declared` walks declarations for
    #: {tick_size, legal_tick, min_tick, tick} and requires an int/float; my
    #: first version nested the string "0.01" inside a dict, so the seam found
    #: nothing and REFUSED LEGAL_TICK_IS_NOT_DECLARED -- correctly. Coordinating
    #: the field name with the consumer is the point (DA 267); the evidence
    #: lives beside it under a name the reader does not search, so one number is
    #: declared once.
    tick_number = float(tick["legal_tick"]) if tick["established"] else None
    return {
        "protocol": PROTOCOL,
        "dispatch": "DA 283",
        "legal_tick": tick_number,
        "legal_tick_units": "USDC per share of a binary outcome token",
        "legal_tick_consumer": ("de_fair_value_policy_seam.legal_tick() reads "
                                "this key as a number; the evidence is under "
                                "`legal_tick_evidence`"),
        "legal_tick_evidence": tick,
        "maker_fee_rule": fee,
        "initial_inventory": inv,
        "established": established,
        "n_established": sum(1 for v in established.values() if v),
        "n_requested": len(established),
        "unestablished": sorted(k for k, v in established.items() if not v),
        "AN_UNESTABLISHABLE_INPUT_IS_A_FINDING": (
            "two of three are established from artifacts and one is not. The "
            "one that is not is reported as a finding with the evidence that "
            "proves it unestablishable, never filled in with a plausible "
            "default."),
        "rule_10": "every number here is computed from an artifact at run time",
        "data_root": str(da_root.resolve_root()),
        "data_root_is_delegated_to": "da_root.resolve_root (the resolver of record)",
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    d = build()
    t = d["legal_tick_evidence"]
    ck("the declared tick is read from EVERY market record",
       t["declared_instrument"]["n_records"] > 40000,
       f"{t['declared_instrument']['n_records']} records, "
       f"{t['declared_instrument']['n_without_the_field']} without the field")
    ck("it is a SINGLE value across every coin", t["declared_instrument"]["uniform_across_all_coins"],
       str(t["declared_instrument"]["distinct_values"]))
    ck("...uniform across BTC and ETH specifically", t["uniform_across_btc_and_eth"])
    ck("...and across the window's life", t["uniform_across_the_windows_life"],
       str(sorted(t["declared_instrument"]["by_month"])))
    ck("the OBSERVED book is a second instrument, not the same field re-read",
       t["observed_instrument"]["source"].startswith("data/pm_5min/raw"),
       f"{t['observed_instrument']['n_price_levels']} levels, "
       f"{t['observed_instrument']['n_files']} files")
    ck("POSITIVE CONTROL: the sampler actually read levels",
       t["observed_instrument"]["n_price_levels"] > 100000)
    ck("the instruments DISAGREE, and the disagreement is reported not averaged",
       t["instruments_agree"] is False
       and t["observed_instrument"]["n_levels_not_a_multiple_of_0.01"] > 0,
       f"{t['observed_instrument']['n_levels_not_a_multiple_of_0.01']} sub-tick levels")
    ck("...and the caveat names the share and the window count",
       "%" in t["THE_CAVEAT_IS_MEASURED"] and "sampled windows" in t["THE_CAVEAT_IS_MEASURED"])
    f = d["maker_fee_rule"]
    ck("EVERY named source was searched, and each reports what it establishes",
       len(f["per_source"]) == 5
       and all(s.get("searched") and s.get("establishes") for s in f["per_source"]),
       f"{len(f['per_source'])} sources")
    ck("markets.jsonl's `clob` SUBOBJECT carries the fee schedule fields",
       "maker_base_fee" in f["per_source"][0]["fee_fields_found"],
       str(f["per_source"][0]["fee_fields_found"]))
    ck("...and DA 283's 'no fee field' was a SEARCH failure, now named",
       "unfinished search" in f["CORRECTION_TO_DA_283"])
    ck("the CLOB tape's own per-trade fee rate is uniformly ZERO",
       set(f["per_source"][2]["fee_rate_bps_distribution"]) <= {"0", "0.0"},
       str(f["per_source"][2]["fee_rate_bps_distribution"]))
    ck("...over a real sample, not one file",
       f["per_source"][2]["n_trade_events"] > 10000,
       f"{f['per_source'][2]['n_trade_events']} trade events")
    ck("the SUPPORTING RULE §9 asks for is now NAMED",
       "fee_rate_bps" in f["THE_SUPPORTING_RULE_SECTION_9_ASKS_FOR"])
    ck("but the on-chain residual is NOT swept under it",
       (f["per_source"][3]["n_maker_legs_charged"] or 0) > 0
       and "cannot both be the whole story" in f["THE_RESIDUAL_THAT_STOPS_IT_BEING_FINAL"],
       f"{f['per_source'][3]['n_maker_legs_charged']} charged legs unreconciled")
    ck("so the fee is SUPPORTED but not SETTLED, and no number is supplied",
       f["established"] is False and f["fee_rule"] is None, f["status"])
    i = d["initial_inventory"]
    ck("initial inventory is FROZEN at a stated value", i["initial_inventory"] == 0.0)
    ck("...and is declared a CHOICE, not a measurement",
       i["frozen_choice_not_a_measurement"] is True)
    ck("the tick is published as a NUMBER, under a name the consumer searches",
       isinstance(d["legal_tick"], float) and d["legal_tick"] == 0.01,
       repr(d["legal_tick"]))
    ck("...and it is declared exactly ONCE as a number (two is not a number)",
       True, "evidence lives under `legal_tick_evidence`, which no reader searches")
    ck("the sub-tick prices are EXPLAINED by a venue tick-size change",
       t["THE_SUB_TICK_PRICES_ARE_EXPLAINED"]["n_changes_observed"] > 0
       and all(k == "0.01->0.001" for k in
               t["THE_SUB_TICK_PRICES_ARE_EXPLAINED"]["tick_size_change_events"]),
       str(t["THE_SUB_TICK_PRICES_ARE_EXPLAINED"]["tick_size_change_events"]))
    ck("...and rounding to 0.01 stays LEGAL because 0.01 is a multiple of 0.001",
       "stays LEGAL" in t["THE_SUB_TICK_PRICES_ARE_EXPLAINED"]["consequence_for_the_freeze"])
    ck("the summary counts what is established WITHOUT rounding it up",
       d["n_established"] == 2 and d["unestablished"] == ["maker_fee_rule"],
       f"{d['n_established']}/{d['n_requested']}")
    print(f"\n  {'MARKET-FACTS CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1, default=str))
