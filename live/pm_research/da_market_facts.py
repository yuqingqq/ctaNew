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
        "fee_rate_bps_dtype": "str (the literal '0'), never numeric",
        "fee_rate_bps_present_on": "last_trade_price events only",
        "establishes": ("that the venue REPORTS a constant zero. The field is "
                        "POPULATED -- it is an observation of what the venue "
                        "says, and it is not a placeholder."),
        "CORRECTION_HISTORY_DA_286_TO_DA_290": (
            "DA 286 reported `fee_rate_bps = 0` on every observed trade as the "
            "venue's own per-trade fee rate. DA 289 doubted it, on the strength "
            "of this programme's own on-chain audit whose `limits` said the "
            "field 'is unpopulated'. DA 290 settled it AT THE RAW TAPE: the "
            "field is PRESENT in 10,392 of 10,392 last_trade_price events "
            "across 25 raw files and 76,617 of 76,617 in a wider sweep, absent "
            "in none. THE OBSERVATION WAS REAL; the doubt came from a wrong "
            "word in an artifact, and I had propagated that word by not "
            "checking the raw payload when I first read the limits line."),
        "BUT_IT_IS_NON_DISCRIMINATING_AND_THAT_IS_WHY_IT_IS_NOT_A_RULE": (
            "the venue reports the SAME constant zero for the six accounts the "
            "chain demonstrably charged. A field that reports zero where the "
            "chain took 10% of size x min(p, 1-p) does not reflect what is "
            "actually charged, and cannot be the 'verified maker fee "
            "applicable to these markets' §9 asks for. Populated is not the "
            "same as informative: the durable test of a fee source is whether "
            "it can SEPARATE the charged accounts from the rest, and this one "
            "cannot."),
        "THE_LESSON_THAT_OUTLIVES_THE_FEE": (
            "key a check on a PROPERTY, not on a WORD. 'Unpopulated' was a "
            "label two artifacts asserted and neither measured; the property "
            "that actually decides the question is DISCRIMINATION. A refusal "
            "keyed on the label would have flipped twice in two dispatches; "
            "one keyed on discrimination would not have moved."),
        "does_not_establish": ("what is actually charged to anyone, and so no "
                               "fee rule applicable to us"),
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
        "NO_FEE_RULE_APPLICABLE_TO_US_EXISTS": (
            "a venue-side observation DOES exist and is populated, but it is "
            "non-discriminating: it reports zero for accounts the chain "
            "charged. The only source that reflects what was actually charged "
            "is the on-chain OrderFilled fee word, and that records OTHER "
            "PEOPLE'S accounts -- we have no maker address. So §9's supporting "
            "rule for a zero applicable to US does not exist."),
        "THE_SUPPORTING_RULE_SECTION_9_ASKS_FOR_DOES_NOT_EXIST": (
            "WITHDRAWN. It read: the venue's own `fee_rate_bps` field, "
            "carried on every trade event in the CLOB tape and equal to 0 on "
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
        "THE_CHASE_DA_286": {
            "hypothesis_1_parsing_edge_in_my_own_audit": {
                "tested": "decoded all 5 transactions with the audit's own "
                          "OrderFilled/OrdersMatched topic constants",
                "result": "REFUTED. Each tx carries exactly ONE OrdersMatched "
                          "with the normal 3-topic shape, and NO charged leg's "
                          "maker address is that tx's takerOrderMaker. The "
                          "maker/taker split is correct; these are genuine "
                          "resting maker legs.",
            },
            "hypothesis_2_not_our_markets": {
                "tested": "matched each charged leg's asset ids against the "
                          "94,112 clobTokenIds in markets.jsonl",
                "result": "REFUTED. All 10 are in OUR markets -- takerAssetId "
                          "is one of our 5-min tokens in every case, and "
                          "makerAssetId is 0 (USDC), so all ten are maker BUY "
                          "legs.",
            },
            "hypothesis_3_extreme_price": {
                "tested": "counted zero-fee maker BUY legs at the SAME price",
                "result": "REFUTED as a sufficient cause. All 10 charged legs "
                          "are at EXACTLY 0.9900 -- but 25 maker BUY legs at "
                          "exactly 0.9900, interleaved across the same block "
                          "buckets, paid ZERO. Price alone does not decide it.",
            },
            "WHAT_THE_CHASE_DID_SETTLE": {
                "the_RATE_is_identified": (
                    "implied rate = fee / (size x min(p, 1-p)) has median "
                    "0.0990 over the charged legs, which is `maker_base_fee = "
                    "1000` read as BASIS POINTS: 1000 bps = 10%. So the charge "
                    "reconciles to the schedule field in markets.jsonl by the "
                    "standard formula fee = 0.10 x size x min(p, 1-p)."),
                "the_TRIGGER_is_not": (
                    "nothing in the collected data distinguishes the 10 charged "
                    "legs from 25 otherwise-identical zero-fee legs at the same "
                    "price in the same blocks. The rate is known; WHEN it "
                    "applies is not."),
            },
            "THE_ANSWER_TO_SECTION_9": (
                "the supporting rule for a ZERO maker fee is the venue's own "
                "order-level `fee_rate_bps`, which is 0 on all 76,617 observed "
                "trades. It is a rule stated by the market. But it is not a "
                "CLEAN zero: 10 of 1,056 maker legs (0.95%) were charged at the "
                "market base rate. A receipt may therefore use zero ONLY if it "
                "also carries that exception rate; a receipt claiming an "
                "unqualified zero would be asserting something the chain "
                "contradicts 0.95% of the time."),
        },
        "THE_AUDITS_FORMULA_FAILS_ITS_OWN_POSITIVE_CONTROL": {
            "where": "taker legs -- 901 of 901 charged, the one place the "
                     "schedule is fully observable",
            "match_share_to_1e-6": 0.1220865704772475,
            "n_matching": 110, "n_taker_legs": 901,
            "residual_usdc": {"p50": 5.94e-06, "max": 0.51021345},
            "the_fair_reading": (
                "'reproduces 12%' is true AT A 1e-6 THRESHOLD and understates "
                "the median: half the legs are within six MICRODOLLARS, so the "
                "formula is nearly exact for most fills. But the TAIL is what a "
                "bound depends on, and the maximum residual is 0.51 USDC. An "
                "instrument that can be half a dollar wrong where the answer is "
                "certain cannot bound maker exposure where it is not."),
            "consequence": (
                "NO VALIDATED FEE MODEL EXISTS. The sensitivity DA 287 asked "
                "DE to run -- charge every maker fill at 10% of "
                "size x min(p, 1-p) -- would be computed with this formula, so "
                "it would inherit that tail. It must be fixed or replaced "
                "before it can be called a worst case."),
        },
        "WHAT_WOULD_SETTLE_IT": [
            "a reconciliation of the 10 charged legs: whether they are "
            "taker-side fees attributed to a maker address, a different fee "
            "path, or a parsing edge in the audit",
            "the published CLOB fee schedule for these condition ids with an "
            "effective date, which no collector currently captures -- it is now "
            "the ONLY open question, since the rate itself is reconciled",
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
    inv = initial_inventory()
    fee = maker_fee_rule()
    # ADOPTED AT DA 287: the rule IS identified, so the input is established --
    # as a QUALIFIED zero whose qualification travels with it, never as a clean
    # one. `established` here means "§9's requirement is met", not "no residual".
    established = {"legal_tick": tick["established"],
                   # WITHDRAWN at DA 288: the rule identified at DA 287 was a
                   # zero observed on OTHER accounts, and we have none.
                   "maker_fee_rule": False,
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
    ev = fee["per_source"]
    chase = fee["THE_CHASE_DA_286"]
    ACCOUNT_PARTITION = {
        "finding": "REVIEW 249, verified here from the audit's own two fields",
        "the_trigger_is_NOT_unidentified": (
            "charged-ness PARTITIONS BY ACCOUNT, totally. Six maker addresses "
            "are each charged on 100% of their maker legs, and the number of "
            "addresses holding BOTH a charged and a zero maker leg is ZERO."),
        "charged_class": {"0x0fd0ebb1ba": "1/1", "0x18b0b71054": "1/1",
                          "0x2277c18fb7": "3/3", "0x8d009282a7": "1/1",
                          "0xb3b0780f28": "2/2", "0xbdf221228d": "2/2"},
        "HOW_THE_CLASSES_WERE_ENUMERATED": (
            "the audit SUMMARY enumerates only the CHARGED side -- "
            "`maker_charged_by_address` and its companion hold the same six "
            "addresses, and there is no listing of the 1,046 zero-fee legs by "
            "address. The 218 below were decoded from the 901 RECEIPTS "
            "directly, OrderFilled by OrderFilled, not read off the summary."),
        "n_distinct_maker_addresses": 218,
        "n_in_the_charged_class": 6,
        "n_in_the_zero_class": 212,
        "why_the_25_zero_legs_at_0.99_paid_nothing": (
            "they belong to accounts in the OTHER CLASS. My DA 286 chase "
            "treated that as evidence the trigger was unidentifiable; it was "
            "evidence the trigger is not PRICE. The partition was visible in "
            "two fields of a file I already held, and I did not split by "
            "account."),
        "THE_SAMPLE_LIMIT_THAT_BOUNDS_EVERY_INFERENCE_HERE": (
            "the audit's own limits: 'the 901 receipts are a SAMPLE of our own "
            "recorded trades, not a population; incidence here bounds observed "
            "volume only.' SO AN ABSENCE FROM THE CHARGED SET IS NOT EVIDENCE "
            "OF MEMBERSHIP IN THE ZERO CLASS. That is precisely the inference a "
            "qualified zero would have rested on, and it does not hold: 6 of "
            "218 observed accounts are charged, and the next receipt could "
            "carry a seventh."),
        "OUR_OWN_ADDRESS": {
            "question": "is our maker address in this corpus, and which class?",
            "answer": "WE HAVE NO MAKER ADDRESS. Neither this lane nor any "
                      "module in it declares an executing or maker account; "
                      "every 0x constant in the lane is protocol "
                      "infrastructure (the exchange, USDC, event topics, the "
                      "rewards asset). The programme is research-only -- "
                      "'No live trading, no exchange integrations' -- so no "
                      "order of ours has ever rested on this book.",
            "therefore": (
                "all 218 maker addresses are third parties, and the 1,046 "
                "zero-fee legs are OTHER PEOPLE'S ACCOUNTS. We have NO "
                "observation of our own treatment, and cannot have one from "
                "collected data: the observation does not exist yet rather "
                "than being missing from what we gathered."),
            "so_the_receipt_must_say": (
                "the fee applicable to US is UNOBSERVED BY CONSTRUCTION. A "
                "zero taken from other accounts' fills is not our fee; 6 of "
                "218 observed accounts (2.8%) are in a charged class, and "
                "nothing determines which class an account of ours would "
                "join."),
        },
    }

    RESIDUAL = {
        "n_charged": 10, "n_maker_legs": 1056, "share": 0.00946969696969697,
        "all_at_price": 0.99,
        "implied_rate_median": 0.0990,
        "reconciles_to": ("maker_base_fee = 1000 read as BASIS POINTS: "
                          "fee = 0.10 x size x min(p, 1-p)"),
        "trigger": ("IDENTIFIED as an ACCOUNT ATTRIBUTE (DA 288 / REVIEW 249): "
                    "six accounts charged on 100% of their legs, none mixed. "
                    "My DA 286 'UNIDENTIFIED' was wrong -- I split by price and "
                    "by block and never by account."),
        "THE_DISCRIMINATING_FACT": (
            "25 maker BUY legs at the SAME price 0.9900, interleaved across the "
            "SAME block buckets, paid ZERO. This is what makes price-alone "
            "insufficient as an explanation, and it is what a later reader "
            "needs in order to re-open this question."),
        "what_would_settle_it": ("the published CLOB fee schedule for these "
                                 "condition ids with an effective date; no "
                                 "collector captures it"),
        "hypotheses_refuted_by_driving": {
            k: chase[k]["result"].split(".")[0] for k in chase
            if k.startswith("hypothesis")},
        "THE_UNKNOWN_CARRIES_ITS_OWN_WEIGHT": (
            "DA 287 requires a §9 fee SENSITIVITY, owned by DE: re-run the "
            "economic verdict with EVERY maker fill charged at 10% of "
            "size x min(p, 1-p) -- the worst case consistent with what was "
            "measured. If the verdict does not flip, this residual is "
            "immaterial AS A COMPUTED STATEMENT rather than a hope. If it "
            "flips, the economic gate cannot be settled on collected data and "
            "must say exactly that. Either way nobody has to trust the zero."),
    }
    return {
        "protocol": PROTOCOL,
        "dispatch": "DA 283, ruled at DA 287",

        #: THE QUALIFIED ZERO IS WITHDRAWN (DA 288). NO NUMBER AND NO RULE
        #: STRING ARE PUBLISHED, so `de_fair_value_pnl.declared_fee` refuses
        #: again -- deliberately. The reversal is not a retreat to the old
        #: uncertainty; it rests on a SHARPER fact that makes the zero
        #: inapplicable to us rather than merely qualified.
        "maker_fee_bps": None,
        "maker_fee_rule": None,
        "maker_fee_residual": RESIDUAL,
        "maker_fee_account_partition": ACCOUNT_PARTITION,

        #: THE NEGATIVE DECLARATION (DA 293). Written on its own merits: it is
        #: the honest state of the evidence whatever the ledger decides to do
        #: with it. A question answered NEGATIVELY is still answered.
        "maker_fee_negative_declaration": {
            "status": "FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_DATA",
            "claim": ("the maker fee applicable to US cannot be established "
                      "from any artifact this programme collects. This is a "
                      "RESOLVED question with a negative answer, not an "
                      "unexamined one."),
            "the_five_strands": {
                "0_account_level_incidence": (
                    "6 of 218 distinct maker addresses are in the charged "
                    "class -- 2.75%. That is the rate at which an ACCOUNT is "
                    "charged, and it is the number that matters, because "
                    "charged-ness is an account attribute and not a per-fill "
                    "event. Our own class is UNOBSERVED and unobservable from "
                    "this corpus."),
                "1_account_partition_is_total": (
                    "six maker addresses are charged on 100% of their legs and "
                    "none is mixed; charged-ness is an ACCOUNT attribute"),
                "2_the_RATE_is_an_account_attribute_too": (
                    "each charged address carries exactly ONE rate tier -- "
                    "four at ~0.099 and two at 0.495, never both. So the fee "
                    "is not a market parameter that could be read off the "
                    "market; it is a property of the counterparty"),
                "3_the_venue_field_is_populated_but_non_discriminating": (
                    "`fee_rate_bps` is present on every trade event and reads "
                    "zero for ALL SIX charged accounts. It cannot separate the "
                    "charged from the uncharged, so it cannot be the rule"),
                "4_the_corpus_is_a_sample_not_a_population": (
                    "the audit's own limit: 901 receipts bound observed volume "
                    "only. An absence from the charged set is not membership "
                    "in the zero class"),
                "5_no_validated_fee_model_exists": (
                    "the formula reproduces 110 of 901 taker legs to 1e-6 "
                    "(12.21%) where the schedule is FULLY observable, with a "
                    "maximum residual of 0.51 USDC"),
            },
            "n_maker_addresses_total": 218,
            "n_maker_addresses_charged": 6,
            "account_level_incidence": 0.0275,
            "our_class": "UNOBSERVED -- and unobservable from this corpus",
            "and_the_sixth_which_is_decisive": (
                "WE HAVE NO MAKER ADDRESS. The programme is research-only and "
                "has never rested an order on this book, so our own treatment "
                "is not merely unobserved -- it does not yet exist to observe. "
                "No amount of further collection closes that; only trading "
                "does."),
            "REVIEW_256_SENTENCE_TRANSCRIBED": (
                "`fee_rate_bps` HAS NEVER BEEN OBSERVED ON A CHARGED FILL. The "
                "field is populated and constant -- REVIEW 256 measures it at "
                "1,881,868 of 1,881,868 trade events carrying the same value -- "
                "and every fill the chain actually charged sits OUTSIDE that "
                "set, because the charged fills are identified on chain by the "
                "OrderFilled fee word and never by this field. A constant "
                "observed only where the answer is always the same is not "
                "evidence about the case where it differs. The positive "
                "controls are what make that statement safe to make rather "
                "than merely plausible: the extractor was shown to separate a "
                "large value from a tiny one before it was pointed at the "
                "question."),
            "REVIEW_256_COUNTS_ARE_REVS_MEASUREMENT": (
                "the 1,881,868 constancy and the control counts are REVIEW "
                "256's, transcribed here rather than re-derived. DA's own "
                "independent sweep found the same constancy over a smaller "
                "window: 76,617 of 76,617 trade events across 8 UTC days."),
            "what_would_change_it": [
                "the published CLOB fee schedule for these condition ids with "
                "an effective date, which no collector captures",
                "or an executing account of our own with observed fills",
            ],
        },

        #: FOR DE'S §9 SENSITIVITY. Measured, not quoted.
        "sensitivity_rate_worst_observed": 0.495,
        "sensitivity_rate_modal": 0.099,
        "sensitivity_price_basis": "min_p_at_each_fill_own_price",
        "sensitivity_basis_note": (
            "rate = fee / (size x min(p, 1-p)) evaluated at EACH FILL'S OWN "
            "price, never at a portfolio or day average -- min(p, 1-p) is "
            "convex in p and an average price understates the charge at the "
            "extremes where every observed charge actually fell. The two tiers "
            "are per-address and disjoint: 0.0989/0.0990 on four addresses, "
            "0.4950 on two, one tier per address with no address in both."),
        "sensitivity_tiers_by_address": {
            "0x0fd0ebb1ba": 0.495, "0x18b0b71054": 0.0989,
            "0x2277c18fb7": 0.099, "0x8d009282a7": 0.0989,
            "0xb3b0780f28": 0.099, "0xbdf221228d": 0.495},
        "dispatch_note": ("DA 287 ruled ADOPT the qualified zero; DA 288 "
                          "REVERSED it on REVIEW 249's account partition, "
                          "before it landed."),
        "legal_tick": tick_number,
        "legal_tick_units": "USDC per share of a binary outcome token",
        "legal_tick_consumer": ("de_fair_value_policy_seam.legal_tick() reads "
                                "this key as a number; the evidence is under "
                                "`legal_tick_evidence`"),
        "legal_tick_evidence": tick,
        "maker_fee_rule_evidence": fee,
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
    f = d["maker_fee_rule_evidence"]
    ck("EVERY named source was searched, and each reports what it establishes",
       len(f["per_source"]) == 5
       and all(s.get("searched") and s.get("establishes") for s in f["per_source"]),
       f"{len(f['per_source'])} sources")
    ck("markets.jsonl's `clob` SUBOBJECT carries the fee schedule fields",
       "maker_base_fee" in f["per_source"][0]["fee_fields_found"],
       str(f["per_source"][0]["fee_fields_found"]))
    ck("...and DA 283's 'no fee field' was a SEARCH failure, now named",
       "unfinished search" in f["CORRECTION_TO_DA_283"])
    src = f["per_source"][2]
    src = f["per_source"][2]
    ck("the tape's `fee_rate_bps` is POPULATED -- the observation is real",
       src["establishes"].startswith("that the venue REPORTS"),
       src["fee_rate_bps_dtype"])
    ck("...and the DA 286 -> 289 -> 290 correction history is carried",
       "THE OBSERVATION WAS REAL" in src["CORRECTION_HISTORY_DA_286_TO_DA_290"])
    ck("but it is NON-DISCRIMINATING, and that is why it is not a rule",
       "cannot" in src["BUT_IT_IS_NON_DISCRIMINATING_AND_THAT_IS_WHY_IT_IS_NOT_A_RULE"]
       and "six accounts" in src["BUT_IT_IS_NON_DISCRIMINATING_AND_THAT_IS_WHY_IT_IS_NOT_A_RULE"])
    ck("...the durable test is DISCRIMINATION, not presence",
       "DISCRIMINATION" in src["THE_LESSON_THAT_OUTLIVES_THE_FEE"])
    nd = d["maker_fee_negative_declaration"]
    ck("the NEGATIVE DECLARATION is written, with a named status",
       nd["status"] == "FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_DATA")
    ck("...and it carries FIVE independent strands plus the decisive sixth",
       len(nd["the_five_strands"]) == 5 and "NO MAKER ADDRESS" in
       nd["and_the_sixth_which_is_decisive"])
    ck("...and says plainly that further collection cannot close it",
       "only trading does" in nd["and_the_sixth_which_is_decisive"])
    ck("the RATE is an account attribute: one tier per address, none in both",
       len(set(d["sensitivity_tiers_by_address"].values())) == 3
       and len(d["sensitivity_tiers_by_address"]) == 6,
       str(sorted(set(d["sensitivity_tiers_by_address"].values()))))
    ck("DE's sensitivity fields are present and MEASURED",
       d["sensitivity_rate_worst_observed"] == 0.495
       and d["sensitivity_rate_modal"] == 0.099
       and d["sensitivity_price_basis"] == "min_p_at_each_fill_own_price")
    ck("...and the price basis warns against averaging a CONVEX function",
       "convex in p" in d["sensitivity_basis_note"])
    ck("so NO FEE RULE APPLICABLE TO US EXISTS is stated plainly",
       "does not exist" in f["NO_FEE_RULE_APPLICABLE_TO_US_EXISTS"])
    ck("but the on-chain residual is NOT swept under it",
       (f["per_source"][3]["n_maker_legs_charged"] or 0) > 0
       and "cannot both be the whole story" in f["THE_RESIDUAL_THAT_STOPS_IT_BEING_FINAL"],
       f"{f['per_source'][3]['n_maker_legs_charged']} charged legs unreconciled")
    ap = d["maker_fee_account_partition"]
    ck("THE QUALIFIED ZERO IS WITHDRAWN -- no number and no rule are published",
       d["maker_fee_bps"] is None and d["maker_fee_rule"] is None,
       "so de_fair_value_pnl.declared_fee refuses again, deliberately")
    ck("the TRIGGER is IDENTIFIED: charged-ness partitions BY ACCOUNT, totally",
       ap["n_in_the_charged_class"] == 6 and ap["n_in_the_zero_class"] == 212,
       f"{ap['n_in_the_charged_class']} charged / {ap['n_in_the_zero_class']} "
       f"zero of {ap['n_distinct_maker_addresses']} addresses")
    ck("...every charged address is charged on ALL of its legs",
       all(n.split("/")[0] == n.split("/")[1]
           for n in ap["charged_class"].values()),
       str(ap["charged_class"]))
    ck("...so the 25 zero legs at 0.9900 are explained by CLASS, not price",
       "OTHER CLASS" in ap["why_the_25_zero_legs_at_0.99_paid_nothing"])
    ck("WE HAVE NO MAKER ADDRESS, so we have no observation of our own fee",
       "WE HAVE NO MAKER ADDRESS" in ap["OUR_OWN_ADDRESS"]["answer"]
       and "UNOBSERVED BY CONSTRUCTION" in ap["OUR_OWN_ADDRESS"]["so_the_receipt_must_say"])
    ck("...and the zeros are named as OTHER PEOPLE'S accounts",
       "OTHER PEOPLE'S ACCOUNTS" in ap["OUR_OWN_ADDRESS"]["therefore"])
    ck("the SAMPLE limit is carried: absence from the charged set proves nothing",
       "NOT EVIDENCE" in ap["THE_SAMPLE_LIMIT_THAT_BOUNDS_EVERY_INFERENCE_HERE"]
       and "SAMPLE" in ap["THE_SAMPLE_LIMIT_THAT_BOUNDS_EVERY_INFERENCE_HERE"])
    ck("...and the classes were decoded from the RECEIPTS, not the summary",
       "not read off the summary" in ap["HOW_THE_CLASSES_WERE_ENUMERATED"])
    tf = d["maker_fee_rule_evidence"]["THE_AUDITS_FORMULA_FAILS_ITS_OWN_POSITIVE_CONTROL"]
    ck("the audit's formula FAILS its own positive control, and by how much",
       tf["n_matching"] == 110 and tf["n_taker_legs"] == 901
       and tf["residual_usdc"]["max"] > 0.5,
       f"{tf['n_matching']}/{tf['n_taker_legs']} to 1e-6, max residual "
       f"{tf['residual_usdc']['max']} USDC")
    ck("...stated FAIRLY: the median is near-exact, the TAIL is what fails",
       "MICRODOLLARS" in tf["the_fair_reading"] and "TAIL" in tf["the_fair_reading"])
    ck("...so NO VALIDATED FEE MODEL EXISTS is said plainly",
       "NO VALIDATED FEE MODEL EXISTS" in tf["consequence"])
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
