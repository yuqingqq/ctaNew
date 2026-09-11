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


def legal_tick() -> dict:
    """THE ESTABLISHED TICK, with the disagreement between its two instruments
    stated rather than averaged away."""
    dec = declared_tick()
    obs = observed_increments()
    one = dec["distinct_values"][0] if len(dec["distinct_values"]) == 1 else None
    return {
        "legal_tick": one,
        "established": one is not None,
        "declared_instrument": dec,
        "observed_instrument": obs,
        "instruments_agree": bool(one) and obs["every_level_is_a_multiple_of_0.01"],
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


def maker_fee_rule() -> dict:
    """THE MAKER FEE -- and why it CANNOT be established from what is collected.

    §9: "the verified maker fee applicable to these markets. A zero fee is used
    only if the receipt identifies the supporting market/account rule."

    The on-chain audit measures what the chain RECORDS. That is not the same as
    a RULE, and §9 asks for the rule.
    """
    ev = {}
    # 1. markets.jsonl carries no fee field at all.
    keys = set()
    with open(_p(MARKETS)) as fh:
        for i, line in enumerate(fh):
            if i >= 2000:
                break
            try:
                keys |= set(json.loads(line).keys())
            except Exception:
                pass
    ev["markets_jsonl_fee_fields"] = sorted(k for k in keys if "fee" in k.lower())
    ev["markets_jsonl_all_keys"] = sorted(keys)
    # 2. the rewards registry is collected as a COUNT, not as contents.
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
    ev["rewards_registry_fields"] = sorted(rk)
    ev["rewards_registry_carries_a_fee_rule"] = any(
        "fee" in k.lower() for k in rk)
    # 3. the on-chain audit: what the chain records.
    try:
        a = json.loads(_p(FEE_AUDIT).read_text())
        ev["onchain_audit"] = {
            "artifact": FEE_AUDIT,
            "sha256": hashlib.sha256(_p(FEE_AUDIT).read_bytes()).hexdigest()[:16],
            "formula_recorded": a.get("formula"),
            "n_maker_legs": a.get("n_maker_legs"),
            "n_maker_legs_zero": a.get("n_maker_legs_zero"),
            "n_maker_legs_charged": a.get("n_maker_legs_charged"),
            "maker_charged_share": a.get("maker_charged_share"),
            "charged_leg_prices": sorted(
                {str(r.get("price")) for r in a.get("maker_charged_detail", [])}),
            "n_taker_legs": a.get("n_taker_legs"),
            "taker_all_charged": a.get("taker_all_charged"),
            "taker_formula_match_share": a.get("taker_formula_match_share"),
            "the_audits_own_role": a.get("role"),
        }
    except Exception as e:
        ev["onchain_audit"] = {"error": f"{type(e).__name__}: {e}"}
    o = ev.get("onchain_audit", {})
    zero_share = ((o.get("n_maker_legs_zero") or 0) / o["n_maker_legs"]
                  if o.get("n_maker_legs") else None)
    return {
        "established": False,
        "status": FEE_UNESTABLISHABLE,
        "fee_rule": None,
        "evidence": ev,
        "maker_zero_share_observed": zero_share,
        "WHY_NOT_ESTABLISHED": [
            "markets.jsonl carries NO fee field -- the market artifact does not "
            "state a fee at all",
            "rewards_registry.jsonl is collected as a COUNT (recv_ns, n) and "
            "carries no fee rule, so the registry cannot supply one either",
            "the on-chain audit measures what the chain RECORDS, and its own "
            "`role` says REPORTED, NOT ENFORCED -- it promotes nothing and "
            "clears no gate. A measurement of charges is not a market or "
            "account rule",
            "and the measurement is NOT uniformly zero: "
            f"{o.get('n_maker_legs_charged')} of {o.get('n_maker_legs')} maker "
            f"legs WERE charged, all at price 0.99, so 'maker fills are free' "
            f"is false as stated even as a description",
        ],
        "WHAT_SECTION_9_REQUIRES": (
            "the VERIFIED maker fee applicable to these markets; a zero fee "
            "only if the receipt identifies the SUPPORTING market/account rule"),
        "THEREFORE": (
            "no fee may be frozen, and zero may NOT be assumed. A zero that no "
            "rule supports is an assumption wearing a number's clothes, and it "
            "would flatter every P&L computed from it. This is a FINDING: the "
            "input is unestablishable from what is collected, and "
            "`freeze_is_effective` stays False honestly rather than being "
            "satisfied by an assumption."),
        "WHAT_WOULD_ESTABLISH_IT": [
            "the CLOB fee schedule for these condition ids, captured as an "
            "artifact with an as-of",
            "or the account-level maker-rebate/fee tier that applies to the "
            "executing account, captured the same way",
            "either one identifies a RULE; neither is currently collected",
        ],
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
    return {
        "protocol": PROTOCOL,
        "dispatch": "DA 283",
        "legal_tick": tick,
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
    t = d["legal_tick"]
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
    ck("THE FEE IS NOT ESTABLISHED, and supplies no number",
       f["established"] is False and f["fee_rule"] is None, f["status"])
    ck("...because markets.jsonl carries no fee field",
       f["evidence"]["markets_jsonl_fee_fields"] == [])
    ck("...and the rewards registry carries no fee rule",
       f["evidence"]["rewards_registry_carries_a_fee_rule"] is False,
       str(f["evidence"]["rewards_registry_fields"]))
    ck("...and the on-chain maker legs are NOT uniformly zero",
       (f["evidence"]["onchain_audit"].get("n_maker_legs_charged") or 0) > 0,
       f"{f['evidence']['onchain_audit'].get('n_maker_legs_charged')} charged of "
       f"{f['evidence']['onchain_audit'].get('n_maker_legs')}")
    ck("a zero fee is NOT substituted anywhere in the output",
       f["fee_rule"] is None and "zero may NOT be assumed" in f["THEREFORE"])
    i = d["initial_inventory"]
    ck("initial inventory is FROZEN at a stated value", i["initial_inventory"] == 0.0)
    ck("...and is declared a CHOICE, not a measurement",
       i["frozen_choice_not_a_measurement"] is True)
    ck("the summary counts what is established WITHOUT rounding it up",
       d["n_established"] == 2 and d["unestablished"] == ["maker_fee_rule"],
       f"{d['n_established']}/{d['n_requested']}")
    print(f"\n  {'MARKET-FACTS CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1, default=str))
