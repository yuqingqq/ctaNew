"""FAIR-VALUE LANE, STEP 1 GATE 1 -- THE SETTLEMENT-LABEL/STATUS READER, FROZEN.

DA 266 / R-917. Fair-value plan §5 (Step 0) with §2's admissibility.

WHAT THIS IS. One reader that turns a market window into EITHER a
label-admissible settled outcome OR a COUNTED STATUS. It never infers a label.
A window is label-admissible ONLY when BOTH hold:

    (a) an OFFICIAL resolution exists for the slug in `resolutions.jsonl`, and
    (b) this checker independently reads VERIFIED_AGREE against it.

Everything else -- not in capture, no official resolution, outage, stale
boundary, margin below the feed's resolution -- is a STATUS WITH A COUNT.
Rule 4: exclusions are statuses, never silent drops.

WHY A SECOND READER WOULD BE A SECOND CONVENTION. The settlement convention
was pinned at Q-DA-142/146 and is implemented once, in
`exp_m6_settlement.read_at` (the boundary reader) and
`da_settlement_audit_v1.load_population` (the counted-exclusion loader). This
module IMPORTS BOTH rather than restating them. A reimplementation here would
be a second convention that could drift from the audited one silently, which is
the failure this programme keeps finding.

THE STATISTIC, as pinned: `X60(T) >= X60(t0)`, TIES UP -- the 60-second TWAP at
expiry against the 60-second TWAP at the window's open. The plan writes it
`S60`; `exp_m6_settlement` keys the same series as `(symbol, 60)`. Same
statistic, three spellings; this module resolves it by the series key, not the
letter.

MARGIN AWARENESS, AND WHY IT IS NOT A FUDGE. `X60(T) - X60(t0)` can be smaller
than the feed can resolve. Scoring those as agreement or disagreement would
manufacture a verdict out of quantisation. They are their own status,
MARGIN_BELOW_RESOLUTION, counted and excluded -- never folded into either side.

HOLD. Any VERIFIED_DISAGREE puts the LANE ON HOLD (plan §10, kill criterion 1)
until a superseding receipt explains it. `hold_state()` computes that from the
declaration chain; it is a predicate, not prose.
"""
from __future__ import annotations

import json
from pathlib import Path

import da_settlement_audit_v1 as AUDIT
import declaration_chain as DC
import exp_m6_settlement as M6

PROTOCOL = "P003_DA_FAIR_VALUE_GATE1_LABELS_V1"
FAMILY = "da_fair_value_gate1_declaration"
HERE = Path(__file__).resolve().parent
DECL_DIR = HERE / "declarations"

#: The settlement series, keyed as `exp_m6_settlement` keys it. 60 == the
#: 60-second TWAP topic `crypto_prices_twap_sixty`.
WINDOW_S = 60

#: Below this the feed cannot resolve the comparison. Declared BEFORE any
#: audit runs (rule 11: choosing after seeing voids the test). 0.5bp is the
#: split Q-DA-146 pre-registered for the big-margin gate; the same number is
#: used here for the same reason and is NOT re-tuned.
MARGIN_EPS_BP = AUDIT.BIG_MARGIN_BP

#: A boundary read older than this was taken against a reference the feed had
#: not refreshed. `read_at` returns the last sample at or before the boundary
#: and says nothing about its age -- the staleness finding already recorded in
#: da_settlement_audit_v1.audit().
STALE_BOUNDARY_MS = 1000

# ---- THE STATUS GRAMMAR -------------------------------------------------
LABEL_ADMISSIBLE = "VERIFIED_AGREE"
DISAGREE = "VERIFIED_DISAGREE"
NO_OFFICIAL = "NO_OFFICIAL_RESOLUTION"
NOT_IN_CAPTURE = "NOT_IN_CAPTURE"
OUTAGE = "OUTAGE"
STALE_BOUNDARY = "STALE_BOUNDARY"
MARGIN_UNRESOLVABLE = "MARGIN_BELOW_RESOLUTION"

STATUSES = (LABEL_ADMISSIBLE, DISAGREE, NO_OFFICIAL, NOT_IN_CAPTURE,
            OUTAGE, STALE_BOUNDARY, MARGIN_UNRESOLVABLE)
#: EXACTLY ONE status admits a label. Stated as data so a reader cannot
#: acquire a second by being written into a branch.
ADMITTING = (LABEL_ADMISSIBLE,)

ROLLING_TOPIC = "crypto_prices_twap_sixty"
RAW_TOPIC = "crypto_prices"

GATE_REFUSED = "FAIR_VALUE_GATE1_REFUSED"
ROLLING_AS_RAW = "ROLLING_TWAP_PASSED_WHERE_A_RAW_AGGREGATE_PATH_IS_REQUIRED"
LANE_ON_HOLD = "FAIR_VALUE_LANE_ON_HOLD_UNTIL_A_SUPERSEDING_RECEIPT"


class GateRefused(RuntimeError):
    """A named refusal."""


def assert_series_is_not_a_rolling_statistic(topic: str, role: str) -> None:
    """`crypto_prices_twap_sixty` IS ALREADY A ROLLING MEAN.

    Integrating it, or passing it where a RAW AGGREGATE PATH is required,
    double-counts the averaging window: each sample already contains the
    preceding 60 seconds. The plan lists the topic as available data; it does
    NOT license using it as the raw path, and nothing else in this lane
    distinguishes the two by type. So the distinction is enforced here, by
    name, at the boundary where a caller could confuse them.
    """
    if role == "raw_aggregate_path" and topic == ROLLING_TOPIC:
        raise GateRefused(
            f"REFUSED {ROLLING_AS_RAW}: `{ROLLING_TOPIC}` is the ROLLING "
            f"60-second statistic, not the raw aggregate path. Each sample "
            f"already contains the preceding 60s, so integrating it "
            f"double-counts the window. The raw path is `{RAW_TOPIC}`.")


def _margin_bp(x_T: float, x_0: float) -> float:
    """Signed margin in basis points of the strike."""
    if not x_0:
        return 0.0
    return (x_T - x_0) / x_0 * 1e4


def classify_window(*, slug, market, winners, streams, coins=None,
                    eps_bp=MARGIN_EPS_BP, stale_ms=STALE_BOUNDARY_MS,
                    gaps=None):
    """ONE window -> a row carrying a STATUS and, only sometimes, a label.

    `winners` is the official `resolutions.jsonl` entry or None. The official
    join is NOT optional: with no official row the status is NO_OFFICIAL and
    there is no label, however confidently the feed reads.
    """
    coins = M6.COINS if coins is None else coins
    sym = coins.get(market["coin"])
    t0_ms, T_ms = market["window_start"] * 1000, market["window_end"] * 1000
    row = {"slug": slug, "coin": market["coin"], "symbol": sym,
           "window_start_ms": t0_ms, "window_end_ms": T_ms,
           "status": None, "label": None, "official_up": None,
           "checker_up": None, "margin_bp": None, "boundary_age_ms": None}

    series = streams.get((sym, WINDOW_S)) if sym else None
    if not series:
        row["status"] = NOT_IN_CAPTURE
        return row

    x_T, t_T = M6.read_at(series, T_ms, False)
    x_0, t_0 = M6.read_at(series, t0_ms, False)
    if x_T is None or x_0 is None:
        row["status"] = NOT_IN_CAPTURE
        return row

    if gaps and _covered_by_gap(gaps, sym, t0_ms, T_ms):
        row["status"] = OUTAGE
        return row

    age = max(T_ms - (t_T or T_ms), t0_ms - (t_0 or t0_ms))
    row["boundary_age_ms"] = age
    if age > stale_ms:
        row["status"] = STALE_BOUNDARY
        return row

    margin = _margin_bp(x_T, x_0)
    row["margin_bp"] = margin
    # TIES UP: `>=`. A tie is a real UP, not a margin failure -- so the
    # margin test is on the ABSOLUTE margin and an exact tie is admitted by
    # the convention, then screened by the resolution test like any other.
    row["checker_up"] = bool(x_T >= x_0)

    if abs(margin) < eps_bp and x_T != x_0:
        row["status"] = MARGIN_UNRESOLVABLE
        return row

    if not winners:
        row["status"] = NO_OFFICIAL
        return row

    official_up = bool(winners.get("Up"))
    row["official_up"] = official_up
    if official_up == row["checker_up"]:
        row["status"] = LABEL_ADMISSIBLE
        row["label"] = "UP" if official_up else "DOWN"
    else:
        row["status"] = DISAGREE
    return row


def _covered_by_gap(gaps, sym, a_ms, b_ms) -> bool:
    for g in gaps or ():
        if g.get("symbol") not in (None, sym):
            continue
        if int(g["start_ms"]) <= b_ms and int(g["end_ms"]) >= a_ms:
            return True
    return False


def tally(rows) -> dict:
    """Counts per status. EVERY row lands in exactly one."""
    out = {s: 0 for s in STATUSES}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    n = len(rows)
    admissible = sum(out[s] for s in ADMITTING)
    return {"n_windows": n, "by_status": out,
            "n_label_admissible": admissible,
            "n_excluded_as_status": n - admissible,
            "every_row_has_exactly_one_status": sum(out.values()) == n,
            "ADMITTING_STATUSES": list(ADMITTING)}


def hold_state(rows, decl_dir: Path | None = None) -> dict:
    """THE LANE IS ON HOLD WHILE ANY DISAGREE IS UNEXPLAINED.

    Computed, not asserted: a HOLD is lifted only by a SUPERSEDING RECEIPT in
    this gate's declaration family whose head names the disagreeing slugs. The
    chain head is resolved by `declaration_chain.resolve_head` -- the one
    implementation (BE 77) -- so this rule cannot drift from the rest of the
    programme's supersession semantics.
    """
    d = Path(decl_dir) if decl_dir else DECL_DIR
    bad = sorted(r["slug"] for r in rows if r["status"] == DISAGREE)
    explained, head = set(), None
    if bad:
        try:
            head = DC.resolve_head(str(d), FAMILY)
        except Exception:                           # noqa: BLE001
            head = None
        if head:
            try:
                doc = json.loads(Path(head["path"]).read_text()) if isinstance(head, dict) else {}
            except Exception:                       # noqa: BLE001
                doc = {}
            explained = set(doc.get("explained_disagreements") or ())
    unexplained = [s for s in bad if s not in explained]
    return {"n_disagree": len(bad), "n_explained": len(bad) - len(unexplained),
            "unexplained": unexplained, "head": head,
            "ON_HOLD": bool(unexplained),
            "refusal_if_run": (f"REFUSED {LANE_ON_HOLD}: {len(unexplained)} "
                               f"VERIFIED_DISAGREE window(s) unexplained: "
                               f"{unexplained[:5]}") if unexplained else None}


def assert_lane_not_on_hold(rows, decl_dir: Path | None = None) -> dict:
    h = hold_state(rows, decl_dir)
    if h["ON_HOLD"]:
        raise GateRefused(h["refusal_if_run"])
    return h


# ---- FALSIFIERS ---------------------------------------------------------
def _fixture(x0=100.0, xT=101.0, up=True, coin="btc", t0=1_000_000):
    sym = M6.COINS[coin]
    t0_ms, T_ms = t0 * 1000, (t0 + 300) * 1000
    series = ([t0_ms, T_ms], [t0_ms, T_ms], [x0, xT])
    market = {"coin": coin, "window_start": t0, "window_end": t0 + 300}
    return sym, series, market, ({"Up": up} if up is not None else None)


def falsify() -> int:
    """MANDATORY TWO-WAY FALSIFIERS (plan §5). Synthetic only -- no heavy I/O."""
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    # (1) POSITIVE CONTROL: an exact Chainlink endpoint label PASSES.
    sym, series, market, win = _fixture(100.0, 101.0, True)
    r = classify_window(slug="s1", market=market, winners=win,
                        streams={(sym, WINDOW_S): series})
    ck("an exact Chainlink endpoint label is VERIFIED_AGREE and admits a label",
       r["status"] == LABEL_ADMISSIBLE and r["label"] == "UP", r["status"])

    sym, series, market, win = _fixture(101.0, 100.0, False)
    r2 = classify_window(slug="s2", market=market, winners=win,
                         streams={(sym, WINDOW_S): series})
    ck("and the DOWN direction too", r2["status"] == LABEL_ADMISSIBLE
       and r2["label"] == "DOWN", r2["status"])

    # TIES UP, as the convention pins it.
    sym, series, market, win = _fixture(100.0, 100.0, True)
    r3 = classify_window(slug="s3", market=market, winners=win,
                         streams={(sym, WINDOW_S): series})
    ck("a TIE is UP (the pinned convention), not a margin failure",
       r3["checker_up"] is True and r3["status"] == LABEL_ADMISSIBLE, r3["status"])

    # (2) KNOWN-BAD: the rolling TWAP passed where a raw aggregate path is
    #     required must REFUSE, by name.
    try:
        assert_series_is_not_a_rolling_statistic(ROLLING_TOPIC, "raw_aggregate_path")
        ck("rolling-TWAP-as-raw-integral REFUSES", False, "admitted")
    except GateRefused as exc:
        ck("rolling-TWAP-as-raw-integral REFUSES by name",
           ROLLING_AS_RAW in str(exc), str(exc)[:64] + "...")
    # and the SAME topic in its OWN role is admitted -- a control that can fail
    try:
        assert_series_is_not_a_rolling_statistic(ROLLING_TOPIC, "settlement_endpoint")
        ck("the same topic in its OWN endpoint role is admitted", True)
    except GateRefused:
        ck("the same topic in its OWN endpoint role is admitted", False)

    # (3) A DELIBERATELY INVERTED settlement/token mapping is DETECTED.
    #     TWO symbols, EACH with ONE series spanning ALL its windows, and the
    #     swap must produce a genuine VERIFIED_DISAGREE.
    #     TWO EARLIER VERSIONS OF THIS CELL WERE WRONG AND BOTH ARE WORTH
    #     RECORDING: the first used ONE symbol, so the inverted map pointed at
    #     a series that did not exist and every row came back NOT_IN_CAPTURE --
    #     detection BY ABSENCE, which passes while proving nothing. The second
    #     wrote each window's series under the SAME (symbol, 60) key, so later
    #     windows clobbered earlier ones and the reads fell outside the
    #     capture. A control that fails for the wrong reason is not a control.
    btc, eth = M6.COINS["btc"], M6.COINS["eth"]
    honest = {"btc": btc, "eth": eth}
    swapped = {"btc": eth, "eth": btc}
    base = 2_000_000
    wins = [(base + i * 600, base + i * 600 + 300) for i in range(2)]
    bumps = [t for w in wins for t in w]
    # btc RISES across every window, eth FALLS across every window.
    btc_v = [100.0 * (1.0 + 0.002 * i) for i in range(len(bumps))]
    eth_v = [200.0 * (1.0 - 0.002 * i) for i in range(len(bumps))]
    ms = [t * 1000 for t in bumps]
    streams2 = {(btc, WINDOW_S): (ms, ms, btc_v),
                (eth, WINDOW_S): (ms, ms, eth_v)}
    markets2 = []
    for (t0, T) in wins:
        markets2.append(("btc", {"coin": "btc", "window_start": t0, "window_end": T}, {"Up": True}))
        markets2.append(("eth", {"coin": "eth", "window_start": t0, "window_end": T}, {"Up": False}))
    rows_ok = [classify_window(slug=f"ok{i}", market=m, winners=w, streams=streams2,
                               coins=honest) for i, (_, m, w) in enumerate(markets2)]
    rows_inv = [classify_window(slug=f"inv{i}", market=m, winners=w, streams=streams2,
                                coins=swapped) for i, (_, m, w) in enumerate(markets2)]
    ok_st = sorted({r["status"] for r in rows_ok})
    inv_st = sorted({r["status"] for r in rows_inv})
    ck("the honest mapping yields only VERIFIED_AGREE",
       ok_st == [LABEL_ADMISSIBLE], ok_st)
    ck("an INVERTED settlement/token mapping is DETECTED",
       all(r["status"] != LABEL_ADMISSIBLE for r in rows_inv), inv_st)
    ck("...and DETECTED AS DISAGREEMENT, not as absence -- both symbols have series",
       inv_st == [DISAGREE], inv_st)

    # (4) STATUSES ARE COUNTED, NEVER INFERRED LABELS.
    sym, series, market, _ = _fixture(100.0, 101.0, True)
    r = classify_window(slug="nores", market=market, winners=None,
                        streams={(sym, WINDOW_S): series})
    ck("a missing official resolution is NO_OFFICIAL and carries NO label",
       r["status"] == NO_OFFICIAL and r["label"] is None, r["status"])
    r = classify_window(slug="nocap", market=market, winners={"Up": True}, streams={})
    ck("a window outside the capture is NOT_IN_CAPTURE, not a label",
       r["status"] == NOT_IN_CAPTURE and r["label"] is None, r["status"])
    sym, series, market, win = _fixture(100.0, 100.0001, True)
    r = classify_window(slug="thin", market=market, winners=win,
                        streams={(sym, WINDOW_S): series})
    ck("a margin below the feed's resolution is its OWN status",
       r["status"] == MARGIN_UNRESOLVABLE and r["label"] is None, r["status"])
    sym, series, market, win = _fixture(100.0, 101.0, True)
    stale = ([series[0][0] - 5000, series[0][1] - 5000],
             [series[1][0] - 5000, series[1][1] - 5000], series[2])
    r = classify_window(slug="stale", market=market, winners=win,
                        streams={(sym, WINDOW_S): stale})
    ck("a stale boundary read is its OWN status", r["status"] == STALE_BOUNDARY, r["status"])

    # (5) THE TALLY CLOSES and exactly one status admits.
    rows = rows_ok + rows_inv
    t = tally(rows)
    ck("every row lands in exactly one status", t["every_row_has_exactly_one_status"],
       f"{t['n_label_admissible']} admissible of {t['n_windows']}")
    ck("exactly ONE status admits a label", len(ADMITTING) == 1, ADMITTING)

    # (6) HOLD: a DISAGREE puts the lane on hold, and the predicate says so.
    h = hold_state(rows_inv)
    ck("any VERIFIED_DISAGREE puts the lane ON HOLD -- driven on REAL disagreements",
       h["n_disagree"] > 0 and h["ON_HOLD"],
       f"n_disagree={h['n_disagree']} ON_HOLD={h['ON_HOLD']}")
    ck("...and the honest rows do NOT hold the lane (a control that can fail)",
       hold_state(rows_ok)["ON_HOLD"] is False)
    forced = [{"slug": "d1", "status": DISAGREE}]
    h2 = hold_state(forced)
    ck("a forced DISAGREE is ON HOLD and names the slug", h2["ON_HOLD"]
       and "d1" in h2["unexplained"], h2["refusal_if_run"][:60] + "...")
    try:
        assert_lane_not_on_hold(forced)
        ck("assert_lane_not_on_hold REFUSES on an unexplained disagreement", False)
    except GateRefused as exc:
        ck("assert_lane_not_on_hold REFUSES by name", LANE_ON_HOLD in str(exc))

    print(f"\n  {'ALL FALSIFIERS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


# ---- THE RECEIPT THE CANCELLATION LANE'S GUARD READS (DA 267) -----------
#
# SECOND CONSUMER, DISCOVERED LIVE. `de_forward_evaluator.py:344` refuses any
# day with no point-estimate receipt, and reads a WINNER-SOURCE BLOCK found
# RECURSIVELY by `_winner_source_blocks`: any dict carrying BOTH a str
# `sha256` AND a dict `chainlink_verification`.
#
# THE SHAPE IS NOT INVENTED HERE. Twenty-five receipts for 09-03..09-06
# already carry it, and this module matches THAT convention field for field.
# Its `reader_module` already points at `exp_m6_settlement.py`, the same
# reader this gate imports -- so one reader, one convention, two lanes.
#
# THE STATUS VOCABULARY IS THE CONSUMER'S, NOT THIS MODULE'S. The guard
# counts exactly five names and puts ANY other name in `unknown_statuses`,
# which blocks finality. So gate-1's finer grammar is MAPPED, and the detail
# it would have carried travels in `verifiability` -- where the landed
# convention already keeps a staleness and a margin field (emitted here as `staleness_ms` and `margin_bp` -- MILLISECONDS and BASIS POINTS, not seconds and absolute; the landed 09-06 receipts use `staleness_s`, and that cover has already misled one seat).
CONSUMER_STATUSES = ("VERIFIED_AGREE", "DISAGREE", "BOUNDARY_NOT_IN_CAPTURE",
                     "CHAINLINK_UNAVAILABLE", "VENUE_UNRESOLVED")

#: gate-1 status -> the consumer's status. Every mapping preserves finality:
#: anything that is not VERIFIED_AGREE leaves the day not-final, which is
#: exactly what each of these means.
STATUS_MAP = {
    LABEL_ADMISSIBLE: "VERIFIED_AGREE",
    DISAGREE: "DISAGREE",
    NOT_IN_CAPTURE: "BOUNDARY_NOT_IN_CAPTURE",
    OUTAGE: "CHAINLINK_UNAVAILABLE",
    NO_OFFICIAL: "VENUE_UNRESOLVED",
    # NO NEW NAMES ARE PROPOSED. A stale boundary and a margin the feed
    # cannot resolve are both "the boundary does not verify", which is what
    # BOUNDARY_NOT_IN_CAPTURE means to the consumer. The DISTINCTION is not
    # lost: it travels as `verifiability.T.staleness_s` and
    # `verifiability.margin`, both of which the landed convention already
    # carries, plus `gate1_status` on the row.
    STALE_BOUNDARY: "BOUNDARY_NOT_IN_CAPTURE",
    MARGIN_UNRESOLVABLE: "BOUNDARY_NOT_IN_CAPTURE",
}

RECEIPT_GLOB = "p003_de_point_estimate_day_{compact}_{revision}__*.json"
DEFAULT_REVISION = "L250ms"


def to_consumer_status(gate1_status: str) -> str:
    if gate1_status not in STATUS_MAP:
        raise GateRefused(
            f"REFUSED {GATE_REFUSED}: gate-1 status {gate1_status!r} has no "
            f"mapping to the consumer's vocabulary. An unmapped status would "
            f"reach `unknown_statuses` and block finality silently.")
    return STATUS_MAP[gate1_status]


def winner_source_block(rows, *, winner_source_sha256, winner_source_path,
                        n_slugs_required, reader_module_sha256,
                        day_slice=None, convention="S60(T) >= S60(t0)"):
    """The block `_winner_source_blocks` will find, counts RECOMPUTED here.

    `counts` is derived from `per_slug` in this function and never taken from
    a caller (the landed convention's own rule, REV 104B §6(2)).
    """
    per_slug = {}
    for r in rows:
        cs = to_consumer_status(r["status"])
        per_slug[r["slug"]] = {
            "status": cs,
            "gate1_status": r["status"],
            "the_conventions_own_reading": (
                None if r["checker_up"] is None else
                ("AGREE" if r["checker_up"] == r["official_up"] else "DISAGREE")
                if r["official_up"] is not None else None),
            "chainlink_up_won": r["checker_up"],
            "venue_up_won": r["official_up"],
            "verifiability": {"margin_bp": r["margin_bp"],
                              "T": {"staleness_ms": r["boundary_age_ms"]}},
        }
    counts = {n: sum(1 for v in per_slug.values() if v["status"] == n)
              for n in CONSUMER_STATUSES}
    every_agree = (bool(per_slug)
                   and counts["VERIFIED_AGREE"] == len(per_slug))
    finality = {
        "a_n_slugs_required": n_slugs_required,
        "a_n_slugs_verified": len(per_slug),
        "a_slug_set_equals_the_days": len(per_slug) == n_slugs_required,
        "b_counts_recomputed_here": True,
        "c_every_status_is_VERIFIED_AGREE": every_agree,
        "d_convention_is_the_pinned_one": convention == "S60(T) >= S60(t0)",
        "f_provenance": {
            "reader_module": {"path": "live/pm_research/exp_m6_settlement.py",
                              "sha256": reader_module_sha256},
            "venue_record": {"path": winner_source_path,
                             "sha256": winner_source_sha256},
            "gate": {"path": "live/pm_research/da_fair_value_gate1_labels.py",
                     "protocol": PROTOCOL}},
        "is_final": bool(every_agree and len(per_slug) == n_slugs_required),
    }
    if day_slice is None:
        raise GateRefused(
            f"REFUSED {NO_SLICE}: a block must carry `day_slice`. The "
            f"whole-file sha is PROVENANCE and is never the match key -- the "
            f"ledger grows, so a whole-file key refuses a correct "
            f"verification for a reason that has nothing to do with the day.")
    return {
        "sha256": winner_source_sha256,
        "sha256_is": "THE WHOLE-FILE DIGEST -- PROVENANCE ONLY, never the match key",
        "day_slice": {"sha256": day_slice["sha256"],
                      "n_day_records": day_slice["n_day_records"]},
        "day_slice_definition": day_slice.get("definition"),
        "path": winner_source_path,
        "n_slugs": len(per_slug),
        "is_final_for_quotation": finality["is_final"],
        "method": ("DA 267 / gate 1: the official `winners` join from "
                   "resolutions.jsonl, verified per window against the pinned "
                   "Chainlink convention by da_fair_value_gate1_labels."),
        "chainlink_verification": {
            "convention": convention,
            "counts": counts,
            "counts_are": ("RECOMPUTED here from `per_slug`, never read from "
                           "a dict handed in"),
            "per_slug": per_slug,
            "per_slug_status": " / ".join(CONSUMER_STATUSES),
            "per_slug_status_IS_AUTHORITATIVE_HERE": (
                "the landed 09-06 receipts document FOUR names while the code "
                "counts FIVE and the data in the SAME block uses the fifth "
                "(BOUNDARY_NOT_IN_CAPTURE). THE CODE AND THE DATA ARE "
                "AUTHORITATIVE; the prose was stale against both. This field "
                "is generated from CONSUMER_STATUSES so it cannot drift "
                "again -- a doc string that is typed can go stale, one that "
                "is derived cannot."),
            "n_slugs_verified": len(per_slug),
            "reader_module": {
                "path": "live/pm_research/exp_m6_settlement.py",
                "sha256": reader_module_sha256},
            "gate1_grammar_mapped_into_this_vocabulary": dict(STATUS_MAP),
            "finality": finality,
            "status": ("VERIFICATION_AGREES" if every_agree
                       else "VERIFICATION_DID_NOT_AGREE"),
        },
    }


def receipt_name(day: str, revision: str = DEFAULT_REVISION,
                 stamp: str = "") -> str:
    """EXACTLY what the evaluator globs -- read from its source, not guessed."""
    return RECEIPT_GLOB.format(compact=day.replace("-", ""),
                               revision=revision).replace("*", stamp or "STAMP")


def falsify_receipt() -> int:
    """The receipt cells. Driven against the CONSUMER'S OWN reader."""
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    _ds = {"sha256": "e" * 64, "n_day_records": 2016,
           "definition": "de_combine_day_cells.day_subset_digest"}
    rows = [{"slug": "s%d" % i, "status": st, "checker_up": True,
             "official_up": True if st == LABEL_ADMISSIBLE else False,
             "margin_bp": 1.0, "boundary_age_ms": 0}
            for i, st in enumerate([LABEL_ADMISSIBLE] * 3 + [STALE_BOUNDARY])]
    blk = winner_source_block(rows, winner_source_sha256="a" * 64,
                              winner_source_path="/x/resolutions.jsonl",
                              n_slugs_required=4, reader_module_sha256="b" * 64,
                              day_slice=_ds)
    ck("the block is what _winner_source_blocks looks for (str sha256 + dict chainlink_verification)",
       isinstance(blk.get("sha256"), str) and isinstance(blk.get("chainlink_verification"), dict))
    cv = blk["chainlink_verification"]
    ck("every per_slug status is in the CONSUMER's five",
       set(r["status"] for r in cv["per_slug"].values()) <= set(CONSUMER_STATUSES),
       sorted({r["status"] for r in cv["per_slug"].values()}))
    ck("counts are recomputed and sum to per_slug",
       sum(cv["counts"].values()) == len(cv["per_slug"]), cv["counts"])
    ck("a STALE_BOUNDARY maps to BOUNDARY_NOT_IN_CAPTURE and keeps its measurement",
       cv["per_slug"]["s3"]["status"] == "BOUNDARY_NOT_IN_CAPTURE"
       and cv["per_slug"]["s3"]["gate1_status"] == STALE_BOUNDARY)
    ck("a day with any non-AGREE is NOT final", blk["is_final_for_quotation"] is False)
    allok = [dict(r, status=LABEL_ADMISSIBLE, official_up=True) for r in rows]
    blk2 = winner_source_block(allok, winner_source_sha256="a" * 64,
                               winner_source_path="/x/resolutions.jsonl",
                               n_slugs_required=4, reader_module_sha256="b" * 64,
                               day_slice=_ds)
    ck("an all-AGREE day IS final (a control that can fail)",
       blk2["is_final_for_quotation"] is True)
    ck("...and the consumer's own finality conjunction accepts it",
       blk2["chainlink_verification"]["finality"]["is_final"] is True
       and blk2["chainlink_verification"]["counts"]["VERIFIED_AGREE"] == 4)
    try:
        to_consumer_status("A_STATUS_NOBODY_DECLARED")
        ck("an unmapped status REFUSES rather than reaching unknown_statuses", False)
    except GateRefused:
        ck("an unmapped status REFUSES rather than reaching unknown_statuses", True)
    ck("every gate-1 status has a mapping",
       set(STATUSES) == set(STATUS_MAP), sorted(set(STATUSES) - set(STATUS_MAP)) or "complete")
    ck("the receipt name matches the evaluator's glob shape",
       receipt_name("2026-09-07", "L250ms", "20260911T000000Z")
       == "p003_de_point_estimate_day_20260907_L250ms__20260911T000000Z.json",
       receipt_name("2026-09-07", "L250ms", "20260911T000000Z"))
    print(f"\n  {'RECEIPT CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


# ---- THE DIGEST COUPLING (DE, answering DA 267) -------------------------
#
# THE DISCLOSURE DOES NOT WANT *A* RECEIPT FOR THE DAY. It wants one carrying
# THE WINNER-SOURCE DIGEST THE DAY'S CELLS USED:
#
#   REFUSED FORWARD_EVALUATOR_NO_VERIFIED_WINNER_RECEIPT_FOR_A_DAY: <day> has
#   no verification receipt for the winner-source digest used by its
#   settlement cells
#
# `resolutions.jsonl` IS APPEND-ONLY AND GROWS. A verifier that re-reads it at
# its own time digests something else and the consumer refuses WITH EVERY
# STATUS CORRECT. Measured 2026-09-11T18:40Z: the file digests 86147ce2 now,
# while 09-08's cells name 455b4132, 09-09's name 9b62cebe, and 09-07 carries
# THREE distinct digests -- its two fwd_v2 arms disagree with each other
# (1e9500d0 / 4f0fc39a) because that day predates the read-once fix.
#
# So the digest is NEVER taken from a fresh read of the ledger. It is read
# from the cells, and a receipt carries ONE BLOCK PER DISTINCT DIGEST --
# `_winner_source_blocks` yields all of them and the matcher takes the first
# whose sha256 equals the cells'.
CELL_GLOB = "de_settle_result_{compact}_*.json"
NO_CELL_DIGEST = "NO_WINNER_SOURCE_DIGEST_IN_THE_DAYS_CELLS"


def cells_winner_digests(derived, day: str) -> dict:
    """The winner-source digest(s) THE DAY'S CELLS NAME. Never a fresh read.

    Returns {sha256: [cell filenames]}. More than one key is not an error --
    day one has three -- and the receipt must then carry a block for each.
    """
    import glob as _glob
    c = day.replace("-", "")
    out: dict = {}
    for f in sorted(_glob.glob(str(Path(derived) / "**" /
                                   CELL_GLOB.format(compact=c)),
                               recursive=True)):
        try:
            rec = json.loads(Path(f).read_text())
        except Exception:                           # noqa: BLE001
            continue
        sha = ((rec.get("winner_source") or {}).get("sha256"))
        if isinstance(sha, str) and len(sha) == 64:
            out.setdefault(sha, []).append(Path(f).name)
    if not out:
        raise GateRefused(
            f"REFUSED {NO_CELL_DIGEST}: no cell under {derived} names a "
            f"winner-source sha256 for {day}. The receipt cannot be pinned to "
            f"a digest the cells never recorded, and guessing one is how a "
            f"verification comes to verify a different snapshot than it "
            f"reports.")
    return out


def receipt_for_day(rows_by_digest: dict, *, day, derived, winner_source_path,
                    n_slugs_required, reader_module_sha256, ledger,
                    n_records, revision=DEFAULT_REVISION, stamp=""):
    """ONE receipt. One block per whole-file snapshot the cells name -- but
    the MATCH KEY is the day_slice, identical across all of them.

    THE SLICE IS COMPUTED HERE, NOT ACCEPTED. It used to be a parameter, so a
    caller could hand in a slice from either of two disagreeing definitions and
    the receipt would carry it. The receipt now derives its own identity from
    (ledger, n_records, day) and there is nothing for a caller to get wrong.
    """
    day_slice = head_resolved_day_slice(ledger, n_records, day)
    blocks = [winner_source_block(
        rows, winner_source_sha256=sha, winner_source_path=winner_source_path,
        n_slugs_required=n_slugs_required, day_slice=day_slice,
        reader_module_sha256=reader_module_sha256)
        for sha, rows in sorted(rows_by_digest.items())]
    return {
        "protocol": PROTOCOL,
        "day": day,
        "revision": revision,
        "produced_by": "live/pm_research/da_fair_value_gate1_labels.py",
        "WHY_ONE_BLOCK_PER_DIGEST": (
            "the consumer matches on the winner-source digest the day's CELLS "
            "used, and resolutions.jsonl is append-only -- so a single block "
            "digested at verification time would refuse with every status "
            "correct. Day one's two arms name different digests."),
        "winner_source_blocks": blocks,
        "whole_file_digests_covered": sorted(rows_by_digest),
        "day_slice": {"sha256": day_slice["sha256"],
                      "n_day_records": day_slice["n_day_records"]},
    }, receipt_name(day, revision, stamp)


def falsify_digest_coupling() -> int:
    """The coupling DE named, driven -- including the failure it prevents."""
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "de_settle_result_20260909_CONDVALUE_X_SKEW.json").write_text(
        json.dumps({"winner_source": {"sha256": "9" * 64}}))
    (d / "de_settle_result_20260909_HAZARD_OVER_SKEWED_REF.json").write_text(
        json.dumps({"winner_source": {"sha256": "9" * 64}}))
    got = cells_winner_digests(d, "2026-09-09")
    ck("the digest is READ FROM THE CELLS, not from a fresh ledger read",
       list(got) == ["9" * 64] and len(got["9" * 64]) == 2, list(got)[0][:12])

    (d / "de_settle_result_20260907_CONDVALUE_X_SKEW.json").write_text(
        json.dumps({"winner_source": {"sha256": "1" * 64}}))
    (d / "de_settle_result_20260907_HAZARD_OVER_SKEWED_REF.json").write_text(
        json.dumps({"winner_source": {"sha256": "4" * 64}}))
    two = cells_winner_digests(d, "2026-09-07")
    ck("a day whose two arms DISAGREE yields BOTH digests (day one really does)",
       len(two) == 2, sorted(x[:8] for x in two))

    rows = [{"slug": f"s{i}", "status": LABEL_ADMISSIBLE, "checker_up": True,
             "official_up": True, "margin_bp": 2.0, "boundary_age_ms": 0}
            for i in range(3)]

    ck("the receipt carries ONE BLOCK PER DIGEST", len(rec["winner_source_blocks"]) == 2,
       rec["digests_covered"][0][:8] + " + " + rec["digests_covered"][1][:8])
    ck("and every block is findable by _winner_source_blocks' shape",
       all(isinstance(b.get("sha256"), str)
           and isinstance(b.get("chainlink_verification"), dict)
           for b in rec["winner_source_blocks"]))

    try:
        cells_winner_digests(Path(tempfile.mkdtemp()), "2026-09-13")
        ck("a day whose cells name NO digest REFUSES rather than guessing", False)
    except GateRefused as exc:
        ck("a day whose cells name NO digest REFUSES rather than guessing",
           NO_CELL_DIGEST in str(exc))
    print(f"\n  {'DIGEST-COUPLING CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


# ---- DAY-SLICE ADDRESSING (DE 357 / DA 269 ruling) ----------------------
#
# THE RECEIPT IS DAY-SLICE-ADDRESSED, NOT WHOLE-LEDGER-ADDRESSED.
# `resolutions.jsonl` is append-only and grows -- measured twice within
# minutes on 2026-09-11: 86147ce2 then 90a546bb. A whole-file sha as the match
# key refuses a correct verification for a reason that has nothing to do with
# the day. The DAY SLICE is stable while the file grows, so the slice is the
# identity and the whole-file sha is PROVENANCE beside it, never the key.
#
# *** A SLICE DIGEST IS ONLY AN IDENTITY IF ITS CANONICAL FORM IS DECLARED. ***
# Two honest implementations of "the records for that day" produce different
# digests. The ruling quotes 7eb54006ebfa1029 for 09-07; THIS canonical form
# gives 26f02bdab1814668 over the same 2,016 records. The COUNT is
# convention-free and matches exactly; the DIGEST cannot match a
# canonicalisation nobody wrote down. So the form is declared here, in the
# module, and any consumer comparing digests must use THIS one or say which
# other it used.
#
# THE CANONICAL FORM, normative:
#   * rows are the day's MARKET SLUGS that have a resolution, sorted by slug
#   * each row is  f"{slug}\t{json.dumps(winners, sort_keys=True, separators=(',',':'))}"
#   * rows joined with "\n", encoded UTF-8, sha256 of those bytes
#   * a market with no resolution contributes NO ROW and is counted separately
#: THE DIGEST DEFINITION IS THE REPO'S, NOT A SECOND ONE.
#: `de_combine_day_cells.day_subset_digest(ledger, n_records, day_start,
#: day_end)`: the FIRST n_records lines of the ledger, keep those whose slug's
#: trailing epoch is in [day_start, day_end), join SORTED with "\n", sha256.
#: Sorted lines are what make two snapshots agree regardless of arrival order.
#:
#: DA WROTE A CANONICAL FORM OF ITS OWN FIRST AND THREW IT AWAY. It gave
#: 26f02bdab1814668 for 09-07 where the instrument gives 7eb54006ebfa1029 --
#: the same 2,016 records, a different digest, because a slice digest is only
#: an identity if everyone uses ONE canonicalisation. That is the whole reason
#: this module imports the definition instead of restating it.
SLICE_DIFFERS = "WINNER_SOURCE_DAY_SLICE_DIFFERS"
NO_SLICE = "RECEIPT_CARRIES_NO_DAY_SLICE_DIGEST"
SLICE_DEF_ABSENT = "DAY_SLICE_DEFINITION_NOT_ON_THIS_TREE"
PRE_RULING_SLICE = "PRE_RULING_SLICE_DEFINITION_IS_NOT_THE_IDENTITY"


def day_slice_digest(*_a, **_k) -> dict:
    """WITHDRAWN. It computed the PRE-RULING slice and called itself THE identity.

    IT IS THE DEFECT I WITHDREW THIS MORNING, RECURRING INSIDE ONE FILE. This
    module briefly held TWO slice functions that disagreed on real data --
    7eb54006ebfa1029 here against 43d6273ba7848627 from
    `head_resolved_day_slice` over the SAME 2,016 records -- while
    `receipt_for_day` took the slice as a CALLER-SUPPLIED parameter, so which
    one reached a receipt depended entirely on the caller. DE measured it
    against the landed blob.

    IT REFUSES RATHER THAN DELEGATING. Silently redirecting would hand a
    caller that asked for the old number a DIFFERENT number without telling
    it, which is how a digest mismatch comes to look like a data problem.
    """
    raise GateRefused(
        f"REFUSED {PRE_RULING_SLICE}: `day_slice_digest` computed the "
        f"PRE-RULING slice (line-sorted, no head resolution) and is withdrawn. "
        f"The identity is `head_resolved_day_slice` -- supersession first, "
        f"then sorted BY SLUG. The two disagree on real data: 7eb54006ebfa1029 "
        f"against 43d6273ba7848627 over the same 2,016 records of 2026-09-07.")


def compare_day_slice(mine: dict, theirs: dict) -> dict:
    """SAME SLICE -> clean whatever the whole-file shas say. DIFFERENT -> named."""
    if not theirs or not theirs.get("sha256"):
        return {"day_slice_matches": False, "status": NO_SLICE,
                "reading": (f"REFUSED {NO_SLICE}: the receipt carries no "
                            f"`day_slice.sha256`. It is refused by name rather "
                            f"than tolerated under a second spelling.")}
    same = mine["sha256"] == theirs["sha256"]
    out = {"day_slice_matches": same,
           "mine": {"sha256": mine["sha256"], "n_day_records": mine["n_day_records"]},
           "theirs": {"sha256": theirs["sha256"],
                      "n_day_records": theirs.get("n_day_records")},
           "n_day_records_delta": (mine["n_day_records"] - theirs["n_day_records"])
           if isinstance(theirs.get("n_day_records"), int) else None,
           "status": None if same else SLICE_DIFFERS}
    if not same:
        out["reading"] = (
            f"REFUSED {SLICE_DIFFERS}: mine {mine['sha256'][:16]} over "
            f"{mine['n_day_records']} day record(s), theirs "
            f"{theirs['sha256'][:16]} over {theirs.get('n_day_records')}. A "
            f"REAL difference inside the day, not ledger growth -- named, both "
            f"digests carried, and lifted by a superseding receipt, never a "
            f"permanent block.")
    return out


def falsify_day_slice() -> int:
    """BOTH WAYS, ON THE REAL LEDGER -- rebuilt to the RULED shape.

    DE's warning, taken: a fixture written against the pre-ruling contract
    goes on passing while production refuses. These cells are built to the
    ruled shape and checked against DE's own published numbers.
    """
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    LED = Path("/home/yuqing/ctaNew/data/pm_5min/resolutions.jsonl")
    if not LED.is_file():
        print("  [SKIP] ledger absent")
        return 0
    try:
        head_resolved_day_slice(LED, 45877, "2026-09-07")
    except GateRefused as exc:
        if SLICE_DEF_ABSENT in str(exc):
            print(f"  [REFUSED BY NAME] {SLICE_DEF_ABSENT} -- this tree lacks "
                  f"de_combine_day_cells; run from a tree that has it.")
            return 0
        raise

    # (a) THE SAME DAY UNDER THREE DIFFERENT WHOLE-FILE SNAPSHOTS.
    #     09-07's cells recorded three: 45877 / 45932 / 46290 records.
    got = [head_resolved_day_slice(LED, n, "2026-09-07") for n in (45877, 45932, 46290)]
    ck("growth OUTSIDE the day: THREE whole-file snapshots, ONE slice",
       len({g["sha256"] for g in got}) == 1 and len({g["n_day_records"] for g in got}) == 1,
       f"{got[0]['sha256'][:16]} / {got[0]['n_day_records']} records, from "
       f"{[g['n_records_in_snapshot'] for g in got]}")
    ck("...and it is the RULED number for 09-07 (not the pre-ruling 7eb54006)",
       got[0]["sha256"].startswith("43d6273ba7848627") and got[0]["n_day_records"] == 2016,
       got[0]["sha256"][:16])
    ck("...and compare_day_slice calls all three clean",
       all(compare_day_slice(got[0], g)["day_slice_matches"] for g in got))

    # (b) 09-09, DE's other published number.
    d9 = head_resolved_day_slice(LED, 46521, "2026-09-09")
    ck("09-09 is the RULED slice, head-resolved from 2,030 lines to 2,016 records",
       d9["sha256"].startswith("235490f17dcb5d2a") and d9["n_day_records"] == 2016,
       f"{d9['sha256'][:16]} / {d9['n_day_records']}")

    # (c) A DIFFERENT DAY is a different slice -- the check is on content.
    cmp = compare_day_slice(d9, got[0])
    ck("a genuinely different slice is DETECTED and NAMED",
       cmp["status"] == SLICE_DIFFERS and cmp["mine"]["sha256"] != cmp["theirs"]["sha256"],
       cmp["reading"][:66] + "...")
    ck("...carrying BOTH digests and the day-record delta",
       cmp["n_day_records_delta"] == d9["n_day_records"] - got[0]["n_day_records"],
       f"delta={cmp['n_day_records_delta']}")

    # (d) A RECEIPT WITH NO day_slice IS REFUSED BY NAME, not tolerated.
    ck("a receipt carrying no day_slice is REFUSED by name",
       compare_day_slice(d9, {})["status"] == NO_SLICE)
    try:
        winner_source_block([], winner_source_sha256="a" * 64,
                            winner_source_path="/x", n_slugs_required=0,
                            reader_module_sha256="b" * 64)
        ck("...and a block cannot be built without one", False)
    except GateRefused as exc:
        ck("...and a block cannot be built without one", NO_SLICE in str(exc))

    # (e) THE BLOCK CARRIES THE RULED FIELD NAMES, no alternates.
    rows = [{"slug": "s0", "status": LABEL_ADMISSIBLE, "checker_up": True,
             "official_up": True, "margin_bp": 2.0, "boundary_age_ms": 0}]
    blk = winner_source_block(rows, winner_source_sha256="f" * 64,
                              winner_source_path="/x/resolutions.jsonl",
                              n_slugs_required=1, reader_module_sha256="b" * 64,
                              day_slice=d9)
    ck("the block carries winner_source.day_slice.{sha256, n_day_records}",
       set(blk["day_slice"]) == {"sha256", "n_day_records"}, sorted(blk["day_slice"]))
    ck("...and winner_source.sha256 is labelled PROVENANCE ONLY",
       "PROVENANCE" in blk["sha256_is"])
    ck("a receipt whose WHOLE-FILE sha is a stranger but whose SLICE matches verifies clean",
       blk["sha256"] == "f" * 64
       and compare_day_slice(d9, blk["day_slice"])["day_slice_matches"] is True)
    print(f"\n  {'DAY-SLICE CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


# ---- HEAD-RESOLVED DAY SLICE (DA 270 ruling) ----------------------------
#
# "SORTING IS NOT ORDERING UNTIL THE KEY IS UNIQUE." The slice is built from
# HEAD-RESOLVED records: the §5 gate-1 supersession rule FIRST, so each slug
# appears exactly once, THEN sort by slug. Out-of-order appends are then
# irrelevant, because order is derived from the resolved key and never from
# file position.
#
# THIS IS NOT HYPOTHETICAL AND IT CHANGES A PUBLISHED NUMBER. Measured on the
# real ledger 2026-09-11: 09-09 holds 2,030 lines for the day but only 2,016
# DISTINCT slugs -- 14 slugs appear twice, each as a `gave_up` stub
# (`closed: null`, `winners: null`) followed at a LATER recv_ns by the real
# resolution. The unresolved slice therefore counts a GIVE-UP BESIDE ITS OWN
# SUPERSESSION, which is not the day's settled state. 09-07 has no duplicates,
# so its digest is unchanged.
#
# THE HEAD RULE: greatest `recv_ns` wins -- the ledger is append-only and the
# later record supersedes. A TIE is not broken by position; it REFUSES BY SLUG.
DUPES_COUNTED = "PRE_RESOLUTION_DUPLICATE_SLUGS"
TWO_HEADS = "SLUG_HAS_TWO_HEADS_AFTER_RESOLUTION"


def head_resolved_day_slice(ledger, n_records: int, day: str) -> dict:
    """The day's HEAD-RESOLVED slice. One row per slug, then sorted."""
    import hashlib
    from collections import defaultdict
    from datetime import datetime, timezone
    start = int(datetime.strptime(day, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp())
    end = start + 86400
    lines = [x for x in Path(ledger).read_text().splitlines() if x.strip()]
    if len(lines) < n_records:
        raise GateRefused(
            f"REFUSED {SLICE_DIFFERS}: the ledger holds {len(lines)} records "
            f"and a snapshot claimed {n_records}; a prefix that does not "
            f"exist cannot be compared.")
    by = defaultdict(list)
    for line in lines[:n_records]:
        try:
            rec = json.loads(line)
        except Exception:                           # noqa: BLE001
            continue
        slug = str(rec.get("slug") or "")
        try:
            t = int(slug.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if start <= t < end:
            by[slug].append((rec.get("recv_ns"), line))
    heads, dupes, two_heads = {}, {}, []
    for slug, rows in by.items():
        if len(rows) > 1:
            dupes[slug] = len(rows)
        top = max(r[0] for r in rows if r[0] is not None) if any(
            r[0] is not None for r in rows) else None
        if top is None:
            heads[slug] = sorted(r[1] for r in rows)[0]
            continue
        winners = [r[1] for r in rows if r[0] == top]
        if len(set(winners)) > 1:
            two_heads.append(slug)
            continue
        heads[slug] = winners[0]
    if two_heads:
        raise GateRefused(
            f"REFUSED {TWO_HEADS}: {len(two_heads)} slug(s) resolve to two "
            f"different heads at the same recv_ns: {sorted(two_heads)[:5]}. "
            f"Picking one by file position is how an ordering that is not an "
            f"identity gets used as one.")
    rows = [heads[s] for s in sorted(heads)]
    return {"day": day, "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
            "n_day_records": len(rows),
            "n_records_in_snapshot": n_records,
            "n_lines_before_resolution": sum(len(v) for v in by.values()),
            DUPES_COUNTED: {"n_slugs": len(dupes), "slugs": sorted(dupes)[:20],
                            "counted_never_silently_deduped": True},
            "definition": ("head-resolved: greatest recv_ns per slug, then "
                           "sorted by slug; ties REFUSE by slug name")}


def falsify_head_resolution() -> int:
    """Shuffle+re-append -> BIT-IDENTICAL. Re-settlement -> changes, named."""
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    import tempfile, random
    base = []
    for i in range(6):
        base.append(json.dumps({"slug": f"btc-updown-5m-{1788980100 + i * 300}",
                                "recv_ns": 1000 + i, "closed": True,
                                "winners": {"Up": bool(i % 2)}}))
    d = Path(tempfile.mkdtemp()); led = d / "r.jsonl"
    led.write_text("\n".join(base) + "\n")
    day = "2026-09-09"
    a = head_resolved_day_slice(led, len(base), day)
    ck("the head-resolved slice has one row per slug", a["n_day_records"] == 6, a["sha256"][:16])

    shuf = list(base); random.shuffle(shuf)
    led.write_text("\n".join(shuf) + "\n")
    b = head_resolved_day_slice(led, len(shuf), day)
    ck("SHUFFLED file order -> BIT-IDENTICAL slice digest",
       b["sha256"] == a["sha256"], b["sha256"][:16])

    reapp = shuf + [shuf[0]]
    led.write_text("\n".join(reapp) + "\n")
    c = head_resolved_day_slice(led, len(reapp), day)
    ck("RE-APPENDING the same record -> BIT-IDENTICAL, and the duplicate is COUNTED",
       c["sha256"] == a["sha256"] and c[DUPES_COUNTED]["n_slugs"] == 1,
       f"dupes={c[DUPES_COUNTED]['n_slugs']} lines_before={c['n_lines_before_resolution']}")

    resettle = list(base)
    resettle.append(json.dumps({"slug": "btc-updown-5m-1788980100",
                                "recv_ns": 999999, "closed": True,
                                "winners": {"Up": True}}))
    led.write_text("\n".join(resettle) + "\n")
    e = head_resolved_day_slice(led, len(resettle), day)
    ck("a GENUINE RE-SETTLEMENT CHANGES the digest at the SAME row count",
       e["sha256"] != a["sha256"] and e["n_day_records"] == 6, e["sha256"][:16])
    ck("...and the superseded slug is NAMED in the counted duplicates",
       "btc-updown-5m-1788980100" in e[DUPES_COUNTED]["slugs"])

    tie = list(base) + [json.dumps({"slug": "btc-updown-5m-1788980100",
                                    "recv_ns": 1000, "closed": True,
                                    "winners": {"Up": True}})]
    led.write_text("\n".join(tie) + "\n")
    try:
        head_resolved_day_slice(led, len(tie), day)
        ck("two heads at the same recv_ns REFUSE by slug name", False)
    except GateRefused as exc:
        ck("two heads at the same recv_ns REFUSE by slug name",
           TWO_HEADS in str(exc) and "btc-updown-5m-1788980100" in str(exc))

    try:
        day_slice_digest("x", 1, "2026-09-07")
        ck("the PRE-RULING slice function REFUSES rather than returning a number", False)
    except GateRefused as exc:
        ck("the PRE-RULING slice function REFUSES by name, never delegates silently",
           PRE_RULING_SLICE in str(exc))
    import inspect as _i
    ck("receipt_for_day COMPUTES its slice -- a caller cannot supply one",
       "day_slice" not in _i.signature(receipt_for_day).parameters,
       sorted(_i.signature(receipt_for_day).parameters))
    LED = Path("/home/yuqing/ctaNew/data/pm_5min/resolutions.jsonl")
    if LED.is_file():
        r7 = head_resolved_day_slice(LED, 45877, "2026-09-07")
        # THE RULING CHANGES 09-07 TOO, AND NOT FOR THE REASON I FIRST
        # ASSERTED. I wrote this cell expecting 7eb54006 to survive because
        # 09-07 has no duplicates. IT DOES NOT SURVIVE: the landed instrument
        # sorts LINES, and a line begins `{"recv_ns":...`, so line order IS
        # recv_ns order, not slug order. The ruling orders by the RESOLVED
        # KEY. Same 2,016 records, different ordering, different digest --
        # measured, not argued.
        ck("09-07 has NO duplicates -- the row count is unchanged",
           r7["n_day_records"] == 2016 and r7[DUPES_COUNTED]["n_slugs"] == 0,
           f"{r7['n_day_records']} rows, {r7[DUPES_COUNTED]['n_slugs']} dupes")
        ck("...but the DIGEST CHANGES ANYWAY, because the ORDERING KEY changed",
           not r7["sha256"].startswith("7eb54006"),
           f"{r7['sha256'][:16]} (line-sorted was 7eb54006ebfa1029)")
        r9 = head_resolved_day_slice(LED, 46521, "2026-09-09")
        ck("09-09 HAS 14 duplicates, so head-resolution CHANGES it: 2030 -> 2016",
           r9["n_day_records"] == 2016 and r9[DUPES_COUNTED]["n_slugs"] == 14
           and not r9["sha256"].startswith("36da8727"),
           f"{r9['sha256'][:16]} / {r9['n_day_records']} (was 36da8727d02cfe3d / 2030)")
    print(f"\n  {'HEAD-RESOLUTION CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    import sys
    n = falsify()
    print()
    n += falsify_receipt()
    print()
    print()
    n += falsify_day_slice()
    print()
    n += falsify_head_resolution()
    sys.exit(1 if n else 0)
