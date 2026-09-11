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


if __name__ == "__main__":
    import sys
    sys.exit(1 if falsify() else 0)
