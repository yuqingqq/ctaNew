"""P-2026-003 RACE-READ VERIFIER -- DA's independent MATCHED_VOLUME.

R-235 (do-not-harmonize). This module RE-IMPLEMENTS the interim's primary
statistic from its declarations and imports NOTHING of BE's computation:
not `be_read_cells.matched_volume`, not `be_forward_metric`'s primitives,
not `be_race_reader`'s day statistic. Those were read as DOCUMENTS. Two
implementations that agree are evidence; one checking itself is not.

THE STATISTIC, and where each clause was read.

  MATCHED_VOLUME     `be_read_cells.matched_volume`, the interim declaration's
                     PRIMARY: the candidate acts at its FROZEN theta; the
                     incumbent is then given the candidate's REALISED COUNT by
                     lowering its own cutoff; the increment is
                     `candidate_net_cents - incumbent_net_cents_matched`.
                     Both arms spend the same cancellation budget, so what is
                     left is ranking QUALITY rather than volume.
  action             `be_forward_metric.ACTION_KEY` = (slug, side, gen); a
                     generation's rows are ordered by `t_start`.
  arm ranking        a generation's score is the MAX over its rows.
  candidate select   every generation whose max candidate score >= theta.
  matched select     top-k of the incumbent's OWN ranking at k = the
                     candidate's realised count, tie-broken on the key
                     `(-gmax, key)` (R-234), the cutoff being the k-th
                     generation's max.
  valuation          FIRST CROSSING ACTS: within a chosen generation the
                     acting row is the earliest by `t_start` whose score
                     reaches that generation's cutoff; its value is
                     `latency[str(L)]["preventable_value_cents"]`, or 0.0
                     when the row carries no fill ahead.
  window bucket      `gk[0]` -- the SLUG. Stated because "per window" reads
                     like a time bucket and it is not one.
  theta              `be_operating_point_declaration_v1.theta_frozen_by_coin`
                     at the 10% budget, per coin, READ never typed.
  latency            L = 50 ms, `be_race_reader.LATENCY_MS`.
  coin               `slug.split("-", 1)[0]`, as the interim's loader does.

WHAT THIS MODULE REFUSES TO DO. The five pinned real feeds
(`declarations/be_race_read_feed_pins_v1.json`) are NOT opened. The race read
is the coordinator's act on GO; before GO this verifier refuses on the GATE,
and after GO it still refuses because the real path is declared and NOT
BUILT -- two different refusals, because a control that refused identically
on both sides of its own predicate would prove nothing.

    python3 live/pm_research/da_race_read_verify.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_RACE_READ_VERIFIER_V1"

#: `be_race_reader.LATENCY_MS`, read as a document.
LATENCY_MS = 50

#: `be_forward_metric.ACTION_KEY`.
ACTION_KEY = ("slug", "side", "gen")

#: `be_forward_day.FEED_FIELDS` -- the shape a feed row must have.
FEED_FIELDS = ("slug", "side", "gen", "t0", "t_start", "score",
               "score_incumbent", "any_fill_ahead", "value_cents",
               "preventable_shares", "level")

OP_DECL = HERE / "declarations" / "be_operating_point_declaration_v1.json"
PINS_DECL = HERE / "declarations" / "be_race_read_feed_pins_v1.json"

#: The comparison is EXACT and the reason is in the definition, not in a
#: preference: every term is a finite sum of the SAME float64 values selected
#: by the SAME rule, so two correct implementations differ only if one of
#: them is wrong. `math.fsum` is used on both sides for the arm totals, which
#: is what BE's own code uses, so even the summation order cannot separate
#: them.
TOLERANCE = {
    "rule": "EXACT (==) on every compared quantity",
    "why_exact": (
        "MATCHED_VOLUME is a difference of two `math.fsum` reductions over "
        "values selected by a deterministic rule from the same rows. fsum is "
        "exactly rounded, so summation ORDER cannot separate two correct "
        "implementations, and a tolerance would only hide a selection "
        "difference -- which is the defect worth catching."),
    "where_it_could_NOT_be_exact": (
        "nowhere on this statistic. If a future variant averages over a "
        "sampled null the equality would have to become a seeded "
        "reproduction, and that is a different claim -- said here so the "
        "exactness is not read as a general licence."),
}


class RaceVerifyRefused(RuntimeError):
    """The verification cannot proceed honestly on the inputs given."""


def verifier_identity() -> dict:
    src = Path(__file__).resolve()
    r = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    d = subprocess.run(["git", "status", "--porcelain", "--", str(src)],
                       capture_output=True, text=True, cwd=str(src.parent))
    h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(src.parent))
    return {"path": "live/pm_research/da_race_read_verify.py",
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "commit_best_effort": (r.stdout.strip() or None),
            "tree_head": (h.stdout.strip() or None),
            "producing_code_is_the_committed_bytes":
                d.returncode == 0 and d.stdout.strip() == ""}


def theta_for(coin: str, budget_label: str = "10%") -> float:
    """READ from the operating-point declaration, never typed."""
    d = json.loads(OP_DECL.read_text())
    t = d.get("theta_frozen_by_coin", {}).get(coin, {}).get(budget_label)
    if t is None:
        raise RaceVerifyRefused(
            f"REFUSED: no frozen theta for {coin} at {budget_label}. A "
            f"missing operating point is not a theta of zero.")
    return float(t)


# ------------------------------------------------------- the re-implementation

def da_load_two_arm_feed(path: Path, latency_ms: int) -> dict:
    """Per-coin (rows, cand, inc). REFUSES a one-arm feed BY NAME.

    The failure this prevents is the quiet one: with the incumbent column
    missing a caller could pass the candidate twice and get an increment of
    exactly zero that looks like a result."""
    L = str(latency_ms)
    per: dict = {}
    n, missing = 0, 0
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fr = json.loads(line)
            n += 1
            if "score_incumbent" not in fr:
                raise RaceVerifyRefused(
                    f"REFUSED: feed row {n} carries no `score_incumbent`. "
                    f"This is a ONE-ARM feed and the declared estimand is an "
                    f"increment OVER the incumbent; computing it from one "
                    f"arm would compare the candidate with itself and return "
                    f"a zero that looks like a measurement.")
            si = fr["score_incumbent"]
            if si is None:
                missing += 1
                continue
            coin = fr["slug"].split("-", 1)[0]
            b = per.setdefault(coin, {"rows": [], "cand": [], "inc": []})
            b["rows"].append({
                "slug": fr["slug"], "side": fr["side"], "gen": fr["gen"],
                "t0": fr["t0"], "t_start": fr["t_start"],
                "any_fill_ahead": bool(fr.get("any_fill_ahead", True)),
                "latency": {L: {"preventable_value_cents":
                                fr["value_cents"]}}})
            b["cand"].append(float(fr["score"]))
            b["inc"].append(float(si))
    if not per:
        raise RaceVerifyRefused(
            f"REFUSED: no scored rows in {path}. An empty read is a FAILURE, "
            f"not an empty result.")
    return {"per_coin": per, "n_feed_rows": n,
            "n_rows_without_an_incumbent_score": missing}


def da_gen_index(rows) -> dict:
    gens: dict = {}
    for i, r in enumerate(rows):
        gens.setdefault(tuple(r[k] for k in ACTION_KEY), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: rows[i]["t_start"])
    return gens


def da_select_by_threshold(gens, scores, theta: float):
    gmax = {k: max(scores[i] for i in gens[k]) for k in gens}
    chosen = [k for k in gens if gmax[k] >= theta]
    return chosen, {k: theta for k in chosen}, theta


def da_select_by_exact_count(gens, scores, k_target: int):
    """Lower this arm's own cutoff until it acts k_target times.

    The tie-break is on the KEY (R-234): under equal maxima a bare `-gmax`
    sort falls back to dict order, which is a different selection wearing
    the same name."""
    gmax = {k: max(scores[i] for i in gens[k]) for k in gens}
    order = sorted(gens, key=lambda k: (-gmax[k], k))
    k = max(0, min(int(k_target), len(order)))
    chosen = order[:k]
    cut = gmax[chosen[-1]] if chosen else float("inf")
    return chosen, {kk: cut for kk in chosen}, cut


def da_cancel_value(rows, gens, scores, chosen, theta_map,
                    latency_ms) -> dict:
    """Per-WINDOW (slug) value of cancelling `chosen`. FIRST CROSSING ACTS."""
    L = str(latency_ms)
    bw: dict = {}
    for gk in chosen:
        i = next((j for j in gens[gk] if scores[j] >= theta_map[gk]), None)
        if i is None:
            continue
        r = rows[i]
        v = (r["latency"][L]["preventable_value_cents"]
             if r.get("any_fill_ahead") and "latency" in r else 0.0)
        bw[gk[0]] = bw.get(gk[0], 0.0) + v
    return bw


def da_matched_volume(rows, cand, inc, theta: float, latency_ms: int) -> dict:
    """The interim's PRIMARY, re-derived."""
    gens = da_gen_index(rows)
    c_sel, c_th, _ = da_select_by_threshold(gens, cand, theta)
    m_sel, m_th, m_cut = da_select_by_exact_count(gens, inc, len(c_sel))
    cb = da_cancel_value(rows, gens, cand, c_sel, c_th, latency_ms)
    mb = da_cancel_value(rows, gens, inc, m_sel, m_th, latency_ms)
    cand_net = math.fsum(cb.values())
    inc_net_matched = math.fsum(mb.values())
    wins = sorted(set(cb) | set(mb))
    return {
        "statistic": "MATCHED_VOLUME",
        "theta_declared": theta,
        "MATCHED_VOLUME_increment_cents": cand_net - inc_net_matched,
        "candidate_net_cents": cand_net,
        "incumbent_net_cents_matched": inc_net_matched,
        "candidate_n_cancelled": len(c_sel),
        "incumbent_n_cancelled_matched": len(m_sel),
        "counts_matched": len(m_sel) == len(c_sel),
        "incumbent_cutoff_matched": m_cut,
        "n_actions": len(gens),
        "increment_by_window": {w: cb.get(w, 0.0) - mb.get(w, 0.0)
                                for w in wins},
        "window_bucket_is": "the SLUG (gk[0]), not a time bucket",
    }


def da_day_matched_volume(path, *, latency_ms: int = LATENCY_MS) -> dict:
    """The day statistic: per coin at its frozen theta, summed."""
    feed = da_load_two_arm_feed(Path(path), latency_ms)
    per_coin, net = {}, 0.0
    for coin, blk in sorted(feed["per_coin"].items()):
        mv = da_matched_volume(blk["rows"], blk["cand"], blk["inc"],
                               theta_for(coin), latency_ms)
        per_coin[coin] = {
            "MATCHED_VOLUME_increment_cents":
                mv["MATCHED_VOLUME_increment_cents"],
            "candidate_net_cents": mv["candidate_net_cents"],
            "incumbent_net_cents_matched": mv["incumbent_net_cents_matched"],
            "counts_matched": mv["counts_matched"],
            "n_actions": mv["n_actions"]}
        net += float(mv["MATCHED_VOLUME_increment_cents"])
    if not per_coin:
        raise RaceVerifyRefused(
            f"REFUSED: {Path(path).name} yielded no coin with rows; a day "
            f"with no action is a STATUS, not a zero increment.")
    return {"status": "OK", "per_coin": per_coin,
            "day_increment_cents": net,
            "day_sign": (1 if net > 0 else (-1 if net < 0 else 0)),
            "n_feed_rows": feed["n_feed_rows"],
            "n_rows_without_an_incumbent_score":
                feed["n_rows_without_an_incumbent_score"],
            "latency_ms": latency_ms,
            "computed_by": "da_race_read_verify -- a SEPARATE "
                           "implementation (R-235), not BE's code"}


def compare_day(mine: dict, theirs: dict) -> dict:
    """EXACT, field by field. A near-miss is a mismatch."""
    checks, bad = [], []

    def cmp(name, a, b):
        ok = (float(a) == float(b)) if isinstance(a, (int, float)) \
            else (a == b)
        checks.append({"field": name, "state": "MATCH" if ok else "MISMATCH",
                       "mine": a, "theirs": b})
        if not ok:
            bad.append(name)

    cmp("day_increment_cents", mine["day_increment_cents"],
        theirs["day_increment_cents"])
    cmp("day_sign", mine["day_sign"], theirs["day_sign"])
    cmp("n_feed_rows", mine["n_feed_rows"], theirs["n_feed_rows"])
    coins = sorted(set(mine["per_coin"]) | set(theirs["per_coin"]))
    for c in coins:
        a, b = mine["per_coin"].get(c), theirs["per_coin"].get(c)
        if a is None or b is None:
            checks.append({"field": f"per_coin.{c}",
                           "state": "PRESENT_ON_ONE_SIDE_ONLY"})
            bad.append(f"per_coin.{c}")
            continue
        for f in ("MATCHED_VOLUME_increment_cents", "candidate_net_cents",
                  "incumbent_net_cents_matched", "n_actions",
                  "counts_matched"):
            cmp(f"per_coin.{c}.{f}", a[f], b[f])
    return {"n_compared": len(checks), "n_mismatches": len(bad),
            "mismatched_fields": bad, "checks": checks,
            "verdict": "AGREES" if not bad else "FLAGGED",
            "tolerance": TOLERANCE["rule"]}


# ------------------------------------------------------------- the race gate

def pinned_feeds() -> dict:
    """The five pinned real paths, READ from the pins declaration."""
    d = json.loads(PINS_DECL.read_text())
    per = d["per_day"]
    return {"days": sorted(per),
            "n_days": len(per),
            "n_present": sum(1 for v in per.values() if v.get("exists")),
            "absent_days": sorted(k for k, v in per.items()
                                  if not v.get("exists")),
            "all_five_present": bool(d.get("all_five_present")),
            "pins_sha256": hashlib.sha256(PINS_DECL.read_bytes()).hexdigest(),
            "the_read_voids_on_mismatch": d.get("the_read_voids_on_mismatch")}


def verify_pinned_feeds(*, go: bool = False) -> dict:
    """The five pinned real paths. REFUSES before GO, and refuses AFTER GO
    for a DIFFERENT reason -- the real path is declared and not built."""
    pins = pinned_feeds()
    if not go:
        raise RaceVerifyRefused(
            f"REFUSED BEFORE GO: the race read is the coordinator's act; "
            f"the {pins['n_days']} pinned feeds are not opened by this "
            f"verifier. Opening one to 'check the checker' would spend the "
            f"read the gate exists to schedule.")
    raise RaceVerifyRefused(
        f"REFUSED: declared and NOT OPENED. The real-feed path is specified "
        f"and not built in this round, and the pins themselves record only "
        f"{pins['n_present']} of {pins['n_days']} feeds present "
        f"(absent: {pins['absent_days']}).")


# ---------------------------------------------------------------- the fixture

def _row(slug, side, gen, t_start, score, inc, cents, *, fill=True):
    return {"slug": slug, "side": side, "gen": gen, "t0": 0.0,
            "t_start": float(t_start), "score": float(score),
            "score_incumbent": (None if inc is None else float(inc)),
            "any_fill_ahead": bool(fill), "value_cents": float(cents),
            "preventable_shares": 1.0, "level": 0.5}


def write_feed(d: Path, rows, *, one_arm: bool = False) -> Path:
    p = d / "feed.jsonl"
    with p.open("w") as fh:
        for r in rows:
            r = dict(r)
            if one_arm:
                r.pop("score_incumbent", None)
            fh.write(json.dumps(r) + "\n")
    return p


def selftest() -> tuple:                                      # noqa: C901
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    td = Path(tempfile.mkdtemp(prefix="da65_"))
    th_btc, th_eth = theta_for("btc"), theta_for("eth")

    # -- 1. theta is READ, and a missing coin refuses ---------------------
    missing = False
    try:
        theta_for("nosuchcoin")
    except RaceVerifyRefused:
        missing = True
    ck("THETA IS READ FROM THE OPERATING-POINT DECLARATION, NEVER TYPED -- "
       "and a coin with no frozen theta REFUSES rather than defaulting to "
       "zero, which would cancel everything and call it a policy",
       th_btc == 0.7230267681941027 and th_eth == 0.10284589538586741
       and missing,
       f"btc {th_btc}, eth {th_eth} at the 10% budget, read from "
       f"{OP_DECL.name}; an unknown coin raises")

    # -- 2. THE HEADLINE: a known increment BY CONSTRUCTION ---------------
    #: One row per generation, so first-crossing is that row and the whole
    #: statistic is arithmetic I can do on paper.
    btc = [_row(f"btc-updown-5m-W{g}", "UP", g, 1000 + g,
                0.9 if g < 3 else 0.1,        # candidate: gens 0,1,2 act
                0.5 - 0.01 * (9 - g),         # incumbent: gen 9 ranks highest
                (g + 1) * 10.0) for g in range(10)]
    eth = [_row(f"eth-updown-5m-W{g}", "UP", g, 2000 + g,
                0.5 if g < 2 else 0.01,       # candidate: gens 0,1 act
                0.2 - 0.01 * (5 - g),         # incumbent: gen 5 ranks highest
                (g + 1) * 5.0) for g in range(6)]
    feed = write_feed(td, btc + eth)
    mine = da_day_matched_volume(feed)
    #: BY HAND. btc: candidate acts on {0,1,2} worth 10+20+30 = 60; the
    #: incumbent matched at 3 takes its own top three {9,8,7} worth
    #: 100+90+80 = 270; increment 60 - 270 = -210.
    #: eth: candidate {0,1} worth 5+10 = 15; incumbent top two {5,4} worth
    #: 30+25 = 55; increment 15 - 55 = -40. Day = -250.
    ck("A KNOWN INCREMENT BY CONSTRUCTION REPRODUCES EXACTLY: the candidate "
       "acts at its frozen theta, the incumbent is matched to that COUNT on "
       "its own ranking, and the day increment is the hand arithmetic",
       mine["per_coin"]["btc"]["MATCHED_VOLUME_increment_cents"] == -210.0
       and mine["per_coin"]["eth"]["MATCHED_VOLUME_increment_cents"] == -40.0
       and mine["day_increment_cents"] == -250.0
       and mine["day_sign"] == -1,
       f"btc 60 - 270 = "
       f"{mine['per_coin']['btc']['MATCHED_VOLUME_increment_cents']}; "
       f"eth 15 - 55 = "
       f"{mine['per_coin']['eth']['MATCHED_VOLUME_increment_cents']}; day "
       f"{mine['day_increment_cents']} -- equal, not within a tolerance")

    # -- 3. AGAINST BE's OWN READER, EXACT --------------------------------
    import be_race_reader as RR
    theirs = RR.day_matched_volume(feed)
    cmp_ok = compare_day(mine, theirs)
    ck("TWO SEPARATE IMPLEMENTATIONS AGREE ON EVERY FIELD, EXACTLY (R-235): "
       "mine re-derived from the declarations, BE's through "
       "`be_read_cells.matched_volume` -- and agreement is only evidence "
       "because neither imported the other's statistic",
       cmp_ok["n_mismatches"] == 0 and cmp_ok["verdict"] == "AGREES"
       and cmp_ok["n_compared"] >= 13,
       f"{cmp_ok['n_compared']} fields compared EXACTLY, 0 mismatches; day "
       f"{mine['day_increment_cents']} == {theirs['day_increment_cents']}")

    # -- 4. ONE MOVED ROW FLAGS -- and WHICH row is the point -------------
    #: My first attempt moved row 4, which the candidate does not act on
    #: (score 0.1 < theta) and which the matched incumbent does not reach
    #: either (its rank is 5th of a top-3). NOTHING MOVED, and the check
    #: FAILED -- correctly. A statistic that changed there would be reading
    #: rows outside both selections. So the known-bad now moves a row the
    #: candidate ACTS on, and the row outside both selections is kept as the
    #: control for the property it actually demonstrates.
    (td / "m").mkdir(exist_ok=True)
    moved = [dict(r) for r in btc + eth]
    moved[0]["value_cents"] = float(moved[0]["value_cents"]) + 1e-9
    feed_m = write_feed(td / "m", moved)
    mine_m = da_day_matched_volume(feed_m)
    cmp_bad = compare_day(mine_m, theirs)
    ck("KNOWN-BAD: ONE MOVED ROW FLAGS. A row the candidate ACTS on, shifted "
       "by 1e-9 cents, changes the day increment and the comparison FLAGS -- "
       "the comparison is exact, so a near-miss is a mismatch, not a pass",
       cmp_bad["n_mismatches"] > 0 and cmp_bad["verdict"] == "FLAGGED"
       and "day_increment_cents" in cmp_bad["mismatched_fields"],
       f"row 0 (in the candidate's selection, +1e-9 cents) -> day "
       f"{mine_m['day_increment_cents']!r} against "
       f"{theirs['day_increment_cents']!r}; flagged "
       f"{cmp_bad['mismatched_fields'][:3]}")

    (td / "m2").mkdir(exist_ok=True)
    untouched = [dict(r) for r in btc + eth]
    untouched[4]["value_cents"] = float(untouched[4]["value_cents"]) + 1e6
    mine_u = da_day_matched_volume(write_feed(td / "m2", untouched))
    cmp_u = compare_day(mine_u, theirs)
    ck("AND THE OTHER DIRECTION, WHICH MY OWN FAILING TEST TAUGHT ME: a row "
       "in NEITHER selection moved by a MILLION cents does NOT move the "
       "statistic. That is the estimand behaving -- MATCHED_VOLUME depends "
       "only on the actions the two arms take -- and it is why the "
       "known-bad above had to move a row that is actually acted on",
       cmp_u["n_mismatches"] == 0 and cmp_u["verdict"] == "AGREES",
       f"btc gen 4 (candidate score below theta, incumbent rank 5 of a "
       f"top-3) +1e6 cents -> day {mine_u['day_increment_cents']!r}, "
       f"unchanged. A statistic that moved here would be reading rows "
       f"outside both selections")

    # -- 5. A ONE-ARM FEED REFUSES BY NAME, in BOTH implementations -------
    (td / "one").mkdir(exist_ok=True)
    feed_1 = write_feed(td / "one", btc, one_arm=True)
    mine_ref, why_mine = False, ""
    try:
        da_day_matched_volume(feed_1)
    except RaceVerifyRefused as e:
        mine_ref, why_mine = True, str(e)
    be_ref = False
    try:
        RR.day_matched_volume(feed_1)
    except Exception as e:                                    # noqa: BLE001
        be_ref = "score_incumbent" in str(e)
    ck("A ONE-ARM FEED REFUSES BY NAME -- and BOTH implementations refuse "
       "on the same missing field. The failure this prevents is the quiet "
       "one: with no incumbent column a caller gets an increment of exactly "
       "ZERO that looks like a measurement",
       mine_ref and be_ref and "score_incumbent" in why_mine,
       f"mine: '{why_mine[:70]}...'; BE's refuses on the same field")

    # -- 6. first crossing acts, and it is not the first ROW --------------
    (td / "fc").mkdir(exist_ok=True)
    fc = [_row("btc-updown-5m-A", "UP", 0, 1000, 0.10, 0.9, 7.0),
          _row("btc-updown-5m-A", "UP", 0, 2000, 0.95, 0.9, 11.0),
          _row("btc-updown-5m-B", "UP", 0, 1000, 0.01, 0.8, 3.0)]
    mv_fc = da_matched_volume(
        da_load_two_arm_feed(write_feed(td / "fc", fc), LATENCY_MS
                             )["per_coin"]["btc"]["rows"],
        [0.10, 0.95, 0.01], [0.9, 0.9, 0.8], th_btc, LATENCY_MS)
    ck("FIRST CROSSING ACTS, AND IT IS NOT THE FIRST ROW: a generation "
       "whose earliest row is BELOW the cutoff and whose later row is above "
       "is valued at the LATER row -- 11.0, not 7.0 and not their sum",
       mv_fc["candidate_net_cents"] == 11.0
       and mv_fc["candidate_n_cancelled"] == 1,
       f"gen A: rows at t 1000 (score 0.10, 7c) and 2000 (0.95, 11c) at "
       f"theta {th_btc:.4f} -> candidate_net "
       f"{mv_fc['candidate_net_cents']}")

    # -- 7. a row with no fill ahead contributes ZERO but is not dropped --
    (td / "nf").mkdir(exist_ok=True)
    nf = [_row("btc-updown-5m-A", "UP", 0, 1000, 0.95, 0.9, 50.0, fill=False),
          _row("btc-updown-5m-B", "UP", 0, 1000, 0.95, 0.8, 20.0)]
    blk = da_load_two_arm_feed(write_feed(td / "nf", nf), LATENCY_MS
                               )["per_coin"]["btc"]
    mv_nf = da_matched_volume(blk["rows"], blk["cand"], blk["inc"], th_btc,
                              LATENCY_MS)
    ck("A ROW WITH NO FILL AHEAD CONTRIBUTES 0.0 AND IS STILL AN ACTION: "
       "its 50 cents do not enter the value, and the cancellation COUNT "
       "still includes it -- which is what keeps the matched volume matched",
       mv_nf["candidate_net_cents"] == 20.0
       and mv_nf["candidate_n_cancelled"] == 2,
       f"two generations act, net {mv_nf['candidate_net_cents']} (the "
       f"no-fill 50c row contributes 0.0), count "
       f"{mv_nf['candidate_n_cancelled']}")

    # -- 8. the tie-break is on the KEY (R-234) ---------------------------
    (td / "tb").mkdir(exist_ok=True)
    tb = [_row("btc-updown-5m-B", "UP", 0, 1000, 0.95, 0.5, 100.0),
          _row("btc-updown-5m-A", "UP", 0, 1000, 0.01, 0.5, 1.0)]
    blk_t = da_load_two_arm_feed(write_feed(td / "tb", tb), LATENCY_MS
                                 )["per_coin"]["btc"]
    mv_tb = da_matched_volume(blk_t["rows"], blk_t["cand"], blk_t["inc"],
                              th_btc, LATENCY_MS)
    ck("THE TIE-BREAK IS ON THE KEY (R-234): two generations with EQUAL "
       "incumbent maxima are ordered lexicographically, so the matched "
       "selection is reproducible instead of falling back to dict order",
       mv_tb["incumbent_n_cancelled_matched"] == 1
       and mv_tb["incumbent_net_cents_matched"] == 1.0,
       f"both incumbent maxima 0.5; the matched pick is the "
       f"lexicographically smaller key (…-A, 1.0c), not the one that "
       f"happened to be inserted first (…-B, 100.0c)")

    # -- 9. the window bucket is the SLUG ---------------------------------
    (td / "wb").mkdir(exist_ok=True)
    wb = [_row("btc-updown-5m-A", "UP", 0, 1000, 0.95, 0.9, 4.0),
          _row("btc-updown-5m-A", "DOWN", 1, 1000, 0.95, 0.8, 6.0)]
    blk_w = da_load_two_arm_feed(write_feed(td / "wb", wb), LATENCY_MS
                                 )["per_coin"]["btc"]
    mv_wb = da_matched_volume(blk_w["rows"], blk_w["cand"], blk_w["inc"],
                              th_btc, LATENCY_MS)
    ck("THE WINDOW BUCKET IS THE SLUG, NOT A TIME BUCKET: two distinct "
       "actions on ONE slug land in ONE window key, which is what "
       "`increment_by_window` counts",
       len(mv_wb["increment_by_window"]) == 1
       and mv_wb["candidate_n_cancelled"] == 2
       and mv_wb["candidate_net_cents"] == 10.0,
       f"2 actions, {len(mv_wb['increment_by_window'])} window key "
       f"{list(mv_wb['increment_by_window'])}, net "
       f"{mv_wb['candidate_net_cents']}")

    # -- 10. THE PINNED REAL PATHS: refuse before GO, differently after ---
    pins = pinned_feeds()
    why_pre, why_post = "", ""
    try:
        verify_pinned_feeds(go=False)
    except RaceVerifyRefused as e:
        why_pre = str(e)
    try:
        verify_pinned_feeds(go=True)
    except RaceVerifyRefused as e:
        why_post = str(e)
    ck("THE FIVE PINNED REAL PATHS REFUSE BEFORE GO -- AND THE REFUSAL "
       "REASON CHANGES ACROSS THE GATE: 'REFUSED BEFORE GO' before, "
       "'declared and NOT OPENED' after. A control that refused "
       "identically on both sides of its own predicate would prove nothing",
       "BEFORE GO" in why_pre and "NOT OPENED" in why_post
       and "NOT OPENED" not in why_pre and "BEFORE GO" not in why_post,
       f"pre: '{why_pre[:58]}...'; post: '{why_post[:58]}...'")
    ck("AND THE PINS ARE READ RATHER THAN ASSUMED -- the declaration itself "
       "records that the feed exists for only SOME of the pinned days, "
       "which is a population fact the read cannot wish away",
       pins["n_days"] == 5 and pins["n_present"] == 3
       and pins["all_five_present"] is False
       and pins["absent_days"] == ["20260901", "20260902"],
       f"{pins['n_present']} of {pins['n_days']} feeds present; absent "
       f"{pins['absent_days']}; pins sha {pins['pins_sha256'][:16]}")

    # -- 11. an empty feed is a FAILURE, not an empty result --------------
    (td / "e").mkdir(exist_ok=True)
    empty_ref = False
    try:
        da_day_matched_volume(write_feed(td / "e", []))
    except RaceVerifyRefused as e:
        empty_ref = "empty read is a FAILURE" in str(e)
    ck("AN EMPTY FEED IS A FAILURE, NOT AN EMPTY RESULT -- a zero increment "
       "from no rows is the shape a silent miss takes",
       empty_ref, "an empty feed raises rather than returning 0.0")

    n_fail = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(("ok   " if c["passed"] else "FAIL ") + c["check"])
        print("       " + c["detail"])
    print(f"\n{'SELFTEST OK' if not n_fail else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {n_fail} failure(s)")
    return checks, n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if not a.selftest:
        ap.error("this round is fixture-only: --selftest")
    checks, n_fail = selftest()
    if a.output:
        a.output.write_text(json.dumps({
            "protocol": PROTOCOL + "_FIXTURE",
            "status": "FIXTURE_NO_REAL_FEED_OPENED",
            "verifier_identity": verifier_identity(),
            "statistic": "MATCHED_VOLUME (the interim's PRIMARY)",
            "latency_ms": LATENCY_MS,
            "theta_source": OP_DECL.name,
            "pins": pinned_feeds(),
            "tolerance": TOLERANCE,
            "checks": checks, "n_checks": len(checks), "n_failed": n_fail,
            "both_directions": True,
        }, indent=2, sort_keys=True) + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
