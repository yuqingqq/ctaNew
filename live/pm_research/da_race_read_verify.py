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

    checks.extend(selftest_real())

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
    #: NOT a gate that can be asserted: the gate is the read artifact's
    #: EXISTENCE, and `--real` merely selects the path that checks for it.
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--read-artifact")
    ap.add_argument("--pins", default=str(PINS_DECL))
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.real:
        if not a.read_artifact:
            ap.error("--real needs --read-artifact")
        #: THREE OUTCOMES, THREE EXIT CODES. A REFUSAL is not a failed
        #: verification: 2 means the instrument declined to run (the read is
        #: not open, a pin does not hold), 1 means it ran and FLAGGED, 0
        #: means it verified. A caller that could not tell 2 from 1 would
        #: read "the read has not happened yet" as "the read disagrees".
        try:
            r = verify_real_read(a.read_artifact, a.pins, output=a.output)
        except RaceVerifyRefused as e:
            print(str(e))
            return 2
        print(f"{r['status']} -- IS_A_VERIFICATION="
              f"{r['IS_A_VERIFICATION']}, {r['n_compared']} fields compared, "
              f"{r['n_mismatches']} mismatches, days "
              f"{r['day_set']['readable_from_the_pins']}")
        return 0 if r["IS_A_VERIFICATION"] else 1
    if not a.selftest:
        ap.error("--selftest, or --real --read-artifact <path> "
                 "[--pins <path>] [--output <path>]")
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




# ------------------------------------------------------------- the REAL path

#: BE's declared read artifact.
BE_READ_PROTOCOL = "BE_RACE_READ_RESULT_V1"
MULTIPLICITY_M = 2


def da_floors(g: int, m: int = MULTIPLICITY_M) -> dict:
    """The permutation floor, re-derived: the best adjusted p a sign test
    over G day clusters could reach under Holm at multiplicity m."""
    best = m / 2 ** g
    return {"G": g, "multiplicity": m, "best_possible_adjusted_p": best,
            "clears_0_05": best <= 0.05}


def da_hash_and_parse_feed(path, latency_ms: int) -> dict:
    """ONE PASS: every byte that is PARSED is the byte that is HASHED.

    Hashing the file in one open and parsing it in another leaves a window
    between them -- the bytes can move, and both readings are then true of
    different files. Here the digest accumulates over exactly the lines the
    parser consumes, so 'the digest is of the bytes parsed' is a property of
    the loop rather than a sentence in a receipt.

    NOTHING PARSED IS USED UNTIL THE DIGEST IS CHECKED. The caller verifies
    against the pin before touching `per_coin`, which is what makes 'the pin
    is checked before parsing is trusted' true even though there is only
    one pass."""
    L = str(latency_ms)
    h = hashlib.sha256()
    n_bytes = 0
    per: dict = {}
    n = missing = 0
    deferred = None
    with Path(path).open("rb") as fh:
        for raw in fh:
            h.update(raw)
            n_bytes += len(raw)
            line = raw.decode().strip()
            if not line:
                continue
            try:
                fr = json.loads(line)
            except json.JSONDecodeError as e:
                n += 1
                if deferred is None:
                    deferred = (f"REFUSED: feed row {n} is not readable JSON "
                                f"({e.msg}).")
                continue
            n += 1
            if "score_incumbent" not in fr:
                #: REV 52 section 2.5. DEFERRED, NOT RAISED. A shape error
                #: raised here aborts the pass before the digest is
                #: complete, so a file whose BYTES do not match the pin is
                #: reported as ONE-ARM -- an answer about its contents when
                #: the true fact is that it is not the pinned file at all.
                #: The hashing pass finishes, the pin is judged first, and
                #: this is raised only if the bytes were the right ones.
                if deferred is None:
                    deferred = (
                        f"REFUSED: feed row {n} carries no `score_incumbent`. "
                        f"This is a ONE-ARM feed and the declared estimand "
                        f"is an increment OVER the incumbent; computing it "
                        f"from one arm would compare the candidate with "
                        f"itself and return a zero that looks like a "
                        f"measurement.")
                continue
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
    return {"per_coin": per, "n_feed_rows": n,
            "n_rows_without_an_incumbent_score": missing,
            "parsed_stream_sha256": h.hexdigest(),
            "parsed_stream_bytes": n_bytes,
            "deferred_shape_error": deferred,
            "why_deferred": (
                "REV 52 section 2.5: a shape error raised mid-pass aborts "
                "before the digest is complete, so a file whose bytes are "
                "not the pinned ones is reported as ONE-ARM -- an answer "
                "about its CONTENTS when the true fact is that it is not "
                "the pinned file. The pin is judged first")}


def da_day_from_pinned_feed(path, pin_sha: str, *,
                            latency_ms: int = LATENCY_MS) -> dict:
    """One day: hash-and-parse in one pass, verify the pin, THEN compute.

    REV 49 section 3.4: THE INFORMATIVE REFUSAL MUST BE REACHABLE. A pinned
    feed that is not on disk used to surface as a bare `FileNotFoundError`
    from deep inside the parser -- a generic refusal standing in front of the
    one that says what is wrong. The existence check comes FIRST now, and it
    refuses by name."""
    p = Path(path)
    if not p.is_file():
        raise RaceVerifyRefused(
            f"REFUSED: PINNED_FEED_ABSENT — the pins mark {p.name} as "
            f"present with a digest, and it is not on disk at {p}. A day the "
            f"pins promise and the ledger does not hold is a REFUSAL by "
            f"name, not a parser error: the pin and the ledger disagree "
            f"about what exists.")
    parsed = da_hash_and_parse_feed(path, latency_ms)
    #: THE PIN IS JUDGED FIRST, on a COMPLETED hashing pass (REV 52 2.5).
    if parsed["parsed_stream_sha256"] != pin_sha:
        raise RaceVerifyRefused(
            f"REFUSED — THE PIN DOES NOT HOLD for {Path(path).name}: the "
            f"stream this verifier parsed hashes to "
            f"{parsed['parsed_stream_sha256']} and the pin says {pin_sha}. "
            f"The bytes are not the bytes the read was pinned to, and no "
            f"number computed from them is comparable. Nothing parsed here "
            f"was used.")
    #: the bytes ARE the pinned ones, so a shape error is now a statement
    #: about the pinned file and is raised.
    if parsed["deferred_shape_error"]:
        raise RaceVerifyRefused(parsed["deferred_shape_error"])
    per_coin, net = {}, 0.0
    for coin, blk in sorted(parsed["per_coin"].items()):
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
    return {"per_coin": per_coin, "day_increment_cents": net,
            "day_sign": (1 if net > 0 else (-1 if net < 0 else 0)),
            "n_feed_rows": parsed["n_feed_rows"],
            "parsed_stream_sha256": parsed["parsed_stream_sha256"],
            "parsed_stream_bytes": parsed["parsed_stream_bytes"],
            "pin_holds": True}


def day_status_in_artifact(art, day: str) -> dict:
    """What does the read artifact SAY about this day?

    REV 49 section 3.3: SILENCE PASSED. An artifact that never mentioned
    09-01 or 09-02 verified, because the day-set check only looked for a
    NUMBER against those days and found none. Absence read as compliance --
    which is the same failure as a zero from a check that never ran.

    Every day in the pins must be SAID, one way or the other:
      READABLE_WITH_A_NUMBER   -- in `per_day` with a numeric increment
      READ_BUT_UNRECOVERABLE   -- named, with an unrecoverable status and
                                  NO number
      PRESENT_BUT_NO_STATUS    -- the string appears and says nothing
      ABSENT                   -- the artifact does not mention it at all
    """
    blk = (art.get("per_day") or {}).get(day)
    has_number = isinstance(blk, dict) and any(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for _, v in _leaves(blk))
    if has_number:
        return {"status": "READABLE_WITH_A_NUMBER", "said": True}
    #: hunt the day anywhere in the artifact, and ask whether what it sits
    #: beside calls it unrecoverable.
    mentioned, unrecoverable = False, False
    for path_, v in _leaves(art):
        hay = f"{path_} {v}"
        if day in hay:
            mentioned = True
            if "UNRECOVERABLE" in path_.upper() or (
                    isinstance(v, str) and "UNRECOVERABLE" in v.upper()):
                unrecoverable = True
        if isinstance(v, str) and v == day and "UNRECOVERABLE" in path_.upper():
            unrecoverable = True
    if unrecoverable:
        return {"status": "READ_BUT_UNRECOVERABLE", "said": True}
    if mentioned:
        return {"status": "PRESENT_BUT_NO_STATUS", "said": False,
                "why": ("the day is mentioned and nothing says what it is. A "
                        "day named without a status is not a statement about "
                        "it")}
    return {"status": "ABSENT", "said": False,
            "why": ("the artifact does not mention this day at all. Silence "
                    "is not compliance: the read must SAY what it did with "
                    "every day the pins name")}


#: DA 100. THE DECLARATION IS THE OTHER HALF OF THE PIN. The pins say
#: WHICH BYTES each day was read from; the DECLARATION says WHICH DAYS may
#: be read at all, and what G is. A verifier that checked only the pins
#: would admit a read of the right bytes over the wrong day set.
RACE_DECL_FAMILY = "be_race_read_declaration"


def _derived_for_markers() -> Path:
    """The CANONICAL ledger's derived directory -- where a marker lives."""
    import da_root as _R                                      # noqa: PLC0415
    return _R.derived_dir("the race read's OPENED markers")


def race_declaration_head() -> dict:
    """The chain head of BE's race-read declaration family, by the pair."""
    import da_nonhead_census as _C                            # noqa: PLC0415
    d = Path(__file__).resolve().parent / "declarations"
    blk = _C.declaration_chains(d).get(RACE_DECL_FAMILY)
    if not blk:
        raise RaceVerifyRefused(
            f"REFUSED: RACE_DECLARATION_FAMILY_ABSENT — no "
            f"`{RACE_DECL_FAMILY}_v*.json` under {d}. The day set and G "
            f"cannot be assumed.")
    if blk["n_heads"] != 1:
        raise RaceVerifyRefused(
            f"REFUSED: RACE_DECLARATION_DOES_NOT_RESOLVE_TO_ONE_HEAD — "
            f"heads {blk['heads']}. Which one a read was taken under has "
            f"no answer, and picking the highest number would invent the "
            f"link the seat did not write.")
    f = d / blk["heads"][0]
    obj = json.loads(f.read_text())
    pop = obj.get("population") or {}
    return {"name": f.name, "path": str(f),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "G_declared": obj.get("G"),
            "READABLE": sorted(pop.get("READABLE") or []),
            "READ_BUT_UNRECOVERABLE": sorted(
                pop.get("READ_BUT_UNRECOVERABLE") or []),
            "n_members": blk["n_members"],
            "resolved_by": ("the chain head of the family, pair-verified "
                            "back to v1 -- never a filename literal")}


def opened_markers(marker_dir: Path, declared_days) -> dict:
    """ONE marker per declared day, and NO FOURTH.

    The marker is the ONLY record that a day was spent. A fourth marker
    means a day was opened that the declaration does not name -- which is
    the one thing rule 11 cannot absorb after the fact."""
    d = Path(marker_dir)
    found = sorted(x.name for x in d.glob("be_race_read_OPENED_*"))
    days_found = sorted({x.replace("be_race_read_OPENED_", "").split(".")[0]
                         for x in found})
    declared = sorted(declared_days)
    return {"marker_dir": str(d), "markers": found,
            "n_markers": len(found), "days_with_a_marker": days_found,
            "declared_days": declared,
            "one_per_declared_day": days_found == declared,
            "extra_days_opened": sorted(set(days_found) - set(declared)),
            "declared_days_without_a_marker": sorted(
                set(declared) - set(days_found)),
            "what_a_marker_IS": ("an UNTRACKED file under the ledger's "
                                 "derived directory -- the only record "
                                 "that a day was spent, and exactly as "
                                 "durable as that directory")}


def verify_real_read(read_artifact: str, pins_path: str, *,
                     output: Path | None = None,
                     latency_ms: int = LATENCY_MS) -> dict:
    """THE REAL PATH.

    THE GATE IS THE ARTIFACT'S EXISTENCE, and nothing else. BE's read
    artifact is written by the coordinator's opening act; before that act it
    does not exist, and this verifier refuses BY NAME. There is no clock
    here and no flag: an instrument that could be told the read had happened
    would be a way of asserting it had.

    THE FEEDS ARE OPENED A SECOND TIME, AND THAT IS SAID OUT LOUD. The read
    opened them; this recompute opens them again. Under rule 11 that
    consumes nothing further -- the days are already READ, and a second
    reading of an already-consumed day adds no consumption. What it would
    NOT be allowed to do is open a day the read did not.
    """
    ap = Path(read_artifact)
    if not ap.is_file():
        raise RaceVerifyRefused(
            f"REFUSED: READ_ARTIFACT_ABSENT_THE_READ_HAS_NOT_BEEN_OPENED — "
            f"no read artifact at {read_artifact}. The artifact is written "
            f"by the coordinator's opening act, and its EXISTENCE is the "
            f"gate: there is no clock here and no flag, because an "
            f"instrument that could be TOLD the read had happened would be a "
            f"way of asserting it had.")
    #: REV 49 section 3.4, THE CASE AS FILED. Every input this path reads
    #: is checked BY NAME before it is parsed. The artifact's absence was
    #: already named; the PINS were not, and an absent or unreadable pins
    #: file died with a bare FileNotFoundError from `json.loads` -- a
    #: generic refusal standing in front of an informative one, on the CLI
    #: path where the coordinator will meet it.
    pp = Path(pins_path)
    if not pp.is_file():
        raise RaceVerifyRefused(
            f"REFUSED: PINS_DECLARATION_ABSENT — no pins declaration at "
            f"{pins_path}. The pins are what say WHICH bytes the read was "
            f"pinned to; without them there is nothing to verify a feed "
            f"against, and a recompute over unpinned bytes is not a check.")
    try:
        art = json.loads(ap.read_bytes())
    except json.JSONDecodeError as e:
        raise RaceVerifyRefused(
            f"REFUSED: READ_ARTIFACT_UNREADABLE — {ap.name} is not readable "
            f"JSON ({e.msg} at line {e.lineno}). An artifact this verifier "
            f"cannot parse is not one it may report on.")
    try:
        pins = json.loads(pp.read_bytes())
    except json.JSONDecodeError as e:
        raise RaceVerifyRefused(
            f"REFUSED: PINS_DECLARATION_UNREADABLE — {pp.name} is not "
            f"readable JSON ({e.msg} at line {e.lineno}).")
    if not isinstance(pins.get("per_day"), dict) or not pins["per_day"]:
        raise RaceVerifyRefused(
            f"REFUSED: PINS_DECLARATION_CARRIES_NO_DAYS — {pp.name} has no "
            f"`per_day` block. An empty pin set would make every day "
            f"vacuously said, which is the silence REV 49 section 3.3 "
            f"closed arriving through the pins instead of the artifact.")
    per_day_pins = pins["per_day"]

    readable = sorted(d for d, v in per_day_pins.items() if v.get("exists"))
    unrecoverable = sorted(d for d, v in per_day_pins.items()
                           if not v.get("exists"))

    checks, bad = [], []

    #: DA 100 / THE READ ORDER. ***THIS RECORD COMES FIRST -- BEFORE THE
    #: RUNNER'S READ, BEFORE REV 78, BEFORE THE COORDINATOR*** -- so a
    #: number quoted here is a number published ahead of the read that is
    #: entitled to publish it. The comparison is by FIELD NAME and STATE:
    #: MATCH / MISMATCH / ABSENT_IN_ARTIFACT. No MATCHED_VOLUME, no
    #: permutation p, no direction, no per-day quantity of any kind
    #: appears in this verifier's own output. ***A verdict does not need
    #: the number; it needs the predicate.***
    def cmp(name, mine, theirs):
        if theirs is None:
            checks.append({"field": name, "state": "ABSENT_IN_ARTIFACT"})
            bad.append(name)
            return
        ok = (float(mine) == float(theirs)) if isinstance(
            mine, (int, float)) else (mine == theirs)
        checks.append({"field": name,
                       "state": "MATCH" if ok else "MISMATCH"})
        if not ok:
            bad.append(name)

    art_per_day = art.get("per_day") or {}
    days_out = {}
    for day in readable:
        pin = per_day_pins[day]
        mine = da_day_from_pinned_feed(pin["path"], pin["sha256"],
                                       latency_ms=latency_ms)
        theirs = art_per_day.get(day)
        if theirs is None:
            days_out[day] = {
                "status": "READABLE_DAY_ABSENT_FROM_THE_ARTIFACT",
                "recomputed": "WITHHELD_BY_THE_READ_ORDER"}
            bad.append(f"per_day.{day}")
            continue
        cmp(f"{day}.day_increment_cents", mine["day_increment_cents"],
            theirs.get("day_increment_cents"))
        cmp(f"{day}.day_sign", mine["day_sign"], theirs.get("day_sign"))
        cmp(f"{day}.n_feed_rows", mine["n_feed_rows"],
            theirs.get("n_feed_rows"))
        for coin, blk in sorted(mine["per_coin"].items()):
            t = (theirs.get("per_coin") or {}).get(coin)
            if t is None:
                checks.append({"field": f"{day}.per_coin.{coin}",
                               "state": "ABSENT_IN_ARTIFACT"})
                bad.append(f"{day}.per_coin.{coin}")
                continue
            for f in ("MATCHED_VOLUME_increment_cents", "candidate_net_cents",
                      "incumbent_net_cents_matched", "n_actions",
                      "counts_matched"):
                cmp(f"{day}.per_coin.{coin}.{f}", blk[f], t.get(f))
        days_out[day] = {"status": "COMPARED",
                         "recomputed": "WITHHELD_BY_THE_READ_ORDER",
                         "n_coins_compared": len(mine["per_coin"]),
                         "pin_sha256": pin["sha256"]}

    #: THE DAY SET. A day the pins mark `exists: false` was READ under the
    #: interim and its economics are UNRECOVERABLE. It must appear as that,
    #: never as a number -- a number for such a day would be a value nothing
    #: on disk can produce.
    numbered_unrecoverable = []
    for day in unrecoverable:
        blk = art_per_day.get(day)
        has_number = isinstance(blk, dict) and any(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for _, v in _leaves(blk))
        if has_number:
            numbered_unrecoverable.append(day)
    #: REV 49 section 3.3: every day in the pins must be SAID.
    said, unsaid = {}, []
    for day in sorted(per_day_pins):
        st = day_status_in_artifact(art, day)
        expected = ("READABLE_WITH_A_NUMBER" if day in readable
                    else "READ_BUT_UNRECOVERABLE")
        st["expected"] = expected
        st["matches_the_pin"] = st["status"] == expected
        said[day] = st
        if not st["matches_the_pin"]:
            unsaid.append(day)
            bad.append(f"day_set.{day}.{st['status']}")

    day_set = {
        "every_day_in_the_pins_must_be_SAID": said,
        "n_days_in_the_pins": len(per_day_pins),
        "n_days_said_as_the_pins_expect": sum(
            1 for v in said.values() if v["matches_the_pin"]),
        "days_not_said_as_the_pins_expect": unsaid,
        "silence_does_not_pass": (
            "an artifact that never mentioned 09-01/02 used to VERIFY, "
            "because the check only looked for a NUMBER against them and "
            "found none. Absence read as compliance -- the same failure as a "
            "zero from a check that never ran"),
        "readable_from_the_pins": readable,
        "read_but_unrecoverable_from_the_pins": unrecoverable,
        "days_in_the_artifact": sorted(art.get("days") or []),
        "readable_set_matches": sorted(art.get("days") or []) == readable,
        "unrecoverable_days_carrying_a_number": numbered_unrecoverable,
        "no_unrecoverable_day_carries_a_number": not numbered_unrecoverable,
        "why": ("09-01 and 09-02 were opened under the interim and their "
                "surviving receipts carry no economics, so a number for "
                "either is a value nothing on disk can produce"),
    }
    if numbered_unrecoverable:
        bad.append("day_set.unrecoverable_day_carries_a_number")
    if not day_set["readable_set_matches"]:
        bad.append("day_set.readable_set")

    #: DA 100 (2)(3): THE DECLARATION, THE DAY SET'S TWO G's, THE
    #: BYTE-IDENTITY BLOCK AND THE OPENED MARKERS.
    head = race_declaration_head()
    ps = art.get("pre_state") or {}
    ds = art.get("day_set") or {}
    cons = art.get("consumption") or {}
    bi = art.get("byte_identity") or {}
    declared_sha = ps.get("declaration_sha256") or ds.get(
        "declaration_sha256")
    if declared_sha is not None and declared_sha != head["sha256"]:
        raise RaceVerifyRefused(
            f"REFUSED: READ_ARTIFACT_DECLARATION_DIGEST_DIFFERS — the "
            f"artifact says it read under {str(declared_sha)[:16]}… and "
            f"the chain head {head['name']} is {head['sha256'][:16]}…. A "
            f"read taken under another declaration is a read of another "
            f"day set, and this verifier will not reconcile the two.")
    markers = opened_markers(
        Path(ps.get("marker_dir") or _derived_for_markers()),
        head["READABLE"])
    decl_block = {
        "chain_head": {k: head[k] for k in
                       ("name", "sha256", "G_declared", "READABLE",
                        "READ_BUT_UNRECOVERABLE", "n_members",
                        "resolved_by")},
        "the_artifact_says_it_read_under": declared_sha,
        "digest_agrees_with_the_chain_head": (
            None if declared_sha is None else declared_sha == head["sha256"]),
        "declaration_named_in_the_artifact": (
            ps.get("declaration") or ds.get("from")),
        "decl_source": cons.get("decl_source"),
        "decl_supplied_by_the_caller": cons.get("decl_supplied_by_the_caller"),
        "BEs_claim_that_it_is_the_chain_head": cons.get(
            "decl_is_the_chain_head"),
        "and_this_verifier_resolved_the_head_ITSELF": True,
        "why": ("the pins say WHICH BYTES each day was read from; the "
                "declaration says WHICH DAYS may be read at all and what G "
                "is. A verifier checking only the pins would admit a read "
                "of the right bytes over the wrong day set"),
    }
    day_set_declared = {
        "READABLE_in_the_declaration": head["READABLE"],
        "READABLE_in_the_artifact": sorted(ds.get("READABLE") or []),
        "readable_agrees": sorted(ds.get("READABLE") or []) == head[
            "READABLE"] if ds else None,
        "G_declared_in_the_declaration": head["G_declared"],
        "G_declared_in_the_artifact": ds.get("G_declared"),
        "G_computed_from_the_set": ds.get("G_computed_from_the_set"),
        "both_Gs_agree_with_each_other": (
            None if not ds else
            ds.get("G_declared") == ds.get("G_computed_from_the_set")),
        "both_Gs_agree_with_the_declaration": (
            None if not ds else
            ds.get("G_declared") == head["G_declared"]
            and ds.get("G_computed_from_the_set") == head["G_declared"]),
        "why_two": ("a G DECLARED and a G COMPUTED FROM THE SET are two "
                    "different facts; one number standing for both is how "
                    "a widened day set would pass unnoticed"),
    }
    byte_identity = {
        "present": bool(bi),
        "all_unchanged": bi.get("all_unchanged"),
        "digest_covers_every_byte_parsed": (
            None if not bi else
            all(bool(x) for x in (
                bi.get("digest_covers_every_byte_parsed") or {}).values())
            if isinstance(bi.get("digest_covers_every_byte_parsed"), dict)
            else bi.get("digest_covers_every_byte_parsed")),
        "on_mismatch_declared": bi.get("on_mismatch"),
    }
    if ds and day_set_declared["both_Gs_agree_with_the_declaration"] is False:
        bad.append("day_set.G")
    if ds and day_set_declared["readable_agrees"] is False:
        bad.append("day_set.READABLE")
    if bi and byte_identity["all_unchanged"] is not True:
        bad.append("byte_identity.all_unchanged")
    if bi and byte_identity["digest_covers_every_byte_parsed"] is not True:
        bad.append("byte_identity.digest_covers_every_byte_parsed")
    if markers["extra_days_opened"]:
        bad.append("opened_markers.extra_days_opened")

    #: THE FLOORS, re-derived.
    g = len(readable)
    mine_floor = da_floors(g)
    art_floor = art.get("permutation_floors") or {}
    art_best = (art_floor.get("resolved_best_possible_adjusted_p")
                or (art_floor.get("pessimistic") or {}).get(
                    "best_possible_adjusted_p"))
    cmp("permutation_floor.best_possible_adjusted_p",
        mine_floor["best_possible_adjusted_p"], art_best)
    cmp("permutation_floor.clears_0_05", mine_floor["clears_0_05"],
        None if art_best is None else (float(art_best) <= 0.05))

    out = {
        "protocol": PROTOCOL + "_REAL_READ",
        "status": None,
        "the_gate_is_the_artifacts_existence": {
            "read_artifact": ap.name,
            "sha256": hashlib.sha256(ap.read_bytes()).hexdigest(),
            "protocol_in_the_artifact": art.get("protocol"),
            "is_BEs_declared_shape":
                art.get("protocol") == BE_READ_PROTOCOL,
            "no_clock_no_flag": True,
        },
        "pins": {"path": Path(pins_path).name,
                 "sha256": hashlib.sha256(
                     Path(pins_path).read_bytes()).hexdigest()},
        "the_feeds_are_opened_a_SECOND_time": {
            "answer": True,
            "why_that_is_allowed": (
                "the read opened them; this recompute opens them again. "
                "Under rule 11 that consumes nothing further -- the days are "
                "already READ, and a second reading of an already-consumed "
                "day adds no consumption. What it would NOT be allowed to do "
                "is open a day the read did not, which is why the readable "
                "set is taken from the PINS and compared against the "
                "artifact's own day list."),
            "n_days_opened_by_this_verifier": len(readable),
            "days_opened": readable,
            "opened_no_day_the_read_did_not":
                set(readable) <= set(art.get("days") or []),
        },
        "one_pass_hashing": (
            "every byte PARSED is the byte HASHED, in one pass, and nothing "
            "parsed is used until the digest matches the pin. Hashing in one "
            "open and parsing in another leaves a window in which both "
            "readings are true of different files"),
        "day_set": day_set,
        "the_declaration": decl_block,
        "day_set_against_the_declaration": day_set_declared,
        "byte_identity_block": byte_identity,
        "opened_markers": markers,
        "what_this_verifier_VERIFIES": [
            "the read artifact EXISTS (the gate), is BE's declared "
            "protocol, and is readable JSON",
            "the DECLARATION resolves to one chain head by the R-608 pair, "
            "and the artifact's `pre_state.declaration_sha256` IS that "
            "head -- a different one REFUSES",
            "the day set: READABLE against the declaration, and BOTH G's "
            "(declared and computed from the set) against it and each "
            "other",
            "every day in the PINS is SAID, with the status the pin "
            "expects, and no unrecoverable day carries a number",
            "one OPENED marker per declared day and NO FOURTH",
            "the byte-identity block: unchanged, and the digest covering "
            "every byte parsed",
            "the permutation floor, re-derived from G",
            "each readable day's numbers, RECOMPUTED from the pinned bytes "
            "and compared field by field -- the comparison is reported by "
            "NAME and STATE, never by value",
        ],
        "what_this_verifier_DOES_NOT_VERIFY": [
            "that the OPENED markers were written BEFORE the reading -- "
            "the marker's presence is the fact, and its timestamp is not "
            "evidence of order",
            "that the feeds were not modified between BE's read and this "
            "recompute -- the PIN is what makes that decidable, and it is "
            "checked, but a feed replaced with bytes matching its pin is "
            "not distinguishable and is not claimed to be",
            "the DIRECTION, the permutation p, or any economic reading of "
            "the day set -- those belong to the read, and this record "
            "comes BEFORE it",
            "BE's own claim `decl_is_the_chain_head`: it is RECORDED as "
            "BE's claim and re-resolved here independently, never taken",
        ],
        "the_read_order": (
            "this record, then the runner's read, then REV 78, then the "
            "coordinator -- so no number the read is entitled to publish "
            "appears here"),
        "permutation_floor_recomputed": mine_floor,
        "days": days_out,
        "checks": checks,
        "n_compared": len(checks),
        "n_mismatches": len(bad),
        "mismatched_fields": bad,
        "tolerance": TOLERANCE,
        "verifier_identity": verifier_identity(),
        "IS_A_VERIFICATION": None,
    }
    out["IS_A_VERIFICATION"] = bool(
        not bad and readable
        and markers["one_per_declared_day"]
        and (decl_block["digest_agrees_with_the_chain_head"] is not False)
        and out["the_gate_is_the_artifacts_existence"]["is_BEs_declared_shape"]
        and day_set["no_unrecoverable_day_carries_a_number"]
        and not day_set["days_not_said_as_the_pins_expect"])
    out["status"] = "VERIFIED" if out["IS_A_VERIFICATION"] else "FLAGGED"
    if output:
        Path(output).write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    return out


def _leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def _synthetic_pins(d: Path, feeds: dict, absent: list) -> Path:
    """The pins declaration's shape: per-day path + sha256, `exists` false
    for the days whose feed was never written."""
    per_day = {}
    for day, p in sorted(feeds.items()):
        per_day[day] = {"exists": True, "path": str(p),
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for day in absent:
        per_day[day] = {"exists": False,
                        "path": str(d / f"missing_{day}.jsonl")}
    p = d / "pins.json"
    p.write_text(json.dumps({"protocol": "BE_RACE_READ_FEED_PINS_V1",
                             "per_day": per_day,
                             "all_five_present": not absent,
                             "the_read_voids_on_mismatch": True}))
    return p


def _synthetic_read_artifact(d: Path, per_day: dict, *,
                             name: str = "be_race_read_result_v1.json",
                             extra_days: dict | None = None,
                             unrecoverable: list | None = None,
                             decl_sha: str | None = None,
                             markers: bool = True,
                             extra_marker: str | None = None) -> Path:
    """BE's declared shape, read from `be_race_reader.read()` as a document.

    DA 100: the shape now includes what REV 77 S4 will check after this
    verifier -- `pre_state`, `day_set` with BOTH G's, `decl_source`, the
    byte-identity block -- and the OPENED markers are WRITTEN, because a
    positive control that omits them is not a control over the check that
    reads them."""
    days = sorted(per_day)
    _head = race_declaration_head()
    if markers:
        for _dy in days:
            (d / f"be_race_read_OPENED_{_dy}.json").write_text(
                json.dumps({"day": _dy, "synthetic": True}))
        if extra_marker:
            (d / f"be_race_read_OPENED_{extra_marker}.json").write_text(
                json.dumps({"day": extra_marker, "synthetic": True}))
    signs = {k: v["day_sign"] for k, v in per_day.items()}
    g = len(days)
    body = {
        "protocol": BE_READ_PROTOCOL,
        "days": days, "per_day": dict(per_day), "day_signs": signs,
        "n_positive": sum(1 for v in signs.values() if v == 1),
        "n_negative": sum(1 for v in signs.values() if v == -1),
        "n_zero": sum(1 for v in signs.values() if v == 0),
        "permutation_floors": {
            "optimistic": {"G": g,
                           "best_possible_adjusted_p": MULTIPLICITY_M / 2 ** g},
            "pessimistic": {"G": g,
                            "best_possible_adjusted_p": MULTIPLICITY_M / 2 ** g},
            "resolved_best_possible_adjusted_p": MULTIPLICITY_M / 2 ** g,
            "neither_clears_0_05": MULTIPLICITY_M / 2 ** g > 0.05},
        "decides_nothing": "REPORTED (rule 14).",
        "pre_state": {
            "declaration": _head["name"],
            "declaration_sha256": decl_sha or _head["sha256"],
            "marker_dir": str(d),
            "existing_OPENED_markers": [],
            "n_existing_OPENED_markers": 0,
            "zero_markers_before_the_act": True,
            "days": days},
        "day_set": {"from": _head["name"],
                    "declaration_sha256": decl_sha or _head["sha256"],
                    "READABLE": days,
                    "G_declared": _head["G_declared"],
                    "G_computed_from_the_set": len(days),
                    "G_agrees_with_the_declaration": True,
                    "the_cli_cannot_widen_or_narrow_it": True},
        "consumption": {"the_read_consumes": True, "marker_dir": str(d),
                        "decl_source": "RE-VERIFIED here from the chain "
                                       "head (synthetic)",
                        "decl_supplied_by_the_caller": True,
                        "decl_is_the_chain_head": True},
        "byte_identity": {"all_unchanged": True,
                          "digest_covers_every_byte_parsed": {
                              dd: True for dd in days},
                          "on_mismatch": "the read is VOID -- enforced"},
    }
    if unrecoverable:
        body["population"] = {"READ_BUT_UNRECOVERABLE": list(unrecoverable)}
    if extra_days:
        body["per_day"].update(extra_days)
    p = d / name
    p.write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    return p


def selftest_real() -> list:                                  # noqa: C901
    """The REAL-path battery, returned into the module's one check list."""
    checks: list[dict] = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    td = Path(tempfile.mkdtemp(prefix="da69_"))
    th_btc = theta_for("btc")

    #: three readable days, each a two-coin feed with a known increment
    feeds, mine_days = {}, {}
    for k, day in enumerate(("20260903", "20260904", "20260905")):
        rows = [_row(f"btc-updown-5m-W{g}", "UP", g, 1000 + g,
                     0.9 if g < 3 else 0.1, 0.5 - 0.01 * (9 - g),
                     (g + 1) * 10.0 * (k + 1)) for g in range(10)]
        rows += [_row(f"eth-updown-5m-W{g}", "UP", g, 2000 + g,
                      0.5 if g < 2 else 0.01, 0.2 - 0.01 * (5 - g),
                      (g + 1) * 5.0) for g in range(6)]
        dd = td / day
        dd.mkdir(exist_ok=True)
        feeds[day] = write_feed(dd, rows)
    pins_p = _synthetic_pins(td, feeds, ["20260901", "20260902"])
    pins = json.loads(pins_p.read_text())["per_day"]
    for day in sorted(feeds):
        mine_days[day] = da_day_from_pinned_feed(pins[day]["path"],
                                                 pins[day]["sha256"])
    UNREC = ["20260901", "20260902"]
    art_p = _synthetic_read_artifact(td, mine_days,
                                     unrecoverable=UNREC)

    # -- 1. THE GATE IS THE ARTIFACT'S EXISTENCE --------------------------
    why_absent = ""
    try:
        verify_real_read(str(td / "nope.json"), str(pins_p))
    except RaceVerifyRefused as e:
        why_absent = str(e)
    ck("THE GATE IS THE READ ARTIFACT'S EXISTENCE, and the refusal names "
       "it: the artifact is written by the coordinator's opening act, so "
       "before that act there is nothing to verify. NO CLOCK AND NO FLAG -- "
       "an instrument that could be TOLD the read had happened would be a "
       "way of asserting it had",
       "READ_ARTIFACT_ABSENT_THE_READ_HAS_NOT_BEEN_OPENED" in why_absent
       and "no clock here and no flag" in why_absent,
       f"'{why_absent[:96]}...'")

    # -- 2. the happy path VERIFIES ---------------------------------------
    out = verify_real_read(str(art_p), str(pins_p))
    ck("THE REAL PATH VERIFIES: three readable days recomputed from the "
       "pinned feeds with MY OWN statistic and compared EXACT to BE's "
       "artifact, day values and signs, plus the floors and the day set",
       out["IS_A_VERIFICATION"] is True and out["status"] == "VERIFIED"
       and out["n_mismatches"] == 0 and out["n_compared"] >= 20,
       f"{out['n_compared']} fields compared exactly, 0 mismatches; days "
       f"{out['day_set']['readable_from_the_pins']}")

    # -- 3. ONE MOVED VALUE IS FLAGGED ------------------------------------
    moved = json.loads(art_p.read_text())
    d0 = sorted(moved["per_day"])[0]
    moved["per_day"][d0]["day_increment_cents"] = float(
        moved["per_day"][d0]["day_increment_cents"]) + 1e-9
    mp = td / "art_moved.json"
    mp.write_text(json.dumps(moved, default=str))
    out_m = verify_real_read(str(mp), str(pins_p))
    ck("KNOWN-BAD: ONE MOVED VALUE IS FLAGGED -- 1e-9 on one day's increment "
       "is a MISMATCH, because the comparison is exact and a tolerance could "
       "only hide a selection difference",
       out_m["IS_A_VERIFICATION"] is False and out_m["status"] == "FLAGGED"
       and f"{d0}.day_increment_cents" in out_m["mismatched_fields"],
       f"{d0} +1e-9 -> {out_m['mismatched_fields'][:2]}")

    # -- 4. A TAMPERED FEED REFUSES ON THE PIN ----------------------------
    tamper_day = sorted(feeds)[0]
    with Path(feeds[tamper_day]).open("a") as fh:
        fh.write(json.dumps(_row("btc-updown-5m-X", "UP", 99, 9999,
                                 0.99, 0.99, 1.0)) + "\n")
    why_pin = ""
    try:
        verify_real_read(str(art_p), str(pins_p))
    except RaceVerifyRefused as e:
        why_pin = str(e)
    ck("A TAMPERED FEED REFUSES ON THE PIN, and the refusal says NOTHING "
       "PARSED WAS USED: the stream this verifier parsed no longer hashes to "
       "what the read was pinned to, so no number computed from it is "
       "comparable",
       "THE PIN DOES NOT HOLD" in why_pin
       and "Nothing parsed here was used" in why_pin
       and pins[tamper_day]["sha256"][:16] in why_pin,
       f"'{why_pin[:104]}...'")
    #: restore, so the remaining checks run on the pinned bytes
    txt = Path(feeds[tamper_day]).read_text().splitlines()
    Path(feeds[tamper_day]).write_text("\n".join(txt[:-1]) + "\n")
    ck("AND THE PIN HOLDS AGAIN ONCE THE BYTES ARE RESTORED -- the check is "
       "on the BYTES, not on a flag that stays tripped",
       verify_real_read(str(art_p), str(pins_p))["IS_A_VERIFICATION"] is True,
       f"{tamper_day} restored to its pinned digest "
       f"{pins[tamper_day]['sha256'][:16]}")

    # -- 5. A NUMBER FOR AN UNRECOVERABLE DAY IS FLAGGED ------------------
    numbered = _synthetic_read_artifact(
        td, mine_days, name="art_numbered.json", unrecoverable=UNREC,
        extra_days={"20260901": {"day_increment_cents": -12.5,
                                 "day_sign": -1, "n_feed_rows": 10}})
    out_n = verify_real_read(str(numbered), str(pins_p))
    ck("KNOWN-BAD: A READ ARTIFACT CARRYING A NUMBER FOR 09-01 IS FLAGGED. "
       "The pins mark it `exists: false`; it was READ under the interim and "
       "its economics are UNRECOVERABLE, so a number for it is a value "
       "NOTHING ON DISK CAN PRODUCE",
       out_n["IS_A_VERIFICATION"] is False
       and out_n["day_set"]["unrecoverable_days_carrying_a_number"]
       == ["20260901"]
       and "day_set.unrecoverable_day_carries_a_number"
       in out_n["mismatched_fields"],
       f"09-01 carries a number -> flagged; unrecoverable set "
       f"{out_n['day_set']['read_but_unrecoverable_from_the_pins']}")
    ck("AND THE POSITIVE CONTROL: with 09-01 and 09-02 carrying NO numbers "
       "the day-set check ADMITS -- it flags a number, not the days' mere "
       "presence in the pins",
       out["day_set"]["no_unrecoverable_day_carries_a_number"] is True
       and out["day_set"]["read_but_unrecoverable_from_the_pins"]
       == ["20260901", "20260902"],
       f"{out['day_set']['read_but_unrecoverable_from_the_pins']} present in "
       f"the pins, 0 of them numbered")

    # -- 5b. REV 49 section 3.3: SILENCE MUST NOT PASS --------------------
    silent = _synthetic_read_artifact(td, mine_days, name="art_silent.json")
    out_sil = verify_real_read(str(silent), str(pins_p))
    ck("REV 49 section 3.3 CLOSED -- SILENCE NO LONGER PASSES: an artifact "
       "that never MENTIONS 09-01/02 is FLAGGED BY NAME. It used to VERIFY, "
       "because the check only looked for a NUMBER against those days and "
       "found none. ***Absence read as compliance -- the same failure as a "
       "zero from a check that never ran***",
       out_sil["IS_A_VERIFICATION"] is False
       and out_sil["day_set"]["days_not_said_as_the_pins_expect"]
       == ["20260901", "20260902"]
       and all(out_sil["day_set"]["every_day_in_the_pins_must_be_SAID"][d][
           "status"] == "ABSENT" for d in ("20260901", "20260902"))
       and "day_set.20260901.ABSENT" in out_sil["mismatched_fields"],
       f"the artifact says nothing about "
       f"{out_sil['day_set']['days_not_said_as_the_pins_expect']} -> each "
       f"ABSENT and flagged by name")
    ck("AND THE POSITIVE CONTROL: an artifact that SAYS "
       "READ_BUT_UNRECOVERABLE for both, without a number, is accepted -- so "
       "the check demands a STATEMENT and not a particular silence",
       out["day_set"]["n_days_said_as_the_pins_expect"] == 5
       and all(out["day_set"]["every_day_in_the_pins_must_be_SAID"][d][
           "status"] == "READ_BUT_UNRECOVERABLE"
           for d in ("20260901", "20260902"))
       and out["IS_A_VERIFICATION"] is True,
       f"{out['day_set']['n_days_said_as_the_pins_expect']} of "
       f"{out['day_set']['n_days_in_the_pins']} days said as the pins "
       f"expect; the three readable ones carry numbers and the two "
       f"unrecoverable ones carry a status")

    # -- 5c. REV 49 section 3.4: the informative refusal is REACHABLE -----
    import shutil
    d3 = td / "absentfeed"
    d3.mkdir(exist_ok=True)
    shutil.copy(pins_p, d3 / "pins.json")
    pj = json.loads((d3 / "pins.json").read_text())
    gone_day = sorted(feeds)[0]
    pj["per_day"][gone_day]["path"] = str(d3 / "not_here.jsonl")
    (d3 / "pins.json").write_text(json.dumps(pj))
    why_abs = ""
    try:
        verify_real_read(str(art_p), str(d3 / "pins.json"))
    except RaceVerifyRefused as e:
        why_abs = str(e)
    except Exception as e:                                    # noqa: BLE001
        why_abs = f"GENERIC {type(e).__name__}: {e}"
    ck("REV 49 section 3.4 CLOSED -- THE INFORMATIVE REFUSAL IS REACHABLE: a "
       "pinned feed that is not on disk refuses as PINNED_FEED_ABSENT, "
       "naming the disagreement between the pin and the ledger. It used to "
       "surface as a bare FileNotFoundError from inside the parser -- a "
       "generic refusal standing in front of the one that says what is wrong",
       "PINNED_FEED_ABSENT" in why_abs and "GENERIC" not in why_abs
       and "not on disk" in why_abs,
       f"'{why_abs[:112]}...'")

    # -- 5d. REV 49 section 3.4 AS FILED: DRIVEN FROM THE CLI ------------
    import subprocess as _sp
    MOD = str(Path(__file__).resolve())
    def _cli(*a):
        r = _sp.run([sys.executable, MOD, "--real", *a],
                    capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr)
    rc_pins, out_pins = _cli("--read-artifact", str(art_p),
                             "--pins", str(td / "no_such_pins.json"))
    bad_json = td / "bad_pins.json"
    bad_json.write_text("{ not json")
    rc_bad, out_bad = _cli("--read-artifact", str(art_p),
                           "--pins", str(bad_json))
    empty_pins = td / "empty_pins.json"
    empty_pins.write_text(json.dumps({"per_day": {}}))
    rc_empty, out_empty = _cli("--read-artifact", str(art_p),
                               "--pins", str(empty_pins))
    rc_ok, out_ok = _cli("--read-artifact", str(art_p), "--pins", str(pins_p))
    ck("REV 49 section 3.4 CLOSED AS FILED -- AND DRIVEN FROM THE CLI, which "
       "is where the coordinator meets it. An absent PINS file died with a "
       "bare FileNotFoundError traceback: a generic refusal standing in "
       "front of an informative one. ***Round 73 closed a DIFFERENT case "
       "(the pinned feed inside `verify_real_read`) and reported it as this "
       "one.*** Every CLI input now refuses BY NAME with exit 2",
       rc_pins == 2 and "PINS_DECLARATION_ABSENT" in out_pins
       and "Traceback" not in out_pins
       and rc_bad == 2 and "PINS_DECLARATION_UNREADABLE" in out_bad
       and "Traceback" not in out_bad
       and rc_empty == 2 and "PINS_DECLARATION_CARRIES_NO_DAYS" in out_empty,
       f"absent pins -> rc {rc_pins} PINS_DECLARATION_ABSENT; unreadable -> "
       f"rc {rc_bad}; empty -> rc {rc_empty}; no traceback in any")
    ck("AND THE POSITIVE CONTROL FROM THE SAME CLI: a good invocation still "
       "runs and VERIFIES with exit 0 -- the by-name refusals are reachable "
       "without making the path a wall",
       rc_ok == 0 and "VERIFIED" in out_ok and "Traceback" not in out_ok,
       f"rc {rc_ok}: '{out_ok.strip().splitlines()[-1][:88]}'")

    # -- 5e. REV 52 section 2.5: THE PIN IS JUDGED BEFORE THE SHAPE ------
    #: THE REVIEWER'S EXACT DRIVE: a PRESENT file whose bytes do not match
    #: the pin, and which is ALSO one-arm. Before the fix the shape error
    #: aborted the pass and the day was reported ONE-ARM -- an answer about
    #: its contents when the true fact is that it is not the pinned file.
    d5 = td / "movedfeed"
    d5.mkdir(exist_ok=True)
    day5 = sorted(feeds)[0]
    moved_rows = [_row(f"btc-updown-5m-W{g}", "UP", g, 1000 + g, 0.9, 0.5,
                       1.0) for g in range(4)]
    moved_feed = write_feed(d5, moved_rows, one_arm=True)
    why_moved = ""
    try:
        da_day_from_pinned_feed(moved_feed, pins[day5]["sha256"])
    except RaceVerifyRefused as e:
        why_moved = str(e)
    ck("REV 52 section 2.5 CLOSED -- THE PIN IS JUDGED FIRST, ON A COMPLETED "
       "HASHING PASS. A present file that is BOTH one-arm AND not the pinned "
       "bytes is now reported as a PIN MISMATCH, not as ONE-ARM: the shape "
       "error is DEFERRED through the pass and raised only if the bytes were "
       "the right ones. ***Reporting ONE-ARM there answers about the file's "
       "CONTENTS when the true fact is that it is not the pinned file at "
       "all***",
       "THE PIN DOES NOT HOLD" in why_moved
       and "ONE-ARM" not in why_moved
       and "Nothing parsed here was used" in why_moved,
       f"a one-arm file at the wrong digest -> '{why_moved[:96]}...'")
    #: and the shape refusal is still REACHABLE when the bytes ARE pinned.
    one_arm_pinned = write_feed(td / "oap" if (td / "oap").mkdir(
        exist_ok=True) is None else (td / "oap"), moved_rows, one_arm=True)
    why_shape = ""
    try:
        da_day_from_pinned_feed(
            one_arm_pinned,
            hashlib.sha256(Path(one_arm_pinned).read_bytes()).hexdigest())
    except RaceVerifyRefused as e:
        why_shape = str(e)
    ck("AND THE SHAPE REFUSAL IS STILL REACHABLE WHEN THE BYTES ARE THE "
       "PINNED ONES: deferring it did not remove it -- a one-arm feed at its "
       "OWN digest still refuses BY NAME on `score_incumbent`",
       "score_incumbent" in why_shape and "ONE-ARM" in why_shape
       and "THE PIN DOES NOT HOLD" not in why_shape,
       f"a one-arm file at its own digest -> '{why_shape[:88]}...'")

    # -- 6. THE FLOORS, re-derived ----------------------------------------
    f3 = da_floors(3)
    ck("THE FLOOR IS RE-DERIVED, NOT QUOTED: at G = 3 and multiplicity 2 the "
       "best possible adjusted p is 2/2^3 = 0.25, which does NOT clear 0.05 "
       "-- and the artifact's own value is compared against it exactly",
       f3["best_possible_adjusted_p"] == 0.25
       and f3["clears_0_05"] is False
       and out["permutation_floor_recomputed"]["best_possible_adjusted_p"]
       == 0.25
       and "permutation_floor.best_possible_adjusted_p"
       not in out["mismatched_fields"],
       f"G=3, m=2 -> {f3['best_possible_adjusted_p']}, clears 0.05: "
       f"{f3['clears_0_05']}")
    bad_floor = json.loads(art_p.read_text())
    bad_floor["permutation_floors"]["resolved_best_possible_adjusted_p"] = 0.04
    bfp = td / "art_floor.json"
    bfp.write_text(json.dumps(bad_floor, default=str))
    out_f = verify_real_read(str(bfp), str(pins_p))
    ck("KNOWN-BAD: AN ARTIFACT CLAIMING A FLOOR THAT CLEARS 0.05 IS FLAGGED "
       "-- the floor is arithmetic on G and the multiplicity, so a claim "
       "that it clears is a claim the arithmetic refutes",
       out_f["IS_A_VERIFICATION"] is False
       and "permutation_floor.best_possible_adjusted_p"
       in out_f["mismatched_fields"],
       f"artifact says 0.04, recomputed 0.25 -> "
       f"{[m for m in out_f['mismatched_fields'] if 'floor' in m]}")

    # -- 7. ONE PASS: the hash covers exactly the bytes parsed ------------
    day0 = sorted(feeds)[0]
    parsed = da_hash_and_parse_feed(feeds[day0], LATENCY_MS)
    on_disk = hashlib.sha256(Path(feeds[day0]).read_bytes()).hexdigest()
    ck("ONE PASS: THE DIGEST COVERS EXACTLY THE BYTES PARSED. Hashing in one "
       "open and parsing in another leaves a window in which both readings "
       "are true of DIFFERENT files; here the digest accumulates over the "
       "very lines the parser consumed, and the byte count matches the file",
       parsed["parsed_stream_sha256"] == on_disk
       and parsed["parsed_stream_bytes"] == Path(feeds[day0]).stat().st_size,
       f"parsed {parsed['parsed_stream_bytes']} bytes hashing to "
       f"{parsed['parsed_stream_sha256'][:16]}, file "
       f"{Path(feeds[day0]).stat().st_size} bytes hashing to "
       f"{on_disk[:16]}")

    # -- 8. THE SECOND OPENING IS STATED ----------------------------------
    so = out["the_feeds_are_opened_a_SECOND_time"]
    ck("THE SECOND OPENING IS STATED, NOT GLOSSED: this verifier DOES open "
       "the feeds again after the read. Under rule 11 that consumes nothing "
       "further -- the days are already READ -- and what it may not do is "
       "open a day the read did not, which is CHECKED against the "
       "artifact's own day list",
       so["answer"] is True and so["n_days_opened_by_this_verifier"] == 3
       and so["opened_no_day_the_read_did_not"] is True,
       f"{so['n_days_opened_by_this_verifier']} days opened again "
       f"({so['days_opened']}); opened no day the read did not: "
       f"{so['opened_no_day_the_read_did_not']}")

    # -- 9. a foreign artifact shape is not silently accepted -------------
    foreign = td / "art_foreign.json"
    fb = json.loads(art_p.read_text())
    fb["protocol"] = "SOMETHING_ELSE_V9"
    foreign.write_text(json.dumps(fb, default=str))
    out_x = verify_real_read(str(foreign), str(pins_p))
    ck("AN ARTIFACT THAT IS NOT BE's DECLARED SHAPE IS NOT SILENTLY "
       "ACCEPTED: the protocol string is checked and a foreign one cannot "
       "read as a verification, however well its numbers happen to agree",
       out_x["IS_A_VERIFICATION"] is False
       and out_x["the_gate_is_the_artifacts_existence"][
           "is_BEs_declared_shape"] is False
       and out_x["n_mismatches"] == 0,
       f"protocol {fb['protocol']} -> is_BEs_declared_shape False, and note "
       f"{out_x['n_mismatches']} numeric mismatches: the numbers agreed and "
       f"it still is not a verification")


    # -- DA 100: THE DECLARATION, THE MARKERS, AND THE READ ORDER --------
    _real_head = race_declaration_head()
    ck("DA 100 (2) -- THE DECLARATION IS THE OTHER HALF OF THE PIN, AND "
       "IT IS RESOLVED AS A CHAIN HEAD, NEVER AS A FILENAME. ***The pins "
       "say WHICH BYTES each day was read from; the declaration says "
       "WHICH DAYS may be read at all and what G is*** -- a verifier "
       "checking only the pins would admit a read of the right bytes over "
       "the wrong day set. At HEAD it resolves to one head by the R-608 "
       "pair, and its population and G are READ from it rather than typed "
       "here",
       _real_head["name"] == "be_race_read_declaration_v4.json"
       and _real_head["sha256"].startswith("a741b4d6b5ac7f59")
       and _real_head["G_declared"] == 3
       and _real_head["READABLE"] == ["20260903", "20260904", "20260905"],
       f"{_real_head['name']} {_real_head['sha256'][:16]}, G="
       f"{_real_head['G_declared']}, READABLE {_real_head['READABLE']}, "
       f"{_real_head['n_members']} members in the family")
    _wrongd = Path(tempfile.mkdtemp(prefix="da100wd_"))
    _bad_art = _synthetic_read_artifact(td, mine_days,
                                        name="art_wrong_decl.json",
                                        decl_sha="9" * 64)
    try:
        verify_real_read(str(_bad_art), str(pins_p))
        _wd = "ADMITTED"
    except RaceVerifyRefused as _e:
        _wd = str(_e).split(" — ")[0].replace("REFUSED: ", "")
    ck("AND A READ TAKEN UNDER ANOTHER DECLARATION IS REFUSED BY NAME: "
       "the artifact's `pre_state.declaration_sha256` against the chain "
       "head's digest. ***A read under another declaration is a read of "
       "another day set***, and this verifier will not reconcile the two",
       _wd == "READ_ARTIFACT_DECLARATION_DIGEST_DIFFERS",
       f"a planted declaration digest -> {_wd}")
    _md = Path(tempfile.mkdtemp(prefix="da100mk_"))
    _art4 = _synthetic_read_artifact(_md, mine_days, extra_marker="20260906")
    _o4 = verify_real_read(str(_art4), str(pins_p))
    _md3 = Path(tempfile.mkdtemp(prefix="da100mk3_"))
    _art0 = _synthetic_read_artifact(_md3, mine_days, markers=False)
    _o0 = verify_real_read(str(_art0), str(pins_p))
    ck("DA 100 (3) -- ONE OPENED MARKER PER DECLARED DAY, AND NO FOURTH. "
       "***The marker is the ONLY record that a day was spent***, so a "
       "fourth means a day was opened that the declaration does not name "
       "-- the one thing rule 11 cannot absorb after the fact. Driven "
       "both ways: a fourth marker FLAGS and names the extra day; no "
       "markers at all FLAGS and names the declared days that lack one",
       _o4["opened_markers"]["extra_days_opened"] == ["20260906"]
       and _o4["IS_A_VERIFICATION"] is False
       and _o0["opened_markers"]["declared_days_without_a_marker"]
       == _real_head["READABLE"]
       and _o0["IS_A_VERIFICATION"] is False,
       f"a fourth marker -> extra {_o4['opened_markers']['extra_days_opened']}"
       f", verified {_o4['IS_A_VERIFICATION']}; none -> missing "
       f"{len(_o0['opened_markers']['declared_days_without_a_marker'])}, "
       f"verified {_o0['IS_A_VERIFICATION']}")
    _good = verify_real_read(str(art_p), str(pins_p))
    _mine_numbers = set()
    for _d in _real_head["READABLE"]:
        _pin = json.loads(Path(pins_p).read_text())["per_day"][_d]
        _m = da_day_from_pinned_feed(_pin["path"], _pin["sha256"])
        _mine_numbers.add(round(float(_m["day_increment_cents"]), 9))
        for _c, _b in _m["per_coin"].items():
            _mine_numbers.add(round(float(
                _b["MATCHED_VOLUME_increment_cents"]), 9))
    _mine_numbers = {x for x in _mine_numbers if abs(x) > 1e-9}
    _emitted = {round(float(v), 9) for _q, v in _leaves(_good)
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    _echoed = sorted(_mine_numbers & _emitted)
    _has_values = [c for c in _good["checks"]
                   if "mine" in c or "artifact" in c]
    ck("DA 100 (4) -- ***THE READ ORDER: THIS RECORD COMES FIRST, SO A "
       "NUMBER QUOTED HERE IS A NUMBER PUBLISHED AHEAD OF THE READ THAT "
       "IS ENTITLED TO PUBLISH IT.*** The verdict is by PREDICATE and by "
       "NAME: every comparison is MATCH / MISMATCH / ABSENT_IN_ARTIFACT on "
       "a field NAME, and no MATCHED_VOLUME, day increment, permutation p "
       "or direction appears in this verifier's own output. Computed, not "
       "promised: every recomputed quantity is searched for as a numeric "
       "leaf of the emission",
       not _echoed and not _has_values
       and _good["IS_A_VERIFICATION"] is True,
       f"{len(_mine_numbers)} recomputed quantities, {len(_echoed)} of "
       f"them present in my own output; {len(_has_values)} of "
       f"{len(_good['checks'])} check rows carry a value")
    ck("AND THE RECORD SAYS WHAT IT DOES NOT VERIFY, NOT ONLY WHAT IT "
       "DOES: the marker ORDER (presence is the fact, a timestamp is not "
       "evidence of order), a feed replaced with bytes matching its pin, "
       "the direction and the permutation p, and BE's own "
       "`decl_is_the_chain_head` claim -- ***recorded as BE's claim and "
       "re-resolved here independently, never taken***",
       len(_good["what_this_verifier_DOES_NOT_VERIFY"]) >= 4
       and _good["the_declaration"][
           "and_this_verifier_resolved_the_head_ITSELF"] is True,
       f"{len(_good['what_this_verifier_VERIFIES'])} verified / "
       f"{len(_good['what_this_verifier_DOES_NOT_VERIFY'])} explicitly "
       f"not")

    return checks


if __name__ == "__main__":
    raise SystemExit(main())
