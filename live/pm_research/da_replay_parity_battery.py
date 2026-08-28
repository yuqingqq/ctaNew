"""Seven-arm replay parity battery, run against TYPED STUBS and, through a
declared data contract, against EXTERNAL (BE-produced) trajectories.

AUTHORISATION: hazard plan §10 item 6 / §10.1 ("the common replay harness may
be developed against typed stub outputs"); design in
`plans/LANE4_REPLAY_PARITY_STUB_BATTERY.md` (`6fc96e2`). Nothing here is scored
and no stub is ever a candidate (TODO §10).

WHAT THIS IS. DA builds the CHECKER; BE's replay arms are the CHECKED. The two
stay separate implementations on purpose (R-235 do-not-harmonize): a checker
that shares code with the thing it checks agrees with it by construction. The
external interface below is therefore a DATA contract and imports nothing from
BE -- the moment it imported BE's serializer, agreement would stop being
evidence.

WHY STUBS FIRST. Every arm returns declared-shape output with NO model behind
it, so the battery must observe ZERO difference. If arms differ while every
predictor is inert, the difference is the HARNESS, and any later result would
inherit it invisibly. This programme has already seen path-coupled overlays
amplify prediction noise 10-20x and produce large replay deltas with zero
ranking improvement -- a battery that cannot first demonstrate zero difference
under zero signal cannot attribute a later difference to signal.

HARDENING ROUND 2 (Codex batch 2 item 5). Five findings, and what each cost:
  * `matched_control` took a `cancels` argument and IGNORED it -- 0, 1, 6 and
    99 all returned 12. See `matched_control`: the fix is not to honour the
    argument but to DELETE it, and to implement the real draw underneath.
  * `battery()` returned two anchors and called that a run. It now emits an
    EVALUATED receipt over an enumerated required set, and refuses when a
    required check is ABSENT rather than reporting the conjunction of what
    happened to be present.
  * no zero-repost anchor: an arm that stops trading has no adverse fills and
    would have WON a harm-share comparison.
  * no rate-limit accounting: a cancel that was REQUESTED but SUPPRESSED never
    bound, and counting it as prevented harm inflates the estimand.
  * no training-reuse guard (rule 11).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Canonical trajectory. "Bit-identical" is undefined without a canonical form:
# two correct harnesses could serialise the same trajectory differently and
# every comparison would fail, or -- worse -- differ in a way a tolerance hides.
# Same recipe as annotation_canon_v1, for the same reason.
CANON = "replay_traj_canon_v1"

EVENT_FIELDS = ("t", "seq", "kind", "slug", "side", "gen", "qty", "price",
                "note")

KINDS = (
    "PLACE",
    "PLACE_WITHHELD",        # rule 4: a withheld quote is a STATUS, not a
                             # silent absence -- see permanent_hold_anchor
    "CANCEL_REQUESTED",
    "CANCEL_EFFECTIVE",
    "CANCEL_SUPPRESSED",     # requested, refused by the limiter, NEVER bound
    "FILL",
    "FILL_STALE",
)


class ParityRefused(Exception):
    """Raised instead of returning a degraded answer. A battery that returns
    a number it cannot stand behind is worse than one that stops."""


@dataclass(frozen=True)
class Event:
    """One trajectory event. Ordered by (t, seq) -- never by dict order."""
    t: float
    seq: int
    kind: str
    slug: str
    side: str
    gen: int
    qty: float = 0.0
    price: float | None = None
    note: str = ""


@dataclass
class Trajectory:
    arm: str
    events: list[Event] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.events.append(Event(seq=len(self.events), **kw))

    def canonical_bytes(self) -> bytes:
        """Byte form the parity comparison is defined over.

        The ARM NAME IS EXCLUDED. Two arms are compared on what they DID; if
        the name were included every arm would trivially differ and the anchor
        could never fail, which is the failure mode a decorative anchor has.
        """
        payload = [
            {"t": e.t, "seq": e.seq, "kind": e.kind, "slug": e.slug,
             "side": e.side, "gen": e.gen, "qty": e.qty, "price": e.price,
             "note": e.note}
            for e in sorted(self.events, key=lambda e: (e.t, e.seq))
        ]
        # FLOATS CARRY NO TOLERANCE, verified: 1.0 vs 1.0+1ULP and 0.1+0.2 vs
        # 0.3 produce different digests, so canonicalization removes
        # REPRESENTATION noise only and R-236's bit-identical requirement
        # survives it.
        #
        # ONE EDGE, recorded rather than silently normalised: 0.0 and -0.0 are
        # IEEE-equal but distinct bit patterns and serialize as "0.0" vs
        # "-0.0", so they DIFFER here. That is correct under
        # bit-identical-no-tolerance -- but two harnesses computing the same
        # zero by different routes would report a difference that is not one.
        # NOT normalised, because normalising it would be a tolerance by
        # another name, and the anchor exists to catch exactly the couplings a
        # tolerance hides. If it ever fires on real arms, it is a REAL
        # signed-zero difference in one of them and worth the investigation.
        return json.dumps({"canon": CANON, "events": payload},
                          sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def counts(self) -> dict[str, int]:
        return {k: sum(1 for e in self.events if e.kind == k) for k in KINDS}


# ---------------------------------------------------------------------------
ARMS = (
    "QR_SKEW_ONLY",
    "QR_CANCEL_HOLD_X_SKEW",
    "HAZARD_ONLY_NEUTRAL",
    "CONDVALUE_NEUTRAL",
    "CONDVALUE_X_SKEW",
    "CONDVALUE_X_SKEW_X_FAIRPRICE",
    "RANDOM_MATCHED",
)

CANCEL_EFFECTIVE_LAG_S = 0.050      # declared; a cancel binds only after it
RATE_LIMIT_WINDOW_S = 1.0           # declared limiter bucket
RANDOM_MATCHED_SEED = 20260828      # declared; see budget_matched_selection
PERMANENT_HOLD_WITHHELD_SHARE = 0.25  # >= this share withheld => flagged


def stub_score(slug: str, side: str, gen: int, salt: str = "") -> float:
    """Deterministic pseudo-score in [0,1).

    sha256, NOT the builtin `hash()`. `hash()` of a str is salted by
    PYTHONHASHSEED, so a stub scorer built on it would produce a DIFFERENT
    cancel set per process -- blocker-7's class exactly (a fixed seed over a
    process-dependent order is an independent draw, not a reproduction). The
    determinism check below would then fail, which is the good outcome; the
    bad outcome is a scorer that is only ALMOST unstable.
    """
    h = hashlib.sha256(f"{salt}|{slug}|{side}|{gen}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def stub_opportunities(n: int = 12, dt: float = 10.0) -> list[dict[str, Any]]:
    """Neutral opportunities, identical for every arm. Deterministic by
    construction -- no RNG, so a difference can never be a seed artifact."""
    return [{"slug": f"btc-updown-5m-{1787650200 + 300 * (i // 3)}",
             "side": "BUY_UP" if i % 2 == 0 else "SELL_UP",
             "gen": i, "t": dt * i, "qty": 5.0, "price": 0.50}
            for i in range(n)]


def opp_key(o: dict[str, Any]) -> tuple[str, str, int]:
    return (o["slug"], o["side"], o["gen"])


def run_stub_arm(arm: str, opps: list[dict[str, Any]], *,
                 predictor_enabled: bool = False,
                 cancel_threshold: float = float("inf"),
                 forced_cancel_keys: Iterable[tuple[str, str, int]] | None = None,
                 rate_limit_per_window: int | None = None,
                 hold_after_first_cancel: bool = False,
                 fill_at: float | None = None,
                 score_salt: str = "") -> Trajectory:
    """A typed stub arm. NO MODEL: with the predictor disabled every arm must
    place the same orders and cancel nothing, so all seven trajectories are
    identical by construction -- and the battery's job is to prove the harness
    does not break that.

    THREE LIFECYCLE FACTS the estimand depends on, all represented here:
      * a cancel REQUESTED is not a cancel EFFECTIVE. It binds only after
        CANCEL_EFFECTIVE_LAG_S, and only if the limiter let it through.
      * a cancel SUPPRESSED by the limiter changes NOTHING. The order stays
        exposed and fills normally. Valuing a suppressed request as prevented
        harm is the inflation this arm exists to make visible.
      * a withheld quote is a STATUS (`PLACE_WITHHELD`), never an absence.
        An arm that simply stopped emitting would be indistinguishable from
        an arm that ran out of opportunities.
    """
    tr = Trajectory(arm=arm)
    forced = None if forced_cancel_keys is None else set(forced_cancel_keys)
    requested: set[tuple[str, str, int]] = set()
    effective: set[tuple[str, str, int]] = set()
    per_window: dict[int, int] = {}
    holding = False

    for o in opps:
        key = opp_key(o)

        if holding:
            tr.add(t=o["t"], kind="PLACE_WITHHELD", slug=o["slug"],
                   side=o["side"], gen=o["gen"], qty=o["qty"],
                   price=o["price"],
                   note="withheld after first cancel (permanent hold)")
            continue

        tr.add(t=o["t"], kind="PLACE", slug=o["slug"], side=o["side"],
               gen=o["gen"], qty=o["qty"], price=o["price"])

        if forced is not None:
            want = key in forced
        elif predictor_enabled:
            want = stub_score(o["slug"], o["side"], o["gen"],
                              score_salt) >= cancel_threshold
        else:
            want = False

        if want:
            if key in requested:
                raise ParityRefused(
                    f"REFUSED: generation {key} cancelled twice. One "
                    f"generation may be cancelled at most once.")
            requested.add(key)
            tr.add(t=o["t"], kind="CANCEL_REQUESTED", slug=o["slug"],
                   side=o["side"], gen=o["gen"])

            w = int(o["t"] // RATE_LIMIT_WINDOW_S)
            used = per_window.get(w, 0)
            allowed = (rate_limit_per_window is None
                       or used < rate_limit_per_window)
            if allowed:
                per_window[w] = used + 1
                effective.add(key)
                tr.add(t=o["t"] + CANCEL_EFFECTIVE_LAG_S,
                       kind="CANCEL_EFFECTIVE", slug=o["slug"],
                       side=o["side"], gen=o["gen"])
            else:
                tr.add(t=o["t"], kind="CANCEL_SUPPRESSED", slug=o["slug"],
                       side=o["side"], gen=o["gen"],
                       note=f"rate limit {rate_limit_per_window}/"
                            f"{RATE_LIMIT_WINDOW_S}s reached in window {w}; "
                            f"the order STAYS EXPOSED")
            if hold_after_first_cancel:
                holding = True

        if fill_at is not None:
            ft = o["t"] + fill_at
            eff = o["t"] + CANCEL_EFFECTIVE_LAG_S
            if key in effective and ft >= eff:
                continue          # cancelled orders cannot fill after effect
            stale = key in effective
            tr.add(t=ft, kind=("FILL_STALE" if stale else "FILL"),
                   slug=o["slug"], side=o["side"], gen=o["gen"],
                   qty=o["qty"], price=o["price"],
                   note=("pre-effectiveness fill on a cancelled generation "
                         "is charged as STALE" if stale else ""))
    return tr


# ---------------------------------------------------------------------------
# Matched control (rule 7).
def cell_of(o: dict[str, Any]) -> tuple[str, int]:
    """The MATCHING CELL: side and hour. These are the decision variables the
    ruling names; matching on anything the treatment chose (score, outcome)
    would match away the thing being measured."""
    return (o["side"], int(o["t"] // 3600))


def _cell_seed(cell: tuple[str, int], seed: int) -> int:
    """Per-cell seed derived by sha256, not by hash() -- see stub_score."""
    h = hashlib.sha256(f"{seed}|{cell[0]}|{cell[1]}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def budget_matched_selection(pool: list[dict[str, Any]], budget: int,
                             seed: int) -> list[tuple[str, str, int]]:
    """Draw exactly `budget` generations UNIFORMLY AT RANDOM, without
    replacement, from `pool`.

    THIS IS THE PRIMITIVE THE OLD `matched_control` CLAIMED TO HAVE. It draws;
    it is seeded; and `budget` demonstrably changes the answer (selftests take
    0, 1, 6 and 99 and get 0, 1, 6 and a REFUSAL).

    ORDER FIRST, THEN DRAW. The pool is sorted into a total order before the
    RNG touches it. `random.Random(seed)` is reproducible, but reproducibly
    sampling an unstably-ordered sequence is not -- that is blocker-7's defect
    with a seed bolted on.

    A budget larger than the pool REFUSES. It does not clamp: a control that
    silently drew fewer actions than the treatment is no longer matched on the
    decision variable, and the shortfall would be invisible in the profile it
    reports.
    """
    if budget < 0:
        raise ParityRefused(f"REFUSED: negative budget {budget}")
    if budget > len(pool):
        raise ParityRefused(
            f"REFUSED: budget {budget} exceeds eligible pool {len(pool)}. A "
            f"control cannot be matched on action count by drawing fewer "
            f"actions than the treatment took.")
    ordered = sorted(pool, key=lambda o: (o["t"], o["slug"], o["side"],
                                          o["gen"]))
    idx = random.Random(seed).sample(range(len(ordered)), budget)
    return [opp_key(ordered[i]) for i in sorted(idx)]


def cancel_profile(tr: Trajectory) -> dict[str, Any]:
    """The DECISION-VARIABLE profile: how many cancel REQUESTS, by side and by
    hour. Requests, not effectives -- the action is the request; whether it
    bound is an outcome, and matching on an outcome would match away the
    difference being measured."""
    c = [e for e in tr.events if e.kind == "CANCEL_REQUESTED"]
    by_cell: dict[str, int] = {}
    for e in c:
        k = f"{e.side}|{int(e.t // 3600)}"
        by_cell[k] = by_cell.get(k, 0) + 1
    return {"n": len(c), "by_cell": dict(sorted(by_cell.items()))}


def matched_control(opps: list[dict[str, Any]], *,
                    cancel_threshold: float = 0.5,
                    seed: int = RANDOM_MATCHED_SEED,
                    score_salt: str = "") -> dict[str, Any]:
    """Arm 7 (RANDOM_MATCHED) matched to the treated arm on ACTION COUNT, SIDE
    and HOUR (rule 7), by an actual draw.

    THE DELETED ARGUMENT. This function used to take `cancels: int` and ignore
    it -- 0, 1, 6 and 99 all produced 12 cancels, so every "matched" it ever
    reported was matched by construction and could not have been otherwise.
    The fix is NOT to honour the argument. A matched control's count is
    DETERMINED by the treated arm; a count the caller can choose is one that
    gets chosen after the numbers are visible. The budget knob lives one level
    down, in `budget_matched_selection`, where it is a tested primitive and not
    a policy lever. Same reasoning as the date-predicate granularity in
    `da_forward_day_verify`.

    WHAT THE CONTROL MUST NOT BE. If it selected the same generations as the
    treatment, the profiles would match perfectly and the comparison would be
    vacuous. `selection_differs` is therefore computed and returned, and the
    battery requires it.
    """
    treated = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                           cancel_threshold=cancel_threshold,
                           score_salt=score_salt)
    prof_t = cancel_profile(treated)

    pools: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for o in opps:
        pools.setdefault(cell_of(o), []).append(o)

    picked: list[tuple[str, str, int]] = []
    per_cell: dict[str, dict[str, int]] = {}
    for cell in sorted(pools):
        want = prof_t["by_cell"].get(f"{cell[0]}|{cell[1]}", 0)
        pool = pools[cell]
        sel = budget_matched_selection(pool, want, _cell_seed(cell, seed))
        picked.extend(sel)
        per_cell[f"{cell[0]}|{cell[1]}"] = {"eligible": len(pool),
                                            "budget": want,
                                            "drawn": len(sel)}

    control = run_stub_arm("RANDOM_MATCHED", opps,
                           forced_cancel_keys=picked)
    prof_c = cancel_profile(control)

    treated_keys = {(e.slug, e.side, e.gen) for e in treated.events
                    if e.kind == "CANCEL_REQUESTED"}
    control_keys = set(picked)
    return {"treated": prof_t, "control": prof_c, "per_cell": per_cell,
            "matched": prof_t == prof_c,
            "non_empty": prof_t["n"] > 0,
            "strict_subset": 0 < prof_t["n"] < len(opps),
            "selection_differs": treated_keys != control_keys,
            "seed": seed,
            "pass": (prof_t == prof_c and prof_t["n"] > 0
                     and 0 < prof_t["n"] < len(opps)
                     and treated_keys != control_keys)}


# ---------------------------------------------------------------------------
# Anchors.
def anchor_parity(opps: list[dict[str, Any]]) -> dict[str, Any]:
    """THE ANCHOR: with the predictor DISABLED every arm is BIT-IDENTICAL to
    QR_SKEW_ONLY. Bit-identical, not within-tolerance -- a tolerance would hide
    exactly the coupling this exists to find, and the summation-order finding
    shows ~1e-11 movement on identical terms, so "close" cannot be
    distinguished from "differently ordered but wrong".
    """
    base = run_stub_arm("QR_SKEW_ONLY", opps).digest()
    per = {a: run_stub_arm(a, opps).digest() for a in ARMS}
    diff = sorted(a for a, d in per.items() if d != base)
    return {"baseline_digest": base, "per_arm": per,
            "arms_differing": diff, "bit_identical": not diff,
            "n_arms": len(ARMS), "pass": (not diff) and len(ARMS) == 7}


def infinite_threshold_parity(opps) -> dict[str, Any]:
    """An INFINITE cancel threshold cancels nothing, so an arm with its
    predictor ENABLED must still be bit-identical to QR_SKEW_ONLY."""
    base = run_stub_arm("QR_SKEW_ONLY", opps).digest()
    got = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                       cancel_threshold=float("inf")).digest()
    return {"baseline_digest": base, "digest": got,
            "bit_identical": got == base, "pass": got == base}


def per_window_effective(tr: Trajectory) -> dict[int, int]:
    """Effective cancels per limiter window, bucketed on the REQUEST time --
    the limiter admits or refuses at request, and CANCEL_EFFECTIVE lands a lag
    later, so bucketing the effective event would smear counts across the
    window boundary and report a limit that was never applied."""
    req_t = {(e.slug, e.side, e.gen): e.t for e in tr.events
             if e.kind == "CANCEL_REQUESTED"}
    out: dict[int, int] = {}
    for e in tr.events:
        if e.kind != "CANCEL_EFFECTIVE":
            continue
        t = req_t.get((e.slug, e.side, e.gen))
        if t is None:
            continue                      # caught by external_lifecycle
        w = int(t // RATE_LIMIT_WINDOW_S)
        out[w] = out.get(w, 0) + 1
    return out


def rate_limit_accounting(opps: list[dict[str, Any]], *,
                          limit: int = 3,
                          cancel_threshold: float = 0.0) -> dict[str, Any]:
    """REQUESTED = EFFECTIVE + SUPPRESSED, as an identity the code evaluates.

    Why this is not bookkeeping. The estimand values cancels that PREVENTED an
    adverse fill. A request the venue never honoured prevented nothing, and the
    trajectory that contains it looks -- to any counter that reads
    CANCEL_REQUESTED -- exactly like one that did. Under a limit of 0 the
    SUPPRESSED-only arm must therefore be identical IN ITS FILLS to the arm
    that never cancelled at all; if it is not, suppression is being credited
    somewhere.
    """
    dense = [dict(o, t=o["t"] * 0.005) for o in opps]   # crowd one window
    tr = run_stub_arm("CONDVALUE_X_SKEW", dense, predictor_enabled=True,
                      cancel_threshold=cancel_threshold,
                      rate_limit_per_window=limit, fill_at=0.200)
    c = tr.counts()
    req, eff, sup = (c["CANCEL_REQUESTED"], c["CANCEL_EFFECTIVE"],
                     c["CANCEL_SUPPRESSED"])

    # limit 0: every request suppressed; fills must equal the no-cancel arm's
    zero = run_stub_arm("CONDVALUE_X_SKEW", dense, predictor_enabled=True,
                        cancel_threshold=cancel_threshold,
                        rate_limit_per_window=0, fill_at=0.200)
    never = run_stub_arm("QR_SKEW_ONLY", dense, fill_at=0.200)

    def fills(t):
        return sorted((e.t, e.kind, e.slug, e.side, e.gen)
                      for e in t.events if e.kind in ("FILL", "FILL_STALE"))

    zc = zero.counts()
    # EXACT, not an upper bound. The first version of this field multiplied the
    # limit by the number of windows spanned and was not in `pass` at all -- a
    # loose bound beside a verdict it does not enter is decoration, and this
    # programme has had a hardcoded verdict contradict its own table three
    # times.
    pw = per_window_effective(tr)
    within = all(v <= limit for v in pw.values())
    return {"limit": limit, "requested": req, "effective": eff,
            "suppressed": sup,
            "identity_holds": req == eff + sup,
            "limiter_bound": sup > 0 and eff > 0,
            "per_window_effective": dict(sorted(pw.items())),
            "max_in_any_window": max(pw.values()) if pw else 0,
            "per_window_within_limit": within,
            "zero_limit_all_suppressed": (zc["CANCEL_EFFECTIVE"] == 0
                                          and zc["CANCEL_SUPPRESSED"] > 0),
            "zero_limit_fills_match_no_cancel": fills(zero) == fills(never),
            "pass": (req == eff + sup and sup > 0 and eff > 0 and within
                     and zc["CANCEL_EFFECTIVE"] == 0
                     and zc["CANCEL_SUPPRESSED"] > 0
                     and fills(zero) == fills(never))}


def permanent_hold_anchor(opps: list[dict[str, Any]], *,
                          cancel_threshold: float = 0.5) -> dict[str, Any]:
    """THE ZERO-REPOST ANCHOR: an arm that stops quoting must be FLAGGED, not
    crowned.

    An arm which cancels once and never reposts has no further adverse fills
    -- and no further fills at all. On harm share, or adverse-per-fill, or
    rho, it wins by NOT TRADING. That is the degenerate policy the whole
    cancel line is exposed to, and it is invisible to any metric normalised by
    activity. The predicate is exposure, not harm: withheld share against the
    declared PERMANENT_HOLD_WITHHELD_SHARE, with a NORMAL arm as the positive
    control so the flag is not simply what this always reports.
    """
    def prof(tr: Trajectory) -> dict[str, Any]:
        c = tr.counts()
        n = c["PLACE"] + c["PLACE_WITHHELD"]
        share = (c["PLACE_WITHHELD"] / n) if n else None
        return {"placed": c["PLACE"], "withheld": c["PLACE_WITHHELD"],
                "opportunities": n, "withheld_share": share,
                "fills": c["FILL"] + c["FILL_STALE"],
                "flagged": bool(n) and share is not None
                           and share >= PERMANENT_HOLD_WITHHELD_SHARE}

    normal = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                          cancel_threshold=cancel_threshold, fill_at=0.010)
    holder = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                          cancel_threshold=cancel_threshold,
                          hold_after_first_cancel=True, fill_at=0.010)
    pn, ph = prof(normal), prof(holder)
    return {"normal": pn, "permanent_hold": ph,
            "threshold": PERMANENT_HOLD_WITHHELD_SHARE,
            "holder_flagged": ph["flagged"],
            "normal_not_flagged": not pn["flagged"],
            "holder_trades_less": ph["fills"] < pn["fills"],
            "pass": ph["flagged"] and not pn["flagged"]
                    and ph["fills"] < pn["fills"]}


def determinism_across_hashseed(script_dir: str) -> dict[str, Any]:
    """Two interpreters under DIFFERENT PYTHONHASHSEED must produce
    byte-identical trajectories AND an identical matched-control draw.
    Blocker-7's class was exactly this -- a fixed RNG seed over a
    process-dependent iteration order is an independent draw, not a
    reproduction -- so the battery must not inherit it. The matched control is
    included BECAUSE it is the only path here that touches an RNG."""
    import subprocess, sys, os
    prog = ("import sys,json;sys.path.insert(0,%r)\n"
            "import da_replay_parity_battery as B\n"
            "o=B.stub_opportunities()\n"
            "d=B.run_stub_arm('CONDVALUE_X_SKEW',o,predictor_enabled=True,"
            "cancel_threshold=0.5,fill_at=0.02).digest()\n"
            "m=B.matched_control(o)\n"
            "print(json.dumps([d,m['control'],m['per_cell']],sort_keys=True))"
            % script_dir)
    outs = []
    for hs in ("0", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", prog], env=env,
                                   capture_output=True, text=True).stdout.strip())
    same = bool(outs[0]) and outs[0] == outs[1]
    return {"outputs": outs, "identical": same, "pass": same}


# ---------------------------------------------------------------------------
# Rule 11: the training-reuse guard.
def assert_no_training_reuse(train_days: Iterable[str],
                             score_days: Iterable[str]) -> dict[str, Any]:
    """Days used to FIT may not be days used to SCORE (rule 11), and scoring
    days must be strictly LATER than every fitting day.

    THE VACUITY REFUSAL IS THE POINT. Two empty sets are disjoint, so a guard
    that only tested `train & score == set()` would pass loudest exactly when
    it had nothing to check -- this programme has already shipped one
    invariance check that compared two empty files and printed IDENTICAL.
    Empty on either side therefore REFUSES.
    """
    tr, sc = sorted(set(train_days)), sorted(set(score_days))
    if not tr or not sc:
        raise ParityRefused(
            f"REFUSED: training-reuse guard is VACUOUS with train={len(tr)} "
            f"score={len(sc)} days. Two empty sets are disjoint; that is not "
            f"evidence of separation.")
    overlap = sorted(set(tr) & set(sc))
    if overlap:
        raise ParityRefused(
            f"REFUSED: {len(overlap)} day(s) used to FIT are also used to "
            f"SCORE: {overlap[:5]}. Choosing after seeing voids the test.")
    if min(sc) <= max(tr):
        raise ParityRefused(
            f"REFUSED: scoring day {min(sc)} is not strictly after the last "
            f"fitting day {max(tr)}. Disjoint is not enough -- validation is "
            f"LATER untouched days, not merely other days.")
    return {"train_days": tr, "score_days": sc, "n_train": len(tr),
            "n_score": len(sc), "overlap": [], "strictly_later": True,
            "pass": True}


# ---------------------------------------------------------------------------
# External arms: the data contract BE-produced trajectories arrive under.
def load_external_trajectory(obj: dict[str, Any]) -> Trajectory:
    """Load a trajectory produced by ANOTHER implementation.

    NOTHING IS DEFAULTED. Every field of every event must be present; an
    absent field REFUSES rather than being filled in here. A checker that
    supplies what the producer failed to produce is testing its own defaults,
    which is the fixture-supplies-the-answer defect this programme has already
    paid for twice.

    UNKNOWN FIELDS ALSO REFUSE. Silently ignoring an extra field would mean
    the digest is computed over a PROJECTION of what the producer actually
    did, and parity would then be asserted on a proxy.

    NO IMPORT OF BE'S CODE (R-235). This reads bytes and rebuilds the event
    list under DA's own canon; if it called BE's serializer the two would
    agree by construction.
    """
    if not isinstance(obj, dict):
        raise ParityRefused("REFUSED: external trajectory is not an object")
    canon = obj.get("canon")
    if canon != CANON:
        raise ParityRefused(
            f"REFUSED: external canon {canon!r} != {CANON!r}. Trajectories "
            f"serialized under different canonical forms cannot be "
            f"byte-compared; a matching digest would be luck and a differing "
            f"one would be uninformative.")
    arm = obj.get("arm")
    if arm not in ARMS:
        raise ParityRefused(f"REFUSED: unknown arm {arm!r}; declared: {ARMS}")
    events = obj.get("events")
    if not isinstance(events, list) or not events:
        raise ParityRefused(
            f"REFUSED: external arm {arm!r} carries no events. An empty "
            f"trajectory trivially matches nothing and must not be scored as "
            f"agreement.")
    tr = Trajectory(arm=arm)
    seen_seq: set[int] = set()
    for i, e in enumerate(events):
        if not isinstance(e, dict):
            raise ParityRefused(f"REFUSED: event {i} is not an object")
        missing = [f for f in EVENT_FIELDS if f not in e]
        if missing:
            raise ParityRefused(
                f"REFUSED: event {i} is MISSING {missing}. Absent fields are "
                f"not defaulted here.")
        extra = [k for k in e if k not in EVENT_FIELDS]
        if extra:
            raise ParityRefused(
                f"REFUSED: event {i} carries UNDECLARED fields {extra}; the "
                f"digest would be taken over a projection of the producer's "
                f"actual output.")
        if e["kind"] not in KINDS:
            raise ParityRefused(f"REFUSED: event {i} kind {e['kind']!r} is "
                                f"not a declared kind")
        if not isinstance(e["seq"], int) or isinstance(e["seq"], bool):
            raise ParityRefused(f"REFUSED: event {i} seq is not an int")
        if e["seq"] in seen_seq:
            raise ParityRefused(f"REFUSED: duplicate seq {e['seq']}; ordering "
                                f"by (t, seq) would be ambiguous")
        seen_seq.add(e["seq"])
        for num in ("t", "qty"):
            v = e[num]
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or not math.isfinite(float(v)):
                raise ParityRefused(
                    f"REFUSED: event {i} {num}={v!r} is not a finite number")
        if e["price"] is not None and (not isinstance(e["price"], (int, float))
                                       or isinstance(e["price"], bool)
                                       or not math.isfinite(float(e["price"]))):
            raise ParityRefused(f"REFUSED: event {i} price={e['price']!r}")
        tr.events.append(Event(**{f: e[f] for f in EVENT_FIELDS}))
    return tr


def external_lifecycle(tr: Trajectory) -> dict[str, Any]:
    """The lifecycle invariants an EXTERNAL arm must satisfy. These are the
    same facts the stub enforces by construction -- which is why they must be
    CHECKED on anything the stub did not build."""
    ev = sorted(tr.events, key=lambda e: (e.t, e.seq))
    req: dict[tuple, int] = {}
    eff: dict[tuple, float] = {}
    sup: set[tuple] = set()
    dbl: list[tuple] = []
    for e in ev:
        k = (e.slug, e.side, e.gen)
        if e.kind == "CANCEL_REQUESTED":
            req[k] = req.get(k, 0) + 1
            if req[k] > 1:
                dbl.append(k)
        elif e.kind == "CANCEL_EFFECTIVE":
            eff[k] = e.t
        elif e.kind == "CANCEL_SUPPRESSED":
            sup.add(k)
    # BOTH fill kinds. The first version checked only FILL, so a producer
    # could emit FILL_STALE after effectiveness and pass -- and STALE is
    # DEFINED as pre-effectiveness, so that is precisely the mislabel the
    # check exists to catch. After the cancel binds the order is off the book
    # and NOTHING may fill against it, under either label.
    late = [(e.kind, e.slug, e.gen, e.t) for e in ev
            if e.kind in ("FILL", "FILL_STALE")
            and (e.slug, e.side, e.gen) in eff
            and e.t >= eff[(e.slug, e.side, e.gen)]]
    # An OUTCOME without its REQUEST: an effective or suppressed cancel on a
    # generation that was never asked to cancel means the producer's
    # accounting is broken, and requested=effective+suppressed would then be
    # satisfied by two compensating errors. Sorted, because set iteration
    # order over string tuples is PYTHONHASHSEED-dependent.
    unrequested = sorted(k for k in (set(eff) | sup) if k not in req)
    n_req = sum(req.values())
    return {"arm": tr.arm, "n_events": len(ev),
            "requested": n_req, "effective": len(eff), "suppressed": len(sup),
            "identity_holds": n_req == len(eff) + len(sup),
            "no_double_cancel": not dbl,
            "no_fill_after_effective": not late,
            "no_unrequested_outcome": not unrequested,
            "pass": (not dbl and not late and not unrequested
                     and n_req == len(eff) + len(sup))}


def check_external_arms(objs: list[dict[str, Any]],
                        reference: Trajectory | None = None) -> dict[str, Any]:
    """Run the battery's invariants over EXTERNAL trajectories, and (when a
    reference is supplied) the bit-identity anchor against it.

    An EMPTY submission is NOT EVALUABLE. Seven arms agreeing on nothing is
    not seven passing arms -- the same refusal `battery()` makes."""
    if not objs:
        return {"evaluable": False, "n_arms": 0, "pass": False,
                "why": "no external trajectories submitted"}
    out, arms = {}, []
    for o in objs:
        tr = load_external_trajectory(o)
        arms.append(tr.arm)
        r = external_lifecycle(tr)
        r["digest"] = tr.digest()
        if reference is not None:
            r["matches_reference"] = (r["digest"] == reference.digest())
        out[tr.arm] = r
    dup = len(arms) != len(set(arms))
    res = {"evaluable": True, "n_arms": len(out), "per_arm": out,
           "duplicate_arms": dup,
           "lifecycle_pass": (not dup)
                             and all(v["pass"] for v in out.values())}
    # When a reference IS supplied, bit-identity ENTERS the verdict. The first
    # version computed matches_reference per arm and left it out of `pass`, so
    # a submission could report pass=True beside a reference mismatch. When no
    # reference is supplied the key is ABSENT, not True -- an unrun comparison
    # must not read as a passed one (N/A-vacuity).
    if reference is not None:
        res["all_match_reference"] = all(v["matches_reference"]
                                         for v in out.values())
        res["pass"] = res["lifecycle_pass"] and res["all_match_reference"]
    else:
        res["pass"] = res["lifecycle_pass"]
    return res


# ---------------------------------------------------------------------------
# The receipt.
REQUIRED_CHECKS = (
    "anchor_parity",
    "infinite_threshold_parity",
    "matched_control",
    "rate_limit_accounting",
    "permanent_hold_anchor",
    "determinism_across_hashseed",
)


def battery(opps: list[dict[str, Any]] | None = None,
            script_dir: str = ".", *,
            skip: Iterable[str] = ()) -> dict[str, Any]:
    """Run the battery and emit an EVALUATED RECEIPT.

    WHAT CHANGED. This used to run two anchors and return them. A reader could
    not tell which checks existed, which had run, or whether the top-level
    boolean summarised all of them or the two that happened to be present --
    the enumerate-vs-derive defect, again.

    `all_pass` is now the conjunction over REQUIRED_CHECKS, and a required
    check that is ABSENT from the receipt makes it FALSE. Absence is the
    failure mode that reads as success, so it is the one that must be
    explicitly refused: `missing_checks` is reported beside the verdict, not
    silently dropped from the conjunction (rule 4). `skip` exists ONLY so the
    selftests can prove that -- it can never make `all_pass` true.
    """
    opps = stub_opportunities() if opps is None else opps
    if not opps:
        return {"evaluable": False, "canon": CANON, "arms_checked": 0,
                "checks": {}, "required_checks": list(REQUIRED_CHECKS),
                "missing_checks": list(REQUIRED_CHECKS), "all_pass": False,
                "why": "no opportunities: zero difference under zero data is "
                       "not parity, and seven arms agreeing on nothing is not "
                       "seven passing arms"}
    skip = set(skip)
    fixture = hashlib.sha256(json.dumps(
        sorted((o["t"], o["slug"], o["side"], o["gen"], o["qty"], o["price"])
               for o in opps), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()
    try:
        code_sha = hashlib.sha256(
            open(__file__, "rb").read()).hexdigest()
    except OSError:
        code_sha = None
    runners = {
        "anchor_parity": lambda: anchor_parity(opps),
        "infinite_threshold_parity": lambda: infinite_threshold_parity(opps),
        "matched_control": lambda: matched_control(opps),
        "rate_limit_accounting": lambda: rate_limit_accounting(opps),
        "permanent_hold_anchor": lambda: permanent_hold_anchor(opps),
        "determinism_across_hashseed":
            lambda: determinism_across_hashseed(script_dir),
    }
    checks: dict[str, Any] = {}
    for name, fn in runners.items():
        if name in skip:
            continue
        try:
            checks[name] = fn()
        except ParityRefused as exc:          # a refusal is a RESULT, not a
            checks[name] = {"pass": False,    # crash to be swallowed upstream
                            "refused": str(exc)}
    missing = [c for c in REQUIRED_CHECKS if c not in checks]
    # The receipt NAMES what produced it. A parity result read six weeks
    # later against an unknown fixture and an unknown build of this file is
    # not verifiable at the artifact it claims (rule 16).
    return {"evaluable": True, "canon": CANON, "arms": list(ARMS),
            "arms_checked": len(ARMS), "n_opportunities": len(opps),
            "fixture_sha256": fixture, "battery_code_sha256": code_sha,
            "checks": checks, "required_checks": list(REQUIRED_CHECKS),
            "missing_checks": missing,
            "all_pass": (not missing
                         and all(bool(checks[c].get("pass"))
                                 for c in REQUIRED_CHECKS))}


# ---------------------------------------------------------------------------
def _selftests() -> int:
    """Every guard RED-FIRST with a positive control (rule 15).

    The pairing is the point. A battery shown only to pass a correct harness
    may be one that passes anything -- "all arms agree" would then be evidence
    of an unrun battery, not a neutral one.
    """
    checks = 0
    fails: list[str] = []
    import os

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    def refuses(fn, needle):
        try:
            fn()
        except ParityRefused as e:
            return needle in str(e)
        return False

    here = os.path.dirname(os.path.abspath(__file__))
    opps = stub_opportunities()

    # ---- THE ANCHOR, both directions ------------------------------------
    a = anchor_parity(opps)
    ok(a["pass"] and a["n_arms"] == 7,
       "ANCHOR: with every predictor disabled, all SEVEN arms are BIT-IDENTICAL "
       "to QR_SKEW_ONLY (positive control)")

    tr = run_stub_arm("QR_SKEW_ONLY", opps)
    perturbed = Trajectory(arm="QR_SKEW_ONLY")
    perturbed.events = list(tr.events)
    perturbed.add(t=999.0, kind="CANCEL_REQUESTED",
                  slug="btc-updown-5m-1787650200", side="BUY_UP", gen=0)
    ok(perturbed.digest() != tr.digest(),
       "PERTURBATION: ONE extra cancel BREAKS parity -- if it did not, the "
       "anchor would be decorative and could never fail")
    ok(run_stub_arm("QR_SKEW_ONLY", opps).digest() == tr.digest(),
       "and a re-run of the SAME arm reproduces its digest exactly (so the "
       "perturbation result is a real difference, not run-to-run noise)")

    t1 = run_stub_arm("QR_SKEW_ONLY", opps)
    t2 = run_stub_arm("CONDVALUE_X_SKEW", opps)
    ok(t1.digest() == t2.digest() and t1.arm != t2.arm,
       "the arm NAME is excluded from the canonical bytes -- including it "
       "would make every arm differ trivially and the anchor unfalsifiable")

    # ---- floats carry NO tolerance, and the signed-zero edge is declared --
    def _dig(v):
        t = Trajectory(arm="x")
        t.add(t=0.0, kind="FILL", slug="s", side="BUY_UP", gen=0, qty=v)
        return t.digest()
    ok(_dig(1.0) != _dig(1.0 + 2 ** -52) and _dig(0.1 + 0.2) != _dig(0.3),
       "floats carry NO tolerance: one-ULP and 0.1+0.2-vs-0.3 differences are "
       "VISIBLE, so canonicalization removes representation noise only")
    ok(_dig(1.0) == _dig(1.0),
       "and identical values reproduce exactly (positive control)")
    ok(_dig(0.0) != _dig(-0.0),
       "DECLARED EDGE: 0.0 and -0.0 are IEEE-equal but distinct bit patterns "
       "and DIFFER here. Correct under bit-identical-no-tolerance, and NOT "
       "normalised -- normalising would be a tolerance by another name")
    nan_refused = False
    try:
        _dig(float("nan"))
    except ValueError:
        nan_refused = True
    ok(nan_refused,
       "NaN REFUSES at serialization -- bare NaN is not JSON, so some readers "
       "reject it and others parse it differently, and that divergence would "
       "surface as a spurious parity failure")

    # ---- the stub scorer must not be PYTHONHASHSEED-dependent -------------
    ok(stub_score("s", "BUY_UP", 3) == stub_score("s", "BUY_UP", 3)
       and stub_score("s", "BUY_UP", 3) != stub_score("s", "BUY_UP", 4),
       "the stub scorer is sha256-derived, NOT builtin hash(): stable within "
       "and across processes, and still discriminating between generations")

    # ---- corollary anchors ------------------------------------------------
    ok(infinite_threshold_parity(opps)["pass"],
       "an INFINITE cancel threshold is bit-identical to QR_SKEW_ONLY "
       "(nothing ever crosses)")
    en = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                      cancel_threshold=0.5)
    ok(en.digest() != run_stub_arm("QR_SKEW_ONLY", opps).digest(),
       "control: an ENABLED predictor that DOES cancel is NOT identical -- the "
       "battery can tell a real difference from none")

    # ---- lifecycle invariants --------------------------------------------
    ok(refuses(lambda: run_stub_arm("CONDVALUE_X_SKEW", opps + [opps[0]],
                                    predictor_enabled=True,
                                    cancel_threshold=0.0), "at most once"),
       "one generation may be cancelled AT MOST ONCE -- a second attempt "
       "REFUSES")

    post = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                        cancel_threshold=0.0, fill_at=0.200)
    ok(not any(e.kind in ("FILL", "FILL_STALE") for e in post.events),
       "a cancelled generation CANNOT fill after simulated effectiveness")
    pre = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                       cancel_threshold=0.0, fill_at=0.010)
    ok(all(e.kind != "FILL" for e in pre.events)
       and any(e.kind == "FILL_STALE" for e in pre.events),
       "a PRE-effectiveness fill on a cancelled generation is charged as "
       "STALE, not as prevented")
    unc = run_stub_arm("QR_SKEW_ONLY", opps, fill_at=0.010)
    ok(any(e.kind == "FILL" for e in unc.events)
       and not any(e.kind == "FILL_STALE" for e in unc.events),
       "positive control: an UNCANCELLED generation fills normally, so STALE "
       "is not simply what this harness always reports")

    # ---- THE DELETED ARGUMENT: budget-matched selection is REAL -----------
    pool = list(opps)
    sizes = {b: len(budget_matched_selection(pool, b, 7)) for b in (0, 1, 6)}
    ok(sizes == {0: 0, 1: 1, 6: 6},
       f"the BUDGET now CHANGES THE ANSWER: 0/1/6 -> {sizes}. The old "
       f"matched_control took `cancels` and ignored it -- 0, 1, 6 and 99 all "
       f"returned 12, so every 'matched' it reported was true by construction")
    ok(refuses(lambda: budget_matched_selection(pool, 99, 7), "exceeds"),
       "a budget LARGER than the eligible pool REFUSES -- it does not clamp, "
       "because a control that silently drew fewer actions than the treatment "
       "is no longer matched on the decision variable")
    ok(refuses(lambda: budget_matched_selection(pool, -1, 7), "negative"),
       "a NEGATIVE budget refuses")
    s1 = budget_matched_selection(pool, 6, 7)
    s2 = budget_matched_selection(pool, 6, 7)
    s3 = budget_matched_selection(pool, 6, 8)
    ok(s1 == s2, "the draw is REPRODUCIBLE under a fixed seed")
    ok(s1 != s3,
       "and a DIFFERENT seed gives a DIFFERENT draw -- so it is a draw, not a "
       "deterministic prefix wearing a seed")
    shuf = list(pool)
    random.Random(99).shuffle(shuf)
    ok(budget_matched_selection(shuf, 6, 7) == s1,
       "ORDER FIRST, THEN DRAW: shuffling the input pool does NOT change the "
       "selection -- reproducibly sampling an unstably-ordered sequence is "
       "blocker-7's defect with a seed bolted on")

    m = matched_control(opps)
    ok(m["pass"],
       f"the matched control agrees on ACTION COUNT, SIDE and HOUR by an "
       f"actual draw (n={m['treated']['n']})")
    ok(m["strict_subset"],
       f"and the treated arm cancels a STRICT SUBSET ({m['treated']['n']} of "
       f"{len(opps)}) -- matching an arm that cancels everything is vacuous")
    ok(m["selection_differs"],
       "the control selects DIFFERENT generations than the treatment -- a "
       "control that reproduced the treated selection would match perfectly "
       "and measure nothing")
    import inspect
    _sig = set(inspect.signature(matched_control).parameters)
    _prim = set(inspect.signature(budget_matched_selection).parameters)
    ok("cancels" not in _sig and "budget" in _prim,
       "the ignored `cancels` argument is GONE, not honoured: a matched "
       "control's count is DETERMINED by the treated arm, and a count the "
       "caller can choose is one that gets chosen after the numbers are "
       f"visible. matched_control{tuple(sorted(_sig))} -- the budget knob "
       f"lives one level down as a tested primitive")

    # ---- rate-limit accounting -------------------------------------------
    rl = rate_limit_accounting(opps)
    ok(rl["identity_holds"] and rl["limiter_bound"],
       f"REQUESTED = EFFECTIVE + SUPPRESSED as an evaluated identity "
       f"({rl['requested']} = {rl['effective']} + {rl['suppressed']}), on a "
       f"run where the limiter actually BOUND (both sides non-zero)")
    ok(rl["zero_limit_all_suppressed"] and rl["zero_limit_fills_match_no_cancel"],
       "under a limit of ZERO every request is SUPPRESSED and the fills are "
       "IDENTICAL to the never-cancelled arm -- a request the venue did not "
       "honour prevented nothing, and crediting it would inflate the estimand")
    ok(rl["per_window_within_limit"] and rl["max_in_any_window"] == 3,
       f"per-window effective cancels are counted EXACTLY and the count "
       f"ENTERS the verdict: {rl['per_window_effective']} against limit "
       f"{rl['limit']}. The first version multiplied the limit by the windows "
       f"spanned and left the field out of `pass` -- a loose bound beside a "
       f"verdict it does not enter is decoration")
    _unl_pw = per_window_effective(run_stub_arm(
        "CONDVALUE_X_SKEW", [dict(o, t=o["t"] * 0.005) for o in opps],
        predictor_enabled=True, cancel_threshold=0.0,
        rate_limit_per_window=None))
    ok(max(_unl_pw.values()) > rl["limit"],
       f"and the per-window predicate CAN return False on real output: "
       f"unlimited, one window carries {max(_unl_pw.values())} > "
       f"{rl['limit']} effectives")
    unl = run_stub_arm("CONDVALUE_X_SKEW",
                       [dict(o, t=o["t"] * 0.005) for o in opps],
                       predictor_enabled=True, cancel_threshold=0.0,
                       rate_limit_per_window=None, fill_at=0.200).counts()
    ok(unl["CANCEL_SUPPRESSED"] == 0 and unl["CANCEL_EFFECTIVE"] > 0,
       "positive control: with NO limiter nothing is suppressed, so "
       "SUPPRESSED is not simply what this harness always emits")

    # ---- the zero-repost / permanent-hold anchor -------------------------
    ph = permanent_hold_anchor(opps)
    ok(ph["holder_flagged"],
       f"ZERO-REPOST ANCHOR: an arm that cancels once and never reposts is "
       f"FLAGGED ({ph['permanent_hold']['withheld']}/"
       f"{ph['permanent_hold']['opportunities']} withheld >= "
       f"{PERMANENT_HOLD_WITHHELD_SHARE})")
    ok(ph["normal_not_flagged"],
       "positive control: the NORMAL arm is NOT flagged, so the anchor is not "
       "simply what this always reports")
    ok(ph["holder_trades_less"],
       f"and the holder takes FEWER fills ({ph['permanent_hold']['fills']} vs "
       f"{ph['normal']['fills']}) -- which is exactly why it would WIN any "
       f"harm-share or adverse-per-fill comparison by not trading. The "
       f"predicate is EXPOSURE, not harm")
    hold_tr = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                           cancel_threshold=0.5, hold_after_first_cancel=True)
    ok(any(e.kind == "PLACE_WITHHELD" for e in hold_tr.events),
       "a withheld quote is recorded as a STATUS (PLACE_WITHHELD), never a "
       "silent absence -- an arm that just stopped emitting would be "
       "indistinguishable from one that ran out of opportunities (rule 4)")

    # ---- training-reuse guard (rule 11) ----------------------------------
    g = assert_no_training_reuse(["20260820", "20260821"],
                                 ["20260826", "20260827"])
    ok(g["pass"] and g["strictly_later"],
       "TRAINING-REUSE GUARD: disjoint fitting days strictly before the "
       "scoring days PASS (positive control)")
    ok(refuses(lambda: assert_no_training_reuse(["20260820", "20260826"],
                                                ["20260826"]), "also used to"),
       "a day used to FIT and to SCORE REFUSES (rule 11)")
    ok(refuses(lambda: assert_no_training_reuse([], []), "VACUOUS"),
       "TWO EMPTY SETS REFUSE: they are disjoint, and a guard that passed "
       "there would pass loudest with nothing to check -- this programme has "
       "already shipped an invariance check that diffed two empty files")
    ok(refuses(lambda: assert_no_training_reuse(["20260827"], ["20260820"]),
               "strictly after"),
       "disjoint but EARLIER scoring days REFUSE -- validation is LATER "
       "untouched days, not merely other days")

    # ---- external-arm interface ------------------------------------------
    ext_tr = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                          cancel_threshold=0.5, rate_limit_per_window=1,
                          fill_at=0.010)
    ext = {"canon": CANON, "arm": "CONDVALUE_X_SKEW",
           "events": [{f: getattr(e, f) for f in EVENT_FIELDS}
                      for e in ext_tr.events]}
    back = load_external_trajectory(ext)
    ok(back.digest() == ext_tr.digest(),
       "EXTERNAL INTERFACE: a declared-shape trajectory round-trips to the "
       "SAME digest -- the contract is data, and DA imports nothing from BE "
       "(R-235: a checker sharing code with the checked agrees by "
       "construction)")
    ok(external_lifecycle(back)["pass"],
       "and the external lifecycle invariants (at-most-once, "
       "requested=effective+suppressed, no fill after effective) hold on it")

    def _mut(**kw):
        o = json.loads(json.dumps(ext))
        for k, v in kw.items():
            o[k] = v
        return o
    ok(refuses(lambda: load_external_trajectory(_mut(canon="other_v1")),
               "canon"),
       "a DIFFERENT canon REFUSES -- trajectories serialized under different "
       "canonical forms cannot be byte-compared")
    ok(refuses(lambda: load_external_trajectory(_mut(arm="NOT_AN_ARM")),
               "unknown arm"),
       "an UNDECLARED arm refuses")
    ok(refuses(lambda: load_external_trajectory(_mut(events=[])), "no events"),
       "an EMPTY external trajectory refuses -- it would trivially satisfy "
       "every invariant and must not be scored as agreement")
    drop = json.loads(json.dumps(ext))
    del drop["events"][0]["qty"]
    ok(refuses(lambda: load_external_trajectory(drop), "MISSING"),
       "a MISSING field REFUSES rather than being defaulted -- a checker that "
       "supplies what the producer failed to produce is testing its own "
       "defaults")
    xtra = json.loads(json.dumps(ext))
    xtra["events"][0]["shadow_pnl"] = 1.0
    ok(refuses(lambda: load_external_trajectory(xtra), "UNDECLARED fields"),
       "an EXTRA undeclared field REFUSES -- ignoring it would compute the "
       "digest over a PROJECTION of what the producer actually did")
    badk = json.loads(json.dumps(ext))
    badk["events"][0]["kind"] = "TELEPORT"
    ok(refuses(lambda: load_external_trajectory(badk), "not a declared kind"),
       "an undeclared event KIND refuses")
    dupseq = json.loads(json.dumps(ext))
    dupseq["events"][1]["seq"] = dupseq["events"][0]["seq"]
    ok(refuses(lambda: load_external_trajectory(dupseq), "duplicate seq"),
       "a DUPLICATE seq refuses -- ordering by (t, seq) would be ambiguous "
       "and the digest would depend on input order")
    badt = json.loads(json.dumps(ext))
    badt["events"][0]["t"] = "12.0"
    ok(refuses(lambda: load_external_trajectory(badt), "finite number"),
       "a non-numeric timestamp refuses")

    # external lifecycle must CATCH a violation the loader cannot see
    dbl_ev = json.loads(json.dumps(ext))
    first = next(e for e in dbl_ev["events"] if e["kind"] == "CANCEL_REQUESTED")
    clone = dict(first, seq=max(e["seq"] for e in dbl_ev["events"]) + 1,
                 t=first["t"] + 0.001)
    dbl_ev["events"].append(clone)
    ok(load_external_trajectory(dbl_ev) is not None
       and not external_lifecycle(load_external_trajectory(dbl_ev))["pass"],
       "a WELL-FORMED external trajectory that cancels one generation TWICE "
       "loads cleanly and FAILS the lifecycle check -- shape validity is not "
       "behavioural validity")
    stale_late = json.loads(json.dumps(ext))
    _eff = next(e for e in stale_late["events"]
                if e["kind"] == "CANCEL_EFFECTIVE")
    stale_late["events"].append(dict(
        _eff, kind="FILL_STALE", t=_eff["t"] + 1.0,
        seq=max(e["seq"] for e in stale_late["events"]) + 1, qty=5.0,
        price=0.5, note=""))
    ok(not external_lifecycle(load_external_trajectory(stale_late))["pass"],
       "a FILL_STALE **after** effectiveness FAILS -- the first version "
       "checked only FILL, so a producer could relabel a post-cancel fill as "
       "STALE and pass, which is exactly the mislabel the check exists for")
    orphan = json.loads(json.dumps(ext))
    orphan["events"] = [e for e in orphan["events"]
                        if not (e["kind"] == "CANCEL_REQUESTED"
                                and e["gen"] == _eff["gen"])]
    ok(not external_lifecycle(load_external_trajectory(orphan))[
           "no_unrequested_outcome"],
       "an EFFECTIVE cancel with NO matching REQUEST fails -- otherwise "
       "requested=effective+suppressed could be satisfied by two compensating "
       "errors")
    ok(external_lifecycle(back)["no_unrequested_outcome"],
       "positive control: the well-formed trajectory has no unrequested "
       "outcome")

    _ref_ok = check_external_arms([ext], reference=ext_tr)
    _other = run_stub_arm("QR_SKEW_ONLY", opps, fill_at=0.010)
    _ref_bad = check_external_arms([ext], reference=_other)
    ok(_ref_ok["pass"] is True and _ref_ok["all_match_reference"] is True
       and _ref_bad["pass"] is False
       and _ref_bad["lifecycle_pass"] is True,
       "when a REFERENCE is supplied, bit-identity ENTERS the verdict: a "
       "lifecycle-clean submission that does NOT match the reference reports "
       "pass=False (the first version left matches_reference out of `pass`)")
    ok("all_match_reference" not in check_external_arms([ext]),
       "and with NO reference the key is ABSENT, not True -- an unrun "
       "comparison must not read as a passed one")

    ok(check_external_arms([])["evaluable"] is False,
       "an EMPTY external submission is NOT EVALUABLE, never a clean pass")
    ok(check_external_arms([ext])["pass"] is True,
       "positive control: a populated, valid external submission passes")
    ok(check_external_arms([ext, json.loads(json.dumps(ext))])[
           "duplicate_arms"] is True,
       "two submissions under the SAME arm name are flagged -- otherwise one "
       "arm could be scored twice and a missing arm would go unnoticed")

    # ---- determinism across processes ------------------------------------
    d = determinism_across_hashseed(here)
    ok(d["pass"],
       "two interpreters under DIFFERENT PYTHONHASHSEED produce IDENTICAL "
       "trajectories AND an identical matched-control draw -- the battery "
       "does not inherit blocker-7's fixed-seed-over-unstable-order class")

    # ---- the receipt -------------------------------------------------------
    r = battery(opps, here)
    ok(len(r.get("fixture_sha256") or "") == 64
       and len(r.get("battery_code_sha256") or "") == 64
       and battery(stub_opportunities(11), here)["fixture_sha256"]
           != r["fixture_sha256"],
       f"the receipt NAMES what produced it -- fixture "
       f"{r['fixture_sha256'][:12]} and battery code "
       f"{r['battery_code_sha256'][:12]} -- and the fixture digest MOVES with "
       f"the fixture, so a parity result read later is verifiable at the "
       f"artifact it claims (rule 16)")
    ok(r["all_pass"] and not r["missing_checks"]
       and set(r["checks"]) == set(REQUIRED_CHECKS),
       f"RECEIPT: battery() emits an EVALUATED receipt over all "
       f"{len(REQUIRED_CHECKS)} required checks, all passing (positive "
       f"control)")
    part = battery(opps, here, skip=["rate_limit_accounting"])
    ok(part["all_pass"] is False
       and part["missing_checks"] == ["rate_limit_accounting"],
       "a MISSING required check makes the receipt FAIL and is NAMED -- the "
       "old battery returned the conjunction of whatever happened to be "
       "present, which reads as success exactly when coverage is lost")
    e = battery([], here)
    ok(e["evaluable"] is False and e["all_pass"] is False
       and e["missing_checks"] == list(REQUIRED_CHECKS),
       "an EMPTY run reports NOT EVALUABLE with every check missing, never "
       "seven passing arms -- zero difference under zero data is not parity")

    print(f"\n{'REPLAY PARITY BATTERY GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftests())
