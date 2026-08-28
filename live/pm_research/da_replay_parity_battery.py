"""Seven-arm replay parity battery, run against TYPED STUBS.

AUTHORISATION: hazard plan §10 item 6 / §10.1 ("the common replay harness may
be developed against typed stub outputs"); design in
`plans/LANE4_REPLAY_PARITY_STUB_BATTERY.md` (`6fc96e2`). Nothing here is scored
and no stub is ever a candidate (TODO §10).

WHAT THIS IS. DA builds the CHECKER; BE's replay arms are the CHECKED. The two
stay separate implementations on purpose (R-235 do-not-harmonize): a checker
that shares code with the thing it checks agrees with it by construction.

WHY STUBS FIRST. Every arm returns declared-shape output with NO model behind
it, so the battery must observe ZERO difference. If arms differ while every
predictor is inert, the difference is the HARNESS, and any later result would
inherit it invisibly. This programme has already seen path-coupled overlays
amplify prediction noise 10-20x and produce large replay deltas with zero
ranking improvement -- a battery that cannot first demonstrate zero difference
under zero signal cannot attribute a later difference to signal.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Canonical trajectory. "Bit-identical" is undefined without a canonical form:
# two correct harnesses could serialise the same trajectory differently and
# every comparison would fail, or -- worse -- differ in a way a tolerance hides.
# Same recipe as annotation_canon_v1, for the same reason.
CANON = "replay_traj_canon_v1"


@dataclass(frozen=True)
class Event:
    """One trajectory event. Ordered by (t, seq) -- never by dict order."""
    t: float
    seq: int
    kind: str                 # PLACE | CANCEL_REQUESTED | CANCEL_EFFECTIVE
                              # | CANCEL_SUPPRESSED | FILL | FILL_STALE
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
        return json.dumps({"canon": CANON, "events": payload},
                          sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


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


def stub_opportunities(n: int = 12) -> list[dict[str, Any]]:
    """Neutral opportunities, identical for every arm. Deterministic by
    construction -- no RNG, so a difference can never be a seed artifact."""
    return [{"slug": f"btc-updown-5m-{1787650200 + 300 * (i // 3)}",
             "side": "BUY_UP" if i % 2 == 0 else "SELL_UP",
             "gen": i, "t": 10.0 * i, "qty": 5.0, "price": 0.50}
            for i in range(n)]


def run_stub_arm(arm: str, opps: list[dict[str, Any]], *,
                 predictor_enabled: bool = False,
                 cancel_threshold: float = float("inf"),
                 fill_at: float | None = None) -> Trajectory:
    """A typed stub arm. NO MODEL: with the predictor disabled every arm must
    place the same orders and cancel nothing, so all seven trajectories are
    identical by construction -- and the battery's job is to prove the harness
    does not break that.
    """
    tr = Trajectory(arm=arm)
    cancelled: set[tuple[str, str, int]] = set()
    for o in opps:
        key = (o["slug"], o["side"], o["gen"])
        tr.add(t=o["t"], kind="PLACE", slug=o["slug"], side=o["side"],
               gen=o["gen"], qty=o["qty"], price=o["price"])
        score = 1.0 if predictor_enabled else 0.0
        if predictor_enabled and score >= cancel_threshold:
            if key in cancelled:
                raise AssertionError(
                    f"REFUSED: generation {key} cancelled twice. One "
                    f"generation may be cancelled at most once.")
            cancelled.add(key)
            tr.add(t=o["t"], kind="CANCEL_REQUESTED", slug=o["slug"],
                   side=o["side"], gen=o["gen"])
            tr.add(t=o["t"] + CANCEL_EFFECTIVE_LAG_S, kind="CANCEL_EFFECTIVE",
                   slug=o["slug"], side=o["side"], gen=o["gen"])
        if fill_at is not None:
            ft = o["t"] + fill_at
            eff = o["t"] + CANCEL_EFFECTIVE_LAG_S
            if key in cancelled and ft >= eff:
                continue          # cancelled orders cannot fill after effect
            tr.add(t=ft, kind=("FILL_STALE" if key in cancelled else "FILL"),
                   slug=o["slug"], side=o["side"], gen=o["gen"],
                   qty=o["qty"], price=o["price"],
                   note=("pre-effectiveness fill on a cancelled generation "
                         "is charged as STALE" if key in cancelled else ""))
    return tr


# ---------------------------------------------------------------------------
# The battery. Each check returns a computed predicate, never a printed verdict.
def anchor_parity(opps: list[dict[str, Any]]) -> dict[str, Any]:
    """THE ANCHOR: with the predictor DISABLED every arm is BIT-IDENTICAL to
    QR_SKEW_ONLY. Bit-identical, not within-tolerance -- a tolerance would hide
    exactly the coupling this exists to find, and today's summation-order
    finding shows ~1e-11 movement on identical terms, so "close" cannot be
    distinguished from "differently ordered but wrong".
    """
    base = run_stub_arm("QR_SKEW_ONLY", opps).digest()
    per = {a: run_stub_arm(a, opps).digest() for a in ARMS}
    diff = sorted(a for a, d in per.items() if d != base)
    return {"baseline_digest": base, "per_arm": per,
            "arms_differing": diff, "bit_identical": not diff,
            "n_arms": len(ARMS)}


def infinite_threshold_parity(opps) -> dict[str, Any]:
    """An INFINITE cancel threshold cancels nothing, so an arm with its
    predictor ENABLED must still be bit-identical to QR_SKEW_ONLY."""
    base = run_stub_arm("QR_SKEW_ONLY", opps).digest()
    got = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                       cancel_threshold=float("inf")).digest()
    return {"baseline_digest": base, "digest": got, "bit_identical": got == base}


def matched_control(opps, cancels: int) -> dict[str, Any]:
    """Arm 7 must match on ACTION COUNT, SIDE and HOUR -- the decision
    variables (rule 7), asserted rather than assumed."""
    def prof(tr):
        c = [e for e in tr.events if e.kind == "CANCEL_REQUESTED"]
        return {"n": len(c),
                "by_side": {s: sum(1 for e in c if e.side == s)
                            for s in sorted({e.side for e in c})},
                "by_hour": {str(int(e.t // 3600)): sum(
                    1 for x in c if int(x.t // 3600) == int(e.t // 3600))
                    for e in c}}
    a = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                     cancel_threshold=0.5)
    b = run_stub_arm("RANDOM_MATCHED", opps, predictor_enabled=True,
                     cancel_threshold=0.5)
    pa, pb = prof(a), prof(b)
    return {"treated": pa, "control": pb, "matched": pa == pb}


def determinism_across_hashseed(script_dir: str) -> dict[str, Any]:
    """Two interpreters under DIFFERENT PYTHONHASHSEED must produce
    byte-identical trajectories. Blocker-7's class was exactly this -- a fixed
    RNG seed over a process-dependent iteration order is an independent draw,
    not a reproduction -- so the battery must not inherit it."""
    import subprocess, sys, os
    prog = ("import sys;sys.path.insert(0,%r)\n"
            "import da_replay_parity_battery as B\n"
            "print(B.run_stub_arm('CONDVALUE_X_SKEW', B.stub_opportunities(),"
            " predictor_enabled=True, cancel_threshold=0.5, fill_at=0.02)"
            ".digest())" % script_dir)
    outs = []
    for hs in ("0", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", prog], env=env,
                                   capture_output=True, text=True).stdout.strip())
    return {"digests": outs, "identical": bool(outs[0]) and outs[0] == outs[1]}


def battery(opps: list[dict[str, Any]] | None = None,
            script_dir: str = ".") -> dict[str, Any]:
    """Run the battery. An EMPTY run must NOT report seven passing arms."""
    opps = stub_opportunities() if opps is None else opps
    if not opps:
        return {"evaluable": False, "arms_checked": 0,
                "why": "no opportunities: zero difference under zero data is "
                       "not parity, and seven arms agreeing on nothing is not "
                       "seven passing arms",
                "bit_identical": False}
    a = anchor_parity(opps)
    return {"evaluable": True, "arms_checked": len(ARMS), "anchor": a,
            "infinite_threshold": infinite_threshold_parity(opps),
            "bit_identical": a["bit_identical"],
            "canon": CANON}


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

    here = os.path.dirname(os.path.abspath(__file__))
    opps = stub_opportunities()

    # ---- THE ANCHOR, both directions ------------------------------------
    a = anchor_parity(opps)
    ok(a["bit_identical"] and a["n_arms"] == 7,
       "ANCHOR: with every predictor disabled, all SEVEN arms are BIT-IDENTICAL "
       "to QR_SKEW_ONLY (positive control)")

    # THE PERTURBATION. If one extra cancel does not break parity, the anchor
    # is decorative and every later 'no difference' means nothing.
    tr = run_stub_arm("QR_SKEW_ONLY", opps)
    perturbed = Trajectory(arm="QR_SKEW_ONLY")
    perturbed.events = list(tr.events)
    perturbed.add(t=999.0, kind="CANCEL_REQUESTED", slug="btc-updown-5m-1787650200",
                  side="BUY_UP", gen=0)
    ok(perturbed.digest() != tr.digest(),
       "PERTURBATION: ONE extra cancel BREAKS parity -- if it did not, the "
       "anchor would be decorative and could never fail")
    ok(run_stub_arm("QR_SKEW_ONLY", opps).digest() == tr.digest(),
       "and a re-run of the SAME arm reproduces its digest exactly (so the "
       "perturbation result is a real difference, not run-to-run noise)")

    # the arm NAME must not enter the comparison, or every arm differs
    # trivially and the anchor can never fail
    t1 = run_stub_arm("QR_SKEW_ONLY", opps)
    t2 = run_stub_arm("CONDVALUE_X_SKEW", opps)
    ok(t1.digest() == t2.digest() and t1.arm != t2.arm,
       "the arm NAME is excluded from the canonical bytes -- including it "
       "would make every arm differ trivially and the anchor unfalsifiable")

    # ---- corollary anchors ------------------------------------------------
    ok(infinite_threshold_parity(opps)["bit_identical"],
       "an INFINITE cancel threshold is bit-identical to QR_SKEW_ONLY "
       "(nothing ever crosses)")
    en = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                      cancel_threshold=0.5)
    ok(en.digest() != run_stub_arm("QR_SKEW_ONLY", opps).digest(),
       "control: an ENABLED predictor that DOES cancel is NOT identical -- the "
       "battery can tell a real difference from none")

    # ---- lifecycle invariants --------------------------------------------
    dbl = False
    try:
        run_stub_arm("CONDVALUE_X_SKEW", opps + [opps[0]], predictor_enabled=True,
                     cancel_threshold=0.5)
    except AssertionError as e:
        dbl = "at most once" in str(e)
    ok(dbl, "one generation may be cancelled AT MOST ONCE -- a second attempt "
            "REFUSES")

    post = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                        cancel_threshold=0.5, fill_at=0.200)
    ok(not any(e.kind in ("FILL", "FILL_STALE") for e in post.events),
       "a cancelled generation CANNOT fill after simulated effectiveness")
    pre = run_stub_arm("CONDVALUE_X_SKEW", opps, predictor_enabled=True,
                       cancel_threshold=0.5, fill_at=0.010)
    ok(all(e.kind != "FILL" for e in pre.events)
       and any(e.kind == "FILL_STALE" for e in pre.events),
       "a PRE-effectiveness fill on a cancelled generation is charged as "
       "STALE, not as prevented")
    unc = run_stub_arm("QR_SKEW_ONLY", opps, fill_at=0.010)
    ok(any(e.kind == "FILL" for e in unc.events)
       and not any(e.kind == "FILL_STALE" for e in unc.events),
       "positive control: an UNCANCELLED generation fills normally, so STALE "
       "is not simply what this harness always reports")

    # ---- matched control (rule 7) ----------------------------------------
    m = matched_control(opps, cancels=6)
    ok(m["matched"] and m["treated"]["n"] > 0,
       "the matched control agrees on ACTION COUNT, SIDE and HOUR -- asserted, "
       "and on a non-empty set so it is not vacuous")

    # ---- determinism across processes ------------------------------------
    d = determinism_across_hashseed(here)
    ok(d["identical"],
       f"two interpreters under DIFFERENT PYTHONHASHSEED produce IDENTICAL "
       f"trajectories ({d['digests'][0][:12]}) -- the battery does not inherit "
       f"blocker-7's fixed-seed-over-unstable-order class")

    # ---- the empty run -----------------------------------------------------
    e = battery([], here)
    ok(e["evaluable"] is False and e["bit_identical"] is False,
       "an EMPTY run reports NOT EVALUABLE, never seven passing arms -- zero "
       "difference under zero data is not parity")
    ok(battery(opps, here)["evaluable"] is True,
       "positive control: a populated run IS evaluable")

    print(f"\n{'REPLAY PARITY BATTERY GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftests())
