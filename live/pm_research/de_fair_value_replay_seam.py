"""GATE 6: the replay seam -- shared INPUTS, free OUTCOMES.

`fair_value_plan.md` §5 gate 6: *"baseline and challenger share all
non-fair-value parameters, input snapshot and initial state, while
preserving their own resulting order paths."*

THE SUBTLE HALF IS THE SECOND ONE, and it is the reason this file exists
rather than a diff of two config dicts. Changing the fair value
LEGITIMATELY changes quotes, queue position, fills and inventory. A seam
that pinned the order path -- replaying the challenger against the
baseline's fills, or reusing its inventory -- would produce two runs that
differ only in a number nobody acted on. That is the decorative seam one
level up, and it is invisible in any comparison that checks only the
inputs.

So the seam enforces BOTH directions:

  * every non-fair-value input IDENTICAL, refusing by the KEY that differs
    (REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS);
  * and, when the consumed values differ, the resulting order paths MUST
    be free to differ -- identical paths under differing values are
    REPLAY_OUTCOME_PATH_IS_PINNED, a refusal, not a curiosity.

THE ENGINE HERE IS MINIMAL ON PURPOSE. Crossing a quote fills it, fills
move inventory, and nothing else. It is not a microstructure model and
must never be read as one: it exists so the seam's two properties are
DRIVEN on real state transitions instead of asserted about a simulator
nobody ran.

Usage:  de_fair_value_replay_seam.py --falsify
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_policy_seam as SEAM         # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_REPLAY_SEAM_V1"
INPUTS_DIFFER = "REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS"
PATH_PINNED = "REPLAY_OUTCOME_PATH_IS_PINNED"
NO_ARMS = "REPLAY_HAS_FEWER_THAN_TWO_ARMS"


class ReplayRefused(ValueError):
    """The comparison cannot be made, so no verdict is reported."""


@dataclass(frozen=True)
class ReplayInputs:
    """THE HALF THAT MUST BE IDENTICAL."""
    non_fair_value_params: dict
    input_snapshot_sha256: str
    initial_state: dict

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class OrderPath:
    """THE HALF THAT MUST BE FREE."""
    fills: tuple
    inventory: float
    n_quotes: int

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(
            asdict(self), sort_keys=True, default=str).encode()).hexdigest()


def replay(quotes, price_path, initial_state: dict) -> OrderPath:
    """A MINIMAL, DETERMINISTIC ENGINE -- crossing fills, fills move
    inventory, nothing else. Its only job is to make the order path a
    real consequence of the quotes rather than a label attached to them.
    """
    inv = float(initial_state.get("inventory", 0.0))
    fills = []
    for q, px in zip(quotes, price_path):
        if px <= q.bid:
            fills.append(("BUY", q.slug, q.generation_id, q.bid))
            inv += 1.0
        elif px >= q.ask:
            fills.append(("SELL", q.slug, q.generation_id, q.ask))
            inv -= 1.0
    return OrderPath(fills=tuple(fills), inventory=inv, n_quotes=len(quotes))


def compare_arms(baseline: dict, challenger: dict) -> dict:
    """Both directions of gate 6, as one computed verdict."""
    for nm, arm in (("baseline", baseline), ("challenger", challenger)):
        for key in ("inputs", "path", "anchors"):
            if key not in arm:
                raise ReplayRefused(
                    f"REFUSED {NO_ARMS}: the {nm} arm carries no {key!r}")
    bi, ci = baseline["inputs"], challenger["inputs"]
    differing = [k for k in ("non_fair_value_params",
                             "input_snapshot_sha256", "initial_state")
                 if json.dumps(getattr(bi, k), sort_keys=True, default=str)
                 != json.dumps(getattr(ci, k), sort_keys=True, default=str)]
    if differing:
        raise ReplayRefused(
            f"REFUSED {INPUTS_DIFFER}: {differing} differ between the "
            f"arms. A replay whose arms do not share their inputs "
            f"measures the inputs, not the fair value.")
    same_anchor = baseline["anchors"] == challenger["anchors"]
    same_path = baseline["path"].digest() == challenger["path"].digest()
    out = {"protocol": PROTOCOL,
           "inputs_identical": True,
           "inputs_digest": bi.digest()[:16],
           "anchors_identical": same_anchor,
           "order_paths_identical": same_path,
           "baseline_path": baseline["path"].digest()[:16],
           "challenger_path": challenger["path"].digest()[:16],
           "baseline_inventory": baseline["path"].inventory,
           "challenger_inventory": challenger["path"].inventory,
           "n_fills": (len(baseline["path"].fills),
                       len(challenger["path"].fills))}
    if same_anchor and same_path:
        out["verdict"] = "NO_OP"
        out["reading"] = ("the arms consumed the same values and produced "
                          "the same path: substitution is a no-op, which "
                          "is what Identity-against-itself must be")
        return out
    if not same_anchor and same_path:
        out["verdict"] = PATH_PINNED
        out["reading"] = (
            "the arms consumed DIFFERENT values and produced the SAME "
            "order path. The challenger's path did not respond to its own "
            "value, so the replay pinned the outcome -- quotes, queue "
            "position, fills and inventory are all downstream of the "
            "value and must be free to move.")
        return out
    if same_anchor and not same_path:
        out["verdict"] = "PATH_MOVED_WITHOUT_A_VALUE_CHANGE"
        out["reading"] = ("identical values produced different paths, so "
                          "something outside the fair value moved: the "
                          "arms are not sharing what they claim to share")
        return out
    out["verdict"] = "SEAM_IS_HONEST"
    out["reading"] = ("the arms shared every input and their order paths "
                      "diverged from the value alone, which is what gate 6 "
                      "asks for")
    return out


def run_arm(actions, value_of, inputs: ReplayInputs, price_path,
            half_spread=0.01) -> dict:
    """One arm: the seam produces quotes, the engine produces the path."""
    seam = SEAM.run_seam(actions, value_of, half_spread=half_spread)
    path = replay(seam["quotes"], price_path, inputs.initial_state)
    return {"inputs": inputs, "path": path,
            "anchors": [q.anchor for q in seam["quotes"]],
            "trajectory": seam["trajectory"], "seam": seam}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    import da_fair_price_identity as FP
    import de_fair_value_actions as A

    W_START = 1788825600
    rows = [{"coin": "btc", "slug": "btc-updown-5m-1788825600",
             "generation_id": f"g{i}", "decision_recv_ns": 1000 + i,
             "quote_side": "BID", "up_probability_consumed": 0.5,
             "window_start": W_START, "on_identity_reference_path": True}
            for i in range(6)]
    acts = sorted(A.build_actions(rows)["actions"],
                  key=lambda a: a.decision_recv_ns)
    prices = [0.52, 0.48, 0.55, 0.45, 0.50, 0.60]
    inputs = ReplayInputs(
        non_fair_value_params={"half_spread": 0.01, "max_inventory": 5},
        input_snapshot_sha256="a" * 64,
        initial_state={"inventory": 0.0, "clock": 0})

    ident = {a.generation_id: 0.50 for a in acts}
    chall = {a.generation_id: 0.58 for a in acts}

    def v_ident(a):
        return ident[a.generation_id], FP.IDENTITY, False

    def v_chall(a):
        return chall[a.generation_id], FP.BN_BOOKTICKER, False

    base = run_arm(acts, v_ident, inputs, prices)
    same = run_arm(acts, v_ident, inputs, prices)
    ck("Identity against itself is a NO_OP: same inputs, same path",
       compare_arms(base, same)["verdict"] == "NO_OP",
       f"path {base['path'].digest()[:16]}")

    chall_arm = run_arm(acts, v_chall, inputs, prices)
    honest = compare_arms(base, chall_arm)
    ck("a DIFFERENT value produces a DIFFERENT order path -- the seam is "
       "honest",
       honest["verdict"] == "SEAM_IS_HONEST"
       and not honest["order_paths_identical"],
       f"fills {honest['n_fills']} inventory "
       f"{honest['baseline_inventory']} -> {honest['challenger_inventory']}")
    ck("  and the arms still share every input",
       honest["inputs_identical"] and honest["inputs_digest"]
       == inputs.digest()[:16], honest["inputs_digest"])

    # THE SUBTLE FAILURE: a replay that hands the challenger the
    # BASELINE's path. Every input matches, the values differ, and the
    # outcome was pinned.
    pinned = dict(chall_arm, path=base["path"])
    ck("a replay that PINS the outcome path is REFUSED by name",
       compare_arms(base, pinned)["verdict"] == PATH_PINNED,
       compare_arms(base, pinned)["verdict"])
    ck("  and the refusal says WHY, in terms of what is downstream of the "
       "value",
       "queue position" in compare_arms(base, pinned)["reading"]
       and "free to move" in compare_arms(base, pinned)["reading"])

    for key, bad in (("non_fair_value_params",
                      ReplayInputs({"half_spread": 0.02,
                                    "max_inventory": 5},
                                   "a" * 64, {"inventory": 0.0, "clock": 0})),
                     ("input_snapshot_sha256",
                      ReplayInputs({"half_spread": 0.01,
                                    "max_inventory": 5},
                                   "b" * 64, {"inventory": 0.0, "clock": 0})),
                     ("initial_state",
                      ReplayInputs({"half_spread": 0.01,
                                    "max_inventory": 5},
                                   "a" * 64, {"inventory": 2.0, "clock": 0}))):
        other = run_arm(acts, v_chall, bad, prices)
        try:
            compare_arms(base, other)
            msg = ""
        except ReplayRefused as exc:
            msg = str(exc)
        ck(f"a differing {key} REFUSES, naming the key",
           INPUTS_DIFFER in msg and key in msg,
           msg[:64] or f"ADMITTED A DIFFERING {key}")

    ck("identical values with DIFFERENT paths is its own verdict, not a "
       "pass",
       compare_arms(base, dict(same, path=chall_arm["path"]))["verdict"]
       == "PATH_MOVED_WITHOUT_A_VALUE_CHANGE",
       compare_arms(base, dict(same, path=chall_arm["path"]))["verdict"])
    ck("the engine's path is a REAL consequence: inventory and fills both "
       "move with the value",
       base["path"].inventory != chall_arm["path"].inventory
       and base["path"].fills != chall_arm["path"].fills,
       f"{base['path'].inventory} vs {chall_arm['path'].inventory}")
    try:
        compare_arms({"inputs": inputs}, chall_arm)
        armless = ""
    except ReplayRefused as exc:
        armless = str(exc)
    ck("an arm missing its path REFUSES rather than comparing nothing",
       NO_ARMS in armless, armless[:52] or "COMPARED A MISSING ARM")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL,
                      "refusals": [INPUTS_DIFFER, PATH_PINNED, NO_ARMS]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
