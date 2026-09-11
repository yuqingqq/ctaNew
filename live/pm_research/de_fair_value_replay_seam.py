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
from dataclasses import dataclass, asdict, field, fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import de_fair_value_policy_seam as SEAM         # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_REPLAY_SEAM_V1"
INPUTS_DIFFER = "REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS"
#: REVIEW 201 drove the defect this constant now guards: a challenger
#: replayed on a FLAT 0.99 tape against a baseline on the real tape
#: returned SEAM_IS_HONEST with a MATCHING inputs digest, because
#: `price_path` and `half_spread` were ARGUMENTS to run_arm, outside
#: ReplayInputs, and `input_snapshot_sha256` was a caller-supplied string
#: this module never computed. Inventory went -1.0 -> -6.0 and every unit
#: of it was the tape. The digest is COMPUTED from what the run actually
#: consumed now, and a declared digest that disagrees with the computed
#: one is its own refusal.
CONSUMED_DIFFER = "REPLAY_ARMS_CONSUMED_DIFFERENT_INPUTS"
DECLARED_NOT_COMPUTED = "REPLAY_DECLARED_SNAPSHOT_IS_NOT_WHAT_WAS_CONSUMED"
PATH_PINNED = "REPLAY_OUTCOME_PATH_IS_PINNED"
NO_ARMS = "REPLAY_HAS_FEWER_THAN_TWO_ARMS"


class ReplayRefused(ValueError):
    """The comparison cannot be made, so no verdict is reported."""


@dataclass(frozen=True)
class ReplayInputs:
    """THE HALF THAT MUST BE IDENTICAL -- INCLUDING THE TAPE.

    `price_path` and `half_spread` live HERE and not in `run_arm`'s
    signature, because an input a comparison cannot see is an input the
    arms do not actually share. That was the whole of REVIEW 201's
    finding, and moving them is the fix rather than adding a check.
    """
    non_fair_value_params: dict
    initial_state: dict
    price_path: tuple
    half_spread: float
    #: OPTIONAL, and CHECKED against the computed digest when present. A
    #: caller may say which snapshot it believes it is replaying; it may
    #: not decide the answer.
    declared_snapshot_sha256: str | None = None

    #: THE ONLY THING EXCLUDED FROM THE DIGEST, and it is excluded
    #: because it is a claim ABOUT the digest: including it would make the
    #: digest depend on itself.
    NOT_AN_INPUT = ("declared_snapshot_sha256",)

    def consumed(self) -> dict:
        """EVERY FIELD OF THIS DATACLASS, FOUND BY INTROSPECTION.

        REVIEW 202: my previous 'backstop' digested a HAND-WRITTEN list of
        four names and `compare_arms` compared a SECOND hand-written list
        of the same four -- so the backstop covered a field the first list
        forgot and not one the second forgot, and its coverage was itself
        a hand-maintained list. That is the thing it was built to replace.

        Now a field added to this dataclass tomorrow is digested with no
        edit anywhere, and its ABSENCE from the digest is IMPOSSIBLE
        rather than unlikely: there is no list to forget.
        """
        out = {}
        for f in dc_fields(self):
            if f.name in self.NOT_AN_INPUT:
                continue
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def digested_field_names(cls) -> tuple:
        """What the digest covers, derived -- for a reader and for a cell."""
        return tuple(f.name for f in dc_fields(cls)
                     if f.name not in cls.NOT_AN_INPUT)

    def digest(self) -> str:
        """THE DIGEST OF WHAT IS ACTUALLY CONSUMED -- computed here."""
        return hashlib.sha256(json.dumps(
            self.consumed(), sort_keys=True, default=str).encode()).hexdigest()

    def __post_init__(self) -> None:
        if (self.declared_snapshot_sha256
                and self.declared_snapshot_sha256 != self.digest()):
            raise ReplayRefused(
                f"REFUSED {DECLARED_NOT_COMPUTED}: the caller declares "
                f"{self.declared_snapshot_sha256[:16]} and the inputs it "
                f"actually carries digest to {self.digest()[:16]}. A "
                f"snapshot identity supplied rather than computed is a "
                f"label, and REVIEW 201 measured a flat tape passing "
                f"behind one.")


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
    # ONE SOURCE FOR WHAT IS COMPARED, and it is the dataclass itself.
    # The second hand-written list lived here; REVIEW 202 measured that
    # having two lists made the backstop defeatable from either side.
    b_consumed, c_consumed = bi.consumed(), ci.consumed()
    differing = sorted(
        set(b_consumed) | set(c_consumed),
        key=lambda k: k)
    differing = [k for k in differing
                 if json.dumps(b_consumed.get(k), sort_keys=True,
                               default=str)
                 != json.dumps(c_consumed.get(k), sort_keys=True,
                               default=str)]
    if differing:
        raise ReplayRefused(
            f"REFUSED {INPUTS_DIFFER}: {differing} differ between the "
            f"arms. A replay whose arms do not share their inputs "
            f"measures the inputs, not the fair value.")
    # THE COMPUTED DIGEST IS THE BACKSTOP: a field added later that the
    # list above forgets still moves this number, so the check does not
    # depend on my remembering to extend a list.
    if bi.digest() != ci.digest():
        raise ReplayRefused(
            f"REFUSED {CONSUMED_DIFFER}: the arms' CONSUMED inputs digest "
            f"to {bi.digest()[:16]} and {ci.digest()[:16]}. Something the "
            f"field list above does not name differs, and an input a "
            f"comparison cannot see is not shared.")
    same_anchor = baseline["anchors"] == challenger["anchors"]
    same_path = baseline["path"].digest() == challenger["path"].digest()
    out = {"protocol": PROTOCOL,
           "inputs_identical": True,
           "inputs_digest": bi.digest()[:16],
           "digest_covers": list(type(bi).digested_field_names()),
           "digest_coverage_is_derived":
               "dataclasses.fields(ReplayInputs) minus NOT_AN_INPUT -- no "
               "hand-written list, so a new field cannot be omitted",
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


def run_arm(actions, value_of, inputs: ReplayInputs) -> dict:
    """One arm. THE TAPE AND THE SPREAD COME FROM `inputs`, so there is no
    way to replay two arms on different tapes and have the comparison
    call them shared."""
    seam = SEAM.run_seam(actions, value_of, half_spread=inputs.half_spread)
    path = replay(seam["quotes"], inputs.price_path, inputs.initial_state)
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
    # GATE 4 NOW REQUIRES THE CANONICAL POPULATION (REVIEW 201): its
    # membership is derived, not read off the row. This fixture supplies
    # one -- and the fact that tightening gate 4 broke this fixture is why
    # the drive is done from the REF's bytes and not only in the tree
    # where the change was made.
    POP = {"actions": [(r["slug"], r["generation_id"]) for r in rows],
           "population": "P003_NEUTRAL_REFERENCE_PATH_FIXTURE",
           "as_of": "2026-09-11T19:00:00Z",
           "source_identity": f"{Path(__file__).name}.falsify fixture"}
    acts = sorted(A.build_actions(
        rows, canonical_population=POP)["actions"],
        key=lambda a: a.decision_recv_ns)
    prices = [0.52, 0.48, 0.55, 0.45, 0.50, 0.60]
    inputs = ReplayInputs(
        non_fair_value_params={"max_inventory": 5},
        initial_state={"inventory": 0.0, "clock": 0},
        price_path=tuple(prices), half_spread=0.01)

    ident = {a.generation_id: 0.50 for a in acts}
    chall = {a.generation_id: 0.58 for a in acts}

    def v_ident(a):
        return ident[a.generation_id], FP.IDENTITY, False

    def v_chall(a):
        return chall[a.generation_id], FP.BN_BOOKTICKER, False

    base = run_arm(acts, v_ident, inputs)
    same = run_arm(acts, v_ident, inputs)
    ck("Identity against itself is a NO_OP: same inputs, same path",
       compare_arms(base, same)["verdict"] == "NO_OP",
       f"path {base['path'].digest()[:16]}")

    chall_arm = run_arm(acts, v_chall, inputs)
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

    # REVIEW 201'S OWN ARM, FIRST: a FLAT tape against the real one.
    flat = ReplayInputs(non_fair_value_params={"max_inventory": 5},
                        initial_state={"inventory": 0.0, "clock": 0},
                        price_path=tuple([0.99] * len(prices)),
                        half_spread=0.01)
    flat_arm = run_arm(acts, v_chall, flat)
    try:
        compare_arms(base, flat_arm)
        flat_msg = ""
    except ReplayRefused as exc:
        flat_msg = str(exc)
    ck("a challenger on a FLAT 0.99 TAPE is REFUSED by name (REVIEW 201's "
       "own drive, which used to return SEAM_IS_HONEST)",
       INPUTS_DIFFER in flat_msg and "price_path" in flat_msg,
       flat_msg[:72] or f"ADMITTED A FLAT TAPE: inventory "
                        f"{flat_arm['path'].inventory}")
    ck("  and the real-tape pair still PASSES, so the guard is not merely "
       "strict",
       compare_arms(base, chall_arm)["verdict"] == "SEAM_IS_HONEST")
    ck("  while the legs remain FREE to produce different order paths",
       base["path"].digest() != chall_arm["path"].digest()
       and base["path"].fills != chall_arm["path"].fills,
       f"{len(base['path'].fills)} vs {len(chall_arm['path'].fills)} fills")
    try:
        ReplayInputs(non_fair_value_params={"max_inventory": 5},
                     initial_state={"inventory": 0.0, "clock": 0},
                     price_path=tuple(prices), half_spread=0.01,
                     declared_snapshot_sha256="f" * 64)
        declared_msg = ""
    except ReplayRefused as exc:
        declared_msg = str(exc)
    ck("a DECLARED snapshot digest that is not what the inputs carry "
       "REFUSES",
       DECLARED_NOT_COMPUTED in declared_msg,
       declared_msg[:64] or "ADMITTED A DECLARED DIGEST")
    ck("  and a declared digest that MATCHES the computed one admits",
       ReplayInputs(non_fair_value_params={"max_inventory": 5},
                    initial_state={"inventory": 0.0, "clock": 0},
                    price_path=tuple(prices), half_spread=0.01,
                    declared_snapshot_sha256=inputs.digest()).digest()
       == inputs.digest(), inputs.digest()[:16])

    for key, bad in (("non_fair_value_params",
                      ReplayInputs({"max_inventory": 9},
                                   {"inventory": 0.0, "clock": 0},
                                   tuple(prices), 0.01)),
                     ("half_spread",
                      ReplayInputs({"max_inventory": 5},
                                   {"inventory": 0.0, "clock": 0},
                                   tuple(prices), 0.02)),
                     ("initial_state",
                      ReplayInputs({"max_inventory": 5},
                                   {"inventory": 2.0, "clock": 0},
                                   tuple(prices), 0.01))):
        other = run_arm(acts, v_chall, bad)
        try:
            compare_arms(base, other)
            msg = ""
        except ReplayRefused as exc:
            msg = str(exc)
        ck(f"a differing {key} REFUSES, naming the key",
           INPUTS_DIFFER in msg and key in msg,
           msg[:64] or f"ADMITTED A DIFFERING {key}")

    # --- REVIEW 202's falsifier, the one that could not be written before:
    # a field NOBODY LISTED anywhere, added to the dataclass at runtime.
    import dataclasses as _dc

    @_dc.dataclass(frozen=True)
    class InputsPlusOne(ReplayInputs):
        latency_model_ms: float = 0.0       # a field no list mentions

    a_plus = InputsPlusOne(
        non_fair_value_params={"max_inventory": 5},
        initial_state={"inventory": 0.0, "clock": 0},
        price_path=tuple(prices), half_spread=0.01, latency_model_ms=0.0)
    b_plus = InputsPlusOne(
        non_fair_value_params={"max_inventory": 5},
        initial_state={"inventory": 0.0, "clock": 0},
        price_path=tuple(prices), half_spread=0.01, latency_model_ms=250.0)
    ck("a NEW field is digested with NO edit to any list",
       "latency_model_ms" in InputsPlusOne.digested_field_names()
       and a_plus.digest() != b_plus.digest(),
       f"covers {len(InputsPlusOne.digested_field_names())} fields: "
       f"{', '.join(InputsPlusOne.digested_field_names())}")
    try:
        compare_arms(run_arm(acts, v_ident, a_plus),
                     run_arm(acts, v_chall, b_plus))
        newfield = ""
    except ReplayRefused as exc:
        newfield = str(exc)
    ck("  and two legs differing ONLY on it are REFUSED, naming it",
       INPUTS_DIFFER in newfield and "latency_model_ms" in newfield,
       newfield[:70] or "ADMITTED A DIFFERENCE NO LIST MENTIONED")
    ck("  while legs AGREEING on it still compare normally",
       compare_arms(run_arm(acts, v_ident, a_plus),
                    run_arm(acts, v_chall, a_plus))["verdict"]
       == "SEAM_IS_HONEST")
    ck("the digest excludes ONLY the claim about itself",
       ReplayInputs.NOT_AN_INPUT == ("declared_snapshot_sha256",)
       and "declared_snapshot_sha256"
       not in ReplayInputs.digested_field_names(),
       str(ReplayInputs.digested_field_names()))

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
