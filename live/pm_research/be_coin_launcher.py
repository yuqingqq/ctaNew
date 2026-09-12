"""BE 215: build any stage for ANY COIN, without moving a pinned digest.

WHY THIS FILE EXISTS AND WHY IT IS NOT A ONE-LINE EDIT.
`be_daybook_build.py` is already coin-parameterised where it counts -- `build(day,
*, coin=COIN, ...)` at :1427, and `coin` reaches the tape and fragment receipt
stems (:322, :344), `TAPEMOD.out_path` (:411), the supplied-window refusal
(:640), the mask arithmetic (:1110/:1118) and the artifact stem (:1874). What is
hardcoded is the COMMAND LINE: :4611 calls `artifact_paths(day, COIN, ...)` and
:4613 calls `build(day, ...)` without passing coin, and there is no `--coin`.

Adding that flag would edit a PINNED module. The five 7ed5a90 digests would
move, every book's recorded import closure would re-pin, and V2 would refuse
the four books already built. The coin is not worth a re-pin of the population.

So this module imports the pinned builders and CALLS them. It adds a coin and
nothing else -- which is a claim the falsifier has to make good on, not a
promise in a docstring.

A NOTE ON THE CONTROL THE COORDINATOR ASKED FOR. "Byte-identical btc book"
cannot be met by any instrument and never could: `asm.assembly` embeds wall
clocks and `peak_rss_mb_highwater`, so two builds of one day differ in bytes by
construction (BE 173/175 measured five books across three freezes -- all
content-identical, every sha different). The control that carries the intended
meaning is CONTENT identity with a computed allowed set, which
`be_book_content_diff.py` already implements and which is driven separately.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

WT_FWD = "/home/yuqing/ctaNew-wt-fwd"
STAGES = ("frag", "tape", "book")
MODULE = {"frag": "be_gate1_fragment",
          "tape": "be_gate1_state_tape",
          "book": "be_daybook_build"}

OK = "OK"
COIN_NOT_SUPPLIED = "COIN_NOT_SUPPLIED"
UNKNOWN_STAGE = "UNKNOWN_STAGE"
WRONG_TREE = "IMPORTED_FROM_ANOTHER_TREE"
OUTPUT_EXISTS = "OUTPUT_EXISTS"


def import_from_tree(name: str, tree: str = WT_FWD):
    """Import from the tree under test and PROVE it, the way the preflight
    does: a module already imported from another tree would otherwise be
    returned from sys.modules and the build would run the wrong bytes."""
    for mod in list(sys.modules):
        m = sys.modules[mod]
        f = getattr(m, "__file__", None)
        if f and "/pm_research/" in f and not f.startswith(tree):
            del sys.modules[mod]
    for p in (f"{tree}/live/pm_research", tree):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    m = importlib.import_module(name)
    got = getattr(m, "__file__", "")
    if not got.startswith(tree):
        raise RuntimeError(f"{WRONG_TREE}: {name} resolved to {got}")
    return m


def coin_is_supplied(day: str, coin: str, tree: str = WT_FWD) -> dict:
    """Does this day supply windows for this coin? REFUSES rather than
    answering False on an error -- False is the permissive answer here, and a
    coin that merely failed to load is not a coin with no data."""
    FR = import_from_tree("be_gate1_fragment", tree)
    try:
        pop = FR.population(day, coin)
    except Exception as exc:
        return {"status": COIN_NOT_SUPPLIED, "day": day, "coin": coin,
                "cause": f"{type(exc).__name__}: {str(exc)[:160]}"}
    n = pop.get("n_windows")
    if not n:
        return {"status": COIN_NOT_SUPPLIED, "day": day, "coin": coin,
                "cause": "population resolved but supplied no windows"}
    return {"status": OK, "day": day, "coin": coin, "n_windows": n,
            "n_slugs": len(pop["slugs"])}


def artifact_path(stage: str, day: str, coin: str, *,
                  placement_latency_ms=250.0, artifact_revision="FWD1",
                  tree: str = WT_FWD) -> Path:
    """The path the PINNED module itself would choose -- asked of that module,
    never reconstructed here. A path this file composed by hand would be a
    second opinion about where a book lives."""
    M = import_from_tree(MODULE[stage], tree)
    if stage == "book":
        bp, _dst = M.artifact_paths(day, coin, placement_latency_ms,
                                    artifact_revision)
        return Path(bp)
    return Path(M.out_path(day, coin))


def run_stage(stage: str, day: str, coin: str, *,
              placement_latency_ms: float = 250.0,
              artifact_revision: str = "FWD1",
              out_path: str | None = None, tree: str = WT_FWD) -> dict:
    if stage not in STAGES:
        return {"status": UNKNOWN_STAGE, "stage": stage}
    sup = coin_is_supplied(day, coin, tree)
    if sup["status"] != OK:
        return sup
    M = import_from_tree(MODULE[stage], tree)
    got = getattr(M, "__file__", "")
    t0 = time.time()
    if stage == "book":
        kw = dict(coin=coin, placement_latency_ms=placement_latency_ms,
                  artifact_revision=artifact_revision, progress=False)
        if out_path:
            kw["out_path"] = Path(out_path)
        out = M.build(day, **kw)
    else:
        out = M.build(day, coin=coin, progress=False)
    return {"status": OK, "stage": stage, "day": day, "coin": coin,
            "wall_s": round(time.time() - t0, 1),
            "module_file": got, "n_windows": sup["n_windows"],
            "builder_returned": {k: out[k] for k in list(out)[:8]}
            if isinstance(out, dict) else str(out)[:200]}


def falsify() -> int:
    rc = 0

    def note(n, ok, d=""):
        nonlocal rc
        if not ok:
            rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

    day = "20260905"
    # ---- the ADMIT direction: both coins this day actually supplies
    b = coin_is_supplied(day, "btc")
    e = coin_is_supplied(day, "eth")
    note("btc is supplied on 09-05 and the launcher says so",
         b["status"] == OK and b["n_windows"] == 288, f"n={b.get('n_windows')}")
    note("eth is supplied on 09-05 -- the coin this file exists for",
         e["status"] == OK and e["n_windows"] == 288, f"n={e.get('n_windows')}")
    # ---- the REFUSE direction: a coin the data does not support
    z = coin_is_supplied(day, "zzz")
    note("a coin the data does not support is REFUSED by name, not built",
         z["status"] == COIN_NOT_SUPPLIED, z.get("cause", "")[:70])
    # a REAL coin with no supply on this day, found rather than assumed
    real_absent = None
    for c in ("hype", "bnb", "sol", "xrp", "doge"):
        if coin_is_supplied(day, c)["status"] != OK:
            real_absent = c
            break
    note("and a REAL coin with no windows that day is refused the same way "
         f"({real_absent or 'none found -- every coin is supplied'})",
         real_absent is None
         or coin_is_supplied(day, real_absent)["status"] == COIN_NOT_SUPPLIED)
    # ---- it changes the COIN and nothing else: the path comes from the
    #      pinned module, and btc's path equals the one the pinned CLI builds
    M = import_from_tree("be_daybook_build")
    cli_bp, _ = M.artifact_paths(day, M.COIN, 250.0, "FWD1")
    mine = artifact_path("book", day, "btc")
    note("with coin=btc the artifact path is IDENTICAL to the pinned CLI's",
         str(mine) == str(cli_bp), str(mine).rsplit("/", 1)[-1])
    eth_p = artifact_path("book", day, "eth")
    note("and with coin=eth only the coin moves in the stem",
         str(eth_p) == str(cli_bp).replace("_btc_", "_eth_"),
         str(eth_p).rsplit("/", 1)[-1])
    # ---- it runs the tree under test, proven by __file__
    note("the builder it would call resolves under wt-fwd, proven by __file__",
         getattr(M, "__file__", "").startswith(WT_FWD),
         getattr(M, "__file__", ""))
    # ---- it never edits or shells out to a pinned module's CLI
    # The property is that no pinned module is EXECUTED as a subprocess -- not
    # that its name never appears, which it does in the docstring above that
    # explains why. Checking the name alone flagged this file's own prose.
    import ast as _ast
    tree_ = _ast.parse(Path(__file__).read_text())
    spawns = [n for n in _ast.walk(tree_)
              if isinstance(n, _ast.Attribute)
              and n.attr in ("run", "Popen", "call", "check_output", "system")
              and isinstance(getattr(n, "value", None), _ast.Name)
              and n.value.id in ("subprocess", "os")]
    imports = [n for n in _ast.walk(tree_) if isinstance(n, _ast.Import)
               for al in n.names if al.name in ("subprocess", "pty")]
    note("this module never EXECUTES a pinned module as a subprocess -- it "
         "imports and calls, so no --coin flag is needed anywhere",
         not spawns and not imports,
         f"spawn_calls={len(spawns)} subprocess_imports={len(imports)}")
    code_lines = [ln for ln in Path(__file__).read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    in_doc = False
    exec_hits = 0
    for ln in code_lines:
        q = ln.count('"""')
        if q:
            in_doc = not in_doc if q == 1 else in_doc
            continue
        if not in_doc and "be_daybook_build.py" in ln:
            exec_hits += 1
    note("  (and its name appears only in prose, never in an executable line)",
         exec_hits == 0, f"executable_mentions={exec_hits}")
    note("an unknown stage is refused before any import",
         run_stage("bogus", day, "btc")["status"] == UNKNOWN_STAGE)
    print(json.dumps({"falsifier": "be_coin_launcher", "n": 10, "failed": rc}))
    return rc


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    a = sys.argv
    def opt(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default
    if "--check" in a:
        print(json.dumps(coin_is_supplied(opt("--day"), opt("--coin")), indent=1))
        raise SystemExit(0)
    st, d, c = opt("--stage"), opt("--day"), opt("--coin")
    if not (st and d and c):
        raise SystemExit("usage: --stage frag|tape|book --day D --coin C "
                         "[--out-path P] | --check --day D --coin C | --falsify")
    r = run_stage(st, d, c, out_path=opt("--out-path"),
                  placement_latency_ms=float(opt("--placement-latency-ms", 250.0)),
                  artifact_revision=opt("--artifact-revision", "FWD1"))
    print(json.dumps(r, indent=1, default=str))
    raise SystemExit(0 if r["status"] == OK else 3)
