"""EVERY FILE THE VALUATION PATH READS, WITH ITS sha256. DA declares it.

The digests come from disk, computed here; nothing is typed. The list is
the valuation path's import closure plus the declarations and data
artifacts the gates resolve -- the set whose bytes decide a number.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")


def build() -> dict:
    import de_forward_value_day, de_preflight_matrix   # noqa: F401
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and "/pm_research/" in str(f) and Path(f).is_file():
            out[Path(f).name] = hashlib.sha256(
                Path(f).read_bytes()).hexdigest()
    # The POST-PROCESSORS live on the other branch line (they read results,
    # they do not value), so they are digested BY PATH rather than imported
    # -- importing them here would fail and, worse, a silent skip would drop
    # them from a list whose whole purpose is completeness.
    post = Path("/home/yuqing/ctaNew-wt-de2/live/pm_research")
    for name in ("de_revaluation_emit.py", "de_window_decomposition.py"):
        f = post / name
        out["post/" + name] = (hashlib.sha256(f.read_bytes()).hexdigest()
                               if f.is_file() else "ABSENT")
    decls = HERE / "declarations"
    for pat in ("de_arm_freeze_v*.json", "de_multiday_gate1_params_v31.json",
                "de_settlement_control_declaration_v*.json",
                "da_forward_test_declaration_v*.json"):
        for p in sorted(decls.glob(pat)):
            out["declarations/" + p.name] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    for pat in ("be_score_neutrality_20260903__EV22_vs_NEUTCHK__68e7d23.json",
                "be137_gap_windows_*.json", "da_blackout_mask_2026091*.json",
                "da_blackout_mask_2026090[7-9].json"):
        for p in sorted(DERIVED.glob(pat)):
            out["data/" + p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"protocol": "P003_DE_POPULATION_FREEZE_LIST_V1",
            "n_files": len(out), "files": out}


if __name__ == "__main__":
    d = build()
    (DERIVED / "de_population_freeze_list.json").write_text(
        json.dumps(d, indent=1))
    print(f"  files: {d['n_files']}  -> {DERIVED}/de_population_freeze_list.json")
