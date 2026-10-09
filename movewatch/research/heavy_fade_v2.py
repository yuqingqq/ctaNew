"""Heavy-volume pumps with a futures discount at +4 h, short +4 h -> +24 h (docs/HEAVY_FADE_V2.md), k = 1.

  python3 -m research.heavy_fade_v2 reconcile   # premium_h4 here == signal-search rows (screen window)
  python3 -m research.heavy_fade_v2 validate    # untouched conf rows (v1 builder), read once
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research.flush_signal as fs
import research.heavy_fade as hf
import research.runners as rn
import research.signal_search as ss

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / "docs" / "HEAVY_FADE_V2.md"
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests" / "test_heavy_fade_v2.py"
RECEIPT = hf.OUT / "receipt_v2_conf.json"
_A3 = next(c for c in json.loads((ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json").read_text())["cells"] if c["id"] == "A3")
DISCOUNT, PREMIUM = float(_A3["cuts"][0]), float(_A3["cuts"][1])
CELL = "S24"


def with_premium(h: pd.DataFrame) -> pd.DataFrame:
    """premium_h4 via runners.first4h (unchanged), joined back on (asset, time)."""
    pop = h[["asset", "time"]].drop_duplicates().reset_index(drop=True)
    got = rn.first4h(pop)[["asset", "time", "premium_h4"]]
    return h.merge(got, on=["asset", "time"], how="left")


def reconcile() -> dict:
    df = pd.read_parquet(hf.OUT / "rows_screen.parquet")
    h = with_premium(hf.heavy(df))
    ref = pd.concat([ss.load_rows(("discovery", "holdout")), ss.load_rows(ss.VAL)], ignore_index=True)
    ref = ref[["asset", "time", "premium_h4"]].drop_duplicates(["asset", "time"]).rename(columns={"premium_h4": "ref"})
    m = h.merge(ref, on=["asset", "time"], how="left")
    both = m["premium_h4"].notna() & m["ref"].notna()
    res = {"rows": int(len(m)), "both": int(both.sum()),
           "mismatch": int((both & ~np.isclose(m["premium_h4"], m["ref"], rtol=0, atol=1e-12)).sum()),
           "only_one_missing": int((m["premium_h4"].isna() ^ m["ref"].isna()).sum())}
    print(json.dumps(res))
    if res["mismatch"] or res["only_one_missing"]:
        raise SystemExit("premium_h4 reconciliation failed")
    return res


def validate() -> None:
    if RECEIPT.exists():
        raise SystemExit("REFUSED: v2 conf already validated (read once)")
    if not all(fs.committed_clean(p) for p in (DECL, SCRIPT, TESTS, hf.FREEZE)):
        raise SystemExit("REFUSED: v2 declaration, script, tests and v1 freeze must be committed clean")
    rec = reconcile()
    df = pd.read_parquet(hf.OUT / "rows_conf.parquet")
    h = with_premium(hf.heavy(df))
    cell = h[h["premium_h4"] <= DISCOUNT]
    fal = hf.falsifiers(cell["time"].reset_index(drop=True), 1)
    c = hf.stats(hf.net(cell, CELL), cell["time"])
    c["verdict"] = hf.verdict(c, 1)
    groups = {"heavy_all": h, "heavy_discount": cell,
              "heavy_middle": h[(h["premium_h4"] > DISCOUNT) & (h["premium_h4"] <= PREMIUM)],
              "heavy_premium": h[h["premium_h4"] > PREMIUM]}
    desc = {g: {k: hf.stats(hf.net(x, k), x["time"]) for k in ("S24", "S72", "S24_stop25", "S72_stop25")}
            for g, x in groups.items()}
    desc["discount_stops"] = {k: hf.stats(hf.net(cell, k), cell["time"]) for k in hf.DESC}
    desc["discount_stress3"] = hf.stats(hf.net(cell, CELL, 3), cell["time"])
    desc["per_year"] = pd.Series(hf.net(cell, CELL), index=cell.index).groupby(cell["time"].dt.year) \
        .agg(["size", "mean", "median"]).round(5).to_dict("index")
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "declaration_commit": fs.last_commit(DECL),
           "script_commit": fs.last_commit(SCRIPT), "k": 1, "premium_reconciliation": rec,
           "heavy_pumps": int(len(h)), "PREMIUM_MISSING": int(h["premium_h4"].isna().sum()), "falsifiers": fal,
           "cell": {"name": "heavy & discount, S24", **c}, "descriptive": desc}
    RECEIPT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "descriptive"}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["reconcile", "validate"])
    {"reconcile": reconcile, "validate": validate}[ap.parse_args().phase]()


if __name__ == "__main__":
    main()
