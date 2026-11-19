"""No-fit verification that the prior coherence finding is repaired in saved v3 outputs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify import DATA, PIN, checkhash, read, save, sha, verified

a, b = DATA / "pipeline_bench_v2_20261119", DATA / "pipeline_bench_v3_20261119"
ja, jb = read(a / "bench.json"), read(b / "bench.json")
assert ja["models"] == jb["models"] and len(jb["models"]) == 44
for name, h in jb["models"].items():
    checkhash(a / "models" / name, h)
    checkhash(b / "models" / name, h)
checkhash(PIN / "predictfun/research/pf_pipeline_bench.py", jb["source_sha256"])
out = dict(model_files_identical_and_hash_verified=44, days={})
for day, expected in (("20261026", 23), ("20261027", 15)):
    for d, j in ((a, ja), (b, jb)):
        checkhash(d / f"pred_{day}.parquet", j["outputs"][f"pred_{day}.parquet"])
    cols = ["oid", "kind", "j", "J_ok", "J_Eq", "J_Eqm", "J_V"]
    old = pd.read_parquet(a / f"pred_{day}.parquet", columns=cols)
    new = pd.read_parquet(b / f"pred_{day}.parquet", columns=cols)
    for c in cols:
        if c != "J_ok":
            assert old[c].equals(new[c]), c
    change = old.J_ok != new.J_ok
    impossible = (new.J_Eq == 0) & (new.J_Eqm.abs() > 1e-9 * np.maximum(1., new.J_Eqm.abs()))
    assert not (impossible & new.J_ok).any()
    assert int(change.sum()) == expected
    assert new.loc[change, "kind"].eq("retain").all()
    triggers = lambda z: z[(z.kind == "retain") & z.J_ok & (z.J_V < 0)].groupby("oid").j.min()
    assert triggers(old).equals(triggers(new))
    policy = []
    for v in (2, 3):
        p = DATA / f"pipeline_policy_v{v}_20261119"
        receipt = read(p / f"policy_{day}.json")
        csv = p / f"policy_{day}.csv"
        checkhash(csv, receipt["outputs"][csv.name])
        policy.append(pd.read_csv(csv))
    assert policy[0].equals(policy[1])
    out["days"][day] = dict(rows=len(new), changed_flags=int(change.sum()),
                            changed_orders=int(new.oid[change].nunique()),
                            incoherent_zero_quantity_accepted=0, first_triggers_changed=0,
                            policy_csv_unchanged=True, policy_csv_rows=len(policy[1]),
                            policy_contract_parity=receipt["parity"])
for name in ("paired.json", "paired.txt", "eval_20261026.json", "eval_20261026.txt", "eval_20261026.csv",
             "eval_20261027.json", "eval_20261027.txt", "eval_20261027.csv"):
    folder = "pipeline_paired_20261119" if name.startswith("paired") else "pipeline_paired_eval_20261119"
    checkhash(PIN / "predictfun/reviews/pipeline_paired_20261119" / name, sha(DATA / folder / name))
out["committed_result_copies_match_data"] = 8
out["prior_fix_script_sha256"] = sha(Path(__file__))
out["verified_hashes"] = verified
save("prior_fix.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "verified_hashes"}, indent=2))
