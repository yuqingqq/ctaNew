#!/bin/bash
# Runs the re-valuation EMIT after the V2 result lands, with a REAL table.
#
# ARMED-BEFORE-FIXED is its own failure: a unit armed at 07:27 invoked the
# emit as it stood then -- no decomposition, no role filter, no zero-D sign
# edge. Landing a commit does not reach a process that already resolved its
# path. So this wrapper DIGESTS the modules it is about to use and REFUSES
# if they are not the landed bytes.
set -u
D=/home/yuqing/ctaNew/data/pm_5min/derived
PY=/home/yuqing/pricer-sol/venv/bin/python3
SRC=/home/yuqing/ctaNew-wt-de2/live/pm_research
V2=$D/fwd_v2/p003_de_forward_value_20260907.json
OLDBOOK=$D/be_daybook_20260907_btc__L250ms__FWD1.superseded_20260911T071439Z.pkl

for m in de_revaluation_emit.py de_window_decomposition.py; do
  a=$(sha256sum "$SRC/$m" | cut -d' ' -f1)
  b=$(git -C /home/yuqing/ctaNew-wt-de2 show "origin/mm-research:live/pm_research/$m" | sha256sum | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "REFUSED EMIT_MODULE_IS_NOT_THE_LANDED_BYTES: $m"; exit 4; }
done
echo "$(date -u +%H:%M:%SZ) emit modules verified against origin/mm-research"

echo "$(date -u +%H:%M:%SZ) waiting for $V2"
for _ in $(seq 1 720); do
  if [ -f "$V2" ]; then
    s1=$(stat -c %s "$V2"); sleep 10; s2=$(stat -c %s "$V2")
    [ "$s1" = "$s2" ] && break
  fi
  sleep 15
done
[ -f "$V2" ] || { echo "$(date -u +%H:%M:%SZ) no V2 result after 3h"; exit 3; }

# the rebuilt book is whatever the V2 result names -- never guessed
NEWBOOK=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['book'])" "$V2")
echo "$(date -u +%H:%M:%SZ) rebuilt book: $NEWBOOK"
cd "$SRC" || exit 2
for b in "$OLDBOOK" "$NEWBOOK"; do
  out=$D/fwd_v2/p003_de_window_decomposition_$(basename "$b" .pkl).json
  [ -f "$out" ] || "$PY" de_window_decomposition.py "$b" || exit 5
done

"$PY" - "$V2" "$D/fwd/p003_de_forward_value_20260907.json" "$OLDBOOK" "$NEWBOOK" <<'PYX'
import json,sys
from pathlib import Path
sys.path.insert(0,"/home/yuqing/ctaNew-wt-de2/live/pm_research")
import de_revaluation_emit as E
v2,v1,oldb,newb=sys.argv[1:5]
D=Path("/home/yuqing/ctaNew/data/pm_5min/derived")
r2=E.read_json(v2,"the V2 re-valuation result")
r1=E.read_json(v1,"day one's V1 result")
def dec(b):
    return E.read_json(D/"fwd_v2"/f"p003_de_window_decomposition_{Path(b).stem}.json",
                       f"the decomposition of {Path(b).name}")
d_old,d_new=dec(oldb),dec(newb)
spine=E.spine_for_day("2026-09-07")
d1={a:c["D"] for a,c in r1["cells"].items()}
d2={a:c["D"] for a,c in r2["cells"].items()}
contrib={}
for arm in d1:
    o=d_old["arms"][arm]["per_window"]; n=d_new["arms"][arm]["per_window"]
    contrib[arm]={int(k):(n.get(k,{}).get("D_contribution_cents",0.0)
                          -o.get(k,{}).get("D_contribution_cents",0.0))
                  for k in set(o)|set(n)}
res=E.emit(d1,d2,contrib,spine)
res["decompositions"]={"superseded":d_old["book_sha256"],"rebuilt":d_new["book_sha256"]}
out=D/"fwd_v2"/"p003_de_revaluation_emit_20260907.json"
out.write_text(json.dumps(res,indent=1,default=str))
for a,v in res["arms"].items():
    print(f"  {a}: DELTA_D={v['DELTA_D_cents']:+.6f}c rows={v['n_rows']} "
          f"sum={v['table_sums_to_cents']:+.6f}c residual={v['residual_cents']:+.6f}c "
          f"band={v['band']} CONCENTRATION_FINDING={v['CONCENTRATION_FINDING']}")
print("  SIGN_CHANGE_HALT:",res["SIGN_CHANGE_HALT"],res["halted_arms"])
print("  written:",out)
PYX
