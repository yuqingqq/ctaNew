#!/usr/bin/env bash
# Leave-one-out validation of the v3 stack: baseline = full frozen v3; each cell reverts ONE new config
# to its v1/default value. Δ vs baseline < 0 (removing hurts) => the lever is validated. Matched pred set
# (hl_lean175 / hl_residrev_lean) + frozen execution; only the one reverted lever changes.
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd "$ROOT"
B=live/state/longtail/v3loo; rm -rf "$B"; mkdir -p "$B"
run(){ local tag=$1; shift; bash live/run_convexity_v3_regime_gate.sh "$B/$tag" "$@" >/dev/null 2>&1; echo "  done $tag"; }
export -f run; export B ROOT

# baseline (no overrides) + 12 LOO reverts. Cost(8a)/funding(8b) are realism, not optimizations -> excluded.
JOBS=(
 "baseline"
 "no_regime_gate|REGIME_GATE=0"
 "K_long2|STRAT_K_LONG=2"
 "beta_neut_on|SIDE_BETA_NEUT=1"
 "no_short_min|SHORT_MIN_RET3D=-999"
 "no_bear_ramp|BEAR_DEPTH_RAMP=0"
 "sizing_equal|SIZING_MODE=equal"
 "no_conc_cap|CONC_CAP=0 CONC_CAP_SINGLE_EXEMPT=0"
 "stop_on_bear|STOP_SKIP_REGIMES="
 "bull_mom_v1|BULL_MODE=mom BULL_DEEP_THR=99 BULL_LONG_MULT=1 BULL_LONG_INSTRUMENT=alt BULL_SHORT_RANK=pred"
 "no_bull_deep|BULL_DEEP_THR=99"
 "bear_flat_v1|BEAR_MODE=flat BEAR_DEPTH_RAMP=0"
 "longwinner_on|LONG_MAX_RET3D=0.20"
)
run_one(){ local spec=$1; local tag=${spec%%|*}; local ov=${spec#*|}; [ "$tag" = "$ov" ] && ov=""
  run "$tag" $ov; }
export -f run_one
printf '%s\n' "${JOBS[@]}" | xargs -P 3 -I{} bash -c 'run_one "$@"' _ {}
echo DONE > "$B/loo.done"
