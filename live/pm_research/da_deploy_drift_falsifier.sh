#!/usr/bin/env bash
# DA round 57: does the midnight unit actually REFUSE when the deployed tree
# has drifted -- and does it still ADMIT when it has not?
#
# WHY THE GUARD'S OWN SELFTEST IS NOT ENOUGH. `da_deploy_guard.py --selftest`
# proves the CHECK discriminates. It cannot prove the LAUNCHER calls it, and
# this programme has already paid for that distinction: SEAT_PROTOCOL 17,
# "suite-green is not pipeline-wired" -- six evaluator functions, all
# falsifier-proven, zero call sites in the runner. So this file drives
# `da_midnight_verify.sh` ITSELF, end to end, on a relocated tree.
#
# BOTH DIRECTIONS, EVERY TIME, on the three surfaces that can be wrong:
#   * the unit REFUSES with rc 7 on planted drift, and WRITES NOTHING
#   * the unit ADMITS a clean deploy and logs the MATCH line
#   * the deploy act REFUSES a dirty tree, a behind tree and an ahead tree,
#     and ADMITS a clean one
#
# It runs the REAL shell and the REAL guard from this directory -- never a
# copy of the logic -- on a COPY OF THE TREE, so nothing it does can touch
# the production record, the installed unit, or the nightly log.
#
#   bash live/pm_research/da_deploy_drift_falsifier.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/yuqing/pricer-sol/venv/bin/python3
SH="$HERE/da_midnight_verify.sh"
GUARD="$HERE/da_deploy_guard.py"
DEPLOY="$HERE/da_deploy_midnight.sh"
for f in "$SH" "$GUARD" "$DEPLOY"; do
  [ -r "$f" ] || { echo "REFUSED: no $f -- a falsifier that cannot find its" \
                        "subject must not report that the subject passed."
                   exit 2; }
done
# THE SUBJECT MUST BE PRESENT IN THE SUBJECT. If the gate is deleted from the
# launcher this REFUSES instead of passing an empty battery.
grep -q 'DEPLOY_DRIFT (REFUSE_TIER_DRIFT)' "$SH" || {
  echo "REFUSED: the deploy gate is not in $SH"; exit 2; }
grep -q 'exit 7' "$SH" || { echo "REFUSED: $SH never exits 7"; exit 2; }

TMP="$(mktemp -d)"; if [ -z "${DA_FALS_KEEP:-}" ]; then trap "rm -rf $TMP" EXIT; else echo "KEEPING $TMP"; fi
fails=0; nchk=0
say() {
  nchk=$((nchk+1))
  if [ "$1" = "$2" ]; then echo "ok   $3"
  else echo "FAIL $3 (got '$2', want '$1')"; fails=$((fails+1)); fi
}

# ---------------------------------------------------------------- the tree
# A COPY, not a symlink farm: the point is to mutate files and see what the
# launcher does, and mutating the real tree is exactly the accident this
# whole instrument exists to prevent.
T="$TMP/tree"; P="$T/live/pm_research"
mkdir -p "$P/systemd"
for f in $("$PY" "$GUARD" compute-set --tree "$(cd "$HERE/../.." && pwd)" \
           | "$PY" -c 'import json,sys
for f in json.load(sys.stdin)["files"]: print(f["path"])'); do
  mkdir -p "$T/$(dirname "$f")"
  cp "$(cd "$HERE/../.." && pwd)/$f" "$T/$f"
done
cp "$HERE/systemd/da-midnight-verify.service" "$P/systemd/"
cp "$HERE/systemd/da-midnight-verify.timer" "$P/systemd/"
REC="$P/systemd/da_deploy_record.json"
MAN="$P/systemd/da_deploy_record.sha256"

# A stub unit file, so the record's installed-unit leg points at something
# this test owns rather than at the real ~/.config/systemd/user copy.
STUBUNIT="$TMP/stub.service"; cp "$P/systemd/da-midnight-verify.service" "$STUBUNIT"

git -C "$T" init -q
git -C "$T" config user.email da@x; git -C "$T" config user.name da
git -C "$T" add -A >/dev/null; git -C "$T" commit -qm initial
# A real `origin/mm-research` ref from the start, so section 3's FIRST check
# is a genuine positive control and not a refusal for want of a remote.
git -C "$T" branch -f mm-research HEAD 2>/dev/null
git -C "$T" remote add origin "$T" 2>/dev/null
git -C "$T" update-ref refs/remotes/origin/mm-research HEAD

# The record is written by the guard's own writer, on the RELOCATED tree, with
# the installed-unit leg pointed at a stub this test owns -- never at the real
# ~/.config/systemd/user copy.
mkrecord() {
  "$PY" - "$GUARD" "$T" "$REC" "$STUBUNIT" <<'PYEOF'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("g", sys.argv[1])
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
rec = g.build_record(sys.argv[2], by="falsifier", at_utc="1970-01-01T00:00:00Z",
                     units=(sys.argv[4],))
open(sys.argv[3], "w").write(json.dumps(rec, indent=2, sort_keys=True) + "\n")
open(sys.argv[3].replace(".json", ".sha256"), "w").write(g.manifest_text(rec))
PYEOF
}
mkrecord

# A stub verifier so a run that PASSES the gate finishes in milliseconds. The
# gate sits before every write, so what happens after it is irrelevant to
# this battery -- only whether the run got past it.
STUBV="$TMP/stubv.py"; printf 'import sys\nsys.exit(9)\n' > "$STUBV"

drive() {   # $1 = scratch tag; echoes "<rc>|<bytes written to the log>"
  local out="$TMP/out.$1" log="$TMP/log.$1" rc
  mkdir -p "$out"; : > "$log"
  DA_DEPLOY_TREE="$T" DA_DEPLOY_RECORD="$REC" \
  DA_MIDNIGHT_OUTDIR="$out" DA_MIDNIGHT_LOG="$log" \
  DA_MIDNIGHT_VERIFY_BIN="$STUBV" \
    bash "$P/da_midnight_verify.sh" >"$TMP/err.$1" 2>&1
  rc=$?
  echo "$rc|$(wc -c < "$log" | tr -d ' ')"
}

echo "== 1. THE LAUNCHER, END TO END =="
r="$(drive clean)"
say "0" "$([ "${r%%|*}" -eq 7 ] && echo 1 || echo 0)" \
    "POSITIVE CONTROL: a CLEAN deploy is ADMITTED -- the launcher does NOT \
exit 7 (a gate shown only to refuse has not been shown to pass)"
say "1" "$(grep -q 'deploy: DEPLOY_RECORD MATCH' "$TMP/log.clean" \
            && echo 1 || echo 0)" \
    "POSITIVE CONTROL: the MATCH line is written into the run log, naming the \
commit and who deployed -- the record is visible in the night's own record"

# The four REFUSE-tier shapes, driven through the launcher one at a time.
for tgt in da_forward_day_verify.py pm_tape_density.py da_race_withdrawals.py \
           da_content_liveness_rule.py da_deploy_guard.py; do
  cp "$P/$tgt" "$TMP/save.$tgt"
  printf '\n# planted drift\n' >> "$P/$tgt"
  r="$(drive "d.$tgt")"
  say "7|0" "$r" \
      "KNOWN-BAD: drift planted in $tgt REFUSES with rc 7 AND writes ZERO \
bytes -- refused before the log header, per this launcher's own lesson that a \
guard which writes before it refuses has already done the thing it refuses"
  cp "$TMP/save.$tgt" "$P/$tgt"
done

# The ExecStart script itself. Mutated in a way that would otherwise change
# behaviour, to make the point that the file under the gate IS the gate's file.
cp "$P/da_midnight_verify.sh" "$TMP/save.sh"
printf '\n# planted drift\n' >> "$P/da_midnight_verify.sh"
r="$(drive d.sh)"
say "7|0" "$r" \
    "KNOWN-BAD: drift in the ExecStart script ITSELF refuses -- the launcher \
digests its own bytes and will not run a version nobody deployed"
cp "$TMP/save.sh" "$P/da_midnight_verify.sh"

# REPORT tier must NOT refuse. If it did, the tiering would be decorative and
# DA's nightly duty would be hostage to DE's and BE's daily edits.
cp "$P/policy_optimizer.py" "$TMP/save.po"
printf '\n# planted drift\n' >> "$P/policy_optimizer.py"
r="$(drive d.po)"
say "0" "$([ "${r%%|*}" -eq 7 ] && echo 1 || echo 0)" \
    "POSITIVE CONTROL: drift in a REPORT-tier module does NOT refuse the night"
say "1" "$(grep -q 'policy_optimizer.py' "$TMP/log.d.po" && echo 1 || echo 0)" \
    "...and IS NAMED in the log -- a status, never a silent drop (rule 4)"
cp "$TMP/save.po" "$P/policy_optimizer.py"

# A missing record is not a pass.
mv "$REC" "$TMP/rec.away"
r="$(drive d.norec)"
say "7|0" "$r" \
    "KNOWN-BAD: NO record at all REFUSES -- 'never deployed by the act' must \
not read the same as 'deployed and unchanged' (rule 11)"
mv "$TMP/rec.away" "$REC"

# The installed unit, edited without touching the repo at all.
printf '\n# planted drift\n' >> "$STUBUNIT"
r="$(drive d.unit)"
say "7|0" "$r" \
    "KNOWN-BAD: the INSTALLED unit edited -- ExecStart repointed with the \
repo untouched -- REFUSES. The tree alone was never the whole deploy"
git -C "$T" checkout -- . 2>/dev/null
mkrecord
r="$(drive d.unitfixed)"
say "0" "$([ "${r%%|*}" -eq 7 ] && echo 1 || echo 0)" \
    "POSITIVE CONTROL: re-deploying against the edited unit ADMITS again -- \
the gate tracks the record, it is not stuck red"

echo
echo "== 2. THE OVERRIDE MAY NOT BE USED ON A CANONICAL RUN =="
# Neither OUTDIR nor LOG set => canonical. With DA_DEPLOY_RECORD set that must
# refuse: a record chosen by the caller certifies nothing.
DA_MIDNIGHT_MODE=production DA_DEPLOY_RECORD="$REC" \
  bash "$P/da_midnight_verify.sh" >"$TMP/err.canon" 2>&1
say "7" "$?" \
    "KNOWN-BAD: DA_DEPLOY_RECORD on a CANONICAL run REFUSES -- the same \
pin-vs-substitution line the verifier binary already draws"

echo
echo "== 3. THE DEPLOY ACT =="
run_deploy() { bash "$DEPLOY" --tree "$T" --no-install --by falsifier \
                 >"$TMP/dep.$1" 2>&1; echo $?; }
# THE ACT REFUSES WHILE ITS OWN RECORD IS UNCOMMITTED -- a real property, not
# a test artefact: the record it just wrote makes live/pm_research dirty, so a
# second deploy without a landing commit in between is refused. Land, then
# re-deploy. The falsifier commits between acts for exactly that reason.
land() { git -C "$T" add -A >/dev/null 2>&1
         git -C "$T" commit -qm land >/dev/null 2>&1
         git -C "$T" update-ref refs/remotes/origin/mm-research HEAD; }
land
say "0" "$(run_deploy clean)" \
    "POSITIVE CONTROL: the deploy act ADMITS a clean, pushed, up-to-date tree"
printf '\n# dirty\n' >> "$P/da_blackout_mask.py"
say "1" "$(run_deploy dirty)" \
    "KNOWN-BAD: the deploy act REFUSES a DIRTY live/pm_research -- digests \
that belong to no commit are not a record of anything"
say "1" "$(grep -q 'DIRTY' "$TMP/dep.dirty" && echo 1 || echo 0)" \
    "...and says DIRTY, so the operator is told which refusal fired"
git -C "$T" checkout -- . 2>/dev/null; land

# BEHIND and AHEAD, built as real refs rather than simulated.
say "0" "$(run_deploy sync)" \
    "POSITIVE CONTROL: HEAD == origin/mm-research ADMITS"
echo "x" > "$T/live/pm_research/extra.txt"
git -C "$T" add -A >/dev/null; git -C "$T" commit -qm ahead
say "1" "$(run_deploy ahead)" \
    "KNOWN-BAD: a tree AHEAD of origin REFUSES -- the recorded commit would be \
unpushed and unresolvable by anyone else"
git -C "$T" update-ref refs/remotes/origin/mm-research HEAD
git -C "$T" reset -q --hard HEAD~1
say "1" "$(run_deploy behind)" \
    "KNOWN-BAD: a tree BEHIND origin REFUSES -- deploying code the branch has \
already moved past"
git -C "$T" update-ref refs/remotes/origin/mm-research HEAD; land
say "0" "$(run_deploy recovered)" \
    "POSITIVE CONTROL: after a pull the act ADMITS again"

echo
if [ "$fails" -eq 0 ]; then
  echo "da_deploy_drift_falsifier: $nchk named checks PASSED"
  exit 0
fi
echo "da_deploy_drift_falsifier: $fails FAILURE(S) of $nchk"; exit 1
