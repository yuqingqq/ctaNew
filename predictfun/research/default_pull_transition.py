"""Finish the active replay, then resume frozen tests with two-slot scheduling."""
import datetime as dt
import fcntl
import json
import re
import subprocess
import time
from pathlib import Path
from default_pull_checkpoint import WORK, ROOT, OUT, REL, sha, need, run as checkpoint
from default_pull_execution import controls as execution_controls, progress

OLD_PID = 2447808
STAGE = 'stress_20261101'


def successful(line, stage):
    return re.fullmatch(
        r'\S+ tag=default_pull_v4_' + re.escape(stage) +
        r' rc=0 peak_rss_mib=\d+ wall_s=[0-9.]+ cpu_user_s=[0-9.]+'
        r' memavail_start_gib=\d+ disk_free_gib=\d+->\d+( slot=codex)?', line
    ) is not None


def controls():
    execution_controls()
    line = '2026-11-12T00:00:00Z tag=default_pull_v4_stress_20261101 rc=0 peak_rss_mib=1 wall_s=1.0 cpu_user_s=1.0 memavail_start_gib=20 disk_free_gib=500->500 slot=codex'
    if not successful(line, STAGE) or successful(line.replace('rc=0', 'rc=1'), STAGE) or successful(line, 'stress_20261102'):
        raise ValueError('exit-line positive/refusal controls failed')


def save(name, value):
    for base in (OUT, WORK / REL, ROOT / REL):
        with (base / name).open('x') as f:
            json.dump(value, f, indent=2); f.write('\n')


def launch(args, log):
    with log.open('xb') as f:
        p = subprocess.Popen(args, cwd=WORK, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(1)
    if p.poll() is not None:
        raise RuntimeError(log.read_text())
    return p.pid


def run():
    controls()
    p = Path(f'/proc/{OLD_PID}/cmdline')
    while p.exists():
        try:
            args = p.read_bytes().decode().strip('\0').split('\0')
        except FileNotFoundError:
            break
        if args != ['bash', 'predictfun/research/run_default_pull_v4.sh']:
            raise ValueError('old driver identity changed')
        time.sleep(5)
    with (OUT / 'driver.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log = OUT / 'logs' / (STAGE + '.log')
        text = log.read_text()
        if not successful(text.splitlines()[-1], STAGE):
            raise ValueError('current replay did not complete successfully')
        features = json.loads((OUT / 'runs/stress/20261101/feature_check.json').read_text())
        count = sum(r['decisions'] for r in features['rows'])
        if '\nVERIFIED_RUN 20261101 stress\n' not in text or f'\nFEATURE_IDENTITIES_PASS 20261101 stress {count}\n' not in text:
            raise ValueError('current replay checks missing')
        done = OUT / 'logs' / (STAGE + '.done')
        recovered = not done.exists()
        if recovered:
            with done.open('x') as f:
                f.write(sha(log) + '  ' + str(log) + '\n')
        try:
            checkpoint('stress', '20261101')
        except BaseException:
            if recovered:
                done.unlink()
            raise
        progress(STAGE, 'complete', 'Verified unchanged replay after scheduler handover; two global research slots, Codex core 8.')
        save('two_slot_transition_completed.json', dict(
            as_of=dt.datetime.now(dt.timezone.utc).isoformat(), old_driver_pid=OLD_PID,
            status='CURRENT_REPLAY_VERIFIED_AND_OLD_DRIVER_RETIRED', completed_stage=STAGE,
            recovered_marker=recovered, log_sha=sha(log), feature_decisions=count,
            source_sha=sha(__file__), authorization_sha=sha(OUT / 'execution_two_slots_20261112.json'),
            scope='Scheduling transition only. Successful current replay and all saved artifact hashes verified; numerical output unchanged.'
        ))
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    driver_log = OUT / 'logs' / f'driver_two_slots_{stamp}.log'
    driver = launch(['bash', 'predictfun/research/run_default_pull_two_slots.sh'], driver_log)
    watcher_log = OUT / 'logs' / f'publication_watcher_two_slots_{stamp}.log'
    watcher = launch(['python3', 'predictfun/research/default_pull_watch.py', '--driver-pid', str(driver)], watcher_log)
    result = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), status='TWO_SLOT_DRIVER_RUNNING',
                  driver_pid=driver, watcher_pid=watcher, driver_log=str(driver_log), watcher_log=str(watcher_log),
                  process_cap=2, slot='codex', core=8, source_sha=sha(__file__))
    save('two_slot_driver_started.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'controls':
        controls(); print('TRANSITION_CONTROLS_PASS')
    else:
        run()
