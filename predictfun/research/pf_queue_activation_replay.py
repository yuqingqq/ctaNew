"""Explicit entry/STOP replay entrypoints for activation-only initialization.

Run under run_light.sh, with sources/inputs frozen before historical execution.
Entry reuses the already fitted local models as a queue-only sensitivity.
STOP reuses movement predictions, but recalibrates its policy in a distinct
artifact namespace. Neither path rebuilds entry training labels implicitly.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest import mock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_queue_activation import CONTRACT
import pf_entry_queue_activation_lane as ENTRY
import pf_stop_queue_activation_lane as STOP
from pf_queue_activation_check import Checks


def checks():
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise ValueError('queue initialization regressions failed')


@contextmanager
def corrected_entry(module):
    with mock.patch.object(module, 'L', ENTRY):
        yield


@contextmanager
def corrected_stop(module):
    # The legacy instrument checks run before this context. Its old-lane
    # equivalence tests are not claims about the deliberately changed queue.
    with mock.patch.object(module.M, 'sim_joint_g', STOP.sim_joint_g), \
            mock.patch.object(module, 'all_selftests', lambda: None), \
            mock.patch.object(module, 'VS', module.VS + '_qinit_activation_v1'):
        yield


def stamp(path, scope):
    """Finish the new run's receipt before returning it to the caller."""
    result = json.loads(path.read_text())
    result['queue_initialization'] = CONTRACT
    result['queue_initialization_scope'] = scope
    result['forward_validation'] = False
    result['promoted'] = False
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def entry(day, out):
    import pf_entry_local as P
    if day not in P.TESTS:
        raise ValueError('undeclared day')
    source = Path(P.K.IO) / 'entry_local_20261115_v3'
    if out.resolve() == source.resolve():
        raise ValueError('separate output namespace required')
    marker = out / 'queue_initialization.json'
    if out.exists() and (not marker.is_file() or json.loads(marker.read_text()).get('contract') != CONTRACT):
        raise ValueError('output namespace belongs to another experiment')
    target = out / ('replay_' + day)
    if target.exists():
        raise FileExistsError(target)
    checks()
    out.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with marker.open('x') as stream:
            json.dump(dict(contract=CONTRACT, purpose='fixed-model queue initialization sensitivity'), stream)
            stream.write('\n')
    for name in [*['rows_' + d for d in P.DAYS], *['fit_' + d for d in P.TESTS]]:
        original = source / name
        if not original.is_dir():
            raise FileNotFoundError(original)
        link = out / name
        if link.is_symlink():
            if link.resolve() != original.resolve():
                raise ValueError('unexpected artifact reuse')
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(original, target_is_directory=True)
    with corrected_entry(P):
        P.replay(day, out)
    stamp(target / 'receipt.json', 'Fixed previously fitted models; replay-only initialization sensitivity. Labels and models were fitted under the historical queue and are not corrected training evidence.')


def stop(command, day):
    import pf_stop_play as P
    if command != 'report' and day not in P.M.TESTS:
        raise ValueError('undeclared day')
    suffix = P.VS + '_qinit_activation_v1'
    folder = Path(P.OUT)
    name = ('stop_play' + suffix if command == 'report' else
            ('play_cal_' if command == 'cal' else 'play_') + day + suffix)
    target = folder / (name + '.json')
    if target.exists() or target.with_suffix('.txt').exists():
        raise FileExistsError(target)
    needed = []
    if command == 'test':
        needed = [folder / ('play_cal_' + day + suffix + '.json')]
    elif command == 'report':
        needed = [folder / (prefix + d + suffix + '.json')
                  for d in P.M.TESTS for prefix in ('play_cal_', 'play_')]
    for path in needed:
        if json.loads(path.read_text()).get('queue_initialization') != CONTRACT:
            raise ValueError('mixed initialization contracts: ' + str(path))
    P.all_selftests()
    checks()
    with corrected_stop(P):
        if command == 'report':
            P.report()
        else:
            P.play_day(day, command)
    stamp(target, 'Movement predictions reused; policy calibration/replay uses activation-only initialization. Separate from all historical STOP-PLAY artifacts.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='lane', required=True)
    e = sub.add_parser('entry'); e.add_argument('day'); e.add_argument('--out', type=Path, required=True)
    s = sub.add_parser('stop'); s.add_argument('command', choices=['cal', 'test', 'report']); s.add_argument('day', nargs='?')
    a = p.parse_args()
    if a.lane == 'entry':
        entry(a.day, a.out)
    else:
        stop(a.command, a.day)
