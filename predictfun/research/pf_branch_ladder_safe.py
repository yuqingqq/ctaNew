"""SAFE SAVE / RESUME for the ladder client (branch_ladder_safe_v1; P-2026-005; 2026-11-21, session aea5bfb9). Review 0065191f finding 5: the
frozen pf_branch_ladder.resume (kept unchanged, bound into the LADDER_REST run's identity; it MUST NOT be used) accepted a changed next_k, a
changed envelope ident, a runtime cap changed inside the pickle and an external path / mark change with a matching envelope tape id while the
executed object kept the original tape. This module binds them:
 - fingerprint_v2: pf_branch_ladder.fingerprint plus the executable fields (configuration, q0, cap, levels, inventory, own_cross, END, the
   engine's END and client mode);
 - save_state_v2 at a tick boundary only: next_k = B.k + 1 with both sides processed (or 0 for an unstarted branch); the envelope carries the
   tick, the processed sides, ident, tape id, mark, mode, unclosed gids, settings and fingerprint_v2;
 - resume_v2 refuses, before any tick: another contract / configuration / END; a path whose tape id differs from the envelope; an object
   whose fingerprint_v2, own tape id, mark, ident, settings, executable fields, mode, unclosed gids, tick or processed sides differ from the
   envelope or the declared configuration; a next_k other than the object's next tick.
Selftest: pf_ladder_audit_selftest.py (the positive control and the review's four falsifiers)."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_branch_ladder as LB                             # noqa: E402  frozen: the client
BE, C = LB.BE, LB.C
CONTRACT = 'branch_ladder_safe_v1'


def exec_fields(B):
    return [B.config, B.q0, B.cap, B.levels, B.inv, B.own_cross, B.end, B.E.end, B.E.client]


def fingerprint_v2(B):
    return hashlib.sha256(json.dumps([LB.fingerprint(B), exec_fields(B)], sort_keys=True, default=str).encode()).hexdigest()


def started(B):
    return bool(B.decisions or B.E.fills or B.E.o or B.k or B.acted)


def save_state_v2(B, next_k):
    """The branch at a tick boundary with its bound envelope (next_k = B.k + 1 with both sides processed, or 0 unstarted)."""
    C.require(isinstance(B, LB.LadderBranch), 'not a ladder branch')
    C.require((int(next_k) == 0 and not started(B)) or (int(next_k) == B.k + 1 and B.acted == {0, 1}), 'a save that is not at a tick boundary')
    return dict(contract=CONTRACT, config=B.config, end=B.end, settings=dict(B.settings), ident=dict(B.ident), tape_id=BE.tape_id(B.E.f, B.E.tr, B.mark),
                mark=B.mark, mode=B.mode, unclosed={str(s): list(v) for s, v in B.unclosed.items()}, k=int(B.k), acted=sorted(B.acted), next_k=int(next_k),
                fingerprint=fingerprint_v2(B), blob=pickle.dumps(B))


def resume_v2(saved, config, path, end):
    """The saved branch for the same configuration, END and tape -> (branch, next tick); every disagreement refused before any tick runs."""
    C.require(isinstance(saved, dict) and saved.get('contract') == CONTRACT, 'not a saved branch_ladder_safe_v1 state')
    C.require(config in LB.CONFIGS and saved['config'] == config and float(saved['end']) == float(end), 'a saved state of another configuration / END')
    C.require(BE.tape_id(path['frames'], path['trades'], path.get('mark')) == saved['tape_id'], "a path other than the saved state's tape")
    B = pickle.loads(saved['blob']); cf = LB.CONFIGS[config]
    C.require(isinstance(B, LB.LadderBranch) and fingerprint_v2(B) == saved['fingerprint'], 'the saved branch changed after it was saved (fingerprint_v2)')
    C.require(BE.tape_id(B.E.f, B.E.tr, B.mark) == saved['tape_id'] and BE.canon_mark(B.mark) == BE.canon_mark(saved['mark']), "the object's own tape / mark differ from the envelope")
    C.require(B.ident == saved['ident'], "the object's ident differs from the envelope")
    want = dict(BE.SETTINGS, contract=LB.CONTRACT, config=config, end_ms=float(end), q0=cf['q0'], cap=cf['cap'], levels=cf['levels'], inventory=cf['inventory'],
                own_cross=cf['own_cross'], qmin=LB.QMIN, tick=LB.TICK)
    C.require(B.settings == saved['settings'] == want, 'the settings differ from the declared configuration')
    C.require(exec_fields(B) == [config, cf['q0'], cf['cap'], cf['levels'], (dict(cf['inventory']) if cf['inventory'] else None), cf['own_cross'], float(end), float(end), True],
              'the executable fields differ from the declared configuration')
    C.require(B.mode == saved['mode'] and {str(s): list(v) for s, v in B.unclosed.items()} == saved['unclosed'], 'the mode / unclosed gids differ from the envelope')
    C.require(int(saved['k']) == int(B.k) and saved['acted'] == sorted(B.acted), 'the tick / processed sides differ from the envelope')
    C.require((int(saved['next_k']) == 0 and not started(B)) or (int(saved['next_k']) == B.k + 1 and B.acted == {0, 1}), "next_k is not the object's next tick")
    return B, int(saved['next_k'])
