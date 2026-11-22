"""AMENDMENT 1 to LONG-VALUE-RETENTION-DISCRIMINATION-v1 (P-2026-005; 2026-11-22, session aea5bfb9). The Oct 26 rest job stopped at the frozen
memory self-guard (1,450 MiB): its VmHWM was 1,451 MiB after its first market, before any output was written (logs/rd_rest_20261026.log). Per
the USER's memory rule (never raise a cap; redesign with a measured buffer), this runner loads ONE market's inputs at a time
(pf_ladder_long_continuation.market_inputs(day, [m])) and frees them (and collects) before the next market. Everything else is the frozen
run_part of pf_ladder_retention_discrimination (34094ec2, unchanged): the same checks, selected states, run_market, market order, output
files and bytes, and the same 1,450 MiB self-guard (not raised). It refuses unless the frozen builder / selftest / declaration / manifest /
census / launcher / checker are the committed ones, this file is committed, and amend1_manifest.json (write-once) binds this file's sha256,
the frozen manifest and the census. Equivalence (tools/rehearse_amend1.py): on two synthetic days the frozen run_part and this runner write
byte-identical pairs / parity / ledgers. Applies to the rest day it is run on (Oct 26); the prefix and the other rest days stand as run.
Usage (run_light): pf_ladder_retention_discrimination_amend1.py manifest | rest <day>"""
import gc
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_retention_discrimination as RD          # noqa: E402  frozen (34094ec2)

LC, AF, pd, need = RD.LC, RD.AF, RD.pd, RD.need
NAME = 'pf_ladder_retention_discrimination_amend1.py'
AMEND = RD.REPO / RD.RECEIPT / 'amend1_manifest.json'


def own_sha():
    return hashlib.sha256((HERE / NAME).read_bytes()).hexdigest()


def build(path=None):
    path = Path(path or AMEND); need(not path.exists(), 'the amendment manifest exists (write-once)'); gm_b = (RD.REPO / RD.RECEIPT / 'manifest.json').read_bytes(); gm = json.loads(gm_b)
    am = dict(amendment='amend1_per_market_loading', source=own_sha(), manifest_sha256=hashlib.sha256(gm_b).hexdigest(), census_sha256=gm['census_sha256'], frozen_sources=gm['sources'],
              self_guard_mib=1450., applies_to=['rest/20261026'], reason='the frozen Oct 26 rest job stopped at the 1,450 MiB self-guard (VmHWM 1,451) before any output',
              asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.write_text(json.dumps(am, indent=1, sort_keys=True) + '\n'); return am


def run_part_amend(part, day, out=None, require_frozen=True, man_path=None, proj_path=None, amend_path=None):
    """The frozen run_part with one market's inputs loaded at a time (header)."""
    if require_frozen:
        need(RD.frozen() and RD.committed(f'predictfun/research/{NAME}') == (HERE / NAME).read_bytes(), 'the frozen builder or this amendment is not the committed one')
    gm_b = Path(man_path or RD.REPO / RD.RECEIPT / 'manifest.json').read_bytes(); gm = json.loads(gm_b); need(gm['sources'] == RD.sources(), 'a source differs from the frozen one')
    am = json.loads(Path(amend_path or AMEND).read_text())
    need(am['source'] == own_sha() and am['manifest_sha256'] == hashlib.sha256(gm_b).hexdigest() and am['census_sha256'] == gm['census_sha256'] and f'{part}/{day}' in am['applies_to'],
         'the amendment manifest does not bind this runner, the manifest, the census or this day')
    need(day in gm['parts'][part], f'{day} is not in the {part} part'); gm_af = LC.af_manifest(); need(RD.af_records(gm_af) == gm['af_records'], 'an ASYNC record differs from the manifest')
    pp = Path(proj_path or RD.REPO / RD.RECEIPT / 'logs' / 'projection.json')
    need(part == 'prefix' or (pp.exists() and json.loads(pp.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')
    P_ = RD.day_paths(part, day, out); need(not P_['rec'].exists(), f'{day}: the output exists (write-once)'); P_['rec'].parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    pairs, par, lgs = [], [], []
    for mp in ('pilot', 'rest'):
        for m in gm['markets'][day][mp]:
            states = [dict(s) for s in gm['selected'] if s['day'] == day and int(s['m']) == m]
            if not states:
                continue
            inp, edges = LC.market_inputs(day, [m], gm_af); path, ident, cache = inp[m]
            a_, b_, c_ = RD.run_market(day, m, mp, path, ident, cache, edges, states); pairs += a_; par.append(b_); lgs += c_
            del inp, path, ident, cache; gc.collect(); need(AF.LS.hwm() < 1450., f'{day} {m}: memory above the self-guard')
    pd.DataFrame(pairs).to_csv(P_['pairs'], index=False); P_['parity'].write_text(json.dumps(par, indent=1, default=str) + '\n')
    P_['ledgers'].write_bytes(gzip.compress('\n'.join(json.dumps(x, default=str) for x in lgs).encode(), mtime=0))
    P_['rec'].write_text(json.dumps(dict(version=RD.VERSION, part=part, day=day, pairs=len(pairs), sha256={k: RD.sha_b(P_[k].read_bytes()) for k in ('pairs', 'parity', 'ledgers')},
                                         sources=dict(RD.sources(), **{NAME: own_sha()}), amendment='amend1_per_market_loading', elapsed_s=time.time() - t0, hwm_mib=AF.LS.hwm(),
                                         asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1) + '\n')
    print(json.dumps(dict(part=part, day=day, pairs=len(pairs), parity_all=all(x['all'] for x in par), statuses=pd.Series([x['status'] for x in pairs]).value_counts().to_dict()), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'manifest':
        print(json.dumps(build(), indent=1))
    elif cmd == 'rest':
        run_part_amend('rest', sys.argv[2])
    else:
        sys.exit('usage: pf_ladder_retention_discrimination_amend1.py manifest | rest <day>')
