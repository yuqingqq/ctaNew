"""pm_ladder_v3 (user 2026-11-07: "Go ahead to freeze") = pm_ladder_v1 + CAP10 + RO60 + S240, i.e. arm
  MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95
Same guard models, thresholds, per-minute lines, clock and rules as v1 (predictfun/freeze/pm_ladder_v1); the three tokens are the
frozen engine's CAP<n> / RO<s> / S<s>, allowed with the ladder and the per-minute lines by pf_ladder_inv_build's patches:
  CAP10   a side (touch and rung) that does not reduce |known net| is withheld at |known net| >= 10 shares.
  RO60    from 60 s only the side that reduces |known net| is quoted, sized min(size rule, |known net|) (never flips); no rung.
  S240    stop at 240 s (cancel all; the rest is carried to settlement). The per-minute lines of minutes 1-3 apply in 60-240 s.
Generates pf_step4_ladder_v3.py = the FROZEN engine (pf_step4_ladder.py sha256 c682229e...) with pf_ladder_inv_build.PATCHES plus
pf_ladder_fwd_build.FWD_PATCH (input-day bound FORWARD from PF_FORWARD, default the frozen '20261104'), and writes
predictfun/freeze/pm_ladder_v3/params.json (arm, comparators, engine hashes, v1 artifact hashes re-verified on disk).
Parity of the generated engine: pf_v3_engine_check.py. Refuses unless every source hash matches.
Usage: python3 -m predictfun.research.pf_ladder_v3_build
"""
import hashlib, json, os, sys
from .pf_ladder_fwd_build import HERE, FROZEN, FWD_PATCH, generate
from .pf_ladder_inv_build import PATCHES

DST = f'{HERE}/pf_step4_ladder_v3.py'
FZ = os.path.normpath(f'{HERE}/../freeze'); OUT = f'{FZ}/pm_ladder_v3/params.json'
ARM = 'MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240|mid|AM:5:25:95'
V1 = 'MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95'
CTRL = 'MICRO|G4+G2+G6+G5+G3+G1+V0+GS+XNA+XMIN1+XV+S60|mid|AV:5:25:0.06'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    generate(PATCHES + [FWD_PATCH], DST, 'D1 inventory tokens (pf_ladder_inv_build) + FORWARD from PF_FORWARD (pf_ladder_fwd_build); pm_ladder_v3')
    p1 = json.load(open(f'{FZ}/pm_ladder_v1/params.json'))
    if p1['arm'] != V1 or p1['control'] != CTRL:
        sys.exit('REFUSED: v1 params arm / control differ from the declared ones')
    for s in ('bid', 'ask'):
        if sha(f"{FZ}/pm_ladder_v1/{p1['guard']['models'][s]}") != p1['guard']['model_sha256'][s]:
            sys.exit(f'REFUSED: v1 guard model {s} hash differs')
    if sha(p1['minute_lines']['source']) != p1['minute_lines']['sha256']:
        sys.exit('REFUSED: minute_lines_t60k.json hash differs from the v1 freeze')
    a3, a1 = ARM.split('|'), V1.split('|')
    if (a3[0], a3[2], a3[3]) != (a1[0], a1[2], a1[3]) or set(a3[1].split('+')) - {'CAP10', 'RO60', 'S240'} != set(a1[1].split('+')) - {'S60'}:
        sys.exit('REFUSED: v3 is not v1 + CAP10 + RO60 with the stop moved to 240 s')
    out = dict(candidate='pm_ladder_v3', arm=ARM, differs_from_v1=['CAP10', 'RO60', 'S240 (v1: S60)'], v1_arm=V1, control=CTRL,
               engine=dict(file='predictfun/research/pf_step4_ladder_v3.py', sha256=sha(DST), source='predictfun/research/pf_step4_ladder.py',
                           source_sha256=FROZEN, builder='predictfun/research/pf_ladder_v3_build.py', builder_sha256=sha(__file__),
                           inv_patches_sha256=sha(f'{HERE}/pf_ladder_inv_build.py'), fwd_patch_sha256=sha(f'{HERE}/pf_ladder_fwd_build.py')),
               guard=dict(from_freeze='predictfun/freeze/pm_ladder_v1', models=p1['guard']['models'], model_sha256=p1['guard']['model_sha256'],
                          threshold_c60=p1['guard']['threshold_c60'], features=p1['guard']['features'], value=p1['guard']['value']),
               minute_lines=p1['minute_lines'],
               rules_added=dict(CAP10='a side (touch and rung) that does not reduce |known net| is withheld at |known net| >= 10 shares',
                                RO60='from 60 s only the side that reduces |known net| is quoted, sized min(size rule, |known net|); no rung',
                                S240='stop at 240 s; cancel all; the rest is carried to settlement; per-minute lines of minutes 1-3 apply in 60-240 s'))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} (engine sha256 {out["engine"]["sha256"]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
