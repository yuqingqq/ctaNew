"""Generate pf_step4_ladder_inv.py = the FROZEN ladder engine of pm_ladder_v1 (pf_step4_ladder.py sha256 c682229e...) with D1's
inventory mechanisms as opt-in rule tokens (user 2026-11-06: "We test these mechanisms like skewing, reducing only after the end, etc,
run v1 with these to check the result"). Every token is OFF unless named; without them every arm is the frozen engine's.
  SK<n>     D1 skew band (pmmmrust research.rs target_reducing): when |known net| > n shares and the spread is >= 2 ticks, the reducing
            side's price moves one tick toward the other side (bid + 1 / ask - 1), unless that would cross. Guards still apply.
  RX<n>     reducing side exempt from the per-minute pull line (PM<key>) when |known net| > n; the Binance guards, GS, G3, G4 and G6
            still apply. Needs a PM rule.
  RL<usd>   D1 A6 exposure limit: a place on the side that does not reduce |net| (touch and rung) is withheld when
            |known net| x r60[bin(sensitivity(mid, t))] >= usd; lookup from PF_R60_FILE (pf_inv_r60.py).
  CAP<n>, RO<s>  the frozen engine's tokens, now allowed with the ladder and per-minute lines; both also hold the rung (RO phase: no rung).
Not with IM<key> (refused). The frozen file is never edited. Refuses unless the source hash matches and each anchor occurs exactly once.
Parity (pf_inv_check.py): the v1 arm run by the generated engine is byte-identical to round i; never-trigger arms (SK1000, RX1000,
CAP1000, RL1000000, RO300) equal v1; an engaged arm differs.
Usage: python3 -m predictfun.research.pf_ladder_inv_build
"""
import sys

PATCHES = [
    # 1. allow CAP / RO (and the new tokens) with the ladder and the per-minute lines; refuse them with IM; RX needs PM
    ("or LEGACY_INTENT or LV['ro'] is not None or LV['capn'] is not None or LV['bk'] is not None):\n",
     "or LEGACY_INTENT or LV['bk'] is not None or (IMK is not None and any(LV[k_] is not None for k_ in ('ro', 'capn', 'sk', 'rx', 'rl')))):   # inv build\n"),
    ("        raise SystemExit('REFUSED: ladder / improve / per-minute rules are modelled for the split-latency live-like arms only')\n",
     "        raise SystemExit('REFUSED: ladder / improve / per-minute rules are modelled for the split-latency live-like arms only')\n"
     "    if LV['rx'] is not None and PMK is None:\n"
     "        raise SystemExit('REFUSED: RX<n> exempts the reducing side from the PM line; it needs a PM<key> rule')   # inv build\n"),
    # 2. tokens
    ("capn=None, ro=None)\n", "capn=None, ro=None, sk=None, rx=None, rl=None)\n"),
    ("        elif r.startswith('RO') and r[2:].isdigit():\n            o['ro'] = float(r[2:])\n",
     "        elif r.startswith('RO') and r[2:].isdigit():\n            o['ro'] = float(r[2:])\n"
     "        elif r.startswith('SK') and r[2:].isdigit():                 # inv build\n            o['sk'] = float(r[2:])\n"
     "        elif r.startswith('RX') and r[2:].isdigit():                 # inv build\n            o['rx'] = float(r[2:])\n"
     "        elif r.startswith('RL') and r[2:].replace('.', '', 1).isdigit():   # inv build\n            o['rl'] = float(r[2:])\n"),
    # 3. the A6 lookup and exposure
    ("LINES = json.load(open(f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']\n",
     "LINES = json.load(open(f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']\n"
     "from .pf_inv_r60 import sensitivity as _inv_sens, bin_of as _inv_bin   # inv build: D1 A6 exposure (RL<usd>)\n"
     "R60 = json.load(open(os.environ['PF_R60_FILE'])) if os.environ.get('PF_R60_FILE') else None\n\n\n"
     "def r60_exposure(net_, mid_c, rel_s):\n"
     "    \"\"\"D1 A6: |net| x r60[bin(sensitivity(mid, t))] in $ (mid in cents; outside the window 0).\"\"\"\n"
     "    if not 0.0 <= rel_s < 300.0:\n"
     "        return 0.0\n"
     "    return abs(net_) * R60['move_per_share'][_inv_bin(R60['edges'], _inv_sens(mid_c / 100.0, rel_s))]\n"),
    # 4. RX: the reducing side ignores the per-minute pull line
    ("            ba = ba or (vv_[0] is not None and vv_[0] <= c_); sa = sa or (vv_[1] is not None and vv_[1] <= c_)\n",
     "            pmf_ = [vv_[0] is not None and vv_[0] <= c_, vv_[1] is not None and vv_[1] <= c_]\n"
     "            if LV['rx'] is not None and abs(knet) > LV['rx'] + 1e-9:     # inv build RX<n>: the reducing side ignores the line\n"
     "                s_r = 0 if knet < 0 else 1\n"
     "                if pmf_[s_r]:\n"
     "                    pmf_[s_r] = False; withheld['RX_exempt'] += 1\n"
     "            ba = ba or pmf_[0]; sa = sa or pmf_[1]\n"),
    # 5. SK: D1 skew band on the touch price
    ("        rung = [None, None]; rz = [0.0, 0.0]\n",
     "        if LV['sk'] is not None and abs(knet) > LV['sk'] + 1e-9 and a1 - b1 >= 2:   # inv build SK<n>: D1 skew band (target_reducing)\n"
     "            s_r = 0 if knet < 0 else 1                       # the reducing side moves one tick toward the other side\n"
     "            if q[s_r] is not None:\n"
     "                np_ = q[s_r] + 1 if s_r == 0 else q[s_r] - 1\n"
     "                if (s_r == 0 and np_ < a1) or (s_r == 1 and np_ > b1):\n"
     "                    q[s_r] = np_; bkside[s_r] = False; withheld['SK_shift'] += 1\n"
     "        rung = [None, None]; rz = [0.0, 0.0]\n"),
    # 6. RO / CAP / RL also hold the rung
    ("                        withheld['LD_G6_hard'] += 1; continue\n",
     "                        withheld['LD_G6_hard'] += 1; continue\n"
     "                red_r = (s_ == 0 and knet < 0) or (s_ == 1 and knet > 0)    # inv build: RO / CAP<n> / RL<usd> also hold the rung\n"
     "                if LV['ro'] is not None and rel >= LV['ro']:\n"
     "                    withheld['LD_RO'] += 1; continue\n"
     "                if LV['capn'] is not None and not red_r and abs(knet) >= LV['capn'] - 1e-9:\n"
     "                    withheld['LD_CAPN'] += 1; continue\n"
     "                if LV['rl'] is not None and not red_r and r60_exposure(knet, (b1 + a1) / 2.0, rel) >= LV['rl']:\n"
     "                    withheld['LD_RL'] += 1; continue\n"),
    # 7. RL on the touch
    ("                q[side] = None; withheld['CAPN'] += 1; continue\n",
     "                q[side] = None; withheld['CAPN'] += 1; continue\n"
     "            if LV['rl'] is not None and not red_ and r60_exposure(knet, (b1 + a1) / 2.0, rel) >= LV['rl']:\n"
     "                q[side] = None; withheld['RL'] += 1; continue          # inv build RL<usd>: D1 A6 exposure limit\n"),
    # 8. RL needs the lookup
    ("if any(r[0] == 'V' for a in arms for r in a[1]) and not (PREDVB and PREDVA):\n",
     "if any(live_opts(set(a[1]))['rl'] is not None for a in arms) and R60 is None:   # inv build\n"
     "    sys.exit('REFUSED: RL<usd> arms need PF_R60_FILE (step4/inv_r60.json from pf_inv_r60.py)')\n"
     "if any(r[0] == 'V' for a in arms for r in a[1]) and not (PREDVB and PREDVA):\n"),
]


def main():
    from .pf_ladder_fwd_build import HERE, generate
    generate(PATCHES, f'{HERE}/pf_step4_ladder_inv.py', 'D1 inventory tokens SK / RX / RL, CAP / RO with the ladder (pf_ladder_inv_build)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
