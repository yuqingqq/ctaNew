import sys; sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import numpy as np
import pf_causal_speed as C, pf_qr_queue_race as Q
# old grid bid at 50 (sent t=0, live 40, pending-cancel to 95) and a birth entry at 51 (decided 15, live 25, p=10).
# One taker order at t=60: legs (51, 2 sh) and (50, 4 sh); 3 sh shown ahead at 50, 2 sh at 51.
X = Q.syn([(-100, [(50, 3)], 53), (10, [(51, 2), (50, 3)], 53)], [(60, 51, 2), (60, 50, 10)], t_end=300.0, kdelay=0.0)
X['best'] = [50] + X['best'][1:]   # grid step 0 knows only the first frame
F, O = C.sim5(X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'], X['tt'],
              X['tp'], X['tq'], 'M', X['t_end'], sched=[(15.0, 51, 15.0, 10.0)])
print('orders (pt, ts, ta, tr):', [(o['pt'], o['ts'], o['ta'], o['tr']) for o in O])
print('fills (trade t, leg price, our price, shares, phase):', [(X['tt'][f[0]], X['tp'][f[0]], f[1], f[2], f[5]) for f in F])
print('our shares from the 10-sh leg at 50:', sum(f[2] for f in F if X['tp'][f[0]] == 50))
