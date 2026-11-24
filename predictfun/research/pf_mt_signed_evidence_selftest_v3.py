"""V2 reuses the frozen v1 controls; adds the historical operator wrapper."""
import pf_mt_signed_evidence_v3 as V
import pf_mt_signed_evidence_selftest as T

T.decode = V.decode
T.main()
v = [123456789, int('11'*20,16), int('11'*20,16), 0, V.E18, 4*V.E18, 10*V.E18, 4102444800, 0, 200, 0, 0]
tk = v.copy(); tk[0] += 1
# Synthetic 8-word wrapper head. Fee/auxiliary args are opaque, not used as fills.
t, m = T.enc_order(tk), V.word(1)+V.word(32)+T.enc_order(v)
head = V.word(256)+V.word(256+len(t))+V.word(123)*6
inp = '0x92895911'+(head+t+m).hex()
assert [r['values'] for r in V.decode(inp)] == [tk,v]
T.expect_bad(lambda: V.decode(inp[:-2]), 'signature length')
assert [r['values'] for r in V.decode(inp + b'{\"created_at\":123}'.hex())] == [tk,v]
bad = bytearray.fromhex(inp[2:]); bad[4:36] = V.word(2**255)
T.expect_bad(lambda: V.decode('0x'+bad.hex()), 'offset')
print('operator_wrapper_positive_and_two_refusals=PASS')
