import copy
import audit as A
def rec(f,mark=50):
 r=dict(fills=f,mark=mark,end=1000);c,v,total=A.independent_fifo(r);r.update(value=total,net_value_c=total,metrics=dict(completed_c=c,residual_c=v));return r
def fill(t,s,q,p,it):return [t,it,s,q,p,it]
def pair(a,b):
 for r in [a,b]:A.verify_record(r)
 z=A.decompose(a,b);A.verify_rows(a,b,z);return {r['category']:r for r in A.group(z)}
# Same long entry, better exit; same short entry, worse cover.
a=rec([fill(0,0,10,40,0),fill(100,1,10,50,1)]);b=rec([fill(0,0,10,40,0),fill(120,1,10,55,2)])
assert pair(a,b)['common_closed']['delta_c']==50
a=rec([fill(0,1,10,60,0),fill(100,0,10,40,1)]);b=rec([fill(0,1,10,60,0),fill(120,0,10,45,2)])
assert pair(a,b)['common_closed']['delta_c']==-50
# Partial entry overlap, residual vs close; extra quantity has its own entry return.
a=rec([fill(0,0,10,40,0)]);b=rec([fill(0,0,15,40,0),fill(100,1,12,45,1)])
z=pair(a,b);assert z['common_close_vs_residual']['delta_c']==-50 and z['new_entry']['delta_c']==40
assert sum(r['delta_c'] for r in z.values())==-10
assert pair(b,a)['foregone_entry']['delta_c']==-40
# Existing fill closes a lot in one path but opens a short in the other.
a=rec([fill(0,0,10,40,0),fill(100,1,10,60,1)]);b=rec([fill(100,1,10,60,1),fill(200,0,10,50,2)])
z=pair(a,b);assert z['new_role']['delta_c']==100 and z['foregone_entry']['delta_c']==-200
# Two prices, partial closes, duplicated public event through different gids.
a=rec([fill(0,0,5,40,0),fill(0,0,5,40,0),fill(20,0,10,45,1),fill(100,1,15,55,2)])
b=rec([fill(0,0,8,40,0),fill(20,0,10,45,1),fill(100,1,4,55,2),fill(150,1,7,53,3)])
pair(a,b);pair(b,a);assert all(abs(z['delta_c'])<1e-9 for z in pair(a,a).values())
print('PASS common long/short exits, residual, extra/foregone volume, role changes, multi-price partial FIFO, same-event orders, identity')
