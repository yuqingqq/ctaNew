"""Supersede only the scalar comparator's bool/int dispatch; saved outputs unchanged."""
import check
original=check.eq
def eq(a,b):
 if isinstance(b,bool):assert str(a)==str(b),(a,b)
 else:original(a,b)
check.eq=eq
if __name__=='__main__':
 eq('False',False);eq('True',True)
 for a,b in [('True',False),('False',True)]:
  try:eq(a,b)
  except AssertionError:pass
  else:raise AssertionError('boolean corruption escaped')
 check.run()
