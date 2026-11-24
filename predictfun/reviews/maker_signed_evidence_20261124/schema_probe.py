import gzip,json,collections
p='/home/yuqing/ctaNew/data/predict_fun/research/onchain/20261023_1400_1415/inputs.jsonl.gz'
lengths=collections.Counter(); examples=[]
with gzip.open(p,'rt') as f:
 for line in f:
  r=json.loads(line); data=bytes.fromhex(r['input'][10:]); lengths[len(data)%32]+=1
  if len(examples)<2:
   ptr=[int.from_bytes(data[i:i+32],'big') for i in (0,32)]
   examples.append({'tx':r['hash'],'byte_count':len(data),'mod32':len(data)%32,'order_offsets':ptr,'tail_hex':data[-(len(data)%32):].hex()})
print(json.dumps({'mod32':dict(lengths),'examples':examples},indent=2))
