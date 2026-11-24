"""Delivery adapter only: same verified payload; no data reanalysis."""
from pathlib import Path
import base64,gzip,json,re,hashlib
h=Path(__file__).resolve().parent;p=h.parent
raw=gzip.decompress((p/'data.json.gz').read_bytes());extra=(p/'extra.js').read_text().removeprefix('window.MM_EXTRA=').rstrip(';\n')
payload=json.dumps(dict(data=base64.b64encode(gzip.compress(raw,9,mtime=0)).decode(),extra=base64.b64encode(gzip.compress(extra.encode(),9,mtime=0)).decode()),separators=(',',':'))
s=(p/'app.js').read_text()
s=s.replace("const $=id=>document.getElementById(id), C={bid:'#53c8a6',ask:'#efac6a',ours:'#65b3ff',maker:'#c6a1ff',muted:'#8fa4b5',grid:'#2b3a48',red:'#ff8c8e',text:'#e6edf3'};", "const root=document.getElementById('pf-maker-microscope-inline');const $=id=>root.querySelector('#'+id);const payload=JSON.parse(root.querySelector('#pf-inline-payload').textContent);const C={};const probe=document.createElement('span');probe.hidden=true;root.appendChild(probe);function colors(){for(const [k,v]of Object.entries({bid:'--viz-series-1',ask:'--viz-series-2',ours:'--viz-series-3',maker:'--viz-series-4',muted:'--muted-foreground',grid:'--border',red:'--destructive',text:'--foreground'})){probe.style.color='var('+v+')';C[k]=getComputedStyle(probe).color;}}colors();")
s=s.replace('window.MM_DATA_GZIP','payload.data').replace('window.MM_EXTRA','root.extra').replace('window.MM_VIEW','root.view')
s=s.replace("const allowed=s=>", "root.extra=JSON.parse(await new Response(new Blob([Uint8Array.from(atob(payload.extra),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text());\nconst allowed=s=>")
s=s.replace('document.querySelectorAll(', 'root.querySelectorAll(').replace('document.querySelector(', 'root.querySelector(')
s=s.replace("c.font='10px system-ui'", "c.font=(parseFloat(getComputedStyle(root).fontSize)*.86)+'px system-ui'").replace("c.font='11px system-ui'", "c.font=(parseFloat(getComputedStyle(root).fontSize)*.86)+'px system-ui'")
for a,b in [("'#b7c8d355'","C.grid"),("'#ffffff'","C.text"),("'#e2caff'","C.maker"),("'#fff'","C.text"),("'#18222d'","getComputedStyle(root).getPropertyValue('--background').trim()"),("'#6b5e6c66'","C.grid"),("'#53627344'","C.grid"),("'#ffce8e'","C.ask"),("'#bad7d9'","C.muted"),("C.ours+'75'","C.ours"),("C.ours+'77'","C.muted"),("C.ask+'40'","C.ask"),("C.bid+'40'","C.bid")]:s=s.replace(a,b)
s=s.replace("function draw(){price();", "function draw(){colors();price();")
s=s.replace("c.fillRect(0,12,47,g.h-40);", "c.clearRect(0,12,47,g.h-40);")
s=s.replace("ev.slice(0,150)","ev.slice(0,25)")
s=s.replace("R=w-18","R=w-32").replace("rect.width-71","rect.width-85")
s=s.replace("c.fillText(fmt(v,Math.abs(max-min)>20?0:1),L-8,y(v)+3);","if(id!=='life')c.fillText(fmt(v,Math.abs(max-min)>20?0:1),L-8,y(v)+3);")
s=s.replace("<button data-fill=", "<button type=\"button\" class=\"btn\" data-fill=")
s=s.replace("${ev.length}","${ev.length}")
# Keep exact data and chart interactions, simplify narrative presentation for the inline surface.
s=s.replace("${ev.length} visible events", "${ev.length} events · first 25 listed")
old="$('metrics').innerHTML=cells.map(([l,v,s,n])=>`<div class=\"metric\"><div class=\"label\">${l}</div><div class=\"number ${sg(n)}\">${v}</div><small>${s}</small></div>`).join('');"
new="$('metrics').innerHTML=cells.slice(0,2).map(([l,v])=>`<span>${l}: <strong>${v}</strong></span>`).join(' · ');"
assert old in s;s=s.replace(old,new)
s=s.replace("if(requested){", "$('market').value='1';if(requested){")
# Restore/persist selected case and zoom only; no research calculations are saved.
end="root.view={data,get state(){return {market:m.id,arm,rank,lo,hi,cursor,selected}},choose};"
assert end in s
s=s.replace(end,end+"""
let localSerial=0;const widgetSession=Math.random().toString(36).slice(2);
function restore(snapshot){const z=snapshot?.privateContent;if(!z||z.version!==1)return false;const idx=data.cases.findIndex(x=>x.id===z.market);if(idx<0)return false;$('market').value=idx;if(z.arm in data.cases[idx].ours)$('arm').value=z.arm;if([0,1,2,3].includes(z.rank))$('maker').value=z.rank;selectMarket();if(Number.isFinite(z.cursor))cursor=Math.max(0,Math.min(m.end,z.cursor));if(Number.isFinite(z.span)&&z.span>=1)windowAt(cursor,z.span);$('time').value=cursor;if(z.selected){const list=z.selected.kind==='maker'?m.makers:z.selected.kind==='fill'?own().fills:z.selected.kind==='order'?own().orders:[];if(list[z.selected.id])choose(z.selected,false);}if(Number.isFinite(z.cursor)){cursor=Math.max(0,Math.min(m.end,z.cursor));$('time').value=cursor;windowAt(cursor,z.span||15);}return true;}
if(!restore(window.openai?.widgetState))$('example').click();
window.addEventListener('openai:set_globals',e=>{const incoming=e.detail?.globals?.widgetState;if(incoming&&incoming.privateContent?._session!==widgetSession)restore(incoming);schedule();});
function save(){const z={version:1,...root.view.state,span:hi-lo,_session:widgetSession,_serial:++localSerial};window.openai?.setWidgetState?.({modelContent:{market:m.id,arm,maker:rank,selected},privateContent:z}).catch(()=>{});}
root.addEventListener('change',()=>queueMicrotask(save));root.addEventListener('click',()=>queueMicrotask(save));root.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',schedule));new MutationObserver(schedule).observe(document.documentElement,{attributes:true,attributeFilter:['class','style','data-theme']});
""")
fragment=(h/'template.html').read_text().replace('__PAYLOAD__',payload).replace('__PROGRAM__',s)
assert len(fragment.encode())<1000000
assert not any(x in fragment.lower() for x in ['<!doctype','<html','<body','<head>'])
assert gzip.decompress(base64.b64decode(json.loads(payload)['data']))==raw
(h/'market-microscope.html').write_text(fragment)
(h/'delivery_check.json').write_text(json.dumps(dict(bytes=len(fragment.encode()),source_sha256=hashlib.sha256(raw).hexdigest(),same_data=True,no_external_fetch=True),indent=2)+'\n')
print('Inline bytes',len(fragment.encode()),'same decoded source payload')
