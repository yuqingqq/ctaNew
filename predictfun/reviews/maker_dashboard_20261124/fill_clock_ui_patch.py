"""One-time presentation correction. Does not change the research payload."""
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'app.js';s=p.read_text()
old="const o=at(own().curve,cursor);$('state').innerHTML=`At ${fmt(cursor,3)} s · book sample ${fmt(row[0],1)} s<br>"
new="const o=at(own().curve,cursor),source=at(m.book,row[0])[6];$('state').innerHTML=`Cursor ${fmt(cursor,3)} s · depth display sample ${fmt(row[0],3)} s<br><b>Underlying book update: ${fmt(source,3)} s (${fmt(1000*(cursor-source),0)} ms before cursor).</b><br>Historical depth, not an execution-time snapshot. The price chart samples every 100 ms; depth samples every 500 ms.<br>"
assert s.count(old)==1;s=s.replace(old,new)
oldnote='The post window is a scenario estimate, not an observed command. No maker pull or queue-front rank is identified.'
newnote='The marker is the settlement upper bound, not an exact match time. The depth panel can show an older sample; a buyer may also consume several ask levels in one trade. The post window is a scenario estimate, not an observed command. No maker pull or queue-front rank is identified.'
assert s.count(oldnote)==1;s=s.replace(oldnote,newnote);p.write_text(s)
p=H/'index.html';s=p.read_text().replace('Book at the cursor','Earlier sampled book').replace('Public update clock · 500 ms display samples','Public update clock · see snapshot age below');p.write_text(s)
p=H/'inline/template.html';s=p.read_text().replace('Book at inspection time','Earlier sampled book');p.write_text(s)
print('Clock labels updated; research payload unchanged')
