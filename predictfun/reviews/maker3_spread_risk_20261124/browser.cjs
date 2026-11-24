const {chromium}=require('/home/yuqing/.npm/_npx/d537ee5ee2a13f03/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const b=await chromium.connectOverCDP('http://127.0.0.1:9222'),errors=[],checks=[];
 for(const [name,viewport,scale]of [['desktop',{width:1440,height:1100},1],['mobile',{width:390,height:844},2]]){
  const context=await b.newContext({viewport,deviceScaleFactor:scale}),p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://127.0.0.1:8765/maker3-research.html');await p.waitForFunction(()=>window.MAKER_RESEARCH);
  await p.locator('#events tr').first().click();
  checks.push(await p.evaluate(()=>({mid:MAKER_RESEARCH.state.mid,markets:MAKER_RESEARCH.data.cases.length,detail:document.querySelector('#selected').innerText.length,overflow:document.body.scrollWidth>innerWidth,canvas:document.querySelector('#price').getBoundingClientRect().height})));
  if(name==='desktop')await p.evaluate(()=>scrollTo(0,0));if(name==='desktop')await p.screenshot({path:path.join(__dirname,'trend.png'),fullPage:false});
  await p.click('#named');await p.check('#ours');await p.selectOption('#role','taker');await p.locator('#events tr').first().click();
  const detail=await p.locator('#selected').innerText();if(!detail.includes('Native action:'))throw Error('native action missing');
  await p.selectOption('#role','all');await p.selectOption('#clock','upper');await p.locator('#closes tr').first().click();await p.click('#last');await p.click('#prev');await p.click('#next');
  checks.push(await p.evaluate(()=>({mid:MAKER_RESEARCH.state.mid,markets:MAKER_RESEARCH.data.cases.length,detail:document.querySelector('#selected').innerText.length,overflow:document.body.scrollWidth>innerWidth,canvas:document.querySelector('#price').getBoundingClientRect().height})));
  await p.screenshot({path:path.join(__dirname,name+'.png'),fullPage:false});await context.close();
 }
 const result={errors,checks};fs.writeFileSync(path.join(__dirname,'browser_v2.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 await b.close();if(errors.length||checks.some(c=>c.overflow||c.detail<100||c.canvas!==330||c.markets!==19))process.exitCode=1;
})().catch(e=>{console.error(e);process.exitCode=1;});
