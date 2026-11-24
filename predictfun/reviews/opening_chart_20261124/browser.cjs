const {chromium}=require('/home/yuqing/.npm/_npx/d537ee5ee2a13f03/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const browser=process.env.CHART_CDP?await chromium.connectOverCDP(process.env.CHART_CDP):await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1280,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8765/opening.html?market=3789484');
 await page.waitForFunction(()=>window.OPENING_VIEW,{timeout:10000});
 const checks=[];
 for(const market of ['3789484','3702960','3877472']){
  await page.selectOption('#market',market);
  await page.click('#jump');
  checks.push(await page.evaluate(()=>({state:OPENING_VIEW.state,rule:document.querySelector('#rule').innerText,events:document.querySelectorAll('#events tr').length,overflow:document.body.scrollWidth>innerWidth})));
  await page.click('[data-span="60"]');
  await page.locator('#defaultOn').uncheck();await page.locator('#defaultOn').check();
  await page.locator('#events tr').first().click();
  if(!(await page.locator('#detail').innerText()).includes('Submit'))throw Error('event inspection did not render order details');
 }
 await page.selectOption('#market','3789484');await page.click('[data-span="15"]');
 await page.screenshot({path:path.join(__dirname,'opening-desktop.png'),fullPage:true});
 await page.click('[data-span="3"]');await page.screenshot({path:path.join(__dirname,'opening-first3s.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.click('#jump');await page.screenshot({path:path.join(__dirname,'opening-mobile.png'),fullPage:true});
 const mobile=await page.evaluate(()=>({overflow:document.body.scrollWidth>innerWidth,canvas:document.querySelector('#price').getBoundingClientRect().width,body:innerWidth}));
 await page.locator('#time').focus();await page.keyboard.press('ArrowRight');
 const result={errors,checks,mobile};fs.writeFileSync(path.join(__dirname,'browser.json'),JSON.stringify(result,null,2));
 await page.close();await browser.close();console.log(JSON.stringify(result));
 if(errors.length||checks.some(x=>x.overflow)||mobile.overflow)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
