// Local Chrome QA. Usage: node qa.mjs <absolute-output-directory>
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const out=path.resolve(process.argv[2]);fs.mkdirSync(out,{recursive:true});
const child=spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port=9224','--user-data-dir='+path.join(out,'chrome-profile'),'about:blank'
],{windowsHide:true,stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let ws;
try{
  let pages;
  for(let i=0;i<80;i++){try{pages=await(await fetch('http://127.0.0.1:9224/json')).json();if(pages.some(p=>p.type==='page'))break;}catch{}await sleep(250);}
  assert(pages,'Chrome debug endpoint did not start');
  ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  let seq=0;const pending=new Map(),errors=[];
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(new Error(JSON.stringify(m.error)));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);};
  const send=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.text+JSON.stringify(r.exceptionDetails.exception));return r.result.value;};
  await send('Runtime.enable');await send('Page.enable');
  await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:out,eventsEnabled:true});
  await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url:'http://127.0.0.1:8765'});
  for(let i=0;i<80;i++){if(await evaluate('!!window.PowerLab'))break;await sleep(100);}
  assert(await evaluate('!!window.PowerLab'),'App failed to initialize');
  const result=await evaluate(`(()=>{
    const failures=[],checks=[];
    const check=(condition,message)=>{if(!condition)failures.push(message);else checks.push(message);};
    const near=(a,b)=>Math.abs(a-b)<1e-8;
    const el=id=>document.getElementById(id);
    const change=(id,value,type='change')=>{el(id).value=value;el(id).dispatchEvent(new Event(type,{bubbles:true}));};
    for(const key of ['ar','mlr']){
      document.querySelector('[name=model][value='+key+']').click();
      const m=POWER_DATA.models[key],metric=m.metrics[Object.keys(m.metrics)[0]],s=PowerLab.stats(m.test);
      check(near(s.mse,metric['MSE (kW²)']),key+' MSE matches Python');
      check(near(s.accuracy,metric['誤差在 ±10% 內 (%)']),key+' accuracy matches Python');
      check(near(s.coverage,metric['95% 區間覆蓋率 (%)']),key+' coverage matches Python');
      const p=PowerLab.scenario();
      check(near(p.mean,m.scenarioReference.mean)&&near(p.lower,m.scenarioReference.lower)&&near(p.upper,m.scenarioReference.upper),key+' scenario mean and interval match Python');
      el('tab-scenario').click();
      const previous=PowerLab.scenario().mean;
      change('input-0',Number(el('input-0').value)+.15,'input');
      check(near(PowerLab.scenario().mean-previous,m.params[1]*.15),key+' scenario input changes linear prediction');
      change('input-0','','input');check(el('export-scenario').disabled&&!el('scenario-error').hidden,key+' empty input rejected');
      change('input-0',-1,'input');check(el('export-scenario').disabled,key+' negative input rejected');
      el('reset-scenario').click();check(near(PowerLab.scenario().mean,m.scenarioReference.mean),key+' reset restores prediction');
    }
    change('date-start','2010-03-01');change('date-end','2010-03-31');
    const marchCount=POWER_DATA.models.mlr.test.filter(r=>r.date>='2010-03-01'&&r.date<='2010-03-31'&&r.actual_kw!==null).length;
    check(el('score-summary').textContent.includes(marchCount+' 個有效評分日'),'date range updates sample count');
    change('date-start','2010-04-01');check(!el('date-error').hidden,'reversed dates rejected');
    el('reset-dates').click();
    const missing=POWER_DATA.models.ar.test.find(r=>r.actual_kw===null).date;
    change('date-start',missing);change('date-end',missing);
    check(el('metric-rmse').textContent==='—','missing-only range has no fabricated score');
    el('reset-dates').click();change('date-end','2010-02-11');
    check(el('metric-r2').textContent==='—','single-day R2 undefined');el('reset-dates').click();
    change('horizon',1,'input');check(el('horizon-value').textContent==='1 天','forecast horizon slider');
    change('horizon',30,'input');
    el('show-compare').click();check(el('test-chart').querySelector('path[stroke-dasharray]')!==null,'comparison curve enabled');
    el('show-interval').click();check(el('test-chart').querySelector('polygon')===null,'interval toggle hides shading');el('show-interval').click();
    el('tab-compare').click();check(!el('panel-compare').hidden,'comparison tab');
    const row=el('comparison-table').rows[1];check(row.cells[3].textContent==='0.2777'&&row.cells[4].textContent==='0.3304','baseline metrics match Python');
    el('tab-data').click();change('data-year','all');change('data-quality','missing');check(el('data-count').textContent==='共 23 日','missing day filter');
    el('next-page').click();check(el('page-info').textContent==='2 / 2','data pagination');
    change('data-quality','valid');change('data-year','2006');check(el('data-count').textContent==='共 15 日','year and valid filters');
    el('tab-forecast').focus();el('tab-forecast').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));check(el('tab-scenario').getAttribute('aria-selected')==='true','keyboard tabs');
    document.querySelector('[name=model][value=ar]').click();el('tab-forecast').click();
    el('show-compare').click();
    el('download-future').click();el('export-report').click();el('export-scenario').click();
    return {checks,failures};
  })()`);
  assert.deepEqual(result.failures,[]);
  await evaluate("document.getElementById('toast').hidden=true;window.scrollTo(0,0)");
  const desktop=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'desktop.png'),Buffer.from(desktop.data,'base64'));
  await evaluate("document.getElementById('tab-scenario').click();document.getElementById('workspace').scrollIntoView()");
  const scenario=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'scenario.png'),Buffer.from(scenario.data,'base64'));
  await send('Emulation.setDeviceMetricsOverride',{width:390,height:1000,deviceScaleFactor:1,mobile:true});
  await evaluate("document.getElementById('tab-forecast').click();window.scrollTo(0,0)");await sleep(200);
  for(const tab of ['forecast','scenario','compare','data']){
    const layout=await evaluate(`(()=>{document.getElementById('tab-${tab}').click();return {width:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}})()`);
    assert(layout.width<=layout.client+1,tab+' mobile overflow: '+JSON.stringify(layout));
  }
  await evaluate("document.getElementById('tab-forecast').click();window.scrollTo(0,0)");
  const mobile=await send('Page.captureScreenshot',{format:'png'});fs.writeFileSync(path.join(out,'mobile.png'),Buffer.from(mobile.data,'base64'));
  await sleep(500);
  for(const name of ['power-ar-30days.csv','power-ar-report.md','power-ar-scenario.json'])assert(fs.existsSync(path.join(out,name)),'Download missing: '+name);
  assert.equal(errors.length,0,'Browser runtime errors: '+JSON.stringify(errors));
  const report={checks:result.checks,downloads:3,mobileOverflow:false,browserErrors:errors};
  fs.writeFileSync(path.join(out,'qa.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
  await send('Browser.close');
}finally{if(ws)ws.close();child.kill();}
