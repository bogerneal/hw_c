'use strict';
(() => {
  const D = window.POWER_DATA;
  const $ = id => document.getElementById(id);
  const labels = {Global_active_power:'有功功率',Global_reactive_power:'無功功率',Voltage:'電壓',Global_intensity:'電流',Sub_metering_1:'廚房分表',Sub_metering_2:'洗衣房分表',Sub_metering_3:'熱水器／空調分表'};
  const units = {Global_active_power:'kW',Global_reactive_power:'原始單位',Voltage:'V',Global_intensity:'A',Sub_metering_1:'Wh／分鐘區間',Sub_metering_2:'Wh／分鐘區間',Sub_metering_3:'Wh／分鐘區間'};
  const colors = {ar:'#187f70',mlr:'#bc7840'};
  const state = {model:'ar',tab:'forecast',start:D.meta.testStart,end:D.meta.end,horizon:30,page:0,scenario:{},filtered:[],scenarioResult:null};
  for (const key of ['ar','mlr']) state.scenario[key] = D.models[key].features.map(f=>f.default);
  const model = () => D.models[state.model];
  const other = () => state.model === 'ar' ? 'mlr' : 'ar';
  const fmt = (x,n=4) => x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
  const escape = s => String(s).replace(/[&<>"']/g, x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
  let toastTimer;
  function toast(text){$('toast').textContent=text;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,3500);}
  function download(name,content,type='text/plain;charset=utf-8'){
    const url=URL.createObjectURL(new Blob([content],{type})); const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已準備下載：'+name);
  }
  function getRows(key=state.model){return D.models[key].test.filter(r=>r.date>=state.start&&r.date<=state.end);}
  function stats(rows,column='predicted_kw'){
    const valid=rows.filter(r=>r.actual_kw!==null&&Number.isFinite(r.actual_kw)&&Number.isFinite(r[column]));
    if(!valid.length)return {n:0,mse:null,rmse:null,mae:null,r2:null,mape:null,smape:null,accuracy:null,coverage:null,width:null};
    const n=valid.length,avg=valid.reduce((s,r)=>s+r.actual_kw,0)/n;
    const mse=valid.reduce((s,r)=>s+(r.actual_kw-r[column])**2,0)/n;
    const total=valid.reduce((s,r)=>s+(r.actual_kw-avg)**2,0);
    return {n,mse,rmse:Math.sqrt(mse),mae:valid.reduce((s,r)=>s+Math.abs(r.actual_kw-r[column]),0)/n,r2:n>1&&total>1e-15?1-mse*n/total:null,
      mape:valid.reduce((s,r)=>s+Math.abs(r.actual_kw-r[column])/Math.max(Math.abs(r.actual_kw),1e-8),0)/n*100,
      smape:valid.reduce((s,r)=>s+2*Math.abs(r.actual_kw-r[column])/Math.max(Math.abs(r.actual_kw)+Math.abs(r[column]),1e-8),0)/n*100,
      accuracy:valid.filter(r=>Math.abs(r.actual_kw-r[column])<=.1*Math.abs(r.actual_kw)).length/n*100,
      coverage:column==='predicted_kw'?valid.filter(r=>r.actual_kw>=r.lower_95_kw&&r.actual_kw<=r.upper_95_kw).length/n*100:null,
      width:column==='predicted_kw'?valid.reduce((s,r)=>s+r.upper_95_kw-r.lower_95_kw,0)/n:null};
  }
  function switchTab(tab,focus=false){
    state.tab=tab;
    document.querySelectorAll('[role=tab]').forEach(b=>{const active=b.dataset.tab===tab;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;if(active&&focus)b.focus();});
    ['forecast','scenario','compare','data'].forEach(k=>$('panel-'+k).hidden=k!==tab);
  }
  function updateModel(){
    const m=model();
    $('feature-count').textContent=`${m.features.length} / ${m.candidateCount}`;
    $('selection-fill').style.width=`${m.features.length/m.candidateCount*100}%`;
    $('model-description').textContent=state.model==='ar'?'從過去 1～30 天的有功功率，選出對預測有幫助的歷史值。':'從 7 個欄位、6 組落後天數的歷史資訊中，選出有幫助的特徵。';
    $('feature-list').innerHTML=m.features.map(f=>`<span class="feature-chip">${labels[f.column]} · ${f.lag} 天前</span>`).join('');
    $('alpha-label').textContent=`Lasso α = ${fmt(m.alpha,6)}`;
    $('cv-label').textContent=`Lasso 選擇階段的時間序列 CV MSE：${fmt(m.lassoCvMse,6)} kW²。這不是最終模型的 CV 分數。`;
    $('coefficient-table').innerHTML=`<tr><td>截距</td><td>—</td><td>—</td><td>${fmt(m.trainIntercept,6)}</td></tr>`+m.features.map(f=>`<tr><td>${labels[f.column]}</td><td>${f.lag} 天</td><td>${fmt(f.lassoCoefficient,6)}</td><td>${fmt(f.trainCoefficient,6)}</td></tr>`).join('');
    buildScenario();updatePeriod();updateFuture();
  }
  function updatePeriod(){
    const rows=getRows();state.filtered=rows;
    const s=stats(rows);
    $('metric-rmse').textContent=fmt(s.rmse);
    $('metric-r2').textContent=fmt(s.r2);
    $('metric-accuracy').textContent=s.accuracy===null?'—':fmt(s.accuracy,2)+'%';
    $('score-summary').textContent=`${state.start} ～ ${state.end} · ${s.n} 個有效評分日 · ${model().features.length} 個特徵`;
    $('date-cursor').max=Math.max(0,rows.length-1);$('date-cursor').value=Math.min(Number($('date-cursor').value),rows.length-1);
    const secondary = $('show-compare').checked ? new Map(getRows(other()).map(r=>[r.date,r.predicted_kw])) : null;
    chart('test-chart',rows,{interval:$('show-interval').checked,secondary,color:colors[state.model],secondaryColor:colors[other()],label:'測試期每日實際與預測功率',onPoint:showPoint});
    showPoint(Number($('date-cursor').value));updateComparison();
  }
  function showPoint(index){
    const rows=state.filtered;if(!rows.length)return;
    index=Math.max(0,Math.min(rows.length-1,index));const r=rows[index];$('date-cursor').value=index;
    $('point-detail').textContent=`${r.date}　實際 ${fmt(r.actual_kw,3)}　預測 ${fmt(r.predicted_kw,3)}　95% 區間 [${fmt(r.lower_95_kw,3)}, ${fmt(r.upper_95_kw,3)}] kW${r.actual_kw===null?' · 真值缺失，不評分':''}`;
  }
  function chart(id,rows,options={}){
    const container=$(id),W=800,H=300,L=48,R=16,T=20,B=40;
    if(!rows.length){container.textContent='這段期間沒有資料。';return;}
    const values=[];
    rows.forEach(r=>{for(const k of ['actual_kw','predicted_kw',...(options.interval?['lower_95_kw','upper_95_kw']:[])])if(Number.isFinite(r[k]))values.push(r[k]);if(options.secondary&&Number.isFinite(options.secondary.get(r.date)))values.push(options.secondary.get(r.date));});
    let min=Math.min(0,...values),max=Math.max(.1,...values);max+=(max-min)*.1;
    const x=i=>L+(rows.length===1?.5:i/(rows.length-1))*(W-L-R),y=v=>H-B-(v-min)/(max-min)*(H-T-B);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(options.label||'用電預測圖')}"><title>${escape(options.label||'用電預測圖')}</title>`;
    for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;svg+=`<path class="grid-line" d="M${L} ${y(v)}H${W-R}"/><text x="${L-9}" y="${y(v)+4}" text-anchor="end">${v.toFixed(1)}</text>`;}
    svg+=`<text x="${L}" y="10">kW</text>`;
    const tickCount=Math.min(5,rows.length);
    for(let j=0;j<tickCount;j++){const i=tickCount===1?0:Math.round(j*(rows.length-1)/(tickCount-1));svg+=`<text x="${x(i)}" y="${H-13}" text-anchor="${j===0?'start':j===tickCount-1?'end':'middle'}">${rows[i].date}</text>`;}
    const color=options.color||colors[state.model];
    if(options.interval){
      let segment=[];
      const flush=()=>{if(segment.length){const points=segment.map(i=>`${x(i)},${y(rows[i].upper_95_kw)}`).concat([...segment].reverse().map(i=>`${x(i)},${y(rows[i].lower_95_kw)}`)).join(' ');svg+=segment.length===1?`<path d="M${x(segment[0])} ${y(rows[segment[0]].upper_95_kw)}V${y(rows[segment[0]].lower_95_kw)}" stroke="${color}" stroke-opacity=".25" stroke-width="8"/>`:`<polygon points="${points}" fill="${color}" opacity=".13"/>`;segment=[];}};
      rows.forEach((r,i)=>{if(Number.isFinite(r.lower_95_kw)&&Number.isFinite(r.upper_95_kw))segment.push(i);else flush();});flush();
    }
    function line(get,color,width,dash=''){
      let path='',open=false;
      rows.forEach((r,i)=>{const v=get(r);if(Number.isFinite(v)){path+=`${open?'L':'M'}${x(i).toFixed(2)} ${y(v).toFixed(2)} `;open=true;}else open=false;});
      svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
      const count=rows.filter(r=>Number.isFinite(get(r))).length;
      if(count===1)rows.forEach((r,i)=>{if(Number.isFinite(get(r)))svg+=`<circle cx="${x(i)}" cy="${y(get(r))}" r="4" fill="${color}"/>`;});
    }
    line(r=>r.actual_kw,'#718079',1.7);
    if(options.secondary)line(r=>options.secondary.get(r.date),options.secondaryColor,1.8,'5 4');
    line(r=>r.predicted_kw,color,2.2);
    svg+=`<line class="hover-line" x1="0" x2="0" y1="${T}" y2="${H-B}" stroke="#789486" stroke-dasharray="3 4" visibility="hidden"/></svg><div class="chart-tooltip" hidden></div>`;
    container.innerHTML=svg;
    const svgEl=container.querySelector('svg'),tooltip=container.querySelector('.chart-tooltip'),hover=container.querySelector('.hover-line');
    svgEl.addEventListener('pointermove',e=>{
      const rect=svgEl.getBoundingClientRect(),ratio=(e.clientX-rect.left)/rect.width;
      const index=Math.max(0,Math.min(rows.length-1,Math.round((ratio*W-L)/(W-L-R)*(rows.length-1))));
      const row=rows[index];hover.setAttribute('x1',x(index));hover.setAttribute('x2',x(index));hover.setAttribute('visibility','visible');
      tooltip.innerHTML=`<b>${row.date}</b><br>實際 ${fmt(row.actual_kw,3)} kW<br>預測 ${fmt(row.predicted_kw,3)} kW${Number.isFinite(row.lower_95_kw)?`<br>區間 ${fmt(row.lower_95_kw,3)} ～ ${fmt(row.upper_95_kw,3)}`:''}`;
      tooltip.hidden=false;tooltip.style.left=Math.max(0,Math.min(rect.width-tooltip.offsetWidth,(e.clientX-rect.left)+12))+'px';tooltip.style.top='20px';
      if(options.onPoint)options.onPoint(index);
    });
    svgEl.addEventListener('pointerleave',()=>{tooltip.hidden=true;hover.setAttribute('visibility','hidden');});
  }
  function updateFuture(){
    const rows=model().future.slice(0,state.horizon);
    $('horizon-value').textContent=`${state.horizon} 天`;
    $('future-mean').textContent=fmt(rows.reduce((s,r)=>s+r.predicted_kw,0)/rows.length,3)+' kW';
    $('future-method').textContent=state.model==='ar'?'AR 使用同一模型遞迴預測；多步區間累積未來誤差，視係數為固定。':'MLR 為每個預測天數分別建模，使用資料終點已知歷史值；區間包含該模型的係數與觀測誤差。';
    const history=D.history.slice(-30).map(r=>({date:r.date,actual_kw:r.Global_active_power}));
    chart('future-chart',[...history,...rows],{interval:true,label:'最近三十日實際觀測與未來預測'});
  }
  function buildScenario(){
    const m=model();
    $('scenario-inputs').innerHTML=m.features.map((f,i)=>{
      const val=state.scenario[state.model][i],step=(f.max-f.min)/200||.001;
      return `<div class="scenario-input"><div class="input-top"><label for="input-${i}">${labels[f.column]}<small>${f.lag} 天前 · ${units[f.column]}</small></label><input id="input-${i}" type="number" min="0" step="any" value="${val}" data-input="${i}" aria-label="${labels[f.column]} ${f.lag} 天前" required></div><input type="range" min="${f.min}" max="${f.max}" step="${step}" value="${Math.max(f.min,Math.min(f.max,val))}" data-slider="${i}" aria-label="調整${labels[f.column]} ${f.lag} 天前"><div class="range-labels"><span>${fmt(f.min,3)}</span><span>訓練範圍</span><span>${fmt(f.max,3)}</span></div></div>`;
    }).join('');
    document.querySelectorAll('[data-input]').forEach(input=>input.addEventListener('input',()=>{
      const i=Number(input.dataset.input),value=input.value===''?NaN:Number(input.value);state.scenario[state.model][i]=value;
      const f=m.features[i];if(Number.isFinite(value))document.querySelector(`[data-slider="${i}"]`).value=Math.max(f.min,Math.min(f.max,value));calculateScenario();
    }));
    document.querySelectorAll('[data-slider]').forEach(slider=>slider.addEventListener('input',()=>{
      const i=Number(slider.dataset.slider),value=Number(Number(slider.value).toFixed(5));state.scenario[state.model][i]=value;$('input-'+i).value=value;calculateScenario();
    }));
    $('equation').innerHTML=`ŷ = ${fmt(m.params[0],6)}`+m.features.map((f,i)=>`<br>${m.params[i+1]>=0?'+':'−'} ${fmt(Math.abs(m.params[i+1]),6)} × ${labels[f.column]}（${f.lag} 天前）`).join('');
    calculateScenario();
  }
  function calculateScenario(){
    const m=model(),inputs=state.scenario[state.model],valid=inputs.every(v=>Number.isFinite(v)&&v>=0&&v<=1e9);
    $('scenario-error').hidden=valid;$('scenario-result').hidden=!valid;$('export-scenario').disabled=!valid;
    if(!valid){$('scenario-error').textContent='請將所有欄位填入 0～1,000,000,000 的有效數字。';$('extrapolation').textContent='';state.scenarioResult=null;return;}
    const x=[1,...inputs],mean=m.params.reduce((s,b,i)=>s+b*x[i],0);
    let variance=m.variance;
    for(let i=0;i<x.length;i++)for(let j=0;j<x.length;j++)variance+=x[i]*m.covariance[i][j]*x[j];
    const margin=m.critical*Math.sqrt(Math.max(0,variance));
    state.scenarioResult={mean,lower:mean-margin,upper:mean+margin};
    $('scenario-mean').textContent=fmt(mean,3);
    $('scenario-interval').textContent=`${fmt(mean-margin,3)} ～ ${fmt(mean+margin,3)} kW`;
    const delta=mean-m.scenarioReference.mean;
    $('scenario-change').textContent=`相較預設情境 ${delta>=0?'+':''}${fmt(delta,3)} kW`;
    const outside=m.features.filter((f,i)=>inputs[i]<f.min||inputs[i]>f.max).length;
    $('extrapolation').textContent=outside?`${outside} 個輸入超出訓練範圍，屬於外推情境；區間未必能反映額外不確定性。`:mean<0?'線性模型產生負預測值，超出功率的實際物理範圍；請檢查情境。':'預測區間未截斷負下界，也不代表保證範圍。';
  }
  function updateComparison(){
    const ar=stats(getRows('ar')),mlr=stats(getRows('mlr')),naive1=stats(getRows('ar'),'naive_1_kw'),naive7=stats(getRows('ar'),'naive_7_kw');
    const entries=[['AR',ar,colors.ar],['多元線性回歸',mlr,colors.mlr],['前一天基準',naive1,'#a1b29a'],['前一週基準',naive7,'#b5c0ab']];
    const max=Math.max(.001,...entries.map(e=>e[1].rmse||0));
    $('compare-period').textContent=`${state.start} ～ ${state.end} · ${ar.n} 個有效評分日 · RMSE（kW）越低越好`;
    $('comparison-bars').innerHTML=entries.map(([name,s,color])=>`<div class="comparison-bar"><span>${name}</span><div><i style="width:${s.rmse===null?0:s.rmse/max*100}%;background:${color}"></i></div><b>${fmt(s.rmse)}</b></div>`).join('');
    const names=[['mse','MSE (kW²)'],['rmse','RMSE (kW)'],['mae','MAE (kW)'],['r2','R²'],['mape','MAPE (%)'],['smape','sMAPE (%)'],['accuracy','誤差在 ±10% 內 (%)'],['coverage','95% 區間覆蓋率 (%)'],['width','平均區間寬度 (kW)']];
    $('comparison-table').innerHTML=names.map(([key,label])=>`<tr><td>${label}</td>${entries.map(e=>`<td>${fmt(e[1][key],key==='accuracy'||key==='coverage'?2:4)}</td>`).join('')}</tr>`).join('');
  }
  function renderData(){
    const year=$('data-year').value,quality=$('data-quality').value;
    const rows=D.history.filter(r=>(year==='all'||r.date.startsWith(year))&&(quality==='all'||(quality==='valid'?r.Global_active_power!==null:r.Global_active_power===null)));
    const pages=Math.max(1,Math.ceil(rows.length/15));state.page=Math.max(0,Math.min(state.page,pages-1));
    $('data-count').textContent=`共 ${rows.length} 日`;
    $('data-table').innerHTML=rows.slice(state.page*15,(state.page+1)*15).map(r=>`<tr><td>${r.date}</td><td>${fmt(r.Global_active_power,3)}</td><td>${fmt(r.Voltage,2)}</td><td>${fmt(r.Global_intensity,3)}</td><td>${r.valid_minutes}${r.Global_active_power===null?' · 不足':''}</td><td>${r.split==='train'?'訓練':'測試'}</td></tr>`).join('')||'<tr><td colspan="6">此條件下沒有資料。</td></tr>';
    $('page-info').textContent=`${state.page+1} / ${pages}`;$('prev-page').disabled=state.page===0;$('next-page').disabled=state.page===pages-1;
  }
  function validateDates(){
    const start=$('date-start').value,end=$('date-end').value;
    const valid=start&&end&&start<=end&&start>=D.meta.testStart&&end<=D.meta.end;
    $('date-error').hidden=!!valid;
    if(!valid){$('date-error').textContent='請選擇 2010-02-11～2010-11-26 之間的日期，起始日不可晚於結束日。目前仍顯示上一次有效範圍。';return;}
    state.start=start;state.end=end;updatePeriod();
  }
  document.querySelectorAll('[name=model]').forEach(input=>input.addEventListener('change',()=>{state.model=input.value;updateModel();}));
  const tabs=[...document.querySelectorAll('[role=tab]')];
  tabs.forEach((b,i)=>{b.addEventListener('click',()=>switchTab(b.dataset.tab));b.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%tabs.length;if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;if(e.key==='Home')next=0;if(e.key==='End')next=tabs.length-1;if(next!==undefined){e.preventDefault();switchTab(tabs[next].dataset.tab,true);}});});
  $('view-data').addEventListener('click',()=>{switchTab('data');$('workspace').scrollIntoView();});
  $('go-scenario').addEventListener('click',()=>{switchTab('scenario');$('panel-scenario').scrollIntoView({block:'start'});});
  $('date-start').addEventListener('change',validateDates);$('date-end').addEventListener('change',validateDates);
  $('reset-dates').addEventListener('click',()=>{$('date-start').value=D.meta.testStart;$('date-end').value=D.meta.end;validateDates();});
  $('show-interval').addEventListener('change',updatePeriod);$('show-compare').addEventListener('change',updatePeriod);
  $('date-cursor').addEventListener('input',()=>showPoint(Number($('date-cursor').value)));
  $('horizon').addEventListener('input',()=>{state.horizon=Number($('horizon').value);updateFuture();});
  $('reset-scenario').addEventListener('click',()=>{state.scenario[state.model]=model().features.map(f=>f.default);buildScenario();});
  ['data-year','data-quality'].forEach(id=>$(id).addEventListener('change',()=>{state.page=0;renderData();}));
  $('prev-page').addEventListener('click',()=>{state.page--;renderData();});$('next-page').addEventListener('click',()=>{state.page++;renderData();});
  $('download-future').addEventListener('click',()=>{
    const header=['date','predicted_kw','lower_95_kw','upper_95_kw'];
    const csv=[header.join(','),...model().future.slice(0,state.horizon).map(r=>header.map(k=>r[k]).join(','))].join('\r\n');
    download(`power-${state.model}-${state.horizon}days.csv`,csv,'text/csv;charset=utf-8');
  });
  $('export-scenario').addEventListener('click',()=>{
    if(!state.scenarioResult)return;
    download(`power-${state.model}-scenario.json`,JSON.stringify({model:model().name,task:'hypothetical next-day daily mean power (kW)',fit:'all available historical data',inputs:Object.fromEntries(model().features.map((f,i)=>[f.name,state.scenario[state.model][i]])),prediction:state.scenarioResult,interval:'95% new-observation interval; selection uncertainty not included',coefficients:model().params},null,2),'application/json;charset=utf-8');
  });
  $('export-report').addEventListener('click',()=>{
    const s=stats(getRows());
    const text=`# Power Lab：${model().name}\n\n## 測試期間\n\n${state.start} ～ ${state.end}；有效評分 ${s.n} 日。固定訓練模型，逐日一步預測。\n\n| 指標 | 結果 |\n|---|---:|\n| MSE (kW²) | ${fmt(s.mse,6)} |\n| RMSE (kW) | ${fmt(s.rmse,6)} |\n| MAE (kW) | ${fmt(s.mae,6)} |\n| R² | ${fmt(s.r2,6)} |\n| MAPE (%) | ${fmt(s.mape,4)} |\n| sMAPE (%) | ${fmt(s.smape,4)} |\n| 誤差在 ±10% 內 (%) | ${fmt(s.accuracy,4)} |\n| 95% 區間覆蓋率 (%) | ${fmt(s.coverage,4)} |\n\n## 特徵選擇\n\nLasso alpha=${model().alpha}；保留 ${model().features.length}/${model().candidateCount} 個特徵。\n\n${model().features.map(f=>'- '+f.name).join('\n')}\n\n## 未來 ${state.horizon} 天預測\n\n自 2010-11-27 起，無真值，不能計算未來準確率。\n\n| 日期 | 預測 kW | 95% 下界 | 95% 上界 |\n|---|---:|---:|---:|\n${model().future.slice(0,state.horizon).map(r=>`| ${r.date} | ${fmt(r.predicted_kw,6)} | ${fmt(r.lower_95_kw,6)} | ${fmt(r.upper_95_kw,6)} |`).join('\n')}\n\n## 說明\n\n目標是每日平均功率，非每日累積電能。兩版候選特徵不同；預測區間依賴模型假設，未校正特徵選擇。情境試算另以全資料重訓模型計算，不影響測試指標。\n\n[資料來源](${D.meta.source})\n`;
    download(`power-${state.model}-report.md`,text,'text/markdown;charset=utf-8');
  });
  updateModel();renderData();
  // Expose read-only calculation entry points for reproducible QA, not credentials.
  window.PowerLab={stats,scenario:()=>state.scenarioResult,model:()=>state.model};
})();
