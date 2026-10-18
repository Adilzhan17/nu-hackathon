// Browser regression checks only. All API requests are mocked; no GPU job is submitted.
const {chromium}=require('../../frontend/node_modules/@playwright/test');
const fs=require('node:fs');const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();let posts=0;const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const id='a'.repeat(32),job={id,name:'long.mp4',status:'failed',message:'Gemma: допустимы видео до 60 секунд',result:null};
  await page.route('https://gemma.test/**',async route=>{
   const u=new URL(route.request().url());
   if(route.request().method()==='POST'){posts++;return route.fulfill({status:500,body:'Unexpected upload'});}
   if(u.pathname==='/demo/gemma/status')return route.fulfill({json:{max_duration_sec:180,max_bytes:67108864,recent:[]}});
   if(u.pathname===`/demo/gemma/jobs/${id}`)return route.fulfill({json:job});
   return route.fulfill({contentType:'text/html',body:fs.readFileSync(__dirname+'/index.html','utf8')});
  });
  await page.goto('https://gemma.test/demo/gemma?job='+id,{waitUntil:'networkidle'});
  assert.equal(await page.locator('#status').innerText(),'Видео не проанализировано');
  assert.equal(await page.locator('#counts').isVisible(),false);
  assert.equal(await page.locator('#workspace .grid').isVisible(),false);
  assert.equal(await page.locator('#progress').isVisible(),false);
  assert.ok(!(await page.locator('#coverage').innerText()).includes('0 из 0'));
  // Stub metadata only: test that an over-limit file cannot reach the upload API.
  await page.evaluate(()=>{window.videoDuration=async()=>181});
  await page.locator('#file').setInputFiles({name:'long.mp4',mimeType:'video/mp4',buffer:Buffer.from('test')});
  await page.locator('#start').click();await page.waitForTimeout(50);
  assert.match(await page.locator('#message').innerText(),/181.0.*180.*не отправлен/);
  assert.equal(posts,0);assert.deepEqual(errors,[]);
  Object.assign(job,{status:'processing',stage:'overview_001',sampled_frames:146,result:{stage:'overview_001',observations:[],rules:[{rule_id:'identity_before_cash',verdict:'insufficient_data',explanation:'placeholder'}],coverage:{total_sampled_frames:146,inspected_sampled_frames:0}}});
  await page.reload({waitUntil:'networkidle'});
  assert.equal(await page.locator('#rules article').count(),0);
  assert.match(await page.locator('#status').innerText(),/Обзор видео.*окно 2 из 15/);
  assert.equal(await page.locator('#counts').isVisible(),false);
  assert.ok(!(await page.locator('#rules').innerText()).includes('placeholder'));
  const frames=[0,1,2,3].map(i=>({id:'f'+String(i).padStart(5,'0'),timestamp:i}));
  Object.assign(job,{status:'completed',duration_sec:4,result:{observations:[{start_sec:0,end_sec:3,action:'employee_leaves',description:'Уход',visibility:'clear',evidence:frames.slice(0,2)}],
    rules:[{rule_id:'valuables_on_departure',verdict:'violation_supported',explanation:'Ключи до и после ухода',evidence:[{start_sec:0,end_sec:3,frames}],
      regulatory_source:{title:'НБРК № 231',section:'пункт 80',url:'https://adilet.zan.kz/rus/docs/V1900019680#z226',legal_note:'Требуется внутренний порядок.'}}],
    coverage:{all_sampled_frames_inspected:true,total_sampled_frames:4,inspected_sampled_frames:4}}});
  await page.reload({waitUntil:'networkidle'});
  assert.equal(await page.locator('#rules article.rule .evidence img').count(),3);
  assert.match(await page.locator('#rules article.rule .source').innerText(),/пункт 80/);
  assert.ok((await page.locator('#rules article.rule .evidence img').last().getAttribute('src')).includes('f00003.jpg'));
  assert.deepEqual(errors,[]);
  console.log('UI: failures stay incomplete; upload guard works; supported findings show law and before/after evidence; no JS errors.');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
