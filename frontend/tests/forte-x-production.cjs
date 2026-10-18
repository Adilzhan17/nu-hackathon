// Real browser upload, real local models, bounded execution. No API response mocks.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const root = '/root/nuhack';
const base = process.env.FORTE_TEST_URL || 'https://forte.perricheno.com';
const out = root + '/runs/forte_x/production_browser_test.json';
const write = data => { fs.writeFileSync(out+'.tmp',JSON.stringify(data,null,2));fs.renameSync(out+'.tmp',out); };
(async()=>{
  const browser = await chromium.launch({headless:true});
  const errors=[];let jid;const started=Date.now();
  try {
    const page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base+'/analyze?pipeline=1&fresh=1');
    await page.getByRole('heading',{name:'Forte X',exact:true}).waitFor();
    const frame=page;
    await frame.locator('#choose-videomaev2').waitFor();
    await page.waitForTimeout(700);
    assert.equal(await frame.locator('#choose-videomaev2').isChecked(),true);
    assert.equal(await frame.locator('#choose-yolo11').isChecked(),true);
    await frame.locator('#choose-x3d').check();assert.equal(await frame.locator('#choose-x3d').isChecked(),true);
    await frame.locator('.pipeline-info summary').click();await frame.locator('#selectPipeline').click();
    assert.equal(await frame.locator('#models input:checked').count(),2);
    await frame.locator('#file').setInputFiles(root+'/data/raw/open/videos/Х-10.mov');
    const created=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/uploads'&&r.request().method()==='POST');
    await frame.locator('#upload').click();const response=await created;
    assert.equal(response.status(),201);const initial=await response.json();jid=initial.id;
    assert.deepEqual([...initial.models].sort(),['videomaev2','yolo11']);
    write({status:'processing',job_id:jid,checked_at:new Date().toISOString()});
    console.log(JSON.stringify({job_id:jid,stage:'uploaded',url:base+'/analyze?job='+jid}));
    let state;
    while(Date.now()-started<350000){
      await page.waitForTimeout(4000);state=await (await page.request.get(base+'/api/uploads/'+jid)).json();
      if(['completed','failed'].includes(state.status))break;
    }
    assert.equal(state.status,'completed',state.message);assert.equal(state.pipeline.name,'Forte X');
    assert.equal(state.pipeline.status,'completed');assert.equal(state.pipeline.ground_truth,false);
    assert.equal(state.pipeline.organizer_answers_in_prompt,false);
    assert.ok(state.pipeline.observations.length>0);assert.equal(state.pipeline.rules.length,11);
    const objects=await (await page.request.get(base+'/api/uploads/'+jid+'/yolo11-objects.json')).json();
    assert.deepEqual(Object.values(objects.classes),['cash','document_id','key','stamp']);
    assert.equal(objects.confidence,.25);assert.equal(objects.native_resolution_tiles,true);
    await page.waitForTimeout(4000);
    assert.ok((await frame.locator('#detectorProvenance').innerText()).includes(objects.checkpoint_sha256.slice(0,12)));
    assert.ok(await frame.locator('#boxes rect').count()>0);
    assert.ok(await frame.locator('#results img').count()>0);
    await frame.locator('#results img').first().scrollIntoViewIfNeeded();
    await page.waitForTimeout(500);
    assert.equal(await frame.locator('#results img').first().evaluate(i=>i.complete&&i.naturalWidth>0),true);
    assert.ok(page.url().includes(jid));
    await page.setViewportSize({width:390,height:844});await page.waitForTimeout(500);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
    assert.deepEqual(errors,[]);
    const result={status:'passed',job_id:jid,checked_at:new Date().toISOString(),wall_seconds:(Date.now()-started)/1000,
      source_duration_sec:state.duration_sec,observations:state.pipeline.observations.length,evidence_images:await frame.locator('#results img').count(),
      checkpoint_sha256:objects.checkpoint_sha256,classes:objects.classes,errors,
      assertions:['Выбор моделей','Реальная загрузка','Forte X: 4 этапа','Рамки и checkpoint','Таймкоды и кадры','Правила и неопределённость','Мобильный экран'],
      independent_quality_test:false,source_is_known_train_video:true};
    write(result);console.log(JSON.stringify(result));
  }catch(error){write({status:'failed',job_id:jid,checked_at:new Date().toISOString(),error:error.message,errors});console.error(error);process.exitCode=1}
  finally{await browser.close()}
})();
