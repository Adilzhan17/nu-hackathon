import test from 'node:test';
import assert from 'node:assert/strict';
import {gemmaProxy} from '../lib/gemma-proxy';
test('Gemma rejects arbitrary paths and cross-origin uploads before forwarding', async()=>{
  assert.equal((await gemmaProxy(new Request('http://localhost/'),'/../secrets')).status,400);
  const r=new Request('http://localhost/api/analysis',{method:'POST',headers:{origin:'https://foreign.example','content-length':'4'},body:'test'});
  assert.equal((await gemmaProxy(r,'/jobs')).status,403);
});
test('Gemma upload has a fixed local destination and does not forward credentials', async(t)=>{
  t.mock.method(globalThis,'fetch',async(url:unknown,init:RequestInit)=>{
    assert.equal(String(url),'http://127.0.0.1:8771/jobs?name=test.mov');
    const h=new Headers(init.headers);assert.equal(h.get('cookie'),null);assert.equal(h.get('authorization'),null);
    assert.equal(h.get('x-gemma-local-proxy'),'1');assert.equal(h.get('content-length'),'4');
    return Response.json({id:'a'.repeat(32)},{status:202});
  });
  const r=new Request('http://localhost/api/analysis?engine=gemma&name=test.mov',{method:'POST',headers:{origin:'http://localhost','content-length':'4',cookie:'private',authorization:'private'},body:'test'});
  assert.equal((await gemmaProxy(r,'/jobs')).status,202);
});
test('Gemma preserves media ranges and enforces upload size',async(t)=>{
  t.mock.method(globalThis,'fetch',async(_url:unknown,init:RequestInit)=>{
    assert.equal(new Headers(init.headers).get('range'),'bytes=0-3');
    return new Response('data',{status:206,headers:{'content-range':'bytes 0-3/20'}});
  });
  const out=await gemmaProxy(new Request('http://localhost/',{headers:{range:'bytes=0-3'}}),'/jobs/'+'a'.repeat(32)+'/media');
  assert.equal(out.status,206);assert.equal(out.headers.get('content-range'),'bytes 0-3/20');
  const r=new Request('http://localhost/',{method:'POST',headers:{origin:'http://localhost','content-length':String(65*1024*1024)},body:'a'});
  assert.equal((await gemmaProxy(r,'/jobs')).status,413);
});
