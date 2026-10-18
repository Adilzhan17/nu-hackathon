"""One bounded private-GPU job. Starts and stops only its own llama.cpp child."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from .protocol import ACTIONS, PROPERTIES, assess, coverage, inspect_answer, memory_before, prompt_memory, windows, parse_result
from .schema import output_schema


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(path)


MAX_INSTRUCTIONS_BYTES = 32 * 1024
OUTPUT_GUIDANCE = (
    'Return one JSON object with exactly the array keys events, states and rule_checks, following the supplied schema. '
    'First record supported visible actions and states, then assess relevant rules. '
    'Unknown roles or object types may be reported as unknown while preserving the visible action. '
    'Use short Russian descriptions and current frame IDs; never copy the input memory as current evidence. '
    'At most 3 events, 3 states and 2 rule checks. There is no minimum quota; omit unsupported observations.'
)
LEGACY_OUTPUT_GUIDANCE = (
    'Return ONLY {"events":[],"states":[],"rule_checks":[]} with current visual evidence. '
    'Never output or copy the input memory schema. Use short Russian descriptions and at most 3 evidence IDs per observation. '
    'At most 3 events, 3 states and 2 rule checks. Empty arrays are valid when nothing can be established.'
)


def prompt_guidance(instructions_path=None):
    """Experimental wording is opt-in; ordinary uploads retain legacy wording."""
    if instructions_path is None:
        return dict(version='legacy_default_v1',
                    current_pass_suffix='. Return salient events, states, relevant rule_checks and unknowns. Keep JSON compact.',
                    final=LEGACY_OUTPUT_GUIDANCE)
    return dict(version='observation_first_explicit_v4',
                current_pass_suffix='. First record supported visible actions and states, then relevant rule_checks. Express unknowns in uncertainty and missing_evidence. Keep JSON compact.',
                final=OUTPUT_GUIDANCE)


def load_instructions(root,instructions_path=None):
    """Select a bounded local prompt without changing the production default.

    Optional CLI overrides are restricted to .txt files inside this component.
    Resolve first so traversal and symlinks cannot read an unrelated local file.
    """
    root=Path(root).resolve();allowed=root/'ml/gemma_lab'
    requested=Path(instructions_path) if instructions_path is not None else Path('ml/gemma_lab/instructions.txt')
    path=(requested if requested.is_absolute() else root/requested).resolve()
    if not path.is_relative_to(allowed) or path.suffix!='.txt' or not path.is_file():
        raise ValueError('Instructions must be a .txt file inside ml/gemma_lab')
    with path.open('rb') as stream:
        data=stream.read(MAX_INSTRUCTIONS_BYTES+1)
    if not data or len(data)>MAX_INSTRUCTIONS_BYTES:
        raise ValueError('Instructions must contain 1–32768 bytes')
    text=data.decode('utf-8')
    if not text.strip():raise ValueError('Instructions must not be blank')
    return text,path


def run(root,job,seconds,instructions_path=None):
    job=job.resolve();root=root.resolve()
    if not job.is_relative_to(root/'runs/gemma_lab'):
        raise ValueError('Invalid job folder')
    plan=json.loads((job/'plan.json').read_text())
    if set(plan)!={'schema_version','duration_sec','source_sha256','frames','detail_frames'}:
        raise ValueError('Non-blind input fields')
    started=time.monotonic();observations=[];claims=[];rejected=[];inspected=[];trace=[]
    assets=root/'artifacts/gemma_lab';provenance=json.loads((assets/'source.json').read_text())
    instructions,selected_instructions=load_instructions(root,instructions_path)
    guidance=prompt_guidance(instructions_path)
    refs=plan['frames'];all_refs=refs+plan['detail_frames']
    for f in all_refs:
        path=(job/f['path']).resolve()
        if not path.is_relative_to(job) or hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:
            raise ValueError('Changed input frame')
    result=dict(status='waiting_gpu',model_id=provenance['repo'],model_revision=provenance['revision'],
        quantization=provenance['quantization'],runtime_release=provenance['runtime_release'],
        prompt_sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        instructions_source=str(selected_instructions.relative_to(root)),
        instructions_bytes=len(instructions.encode('utf-8')),
        prompt_guidance_version=guidance['version'],
        prompt_guidance_sha256=hashlib.sha256(json.dumps(guidance,sort_keys=True).encode()).hexdigest(),
        observations=[],rules=[],model_rule_claims=[],memory=[],rejections=[],review_status='proposed',
        ground_truth=False,organizer_answers_in_prompt=False,cloud_ai_used=False)
    frame_step=min((b['timestamp']-a['timestamp'] for a,b in zip(refs,refs[1:])
                    if b['timestamp']>a['timestamp']),default=.5)
    result['inference_config']=dict(context_tokens=16384,image_max_tokens=512,microbatch=512,
                                    frame_step_sec=round(frame_step,6))
    result['output_schema_version']='bounded_observations_v1'
    result['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'ml/gemma_lab').glob('*.py')}
    (job/'instructions.txt').write_text(instructions)
    def save(stage,status='processing',error=None):
        result.update(status=status,stage=stage,observations=observations,rules=assess(observations,claims) if observations else [],
            model_rule_claims=claims,memory=trace,rejections=rejected,coverage=coverage(refs,inspected),
            elapsed_sec=round(time.monotonic()-started,2),error=error)
        write(job/'result.json',result)
        print(json.dumps(dict(stage=stage,status=status,frames=len(set(inspected)),elapsed=result['elapsed_sec'])),flush=True)
    save('waiting_gpu')
    server=None
    def interrupted(*_):
        raise TimeoutError('Bounded GPU job interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    with (root/'.gpu-work.lock').open('a') as lease:
        try:
            until=time.monotonic()+15
            while True:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>until:raise TimeoutError('GPU занят другой задачей; повторите позже')
                    time.sleep(.5)
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
                raise RuntimeError('GPU занят процессом, не использующим общий lock; запуск отложен')
            runtime=assets/'runtime';binary=next(runtime.rglob('llama-server'))
            libdirs={str(p.parent) for p in runtime.rglob('*.so*')}
            private=assets/'private-sysroot/usr/lib/x86_64-linux-gnu'
            libpath=':'.join([str(private),*sorted(libdirs)])
            env=dict(os.environ,OMP_NUM_THREADS='4')
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            base=f'http://127.0.0.1:{port}'
            command=[str(private/'ld-linux-x86-64.so.2'),'--library-path',libpath,str(binary),'-m',str(assets/'gemma-4-12b-it-qat-q4_0.gguf'),
                '--mmproj',str(assets/'mmproj-gemma-4-12b-it-qat-q4_0.gguf'),'--host','127.0.0.1','--port',str(port),
                '-ngl','99','-c','16384','-np','1','-t','4','-tb','4','-b','512','-ub','512',
                '--flash-attn','on','--jinja','--no-webui','--reasoning','off','--image-max-tokens','512']
            with (job/'llama.log').open('w') as log:
                server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            save('loading_model')
            while True:
                if server.poll() is not None:raise RuntimeError('Gemma runtime failed; see llama.log')
                if time.monotonic()-started>min(120,seconds/2):raise TimeoutError('Model load timeout')
                try:
                    with urllib.request.urlopen(base+'/health',timeout=2) as response:
                        if response.status==200:break
                except OSError:time.sleep(1)
            def ask(stage,block):
                remaining=seconds-(time.monotonic()-started)
                if remaining<25:raise TimeoutError('Лимит времени: сохранён частичный проход')
                mem=memory_before(observations,block[0]['timestamp'])
                trace.append(dict(stage=stage,start_sec=block[0]['timestamp'],memory=mem,
                                  current_state_of_prior_facts='unknown_until_observed_again'))
                prompt=instructions+'\nAllowed actions: '+','.join(ACTIONS)+'\nAllowed state properties: '+','.join(PROPERTIES)
                prompt+='\nStructured last-known memory (NOT established current state): '+json.dumps(prompt_memory(mem),ensure_ascii=False)
                prompt+='\nCurrent pass: '+stage+guidance['current_pass_suffix']
                content=[dict(type='text',text=prompt)]
                for f in block:
                    content.append(dict(type='text',text=f"Frame {f['id']}, time {f['timestamp']:.6f} s"))
                    data=base64.b64encode((job/f['path']).read_bytes()).decode()
                    content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+data)))
                content.append(dict(type='text',text=guidance['final']))
                save(stage)
                request=dict(messages=[dict(role='user',content=content)],max_tokens=2200,temperature=0,
                             response_format=dict(type='json_object',schema=output_schema(block)),stream=False,cache_prompt=False,
                             chat_template_kwargs=dict(enable_thinking=False))
                req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=min(180,remaining-5)) as response:answer=json.load(response)
                raw=answer['choices'][0]['message']['content']
                record=dict(stage=stage,frame_ids=[f['id'] for f in block],raw_response=raw,
                            finish_reason=answer['choices'][0].get('finish_reason'),usage=answer.get('usage'))
                write(job/(stage+'.json'),record)
                try:
                    parsed=parse_result(raw)
                    obs,new_claims,errors=inspect_answer(parsed,block,'gemma_'+plan['source_sha256'][:16],stage)
                    observations.extend(obs);claims.extend(new_claims);rejected.extend(dict(stage=stage,**e) for e in errors)
                    inspected.extend(f['id'] for f in block if f['id'].startswith('f'))
                except (ValueError,TypeError,KeyError) as e:
                    rejected.append(dict(stage=stage,error='Invalid model JSON: '+str(e)))
                save(stage+'_done')
            for i,block in enumerate(windows(refs)):
                ask(f'overview_{i:03d}',block)
            # A second pass checks up to two transitions with actual before/after frames.
            centers=[]
            priority={'employee_leaves','cash_received','person_enters','cash_counted_by_hand','cash_counter_used','phone_used','drink_used'}
            for o in observations:
                if o['action'] in priority and all(abs(o['start_sec']-c)>2 for c in centers):centers.append(o['start_sec'])
            if not centers and refs:centers=[refs[len(refs)//2]['timestamp']]
            for i,c in enumerate(centers[:2]):
                block=[f for f in plan['detail_frames'] if max(0,c-1)<=f['timestamp']<=min(plan['duration_sec'],c+1)]
                if len(block)>=2:ask(f'detail_{i:03d}',block)
            complete=coverage(refs,inspected)['all_sampled_frames_inspected']
            save('done','completed' if complete else 'partial')
        except Exception as e:
            save('interrupted','partial' if inspected else 'failed',str(e))
        finally:
            if server and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            fcntl.flock(lease,fcntl.LOCK_UN)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--instructions',type=Path,help='Optional .txt prompt inside ml/gemma_lab; default unchanged')
    a=p.parse_args()
    if not 60<=a.seconds<=900:raise ValueError('Bounded jobs only')
    run(Path.cwd(),a.job,a.seconds,instructions_path=a.instructions)
