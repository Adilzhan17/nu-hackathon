"""Bounded T4 continuation with exact cache-input checks and per-response runtime provenance.
Run only on a NEW reconciled output, after the previous worker has exited.
The original analyzer and migration wrapper are not modified.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from .reconcile import CODE_HASH,CONTEXT_HASH,validate_plan


def patched_source(original):
    if hashlib.sha256(original.encode()).hexdigest()!=CODE_HASH:raise ValueError('Analyzer code changed; audit before resuming')
    old="if output.exists():return json.loads(output.read_text())"
    new="""if output.exists():
                    cached=json.loads(output.read_text())
                    if cached.get('prompt')!=prompt or cached.get('frames')!=refs or cached.get('parse_error'):
                        raise ValueError('Cached response input mismatch; preserve run and reconcile again: '+str(output))
                    if (cached.get('model_id'),cached.get('model_revision'))!=(source['repo'],source['revision']):
                        raise ValueError('Cached model revision mismatch')
                    if parse_answer(cached['raw_response'])!=cached.get('answer'):
                        raise ValueError('Cached raw and parsed responses disagree')
                    return cached"""
    if original.count(old)!=1:raise ValueError('Unexpected cache code')
    patched=original.replace(old,new)
    patched=patched.replace('dtype=torch.bfloat16,attn_implementation=','dtype=torch.float16,attn_implementation=')
    old="write_json(output,record);return record"
    if patched.count(old)!=1:raise ValueError('Unexpected record writer')
    patched=patched.replace(old,"record['runtime_ref']=RUNTIME_REF;record['generated_at_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();write_json(output,record);return record")
    return patched


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=1440);p.add_argument('--videos',nargs='*')
    args=p.parse_args();args.max_windows=0
    if not 1<=args.max_seconds<=1440:raise ValueError('Use bounded batches, max 1440 seconds')
    run=args.run.resolve();root=args.root.resolve()
    if not (run.parent/'reconciliation_summary.json').exists():raise ValueError('Requires a separately reconciled run')
    plan=json.loads((run/'plan.json').read_text())
    proof=json.loads((run.parent/'source_identity_verification.json').read_text())
    if proof.get('all_aliases_verified_on_main') is not True or proof['plan_sha256']!=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest():
        raise ValueError('Missing main-server duplicate verification')
    validate_plan(root,plan,allow_missing_aliases=True)
    if hashlib.sha256((root/'ml/narration_training/organizer/context.py').read_bytes()).hexdigest()!=CONTEXT_HASH:raise ValueError('Context schema changed')
    original=(root/'ml/narration_training/organizer/analyze.py').read_text();patched=patched_source(original)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    import av,PIL
    if av.__version__!='16.0.1' or PIL.__version__!='12.3.0':raise ValueError('Frame decoder/encoder version mismatch')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    # Transformers 4.57 enables native GQA even on T4, where it falls back to the
    # quadratic math kernel. Explicit KV repetition permits PyTorch's efficient
    # attention kernel. This changes only numerical execution, not tokens/frames.
    import transformers.integrations.sdpa_attention as sdpa
    from torch.nn.attention import sdpa_kernel,SDPBackend
    original_gqa=sdpa.use_gqa_in_sdpa
    sdpa.use_gqa_in_sdpa=lambda attention_mask,key:False
    relative='runtime/'+stamp+'.json';runtime=run/relative;runtime.parent.mkdir(exist_ok=True)
    record=dict(source_sha256=CODE_HASH,context_sha256=CONTEXT_HASH,executed_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='float16',gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,av=av.__version__,pillow=PIL.__version__,max_seconds=args.max_seconds,
        attention_backend='pytorch_efficient_attention_explicit_kv_repeat',native_gqa=False,
        transformers_sdpa_source_sha256=hashlib.sha256(Path(sdpa.__file__).read_bytes()).hexdigest(),
        processor_min_pixels=64*32*32,processor_max_pixels=128*32*32,
        numerical_equivalence_claim='No bitwise-equivalence claim. Synthetic FP16 comparison atol=rtol=0.005 passed; original cached responses remain unchanged.',
        cache_policy='Exact prompt, memory, frame IDs, PTS, paths, hashes and model revision; no implicit stage merge',
        changes=['T4 FP16','reject incompatible cache','per-response runtime provenance','explicit KV repeat + efficient SDPA to avoid T4 GQA math OOM'],human_annotations_read=False,organizer_answers_read=False)
    runtime.write_text(json.dumps(record,indent=2)+'\n')
    scope={'__name__':'organizer_bounded_resume','__file__':str(root/'ml/narration_training/organizer/analyze.py'),
           '__package__':'ml.narration_training.organizer','RUNTIME_REF':relative}
    exec(compile(patched,scope['__file__'],'exec'),scope)
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):scope['run'](args)
    except TimeoutError as exc:
        state=dict(status='bounded_batch_stopped',reason=str(exc),runtime_ref=relative,
                   completed_total=len(list(run.glob('video_*/complete.json'))),remaining_work_preserved=True)
        (run/'batch_status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
        return 75
    finally:sdpa.use_gqa_in_sdpa=original_gqa
    return 0

if __name__=='__main__':sys.exit(main())
