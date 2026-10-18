"""Positive text-alignment adapter; group-held-out diagnosis, never legal labels."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

def fit(x, base, targets, groups, ridge=1.0):
    # Positive embedding targets only: unmentioned actions are not negative labels.
    counts=Counter(groups); w=np.array([1/counts[g] for g in groups])**.5
    a=x*w[:,None]; residual=(targets-base)*w[:,None]
    return a.T @ np.linalg.solve(a@a.T+ridge*np.eye(len(a)), residual)

def evaluate(x, base, targets, prototypes, labels, groups):
    scores=np.zeros((len(x),len(prototypes))); folds=[]
    for group in sorted(set(groups)):
        test=np.array([g==group for g in groups]); train=~test
        if not train.any(): raise ValueError('Need at least two video groups')
        adapter=fit(x[train],base[train],targets[train],np.array(groups)[train])
        scores[test]=norm(base[test]+x[test]@adapter)@prototypes.T
        folds.append(dict(test_group=group,train_groups=sorted(set(np.array(groups)[train])),test_examples=int(test.sum())))
    return scores,folds

def run(args):
    import torch
    from transformers import SiglipModel,AutoTokenizer,AutoImageProcessor
    from ml.autolabel.sampling import frames
    from ml.narration_training.extract import gpu_lock
    from ml.narration_training.prepare import MODEL_ID,MODEL_REVISION
    root=Path.cwd();out=args.output;out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (args.snapshot/'manifest.jsonl').read_text().splitlines()]
    taxonomy=json.loads((args.snapshot/'label_map.json').read_text())['actions']
    taxonomy=[a for a in taxonomy if a['action']!='other_observed_action']
    ids=[a['action'] for a in taxonomy]
    # 'Other' has no consistent visual semantics; preserve it in snapshot, not a class target.
    usable=[r for r in rows if r['observed_action'] in ids]
    model_path=root/'artifacts/assets/quality_siglip'
    source=json.loads((model_path/'source.json').read_text())
    assert (source['repo'],source['revision'])==(MODEL_ID,MODEL_REVISION)
    for name,sha in source['files'].items():
        assert hashlib.sha256((model_path/name).read_bytes()).hexdigest()==sha,name
    all_groups={r['group_id'] for r in usable}
    assert len(all_groups)>=2
    checked=set()
    for r in usable:
        assert r['eligible_for_action_training'] and r['split']=='train'
        p=root/r['source_path']
        if p not in checked:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['video_sha256'];checked.add(p)
    with gpu_lock(root/'.gpu-work.lock'):
        torch.set_num_threads(4)
        model=SiglipModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16).eval().cuda()
        parameters=sum(p.numel() for p in model.parameters());assert parameters<=3_000_000_000
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        processor=AutoImageProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False)
        prompts=['Видео банковской кассы. '+a['label'] for a in taxonomy]
        with torch.inference_mode():
            tokens=tokenizer(prompts,padding='max_length',max_length=model.config.text_config.max_position_embeddings,truncation=True,return_tensors='pt')
            prototypes=norm(model.get_text_features(input_ids=tokens['input_ids'].cuda()).float().cpu().numpy())
        xs=[];bases=[];evidence=[]
        for r in usable:
            start,end=r['start_sec'],r['end_sec']
            sampled=list(frames(root/r['source_path'],start,end-1e-4,16/(end-start),max_side=1024))[:16]
            assert sampled
            with torch.inference_mode():
                pixels=processor(images=[im for _,im in sampled],return_tensors='pt')['pixel_values'].cuda().bfloat16()
                v=norm(model.get_image_features(pixel_values=pixels).float().cpu().numpy())
            base=norm(v.mean(0));early=v[:max(1,len(v)//2)].mean(0);late=v[len(v)//2:].mean(0)
            xs.append(norm(np.concatenate([base,early,late])));bases.append(base)
            evidence.append(dict(task_id=r['task_id'],timestamps=[float(t) for t,_ in sampled]))
            print(json.dumps(dict(encoded=len(xs),total=len(usable))),flush=True)
        del model;torch.cuda.empty_cache()
    x=np.array(xs);base=np.array(bases);labels=np.array([ids.index(r['observed_action']) for r in usable]);groups=[r['group_id'] for r in usable]
    targets=prototypes[labels]
    scores,folds=evaluate(x,base,targets,prototypes,labels,groups)
    baseline=base@prototypes.T
    def metrics(s):
        return dict(top1_annotation_agreement=float(np.mean(s.argmax(1)==labels)),top3_annotation_agreement=float(np.mean([y in np.argsort(z)[-3:] for y,z in zip(labels,s)])),examples=len(labels),groups=len(set(groups)))
    before,after=metrics(baseline),metrics(scores)
    # Final artifact uses all confirmed, specific action examples AFTER the grouped diagnosis.
    adapter=fit(x,base,targets,groups)
    np.savez_compressed(out/'adapter.npz',adapter=adapter,prototypes=prototypes,action_ids=np.array(ids))
    np.savez_compressed(out/'features.npz',x=x,base=base,labels=labels,groups=np.array(groups))
    predictions=[dict(task_id=r['task_id'],group_id=r['group_id'],observed_action=r['observed_action'],prediction=ids[int(s.argmax())],baseline_prediction=ids[int(b.argmax())]) for r,s,b in zip(usable,scores,baseline)]
    report=dict(status='completed',model=MODEL_ID,model_revision=MODEL_REVISION,parameters=parameters,backbone_frozen=True,trained_component='positive text-alignment residual adapter',snapshot=str(args.snapshot),submitted=len(rows),used=len(usable),excluded_generic_other=len(rows)-len(usable),evaluation='leave-one-source-video-out annotation agreement, not violation accuracy',independent_test=False,unannotated_actions_are_negatives=False,baseline=before,adapted=after,folds=folds,predictions=predictions,auto_confirmation_allowed=False,promoted_to_production=False,ridge=1.0)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'frame_timestamps.json').write_text(json.dumps(evidence))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
