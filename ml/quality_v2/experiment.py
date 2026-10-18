"""Video-level MIL; train-only nested selection, one frozen holdout evaluation."""
import argparse,csv,hashlib,json,math,os,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_x3d import atomic_json,metrics
from infer_video import windows

ROOT=Path.cwd();RUN=ROOT/'runs/quality_v2';MANIFEST=ROOT/'data/training/manifest.csv'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    import av
    rows=list(csv.DictReader(MANIFEST.open()));groups={}
    source={r['video']:r for r in csv.DictReader((ROOT/'data/review/event_review.csv').open(encoding='utf-8-sig'))}
    for r in rows:
        key=r['sha256']
        if key in groups:
            assert (groups[key]['split'],groups[key]['target'])==(r['split'],int(r['target']))
            continue
        path=ROOT/'data/raw/open'/r['video'];assert sha(path)==key
        with av.open(str(path)) as v:
            st=v.streams.video[0];duration=float(st.duration*st.time_base)
            width,height=st.width,st.height
        groups[key]=dict(video=r['video'],sha256=key,split=r['split'],target=int(r['target']),duration=duration,width=width,height=height)
    bags=list(groups.values());clips=[]
    for bi,b in enumerate(bags):
        b['indices']=[]
        for start,end in windows(b['duration']):
            b['indices'].append(len(clips));clips.append(dict(bag=bi,video=b['video'],start_sec=start,end_sec=end,kind='coverage'))
    legacy=[]
    for r in rows:
        if r['split']=='val':
            legacy.append(dict(r,index=len(clips)));clips.append(dict(bag=-1,video=r['video'],start_sec=float(r['start_sec']),end_sec=float(r['end_sec']),kind='legacy'))
    plan=dict(manifest_sha256=sha(MANIFEST),bags=bags,clips=clips,legacy=legacy,
              label_policy='Whole-video labels only; no positive label assigned to individual windows.',
              selection_policy='Nested 4-fold video-group CV inside 32 train videos; fixed 8-video reused holdout evaluated once after selection.')
    RUN.mkdir(parents=True,exist_ok=True);atomic_json(RUN/'plan.json',plan);print('Prepared',len(bags),'videos',len(clips),'windows')

def extract():
    import torch,numpy as np
    from quality_features import QualityEncoder,load_views
    from model_suite import Extractor,load_frames
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads((RUN/'plan.json').read_text());assert sha(MANIFEST)==plan['manifest_sha256']
    started=time.time();features={}
    for family in ['action','dino','ssl']:
        out=RUN/(family+'.npz')
        if out.exists():continue
        model=QualityEncoder(family,ROOT/'artifacts/assets',True,'cuda') if family!='ssl' else Extractor('videomaev2',ROOT/'artifacts/assets',True,'cuda')
        data=[]
        for i,r in enumerate(plan['clips']):
            if time.time()-started>1800:raise TimeoutError('30 minute extraction budget')
            path=ROOT/'data/raw/open'/r['video']
            x=load_views(path,r['start_sec'],r['end_sec']) if family!='ssl' else load_frames(path,r['start_sec'],r['end_sec'],16,224)
            data.append(model.encode(x.unsqueeze(0)).cpu().numpy())
            atomic_json(RUN/'status.json',dict(status='running',stage='features',family=family,processed=i+1,total=len(plan['clips']),updated_at=time.time(),elapsed_sec=time.time()-started))
            if i%20==0:print(family,i+1,len(plan['clips']),flush=True)
        np.savez_compressed(out,features=np.concatenate(data),plan_sha256=sha(RUN/'plan.json'))
        del model;torch.cuda.empty_cache()
    atomic_json(RUN/'status.json',dict(status='features_ready',updated_at=time.time(),elapsed_sec=time.time()-started))

def folds(ids,bags,count=4):
    result=[[] for _ in range(count)];rng=random.Random(144)
    for label in [0,1]:
        items=sorted(i for i in ids if bags[i]['target']==label);rng.shuffle(items)
        for j,i in enumerate(items):result[j%count].append(i)
    return result

def evaluate(targets,probs,threshold):
    m=metrics(targets,[int(p>=threshold) for p in probs]);m['accuracy']=sum(y==int(p>=threshold) for y,p in zip(targets,probs))/len(targets)
    return m

def fit(x,bags,ids,regularization):
    import torch
    if not ids or any(bags[i]['split']!='train' for i in ids):raise ValueError('Training on holdout is forbidden')
    torch.manual_seed(81)
    allidx=[j for i in ids for j in bags[i]['indices']]
    mean=x[allidx].mean(0,keepdim=True);std=x[allidx].std(0,keepdim=True).clamp(min=.1)
    z=(x-mean)/std/math.sqrt(x.shape[1]);w=torch.zeros(x.shape[1],1,requires_grad=True);bias=torch.zeros(1,requires_grad=True)
    optimizer=torch.optim.Adam([w,bias],lr=.035)
    y=torch.tensor([bags[i]['target'] for i in ids],dtype=torch.float32)
    weights=torch.where(y>0,1/y.sum(),1/(1-y).sum())/2
    for epoch in range(240):
        optimizer.zero_grad();raw=(z@w+bias).flatten()
        logits=torch.stack([raw[bags[i]['indices']].max() for i in ids])
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).sum()+regularization*w.square().sum()
        loss.backward();optimizer.step()
    return dict(mean=mean,std=std*math.sqrt(x.shape[1]),weight=w.detach().T,bias=bias.detach())

def scores(x,model,bags,ids):
    import torch
    with torch.no_grad():
        p=(((x-model['mean'])/model['std'])@model['weight'].T+model['bias']).flatten().sigmoid()
    return [p[bags[i]['indices']].max().item() for i in ids],p

def choose(xs,bags,ids,recipes):
    results=[]
    for family,reg in recipes:
        predictions={}
        for val in folds(ids,bags):
            train=[i for i in ids if i not in val];model=fit(xs[family],bags,train,reg)
            p,_=scores(xs[family],model,bags,val);predictions.update(zip(val,p))
        y=[bags[i]['target'] for i in ids];p=[predictions[i] for i in ids]
        # Threshold chosen only from out-of-fold training predictions.
        options=[(evaluate(y,p,t),t) for t in [.2,.35,.5,.65,.8]]
        metric,threshold=max(options,key=lambda pair:(pair[0]['balanced_accuracy'],pair[0]['f1'],-abs(pair[1]-.5)))
        results.append(dict(family=family,regularization=reg,threshold=threshold,metrics=metric,oof_predictions=[dict(bag=i,target=bags[i]['target'],probability=predictions[i]) for i in ids]))
    return max(results,key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['f1'])),results

def train():
    import numpy as np,torch
    torch.set_num_threads(2)
    plan=json.loads((RUN/'plan.json').read_text());assert sha(MANIFEST)==plan['manifest_sha256'];bags=plan['bags'];xs={}
    for family in ['action','dino','ssl']:
        data=np.load(RUN/(family+'.npz'));assert str(data['plan_sha256'])==sha(RUN/'plan.json');xs[family]=torch.from_numpy(data['features']).float()
    recipes=[(f,r) for f in xs for r in [.001,.01]]
    trainids=[i for i,b in enumerate(bags) if b['split']=='train'];valids=[i for i,b in enumerate(bags) if b['split']=='val']
    outer=[];start=time.time()
    for k,val in enumerate(folds(trainids,bags)):
        ids=[i for i in trainids if i not in val];chosen,_=choose(xs,bags,ids,recipes)
        model=fit(xs[chosen['family']],bags,ids,chosen['regularization']);probs,_=scores(xs[chosen['family']],model,bags,val)
        outer.extend(dict(bag=i,target=bags[i]['target'],probability=p,threshold=chosen['threshold'],fold=k) for i,p in zip(val,probs))
        atomic_json(RUN/'status.json',dict(status='running',stage='nested_validation',fold=k+1,total=4,updated_at=time.time()));print('Nested fold',k+1,flush=True)
    selected,candidates=choose(xs,bags,trainids,recipes)
    atomic_json(RUN/'selection.json',dict(selected=selected,candidates=candidates,selected_before_holdout=True))
    model=fit(xs[selected['family']],bags,trainids,selected['regularization']);probs,allprobs=scores(xs[selected['family']],model,bags,valids)
    nested=metrics([r['target'] for r in outer],[int(r['probability']>=r['threshold']) for r in outer])
    holdout=evaluate([bags[i]['target'] for i in valids],probs,selected['threshold'])
    legacy_probs=[allprobs[r['index']].item() for r in plan['legacy']]
    legacy=evaluate([int(r['target']) for r in plan['legacy']],legacy_probs,selected['threshold'])
    report=dict(schema_version=1,manifest_sha256=plan['manifest_sha256'],selected=selected,nested_train_metrics=nested,nested_predictions=outer,
        holdout_video_metrics=holdout,holdout_videos=[dict(video=bags[i]['video'],target=bags[i]['target'],probability=p) for i,p in zip(valids,probs)],
        legacy_window_metrics=legacy,legacy_predictions=[dict(video=r['video'],start_sec=r['start_sec'],target=int(r['target']),probability=p) for r,p in zip(plan['legacy'],legacy_probs)],
        limitations=['8 holdout videos previously used for older models; not an independent test.','No verified event intervals; MIL peaks are review proposals.','Nested CV groups exact duplicate videos, not camera/actor families.'],
        training_videos=len(trainids),validation_videos=len(valids),elapsed_sec=time.time()-start)
    torch.save(dict(**model,selected=selected,manifest_sha256=plan['manifest_sha256']),RUN/'head.pt')
    atomic_json(RUN/'report.json',report);atomic_json(RUN/'status.json',dict(status='completed',stage='evaluated',updated_at=time.time(),metrics=holdout,legacy_metrics=legacy,nested_metrics=nested));print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','train']);a=p.parse_args();globals()[a.stage]()
