"""Train-only nested comparison of whole-video temporal representations."""
import json,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_v2.experiment import RUN,MANIFEST,sha,folds,evaluate
from train_x3d import atomic_json,metrics

def aggregate(x,indices,mode):
    import torch
    f=x[indices]
    if mode=='statistics':return torch.cat([f.mean(0),f.std(0,unbiased=False)])
    # Ordered bins preserve beginning / middle / end without inventing window labels.
    bins=3 if mode=='order3' else 5
    chunks=[f[math.floor(i*len(f)/bins):max(math.floor(i*len(f)/bins)+1,math.floor((i+1)*len(f)/bins))].mean(0) for i in range(bins)]
    return torch.cat(chunks+[f.std(0,unbiased=False)])

def fit(x,bags,ids,alpha):
    import torch
    if any(bags[i]['split']!='train' for i in ids):raise ValueError('Holdout training forbidden')
    mean=x[ids].mean(0,keepdim=True);std=x[ids].std(0,keepdim=True).clamp(min=.1)*math.sqrt(x.shape[1]);z=(x[ids]-mean)/std
    y=torch.tensor([2*bags[i]['target']-1 for i in ids],dtype=torch.float32);bias=y.mean();y=y-bias
    dual=torch.linalg.solve(z@z.T+alpha*torch.eye(len(ids)),y)
    weight=(z.T@dual).unsqueeze(0)
    return dict(mean=mean,std=std,weight=weight,bias=bias.reshape(1),alpha=alpha)

def score(x,model):return (((x-model['mean'])/model['std'])@model['weight'].T+model['bias']).flatten().mul(2).sigmoid()

def choose(xs,bags,ids,recipes):
    results=[]
    for feature,alpha in recipes:
        ps={}
        for val in folds(ids,bags):
            train=[i for i in ids if i not in val];m=fit(xs[feature],bags,train,alpha);p=score(xs[feature][val],m)
            ps.update(zip(val,p.tolist()))
        y=[bags[i]['target'] for i in ids];p=[ps[i] for i in ids]
        options=[(evaluate(y,p,t),t) for t in [.35,.5,.65]]
        metric,threshold=max(options,key=lambda pair:(pair[0]['balanced_accuracy'],pair[0]['f1'],-abs(pair[1]-.5)))
        results.append(dict(feature=feature,alpha=alpha,threshold=threshold,metrics=metric,oof_predictions=[dict(bag=i,target=bags[i]['target'],probability=ps[i]) for i in ids]))
    return max(results,key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['f1'])),results

def main():
    import torch,numpy as np
    torch.set_num_threads(2);start=time.time();plan=json.loads((RUN/'plan.json').read_text());assert sha(MANIFEST)==plan['manifest_sha256'];bags=plan['bags'];xs={}
    for family in ['action','dino','ssl']:
        data=np.load(RUN/(family+'.npz'));assert str(data['plan_sha256'])==sha(RUN/'plan.json')
        windows=torch.from_numpy(data['features']).float()
        for mode in ['statistics','order3','order5']:xs[family+'_'+mode]=torch.stack([aggregate(windows,b['indices'],mode) for b in bags])
    trainids=[i for i,b in enumerate(bags) if b['split']=='train'];valids=[i for i,b in enumerate(bags) if b['split']=='val']
    recipes=[(k,a) for k in xs for a in [.01,.1,1.]]
    outer=[]
    for k,val in enumerate(folds(trainids,bags)):
        ids=[i for i in trainids if i not in val];chosen,_=choose(xs,bags,ids,recipes)
        m=fit(xs[chosen['feature']],bags,ids,chosen['alpha']);p=score(xs[chosen['feature']][val],m)
        outer.extend(dict(bag=i,target=bags[i]['target'],probability=float(prob),threshold=chosen['threshold'],fold=k) for i,prob in zip(val,p))
        atomic_json(RUN/'sequence_status.json',dict(status='running',stage='nested_validation',fold=k+1,total=4,updated_at=time.time()));print('Sequence fold',k+1,flush=True)
    selected,candidates=choose(xs,bags,trainids,recipes)
    atomic_json(RUN/'sequence_selection.json',dict(selected=selected,candidates=candidates,selected_before_holdout=True))
    model=fit(xs[selected['feature']],bags,trainids,selected['alpha']);p=score(xs[selected['feature']][valids],model)
    nested=metrics([r['target'] for r in outer],[int(r['probability']>=r['threshold']) for r in outer])
    held=evaluate([bags[i]['target'] for i in valids],p.tolist(),selected['threshold'])
    report=dict(selected=selected,nested_train_metrics=nested,nested_predictions=outer,holdout_video_metrics=held,
        holdout_videos=[dict(video=bags[i]['video'],target=bags[i]['target'],probability=float(prob)) for i,prob in zip(valids,p)],
        train_videos=32,val_videos=8,manifest_sha256=plan['manifest_sha256'],elapsed_sec=time.time()-start,
        scope='whole_video',limitations=['No interval-level ground truth or validated localization.','Same 8-video reused holdout as previous experiments; not an independent test.'])
    torch.save(dict(**model,selected=selected,manifest_sha256=plan['manifest_sha256']),RUN/'sequence_head.pt')
    atomic_json(RUN/'sequence_report.json',report);atomic_json(RUN/'sequence_status.json',dict(status='completed',updated_at=time.time(),metrics=held,nested_metrics=nested));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
