"""Grouped few-shot sequence matching with DTW; no holdout neighbors in training."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_v2.experiment import RUN,MANIFEST,sha,folds,evaluate
from train_x3d import atomic_json,metrics

def dtw(a,b):
    import numpy as np
    cost=np.maximum(0,1-a@b.T);n,m=cost.shape
    table=np.full((n+1,m+1),np.inf);table[0,0]=0
    for i in range(n):
        for j in range(m):table[i+1,j+1]=cost[i,j]+min(table[i,j],table[i,j+1],table[i+1,j])
    return float(table[n,m]/(n+m))

def predict(dist,bags,train,val,k):
    import numpy as np
    if set(train)&set(val) or any(bags[i]['split']!='train' for i in train):raise ValueError('Reference leakage')
    scores=[]
    for i in val:
        nearest=sorted(train,key=lambda j:dist[i,j])[:k]
        weights=np.array([1/max(.001,dist[i,j]) for j in nearest]);labels=np.array([bags[j]['target'] for j in nearest])
        scores.append(float((weights*labels).sum()/weights.sum()))
    return scores

def choose(distances,bags,ids):
    results=[]
    for feature,dist in distances.items():
        for k in [1,3,5,7]:
            ps={}
            for val in folds(ids,bags):
                train=[i for i in ids if i not in val];ps.update(zip(val,predict(dist,bags,train,val,k)))
            y=[bags[i]['target'] for i in ids];p=[ps[i] for i in ids]
            metric=evaluate(y,p,.5)
            results.append(dict(feature=feature,k=k,threshold=.5,metrics=metric,oof_predictions=[dict(bag=i,target=bags[i]['target'],probability=ps[i]) for i in ids]))
    return max(results,key=lambda r:(r['metrics']['balanced_accuracy'],r['metrics']['f1'])),results

def main():
    import numpy as np
    start=time.time();plan=json.loads((RUN/'plan.json').read_text());bags=plan['bags'];assert sha(MANIFEST)==plan['manifest_sha256'];distances={}
    arrays={f:np.load(RUN/(f+'.npz'))['features'] for f in ['action','dino','ssl']}
    data=np.load(RUN/'semantic.npz');arrays.update(siglip=data['visual'],concepts=data['features'])
    for family,x in arrays.items():
        for centered in [False,True]:
            sequences=[]
            for b in bags:
                seq=x[b['indices']].copy()
                if centered:seq-=seq.mean(0,keepdims=True)
                seq/=np.maximum(1e-8,np.linalg.norm(seq,axis=1,keepdims=True));sequences.append(seq)
            dist=np.zeros((len(bags),len(bags)),np.float32)
            for i in range(len(bags)):
                for j in range(i):dist[i,j]=dist[j,i]=dtw(sequences[i],sequences[j])
            distances[family+('_motion' if centered else '_raw')]=dist
    trainids=[i for i,b in enumerate(bags) if b['split']=='train'];valids=[i for i,b in enumerate(bags) if b['split']=='val'];outer=[]
    for k,val in enumerate(folds(trainids,bags)):
        train=[i for i in trainids if i not in val];chosen,_=choose(distances,bags,train)
        p=predict(distances[chosen['feature']],bags,train,val,chosen['k'])
        outer.extend(dict(bag=i,target=bags[i]['target'],probability=s,fold=k) for i,s in zip(val,p))
    selected,candidates=choose(distances,bags,trainids);atomic_json(RUN/'retrieval_selection.json',dict(selected=selected,candidates=candidates,selected_before_holdout=True))
    p=predict(distances[selected['feature']],bags,trainids,valids,selected['k']);nested=evaluate([r['target'] for r in outer],[r['probability'] for r in outer],.5);held=evaluate([bags[i]['target'] for i in valids],p,.5)
    report=dict(selected=selected,nested_train_metrics=nested,nested_predictions=outer,holdout_video_metrics=held,
        holdout_videos=[dict(video=bags[i]['video'],target=bags[i]['target'],probability=s,nearest=[dict(video=bags[j]['video'],target=bags[j]['target'],distance=float(distances[selected['feature']][i,j])) for j in sorted(trainids,key=lambda j:distances[selected['feature']][i,j])[:selected['k']]]) for i,s in zip(valids,p)],
        train_videos=32,val_videos=8,manifest_sha256=plan['manifest_sha256'],elapsed_sec=time.time()-start,scope='whole_video',
        limitations=['Matching similar videos does not confirm the type or location of a violation.','Camera/actor similarity may create shortcuts.','8-video holdout has been reused; not an independent test.'])
    atomic_json(RUN/'retrieval_report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
