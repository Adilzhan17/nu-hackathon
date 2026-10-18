"""Frozen local SigLIP2 semantic features; descriptions are hypotheses, never labels."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_v2.experiment import RUN,MANIFEST,sha,folds,evaluate
from quality_v2.sequence import aggregate,fit,score,choose
from train_x3d import atomic_json,metrics

PAIRS=[
 ('safe','an open metal safe with its door open','a closed metal safe with its door shut'),
 ('desk','an unattended bank teller desk with no employee','a bank employee sitting at a teller desk'),
 ('cash','banknotes and bundles of cash lying on a desk','an empty desk with no banknotes'),
 ('keys','a key ring and metal keys lying on a desk','a clean empty desk surface'),
 ('door','an open office door','a closed office door'),
 ('cash_transfer','a customer handing banknotes to a bank employee','a customer talking to a bank employee without exchanging objects'),
 ('document_transfer','a customer handing an identity card to a bank employee','a customer handing banknotes to a bank employee'),
 ('counting','a bank employee counting banknotes by hand','a bank employee typing on a keyboard'),
 ('counter','a bank employee placing banknotes in a money counting machine','an unused money counting machine sitting on a desk'),
 ('detector','a bank employee inspecting a banknote in a currency detector','a bank employee putting banknotes directly into a drawer'),
 ('id','a bank employee examining a small identity card','a bank employee examining a large sheet of paper'),
 ('computer','a bank employee working at a computer','a bank employee standing away from the computer'),
 ('phone','a person holding and using a mobile phone','a person handling documents with both hands'),
 ('drink','a person drinking from a cup at a desk','a person handling banknotes at a desk'),
 ('customer','a customer standing in front of a bank teller','a bank teller working alone without a customer'),
 ('people','several people standing around a teller desk','one employee alone at a teller desk'),
 ('stamp','a person stamping a document with a rubber stamp','a person signing a document with a pen'),
 ('drawer','an open desk drawer with cash inside','a closed desk drawer'),
 ('safe_interaction','a person reaching into a metal safe','a person standing away from a metal safe'),
 ('leaving','a person walking out of the office','a person sitting at a desk'),
]

def extract():
    import torch,numpy as np
    from transformers import SiglipModel,AutoTokenizer
    from quality_features import load_views
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads((RUN/'plan.json').read_text());assert sha(MANIFEST)==plan['manifest_sha256'];folder=Path('artifacts/assets/quality_siglip')
    model=SiglipModel.from_pretrained(folder,local_files_only=True).cuda().eval()
    params=sum(p.numel() for p in model.parameters());assert params<=3_000_000_000
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    texts=['This is a photo of '+description+'.' for _,a,b in PAIRS for description in (a,b)]
    tokens=tokenizer(texts,padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    with torch.inference_mode():
        text=model.get_text_features(input_ids=tokens['input_ids'].cuda());text=torch.nn.functional.normalize(text,dim=-1)
    rows=[];visual=[];start=time.time()
    for i,r in enumerate(plan['clips']):
        if time.time()-start>1200:raise TimeoutError('Semantic extraction budget exceeded')
        x=load_views(Path('data/raw/open')/r['video'],r['start_sec'],r['end_sec'],16,384)
        x=x[:,:, [2,6,10,14]].permute(0,2,1,3,4).reshape(8,3,384,384).cuda()
        with torch.inference_mode():
            f=model.get_image_features(pixel_values=(x-.5)/.5).float();f=torch.nn.functional.normalize(f,dim=-1)
            similarities=f@text.T;delta=(similarities[:,0::2]-similarities[:,1::2])*10
            rows.append(torch.cat([delta.mean(0),delta.std(0,unbiased=False),delta.max(0).values,delta.min(0).values]).cpu().numpy())
            visual.append(torch.cat([f.mean(0),f.std(0,unbiased=False)]).cpu().numpy())
        atomic_json(RUN/'semantic_status.json',dict(status='running',stage='features',family='SigLIP2 · 384px + semantic contrasts',processed=i+1,total=len(plan['clips']),parameters=params,updated_at=time.time(),elapsed_sec=time.time()-start))
        if i%20==0:print('SigLIP2',i+1,len(plan['clips']),flush=True)
    np.savez_compressed(RUN/'semantic.npz',features=np.stack(rows),visual=np.stack(visual),plan_sha256=sha(RUN/'plan.json'))
    atomic_json(RUN/'semantic_prompts.json',dict(pairs=PAIRS,parameters=params,model_source=json.loads((folder/'source.json').read_text()),note='Unverified semantic similarities, not ground-truth object states.'))
    atomic_json(RUN/'semantic_status.json',dict(status='features_ready',parameters=params,updated_at=time.time(),elapsed_sec=time.time()-start))

def train():
    import torch,numpy as np
    torch.set_num_threads(2);start=time.time();plan=json.loads((RUN/'plan.json').read_text());bags=plan['bags'];assert sha(MANIFEST)==plan['manifest_sha256']
    data=np.load(RUN/'semantic.npz');assert str(data['plan_sha256'])==sha(RUN/'plan.json');xs={}
    for family,array in [('concepts','features'),('visual','visual')]:
        windows=torch.from_numpy(data[array]).float()
        for mode in ['statistics','order3','order5']:xs[family+'_'+mode]=torch.stack([aggregate(windows,b['indices'],mode) for b in bags])
    trainids=[i for i,b in enumerate(bags) if b['split']=='train'];valids=[i for i,b in enumerate(bags) if b['split']=='val']
    recipes=[(k,a) for k in xs for a in [.01,.1,1.]];outer=[]
    for k,val in enumerate(folds(trainids,bags)):
        ids=[i for i in trainids if i not in val];chosen,_=choose(xs,bags,ids,recipes);m=fit(xs[chosen['feature']],bags,ids,chosen['alpha']);p=score(xs[chosen['feature']][val],m)
        outer.extend(dict(bag=i,target=bags[i]['target'],probability=float(prob),threshold=chosen['threshold'],fold=k) for i,prob in zip(val,p))
        atomic_json(RUN/'semantic_status.json',dict(status='running',stage='nested_validation',fold=k+1,total=4,updated_at=time.time()));print('Semantic fold',k+1,flush=True)
    selected,candidates=choose(xs,bags,trainids,recipes);atomic_json(RUN/'semantic_selection.json',dict(selected=selected,candidates=candidates,selected_before_holdout=True))
    model=fit(xs[selected['feature']],bags,trainids,selected['alpha']);p=score(xs[selected['feature']][valids],model)
    nested=metrics([r['target'] for r in outer],[int(r['probability']>=r['threshold']) for r in outer]);held=evaluate([bags[i]['target'] for i in valids],p.tolist(),selected['threshold'])
    report=dict(selected=selected,nested_train_metrics=nested,nested_predictions=outer,holdout_video_metrics=held,
        holdout_videos=[dict(video=bags[i]['video'],target=bags[i]['target'],probability=float(prob)) for i,prob in zip(valids,p)],
        train_videos=32,val_videos=8,manifest_sha256=plan['manifest_sha256'],elapsed_sec=time.time()-start,scope='whole_video',
        limitations=['Semantic similarities are not confirmed actions or violations.','Reused 8-video holdout; no independent test.','Temporal localization not validated.'])
    torch.save(dict(**model,selected=selected,manifest_sha256=plan['manifest_sha256']),RUN/'semantic_head.pt')
    atomic_json(RUN/'semantic_report.json',report);atomic_json(RUN/'semantic_status.json',dict(status='completed',updated_at=time.time(),metrics=held,nested_metrics=nested));print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','train']);a=p.parse_args();globals()[a.stage]()
