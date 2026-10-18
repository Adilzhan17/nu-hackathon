"""Expose real held-out misses for human inspection, with model and reference boxes."""
import argparse
import json
from pathlib import Path

def build(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text());examples=[];questions=[]
    for condition in ['old_tiles','new_tiles']:
        if condition not in report['conditions']:continue
        rows=report['conditions'][condition]['errors'];selected=[]
        for cls in [0,1]:
            found=[r for r in rows if any(g['class_id']==cls for g in r['missed_targets'])]
            selected.extend(found[:2])
        if not selected:selected=rows[:2]
        seen=set()
        for r in selected:
            if r['image_id'] in seen:continue
            seen.add(r['image_id']);raw=json.loads((folder/condition/(r['image_id']+'.json')).read_text());image_id=r['image_id']
            url='/api/human-objects/images/'+image_id+'.jpg' if image_id.startswith('human_') else '/autolabel/objects_v1/images/'+image_id+'.jpg'
            examples.append(dict(condition=condition,image_id=image_id,image_url=url,group_id=r['group_id'],
                checkpoint_sha256=report['conditions'][condition]['checkpoint_sha256'],targets=raw['targets'],
                predictions=[p for p in raw['predictions'] if p['score']>=.25],missed_targets=r['missed_targets'],
                review_url='/label/objects?image_id='+image_id))
            questions.append(dict(image_id=image_id,group_id=r['group_id'],question='Для каждой рамки денег/документа укажите: на столе, в руках, перекрыт рукой или неразличимо. Это отдельный контекст, не новая метка присутствия предмета.',status='unreviewed',eligible_for_training=False))
    report['examples']=examples;(folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    questions=list({q['image_id']:q for q in questions}.values())
    (folder/'hand_table_questions.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in questions))
    return len(examples)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(build(a.folder))
