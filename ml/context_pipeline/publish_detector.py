"""Validated atomic experiment checkpoint switch, with a saved rollback registry."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from ml.live_console.publish import atomic

def check_promotion(report):
    """Conservative regression check, not a claim of independent-test accuracy."""
    old=report['conditions']['old_tiles'];new=report['conditions']['new_tiles']
    for cls in ['cash','document_id']:
        previous=old['per_class'][cls];candidate=new['per_class'][cls]
        if not candidate.get('support') or not candidate.get('tp'):
            raise ValueError('Promotion refused: no true positive at fixed threshold for '+cls)
        if any(candidate[k]<previous[k] for k in ['precision','recall','map50_95']):
            raise ValueError('Promotion refused: regression against the retained baseline for '+cls)

def activate(root,checkpoint,evaluation,name):
    root=Path(root).resolve();checkpoint=Path(checkpoint).resolve();evaluation=Path(evaluation)
    if not checkpoint.is_relative_to(root/'runs'):raise ValueError('Expected an experiment checkpoint')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();report=json.loads(evaluation.read_text())
    condition=report['conditions']['new_tiles']
    if condition['checkpoint_sha256']!=sha:raise ValueError('Evaluation does not belong to this checkpoint')
    if condition['source_rows_sha256']!=report['conditions']['old_tiles']['source_rows_sha256']:raise ValueError('Different evaluation inputs')
    if condition['precision_recall_confidence']!=.25:raise ValueError('Unexpected confidence protocol')
    check_promotion(report)
    from ultralytics import YOLO
    model=YOLO(str(checkpoint),task='detect')
    classes=[model.names[i] for i in range(len(model.names))]
    if classes!=['cash','document_id','key','stamp']:raise ValueError('Unexpected detector taxonomy')
    config=dict(name=name,checkpoint=str(checkpoint.relative_to(root)),sha256=sha,classes=classes,confidence=.25,tiled=True,
        native_tile_size=384,model_input_size=960,experimental=True,evaluation=str(evaluation.relative_to(root)),
        published_at=time.time(),limitations=report['limitations'],score_semantics='uncalibrated_detector_score')
    registry=root/'runs/context_pipeline/detector_active.json';history=registry.parent/'registry_history';history.mkdir(exist_ok=True)
    if registry.exists():
        previous=json.loads(registry.read_text());path=history/(str(time.time_ns())+'_'+previous['sha256'][:12]+'.json');atomic(path,json.dumps(previous,ensure_ascii=False,indent=2));config['rollback_registry']=str(path.relative_to(root))
    atomic(registry,json.dumps(config,ensure_ascii=False,indent=2));return config

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--evaluation',type=Path);p.add_argument('--name',default='YOLO11m human v2');p.add_argument('--rollback',type=Path);a=p.parse_args();root=Path.cwd()
    if a.rollback:
        saved=json.loads(a.rollback.read_text());path=root/saved['checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['sha256']:raise ValueError('Rollback checkpoint changed')
        atomic(root/'runs/context_pipeline/detector_active.json',json.dumps(saved,ensure_ascii=False,indent=2));print('Rollback registry restored')
    else:
        if not a.checkpoint or not a.evaluation:p.error('--checkpoint and --evaluation required')
        print(json.dumps(activate(root,a.checkpoint,a.evaluation,a.name),ensure_ascii=False))
