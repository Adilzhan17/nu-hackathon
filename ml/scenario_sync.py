"""Copy bounded pilot outputs and preserve raw versus stricter rule assessments."""
import json,shlex,subprocess,tarfile,time
from pathlib import Path
from suite_sync import ROOT,REMOTE,SSH
from scenario import hypotheses
from train_x3d import atomic_json

run=ROOT/'runs/scenario_v2';start=time.time()
while time.time()-start<1800:
    r=subprocess.run(SSH+['cat '+shlex.quote(REMOTE+'/runs/scenario_v2/status.json')],capture_output=True,text=True,timeout=30)
    if r.returncode==0:
        state=json.loads(r.stdout);atomic_json(run/'status.json',state)
        if state.get('status') in ('completed','failed','stopped'):
            archive=run/'download.tar.gz'
            with archive.open('wb') as out:subprocess.run(SSH+['tar -C '+shlex.quote(REMOTE+'/runs/scenario_v2')+' -czf - .'],stdout=out,check=True,timeout=180)
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    if not (run/m.name).resolve().is_relative_to(run.resolve()) or not (m.isfile() or m.isdir()):raise ValueError('Unsafe archive entry')
                t.extractall(run,filter='data')
            archive.unlink()
            if (run/'report.json').exists():
                report=json.loads((run/'report.json').read_text());atomic_json(run/'report_raw.json',report)
                for video in report['videos']:
                    allowed={(c['kind'],tuple(c['supporting_event_ids'])) for c in hypotheses(video['events'])}
                    for c in video['hypotheses']:
                        if (c['kind'],tuple(c['supporting_event_ids'])) not in allowed:
                            c['status']='rejected_by_strict_rule';c['rejection']='Visible state must be observed after departure, not only before or during it.'
                report['postprocessing']='Strict after-departure evidence check; raw model outputs preserved in report_raw.json.'
                atomic_json(run/'report.json',report)
            break
    time.sleep(8)
else:raise SystemExit('Pilot synchronization timeout')
