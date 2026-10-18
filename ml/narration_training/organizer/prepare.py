"""Prepare immutable source/mapping and a claim-free visual plan for independent inspection."""
import argparse
from collections import Counter
import csv
import datetime as dt
import json
from pathlib import Path
import re

from ml.autolabel.dataset import registry, digest, snapshot
from ml.autolabel.sampling import metadata
from ml.import_open_labels import read_rows
from ml.narration_training.prepare import write_json, write_jsonl, new_output
from .source import RAW_MESSAGE, SPECS

def prepare(root, output):
    root=root.resolve();out=new_output(root,output)
    (out/'source').mkdir();(out/'blind').mkdir()
    reg,reviews,observations=registry(root)
    source=out/'source/original_message.txt';source.write_text(RAW_MESSAGE,encoding='utf-8')
    now=dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out/'source/provenance.json',dict(source_type='organizer_provided',authority='organizer_claim',
       received_via='user_supplied_verbatim_message_in_this_conversation',organizer_origin_independently_verified=False,
       original_message_sha256=digest(source),format='Original pasted Markdown escapes and spelling retained',
       captured_at=now,scope='whole_video_scenario_not_frame_annotation',version=1,
       mapping_confirmation=dict(question='Номера — номера в именах исходных файлов, например 2 → X-2.mov, 43 → Х-43.mov?',
                                 response_verbatim='да',interpretation='user_confirmed_filename_numbers',
                                 confirms_scenario_content=False,confirms_temporal_boundaries=False)))
    protected=snapshot(root)
    for path in ['audio.txt','data/raw/open/open_labels.xlsx','Кейс_ForteBank.pdf']:
        protected[path]=digest(root/path)
    write_json(out/'source/protected_inputs.json',protected)
    write_json(out/'source/protected_autolabel.json',{str(p.relative_to(root)):digest(p)
       for p in sorted((root/'data/autolabel').rglob('*')) if p.is_file()})
    sheets=list(read_rows(root/'data/raw/open/open_labels.xlsx'))
    names={row['A'].strip():i+2 for i,row in enumerate(sheets)}
    mapping=[];claims=[];ambiguous=[];videos=[];exposed=[]
    for lineno,line in enumerate(RAW_MESSAGE.splitlines(),1):
        match=re.match(r'^(\d+)\\\)(.*?)(?:\\)?$',line)
        if not match: raise ValueError('Unrecognized original message line')
        number=int(match[1]);raw_text=match[2]
        candidates=[r for name,r in reg.items() if re.fullmatch(r'[XХ]-(\d+)\.(mov|mp4)',Path(name).name,re.I)
                    and int(re.search(r'-(\d+)',name)[1])==number]
        if len(candidates)!=1:
            ambiguous.append(dict(number=number,candidates=[r['video'] for r in candidates],reason='missing_or_non_unique_filename'))
            continue
        row=candidates[0]
        if Path(row['video']).stem not in names: raise ValueError('Filename absent from original organizer workbook')
        duration=metadata(root/'data/raw/open'/row['video'])['duration_sec']
        mapped=dict(number=number,video=row['video'],group_id=row['group_id'],split=row['split'],
          video_sha256=row['sha256'],aliases=row['aliases'],duration_sec=duration,
          mapping_status='user_confirmed_filename_number',
          evidence=['user_reply_da_to_filename_mapping','unique_real_filename','explicit_filename_in_original_xlsx','verified_video_sha256_and_frozen_manifest'],
          xlsx_row=names[Path(row['video']).stem],
          ordinal_position_is_not_used=True)
        mapping.append(mapped)
        claim=dict(claim_id=f'organizer_v1_{number:02d}',source_type='organizer_provided',
          authority='organizer_claim',original_text=raw_text,raw_source_line=line,source_line=lineno,
          source_ref='source/original_message.txt',source_sha256=digest(source),
          **mapped,**SPECS[number],human_confirmed=False,visual_evidence_confirmed=False,
          temporal_scope='whole_video',event_start_sec=None,event_end_sec=None,
          binary_window_target=None,eligible_for_strong_training=False)
        if number==43:
            claim['exception']=dict(scope='repeat_counting_after_authenticity_check',is_error=False,
                                    applies_to_other_rules=False,whole_video_normal_label=None)
        claims.append(claim)
        # Only identity, bytes, split and length may cross the visual-model input boundary.
        videos.append(dict(video=row['video'],group_id=row['group_id'],sha256=row['sha256'],aliases=row['aliases'],
                           split=row['split'],duration_sec=duration))
        if row['split']=='val':
            exposed.append(dict(group_id=row['group_id'],videos=row['aliases'],claim_ids=[claim['claim_id']],
               disclosure='organizer_scenario_explanation_seen_by_developers',
               training_allowed=False,test_prompt_use_allowed=False,threshold_tuning_allowed=False,
               independent_test=False,mapping_status=mapped['mapping_status']))
    if len(mapping)+len(ambiguous)!=22: raise ValueError('Unexpected source count')
    write_jsonl(out/'source/organizer_claims_v1.jsonl',claims)
    write_jsonl(out/'number_mapping.jsonl',mapping);write_jsonl(out/'mapping_ambiguities.jsonl',ambiguous)
    with (out/'number_mapping.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['number','video','group_id','split','video_sha256','duration_sec','mapping_status','xlsx_row'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in mapping)
    write_jsonl(out/'disclosed_validation.jsonl',exposed)
    write_json(out/'blind/plan.json',dict(schema_version=1,manifest_sha256=protected['data/training/manifest.csv'],
       videos=videos,coarse_fps=1.,dense_fps=4.,max_refinement_windows=5,
       model_id='Qwen/Qwen3-VL-2B-Instruct',model_revision='89644892e4d85e24eaac8bacfd4f463576704203',
       organizer_claims_in_prompts=False,human_annotations_in_prompts=False,organizer_anchors_in_prompts=False,
       purpose='blind_visual_audit_not_test_scoring',max_parameters=3_000_000_000))
    summary=dict(claims=len(claims),mapped=len(mapping),ambiguous_mapping=len(ambiguous),
       mapped_splits=dict(Counter(r['split'] for r in mapping)),
       disclosed_val_groups=len(exposed),disclosed_val_files=sum(len(r['videos']) for r in exposed),
       video_minutes=sum(v['duration_sec'] for v in videos)/60,
       original_source_sha256=digest(source),human_labels_changed=0)
    write_json(out/'source_summary.json',summary);print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.root,a.output)
