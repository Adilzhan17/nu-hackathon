"""Lossless schema normalization of saved local responses; never invent labels or timestamps."""
import json
from .deep import parse_deep,convert_event


def normalize_record(record,candidate,plan):
    raw=record['raw_response'];text=raw.strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):text=text[3:-3].strip()
    repairs=[];props=[];details=[];errors=[];summary=''
    try:
        # The model occasionally omits only the optional explanatory prose.
        # Preserve its absence; frame evidence/action/object/confidence remain mandatory.
        data=json.loads(text)
        if isinstance(data,dict) and isinstance(data.get('events'),list):
            for index,e in enumerate(data['events']):
                if isinstance(e,dict) and 'evidence_description' not in e:
                    e['evidence_description']=''
                    repairs.append(dict(event_index=index,field='evidence_description',operation='missing_optional_description_preserved_as_empty'))
        parsed=parse_deep(json.dumps(data,ensure_ascii=False,allow_nan=False));summary=parsed['sequence_summary']
        for i,e in enumerate(parsed['events']):
            try:
                p=convert_event(e,candidate,plan,f"{candidate['candidate_id']}_event_{i:02d}")
                props.append(p);details.append(dict(proposal_id=p['proposal_id'],evidence_description=e['evidence_description'],
                                                   missing_event_description=not bool(e['evidence_description']),
                                                   boundary_basis='local_model_frame_indices_not_human_verified'))
            except (ValueError,TypeError,KeyError) as exc:errors.append(str(exc))
    except (ValueError,TypeError,KeyError) as exc:errors.append(str(exc))
    status=('partially_invalid' if props else 'invalid') if errors else ('ok' if props else 'empty')
    return dict(candidate_id=candidate['candidate_id'],video=candidate['video'],group_id=candidate['group_id'],
                start_sec=candidate['start_sec'],end_sec=candidate['end_sec'],sequence_summary=summary,
                proposals=props,event_details=details,normalizations=repairs,errors=errors,
                inference_status=status,elapsed_sec=record['elapsed_sec'])
