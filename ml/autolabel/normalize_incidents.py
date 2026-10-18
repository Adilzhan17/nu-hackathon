"""Recover optional prose omissions while rejecting invalid state/frame evidence."""
import json
from .inspect_incidents import decode


def normalize(record,check):
    text=record['raw_response'].strip();repairs=[];errors=[]
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):text=text[3:-3].strip()
    try:
        d=json.loads(text)
        if isinstance(d,dict) and 'uncertainty_reason' not in d:
            d['uncertainty_reason']='';repairs.append(dict(field='uncertainty_reason',operation='missing_optional_prose_preserved_empty'))
        if not isinstance(d,dict) or not isinstance(d.get('states'),list):raise ValueError('Invalid state envelope')
        # Validate departure and the whole envelope before considering each state.
        decode(json.dumps(dict(d,states=[]),allow_nan=False),check)
        accepted=[]
        for index,state in enumerate(d['states']):
            if isinstance(state,dict) and 'uncertainty_reason' not in state:
                state['uncertainty_reason']='';repairs.append(dict(state_index=index,field='uncertainty_reason',operation='missing_optional_prose_preserved_empty'))
            try:
                decode(json.dumps(dict(d,states=[state]),allow_nan=False),check);accepted.append(state)
            except (ValueError,TypeError,KeyError) as e:errors.append(dict(state_index=index,error=str(e)))
        parsed=dict(d,states=accepted)
    except (ValueError,TypeError,KeyError) as e:
        parsed=None;errors.append(dict(error=str(e)))
    status='invalid' if parsed is None else 'partially_invalid' if errors else 'ok' if parsed['states'] or parsed['departure_frames'] else 'empty'
    return dict(check_id=check['check_id'],parsed=parsed,normalizations=repairs,errors=errors,status=status)
