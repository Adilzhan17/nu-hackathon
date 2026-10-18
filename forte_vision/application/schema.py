"""Bound the generated response, not merely its eventual parser.
llama.cpp b11716 /v1/chat/completions supports response_format.schema.
"""
from .analysis import ACTIONS, PROPERTIES, RULES


def output_schema(frames):
    def string(limit=48):return {'type':'string','maxLength':limit}
    def enum(values):return {'type':'string','enum':list(values)}
    def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    def array(items,maximum,minimum=0):return {'type':'array','items':items,'maxItems':maximum,'minItems':minimum}
    frame=enum(f['id'] for f in frames)
    shared=dict(actor=string(),role=enum(['employee','customer','unknown']),object=string(),track_id=string(),
        visibility=enum(['clear','partial','unclear']),description=string(160),uncertainty=string(120))
    event=obj(dict(action=enum(ACTIONS),from_actor=string(),to_actor=string(),**shared,evidence_frames=array(frame,3,2)))
    state=obj(dict(property=enum(PROPERTIES),value=string(),**shared,evidence_frames=array(frame,3,1)))
    rule=obj(dict(rule_id=enum(RULES),verdict=enum(['violation_supported','no_violation_observed','insufficient_data','not_applicable']),
        evidence_frames=array(frame,3),triggering_event=string(120),explanation=string(200),missing_evidence=array(string(100),3)))
    return obj(dict(events=array(event,3),states=array(state,3),rule_checks=array(rule,2)))
