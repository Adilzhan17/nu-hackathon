"""Evidence-linked memory and conservative, operational (not legal) rule assessments."""
import json
import re

ACTIONS = ['person_enters','person_leaves','employee_leaves','employee_returns','employee_absent',
 'document_transfer','document_checked','cash_received','cash_transferred','cash_unpacking','cash_bundle_split',
 'cash_placed','cash_stored','cash_counted_by_hand','cash_counter_used','banknote_detector_used',
 'safe_opening','safe_closing','door_opening','door_closing','key_handling','stamp_handling',
 'phone_put_away','phone_used','drink_used','computer_work','equipment_operation','object_handling','occluded']
PROPERTIES = ['person_presence','cash_location','cash_holder','document_location','key_location','stamp_location',
              'safe_door','room_door','phone_location','bag_location','clothing_location','computer_screen','equipment_visible']
VERDICTS = ['violation_supported','no_violation_observed','insufficient_data','not_applicable']
RULES = {
 'identity_before_cash': dict(text='Check the sequence of identity-document inspection and cash acceptance. A visible document transfer is not proof of identity inspection. Earlier unseen checks remain possible.',
                              sources=['organizer_train_2','organizer_train_17','organizer_train_34']),
 'previous_cash_before_next_customer': dict(text='Look for the same previously received cash still on the work surface when a different customer arrives; require a linked transfer/custody history, not similar-looking bundles.',sources=['organizer_train_38']),
 'phone_before_counting': dict(text='Put the phone away before counting cash. A phone absent from selected images is not proof it was put away.',sources=['organizer_train_10']),
 'safe_on_departure': dict(text='Check the same safe door before and after an employee departure; distinguish cabinet from safe, closed from locked, leaving the frame from leaving the room.',sources=['preexisting_ml/autolabel/rules.py']),
 'valuables_on_departure': dict(text='Check cash, keys and other observed valuable materials before/after an employee departure. Require visible object identity, departure and post-departure evidence.',sources=['organizer_train_11','organizer_train_15','organizer_train_22','organizer_train_26','preexisting_project_action_taxonomy']),
 'authenticity_after_counting': dict(text='Counting and authenticity-device inspection are different actions. Never judge banknote authenticity from appearance. A missing check needs verified complete operation boundaries, stage visibility and explicit absence evidence; sampled frames cannot prove absence.',sources=['preexisting_ml/autolabel/rules.py:banknote_check_not_observed']),
 'door_after_entry': dict(text='Inspect room-door state after entry. If the door is occluded or outside view, current state is unknown; a later closure and undefined allowed delay limit conclusions.',sources=['organizer_train_30','organizer_train_32','organizer_train_33']),
 'equipment_on_departure': dict(text='Inspect observable equipment interactions before departure. Dark screen, closed laptop lid, or visible lit screen cannot independently establish equipment power or shutdown.',sources=['organizer_train_11','organizer_train_28']),
 'items_before_work': dict(text='Inspect bags/clothing and the start of equipment work. Personal ownership and whether a location is prohibited need separate evidence.',sources=['organizer_train_40']),
 'cash_before_drink': dict(text='Inspect whether the same cash remains on the work surface when drinking starts. Do not invent an allowable delay or infer that storage never happens later.',sources=['organizer_train_46']),
 'cash_workflow': dict(text='Consider unpacking/separating cash, machine counting, storage, and computer work as distinct ordered steps. Missing visible steps and invisible account records are unknown, not violations.',sources=['organizer_train_13']),
}

def general_rules_text():
    return '\n'.join(k+': '+v['text'] for k,v in RULES.items())

def parse_answer(raw):
    text=raw.strip()
    if text.startswith('```'): text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    obj=json.loads(text)
    if not isinstance(obj,dict): raise ValueError('Expected JSON object')
    return obj

def validate_observations(answer,frames,group,stage,focus):
    lookup={f['id']:f for f in frames};accepted=[];errors=[]
    for kind,key in [('event','events'),('state','states')]:
        for index,raw in enumerate(answer.get(key,[]) if isinstance(answer.get(key),list) else []):
            try:
                refs=raw['evidence_frames']
                if not isinstance(refs,list) or not refs or len(set(refs))!=len(refs) or any(f not in lookup for f in refs):
                    raise ValueError('Invalid evidence references')
                ts=sorted(lookup[f]['timestamp'] for f in refs)
                if kind=='event' and (len(ts)<2 or ts[-1]<=ts[0]): raise ValueError('Action requires multiple real frames')
                if kind=='event' and raw.get('action') not in ACTIONS: raise ValueError('Unknown action')
                if kind=='state' and raw.get('property') not in PROPERTIES: raise ValueError('Unknown property')
                if raw.get('visibility') not in ['clear','partial','unclear']: raise ValueError('Unknown visibility')
                if ts[0]<focus[0]-1e-6 or ts[-1]>focus[1]+1e-6: raise ValueError('Evidence outside supplied context')
                value=dict(observation_id=f'{group}:{stage}:{kind}:{index}',kind=kind,
                  action=raw.get('action') if kind=='event' else None,property=raw.get('property') if kind=='state' else None,
                  value=str(raw.get('value','unknown'))[:120] if kind=='state' else None,
                  actor=str(raw.get('actor','unknown'))[:40],role=str(raw.get('role','unknown'))[:40],
                  object=str(raw.get('object','unknown'))[:40],track_id=str(raw.get('track_id','unknown'))[:40],
                  recipient=str(raw.get('recipient','unknown'))[:40],visibility=raw['visibility'],
                  from_actor=str(raw.get('from_actor','unknown'))[:40],to_actor=str(raw.get('to_actor','unknown'))[:40],
                  start_sec=ts[0],end_sec=ts[-1],last_observed_sec=ts[-1],
                  description=str(raw.get('description',''))[:700],uncertainty=str(raw.get('uncertainty',''))[:500],
                  source='local_qwen_visual_proposal',authority='visual_evidence',human_confirmed=False,
                  evidence=[lookup[f] for f in refs],stage=stage,
                  identity_link_status='model_proposed_not_verified',legal_cash_owner='unknown')
                for person_field in ['actor','from_actor','to_actor']:
                    if not re.fullmatch(r'person_\d+|unknown',value[person_field]):value[person_field]='unknown'
                if value['role'] not in ['employee','customer','unknown']:value['role']='unknown'
                accepted.append(value)
            except (KeyError,TypeError,ValueError) as e: errors.append(dict(kind=kind,index=index,error=str(e)))
    return accepted,errors

def memory(observations,until):
    # Exclude future coarse observations; current state is NOT obtained by carrying last-seen state forward.
    past=sorted((o for o in observations if o['end_sec']<until),key=lambda o:(o['end_sec'],o['stage'].startswith('focus')))
    facts={};transfers=[];actions=[];links=[]
    for o in past:
        f=dict(observation_id=o['observation_id'],source=o['source'],authority=o['authority'],
               last_observed_sec=o['last_observed_sec'],visibility=o['visibility'],uncertainty=o['uncertainty'],
               actor=o['actor'],role_candidate=o['role'],object=o['object'],track_candidate=o['track_id'],
               current_state='unknown_until_reobserved',identity_link_status=o['identity_link_status'],
               human_confirmed=False,evidence_timestamps=[x['timestamp'] for x in o['evidence']])
        if o['kind']=='state':
            key=o['property']+':'+(o['track_id'] if o['track_id']!='unknown' else o['object'])
            f['property']=o['property'];f['last_known_value']=o['value']
            if key in facts:links.append(dict(before=facts[key]['observation_id'],after=o['observation_id'],relation='later_observation_of_candidate_track',verified=False))
            facts[key]=f
        else:
            f['action']=o['action'];f['recipient']=o['recipient'];actions.append(f)
            if o['action'] in ['cash_received','cash_transferred','document_transfer']:
                f['ownership']='unknown';f['claimed_custody']='visible-transfer candidate only'
                f['from_actor_candidate']=o.get('from_actor','unknown');f['to_actor_candidate']=o.get('to_actor','unknown')
                transfers.append(f)
            if o['action'] in ['person_enters','person_leaves','employee_leaves','employee_returns','employee_absent','occluded']:
                facts['presence:'+o['actor']]=dict(f,last_known_value=o['action'])
    return dict(as_of_sec=until,source='derived_only_from_prior_visual_proposals',human_confirmed=False,
      current_state_policy='Every carried state is last known, not established now. Occlusion/out-of-view leaves current state unknown.',
      ownership_policy='Role/cash tracks are fallible local hypotheses, never person identification or legal ownership.',
      last_known_facts=list(facts.values())[-24:],transfer_history=transfers[-12:],
      people_last_seen=[f for k,f in facts.items() if k.startswith('presence:') or k.startswith('person_presence:')],
      cash_customer_links=[dict(observation_id=f['observation_id'],from_actor_candidate=f['from_actor_candidate'],
        to_actor_candidate=f['to_actor_candidate'],legal_owner='unknown',status='unverified_transfer_link',
        last_observed_sec=f['last_observed_sec'],visibility=f['visibility'],source=f['source']) for f in transfers if f['object']=='cash'][-8:],
      counting_history=[f for f in actions if f['action'] in ['cash_counted_by_hand','cash_counter_used']][-8:],
      authenticity_check_history=[f for f in actions if f['action']=='banknote_detector_used'][-8:],
      unobserved_fields_mean='unknown_not_absent',recent_actions=actions[-16:],temporal_links=links[-20:])

def rule_assessments(observations):
    events=sorted([o for o in observations if o['kind']=='event'],key=lambda o:o['start_sec'])
    states=[o for o in observations if o['kind']=='state']
    def find(actions): return [o for o in events if o['action'] in actions]
    result=[]
    for rule in RULES:
        trigger=[];evidence=[];missing=[];verdict='insufficient_data';explanation='Наблюдения недостаточны для оценки правила; отсутствие события в выборке не доказывает его отсутствие.'
        if rule=='identity_before_cash':
            trigger=find(['cash_received']);checks=find(['document_checked']);transfers=find(['document_transfer'])
            for cash in trigger:
                earlier=[o for o in checks if o['end_sec']<cash['start_sec'] and o['visibility']=='clear']
                if earlier and cash['visibility']=='clear':
                    evidence=[earlier[-1],cash];verdict='no_violation_observed';explanation='Предложено видимое рассмотрение документа до приёма денег в этом эпизоде; тип документа и содержание проверки требуют подтверждения.';break
            if not evidence:evidence=(transfers+trigger+checks)[:8];missing=['надёжная идентификация удостоверения и действия проверки','полнота этапа до первого получения денег']
        elif rule=='phone_before_counting':
            trigger=find(['cash_counter_used','cash_counted_by_hand']);away=find(['phone_put_away'])
            prior=[(a,c) for a in away for c in trigger if a['end_sec']<c['start_sec'] and a['visibility']==c['visibility']=='clear']
            if prior:evidence=list(prior[0]);verdict='no_violation_observed';explanation='Есть предложенная последовательность убирания телефона до пересчёта; вывод ограничен этим эпизодом.'
            else:evidence=(away+trigger)[:8];missing=['различимый телефон','связанные события убирания телефона и начала пересчёта']
        elif rule=='authenticity_after_counting':
            trigger=find(['cash_counter_used','cash_counted_by_hand']);checks=find(['banknote_detector_used'])
            ordered=[(c,a) for c in trigger for a in checks if c['end_sec']<a['start_sec'] and c['visibility']==a['visibility']=='clear']
            if ordered:evidence=list(ordered[0]);verdict='no_violation_observed';explanation='Предложено использование аппарата проверки после пересчёта. Подлинность самих банкнот не оценивалась.'
            else:evidence=(trigger+checks)[:8];missing=['подтверждённые границы всей операции','полная видимость этапа проверки без перекрытия','проверенное человеком отсутствие проверки','назначение и встроенные функции аппарата']
        elif rule in ['safe_on_departure','valuables_on_departure']:
            trigger=find(['employee_leaves']);absent=find(['employee_absent'])
            properties=['safe_door'] if rule=='safe_on_departure' else ['cash_location','key_location','stamp_location','document_location']
            possible=[s for s in states if s['property'] in properties and s['visibility']=='clear']
            for leave in trigger:
                before=[s for s in possible if s['end_sec']<=leave['start_sec']]
                after=[s for s in possible if leave['end_sec']<=s['start_sec']<=leave['end_sec']+10]
                departed=[a for a in absent if leave['end_sec']<=a['start_sec']<=leave['end_sec']+10 and a['visibility']=='clear' and a['actor']==leave['actor']]
                for a in before:
                    for b in after:
                        same=a['property']==b['property'] and a['track_id']==b['track_id']!='unknown'
                        unsafe=(a['value']==b['value']=='open') if rule=='safe_on_departure' else (a['value']==b['value'] and a['value'] in ['desk','work_surface','safe_top','in_lock'])
                        intervening=[e for e in events if e['action'] in ['occluded','employee_returns','safe_closing','cash_stored','key_handling','stamp_handling','document_transfer'] and a['end_sec']<e['end_sec']<b['start_sec']]
                        if same and unsafe and departed and leave['visibility']=='clear' and not intervening:
                            evidence=[a,leave,departed[0],b];verdict='violation_supported';explanation='Положительные визуальные предложения показывают состояние до и после ухода. Это гипотеза по правилу сценария, не подтверждённое нарушение и не правовая квалификация.'
            if not evidence:evidence=(trigger+possible+absent)[:10];missing=['однозначная роль уходящего сотрудника','связанные различимые состояния того же предмета до и после ухода','положительное свидетельство отсутствия после ухода','отсутствие разрыва наблюдаемой связи предмета']
        elif rule=='previous_cash_before_next_customer':
            trigger=find(['person_enters']);evidence=(find(['cash_received','cash_stored'])+[s for s in states if s['property'] in ['cash_location','cash_holder']]+trigger)[-12:]
            missing=['подтверждённая смена клиентов','надёжная непрерывная связь конкретной пачки с предыдущей операцией','положительная видимость этой пачки на столе после прихода следующего клиента']
        elif rule=='door_after_entry':
            trigger=find(['person_enters','employee_returns']);evidence=(trigger+find(['door_closing'])+[s for s in states if s['property']=='room_door'])[-10:]
            missing=['какая дверь и её положение после входа','границы оцениваемого эпизода','допустимый срок закрытия не задан']
        elif rule=='equipment_on_departure':
            trigger=find(['employee_leaves']);evidence=(trigger+find(['equipment_operation','computer_work'])+[s for s in states if s['property']=='computer_screen'])[-10:]
            missing=['прямое доказательство состояния питания; экран и крышка недостаточны']
        elif rule=='items_before_work':
            trigger=find(['computer_work','equipment_operation']);evidence=(trigger+[s for s in states if s['property'] in ['clothing_location','bag_location']])[-8:]
            missing=['принадлежность вещей','применимость требования к этому месту и этапу']
        elif rule=='cash_before_drink':
            trigger=find(['drink_used']);evidence=(find(['cash_counter_used','cash_counted_by_hand','cash_stored'])+[s for s in states if s['property']=='cash_location']+trigger)[-12:]
            missing=['связь тех же денег между пересчётом и питьём','положительное одновременное наблюдение денег и питья','допустимое время хранения не определено']
        elif rule=='cash_workflow':
            trigger=find(['cash_unpacking','cash_bundle_split']);evidence=(trigger+find(['cash_counter_used','cash_stored','computer_work']))[-12:]
            missing=['временная связь всех этапов','содержание отчёта невидимо','конкретное нарушение этой ожидаемой последовательности не задано']
        result.append(dict(rule_id=rule,verdict=verdict,triggering_event=[e['observation_id'] for e in trigger],
          evidence=[dict(observation_id=e['observation_id'],start_sec=e['start_sec'],end_sec=e['end_sec'],
                         observation=e['description'],visibility=e['visibility'],source=e['source'],frames=e['evidence']) for e in evidence],
          missing_evidence=missing,explanation=explanation,review_status='proposed',human_confirmed=False,
          whole_video_normal=False,legal_violation_target=None,source='evidence_gated_scenario_rule',
          rule_provenance=RULES[rule]['sources']))
    return result
