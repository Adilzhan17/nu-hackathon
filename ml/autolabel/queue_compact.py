"""Deduplicate highly overlapping review tasks without creating long review intervals."""
def compact(tasks,max_seconds=12.):
    result=[]
    for task in sorted(tasks,key=lambda t:(-t['priority'],t['video'],t['start_sec'])):
        match=None
        for other in result:
            if other['group_id']!=task['group_id']:continue
            overlap=max(0,min(task['end_sec'],other['end_sec'])-max(task['start_sec'],other['start_sec']))
            shorter=min(task['end_sec']-task['start_sec'],other['end_sec']-other['start_sec'])
            union=max(task['end_sec'],other['end_sec'])-min(task['start_sec'],other['start_sec'])
            if shorter>0 and overlap/shorter>=.7 and union<=max_seconds:
                match=other;break
        if match is None:
            result.append(dict(task,observations=list(task['observations']),merged_task_ids=[task['review_id']]))
        else:
            match['start_sec']=min(match['start_sec'],task['start_sec']);match['end_sec']=max(match['end_sec'],task['end_sec'])
            match['context_start_sec']=min(match['context_start_sec'],task['context_start_sec']);match['context_end_sec']=max(match['context_end_sec'],task['context_end_sec'])
            for field in ('proposal_ids','conflict_ids','reasons'):match[field]=sorted(set(match[field]+task[field]))
            obs={o['observation_id']:o for o in match['observations']+task['observations']};match['observations']=list(obs.values())
            match['merged_task_ids'].append(task['review_id'])
    return result
