"""Materialize manually normalized observations, preserving uncertain transcript fragments."""
import csv
import hashlib
import json
import re
from pathlib import Path


def convert(root):
    destination = root / 'data/transcripts'
    plan = json.loads((destination / 'audio_event_plan.json').read_text())
    raw = (root / 'audio.txt').read_text()
    lines = raw.splitlines()
    with (root / 'data/review/event_review.csv').open(encoding='utf-8-sig') as f:
        videos = {int(re.search(r'(\d+)\.mov$', r['video'])[1]): r for r in csv.DictReader(f)}
    sections = [('numbered_20',20,1),('numbered_21',21,2),('numbered_22',22,3),
                ('numbered_23',23,4),('numbered_24',24,5),('numbered_25',25,6),
                ('numbered_26',26,7),('numbered_27',27,8),('numbered_48',48,9),
                ('numbered_47',47,11),('numbered_46',46,13),('audio_00_01',11,15),
                ('audio_01_07',10,28),('audio_02_01',3,39),('audio_03_48',8,52),
                ('audio_05_46',2,67),('audio_07_29',5,82),('audio_11_46',1,120),('audio_14_02',4,140)]
    events, sources = [], []
    for i, (key, number, startline) in enumerate(sections):
        endline = sections[i+1][2]-1 if i+1 < len(sections) else len(lines)
        video = videos[number]
        sources.append(dict(section=key, video=video['video'], start_line=startline, end_line=endline,
                            text='\n'.join(lines[startline-1:endline]),
                            mapping='user_confirmed' if number in (1,21) else 'explicit_heading'))
        for index, (start, end, action, observation) in enumerate(plan['blocks'].get(key, [])):
            quarantine = key == 'audio_07_29' and index >= len(plan['blocks'][key])-3
            events.append(dict(video=video['video'], group_id=video['group_id'], section=key,
                               start_sec=start, end_sec=end, action=action, observation=observation,
                               timing_precision='approximate', violation_label=None,
                               source_verdict=video['source_verdict'],
                               status='quarantined' if quarantine else 'transcribed',
                               reason='Неясно, относятся ли последние действия к Х-5 или Х-4/47.' if quarantine else '',
                               source='audio.txt', source_lines=[startline,endline]))
    payload = dict(source_sha256=hashlib.sha256((root/'audio.txt').read_bytes()).hexdigest(),
                   sections=sources, notes=plan['notes'],
                   mapping_confirmations={'numbered_21':21,'audio_11_46':1},
                   unresolved=['Последние три действия Х-5: ответ «да» не выбирает между двумя видео.'])
    payload['notes']['numbered_21'] = ['Сопоставление с Х-21 подтверждено пользователем.']
    payload['notes']['audio_11_46'] = ['Сопоставление с X-1 подтверждено пользователем. Таймкоды заголовка относятся к аудиозаписи.']
    (destination/'sources.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (destination/'observations.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in events))
    summary = dict(narrated_videos=len({e['video'] for e in events}), observations=len(events),
                   quarantined=sum(e['status']=='quarantined' for e in events),
                   note='Описания действий не преобразованы автоматически в юридические типы нарушений.')
    (destination/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    print(json.dumps(convert(Path(__file__).resolve().parent.parent), ensure_ascii=False, indent=2))
