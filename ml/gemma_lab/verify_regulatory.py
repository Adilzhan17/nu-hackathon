"""Retrieve official legal sources; save bounded paragraphs and content hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request

DOCUMENTS = {
    'V1900019680': ('c4cd4052-63e3-475b-ba9e-93f23de13fdb', {12, 19, 20, 22, 76, 80}),
    'V2000020075': ('4bd1c354-b40c-4b38-ab7d-53aaa7b4f39c', {16}),
    'V2600038336': ('315d5902-73a2-45bf-a03d-05bdb8e1db76', {53, 71}),
}


def verify():
    sources = []
    for code, (uid, sections) in DOCUMENTS.items():
        api_url = 'https://adilet.zan.kz/api/documents/' + uid
        with urllib.request.urlopen(api_url, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024)
        document = json.loads(raw)['document']
        if document.get('is_actual') is not True:
            raise ValueError('Source is not current: ' + code)
        content = document['text_content']
        found, active = {}, None
        for attrs, paragraph in re.findall(r'<p\b([^>]*)>(.*?)</p>', content, re.S):
            text = ' '.join(html.unescape(re.sub('<[^>]+>', ' ', paragraph)).split())
            number = re.match(r'(\d+)\.', text)
            if number:
                active = int(number[1])
                if active in found:
                    active = None  # Do not mix appendix numbering into the main rules.
                elif active in sections:
                    anchor = re.search(r'id="([^"]+)"', attrs)
                    found[active] = dict(anchor=anchor[1] if anchor else None, text=[])
            if active in sections:
                found[active]['text'].append(text)
        if set(found) != sections:
            raise ValueError('Missing official paragraphs: ' + code)
        sources.append(dict(code=code, url='https://adilet.zan.kz/rus/docs/' + code,
                            official_api_url=api_url, is_actual=True, updated_at=document.get('updated_at'),
                            response_sha256=hashlib.sha256(raw).hexdigest(),
                            text_sha256=hashlib.sha256(content.encode()).hexdigest(), paragraphs=found))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), sources=sources,
                scope='Source text verified; case applicability still requires visible facts and bank rules.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Verified official sources:', ', '.join(s['code'] for s in result['sources']))
