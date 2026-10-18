"""Publish an escaped, factual experiment status in the independent Gemma page.

No service restart, production checkpoint changes, or private data publication.
Only whitelisted aggregate values are rendered, never sample text or file paths.
"""
import argparse
from html import escape
import json
from pathlib import Path
import re

START = '<!-- GEMMA_ADAPTATION_STATUS_START -->'
END = '<!-- GEMMA_ADAPTATION_STATUS_END -->'


def table(metrics):
    rows = ''.join('<tr>' + ''.join('<td style="padding:6px 12px">' + escape(str(row.get(k, '—'))) + '</td>'
                                  for k in ('model', 'videos', 'answered', 'accuracy', 'correct_all')) + '</tr>'
                   for row in metrics)
    return ('<div style="overflow-x:auto"><table><thead><tr><th>Модель</th><th>Видео</th>'
             '<th>Дала ответ</th><th>Accuracy среди ответов</th><th>Верно / все видео</th></tr></thead>'
             '<tbody>' + rows + '</tbody></table></div>') if rows else ''


def render(status):
    title = escape(str(status.get('title', 'Дообучение Gemma · эксперимент')))
    message = escape(str(status['message']))
    main_table = table(status.get('metrics', []))
    details = ''.join('<p>' + escape(str(value)) + '</p>' for value in status.get('details', []))
    additional = status.get('additional_protocol')
    extra = ''
    if additional:
        extra = ('<section class="card" aria-label="Дополнительный протокол"><h3>'
                 + escape(str(additional.get('title', 'Дополнительная проверка'))) + '</h3><p>'
                 + escape(str(additional['message'])) + '</p>' + table(additional.get('metrics', []))
                 + ''.join('<p>' + escape(str(value)) + '</p>' for value in additional.get('details', [])) + '</section>')
    return (START + '\n<details class="secondary" id="adaptation"><summary>' + title + '</summary>'
            '<div class="secondary-body"><p role="status">' + message + '</p>' + main_table + details
            + extra + '</div></details>\n' + END)


def publish(page, status):
    original = page.read_text()
    block = render(status)
    if START in original:
        updated = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: block, original, flags=re.S)
    else:
        anchor = '<section id="workspace" hidden>'
        if original.count(anchor) != 1:
            raise ValueError('Expected exactly one Gemma workspace anchor')
        updated = original.replace(anchor, block + '\n' + anchor)
    temporary = page.with_suffix('.html.tmp')
    temporary.write_text(updated)
    temporary.replace(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--page', type=Path, default=Path('ml/gemma_lab/index.html'))
    args = parser.parse_args()
    publish(args.page, json.loads(args.status.read_text()))


if __name__ == '__main__':
    main()
