"""Publish links to cached proposals in the live object editor; no model calls."""
import argparse
import html
import json
from pathlib import Path
import re

BEGIN = '<!-- TERRA_REVIEW_PILOT_BEGIN -->'
END = '<!-- TERRA_REVIEW_PILOT_END -->'


def publish(results, page):
    rows = json.loads(results.read_text())
    links = []
    for row in rows:
        if row['status'] != 'completed':
            continue
        url = row['review_url']
        if not url.startswith('/label/objects?image_id='):
            raise ValueError('Unexpected review URL')
        title = f"{row['video']} · {row['timestamp_sec']:.2f} с · рамок: {row['boxes']}"
        links.append('<li><a href="'+html.escape(url, quote=True)+'">'+html.escape(title)+'</a></li>')
    section = BEGIN + '\n<section class="card" id="terra-review-pilot"><h2>Terra · спорные кадры</h2>'
    section += '<p>Готовые предложения координат. Открытие ссылок не запускает модель. Пунктирные рамки требуют проверки; ручные решения сохранены отдельно.</p><ul>'
    section += ''.join(links) + '</ul><small>Только помощник разметки. Анализ загруженных видео использует локальные модели. Один кадр не подтверждает действие или нарушение.</small></section>\n' + END
    original = page.read_text()
    if BEGIN in original:
        updated = re.sub(re.escape(BEGIN)+'.*?'+re.escape(END), lambda _: section, original, flags=re.S)
    else:
        anchor = '<section id="suggestions"'
        if original.count(anchor) != 1:
            raise ValueError('Object editor layout changed; refusing replacement')
        updated = original.replace(anchor, section+'\n'+anchor)
    temporary = page.with_suffix('.terra.tmp')
    temporary.write_text(updated)
    temporary.replace(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--page', type=Path, default=Path('ml/human_objects.html'))
    args = parser.parse_args()
    publish(args.results, args.page)
