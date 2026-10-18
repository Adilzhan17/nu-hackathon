"""Build an offline report and a compact review queue; no server changes or uploads."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from ml.autolabel.dataset import read_jsonl, digest
from .prepare import write_json

def esc(value): return html.escape(str(value))

def build(root,data,features,experiment):
    audit=json.loads((data/'audit_summary.json').read_text())
    report=json.loads((experiment/'report.json').read_text())
    extraction=json.loads((features/'extraction.json').read_text())
    pilot=json.loads((features/'pilot_extraction.json').read_text()) if (features/'pilot_extraction.json').is_file() else None
    tokens=json.loads((features/'text_token_audit.json').read_text())
    queue=read_jsonl(data/'review_queue.jsonl')
    bags=read_jsonl(data/'bags.jsonl')
    train_count=sum(b['split']=='train' for b in bags);diagnostic_count=sum(b['split']=='val' for b in bags)
    def rel(p): return quote(os.path.relpath(p,experiment),safe='/')
    table=''
    for scope,title in [('train_oof',f'Внешние OOF folds внутри train, {train_count} групп'),('reused_val_diagnostic',f'Повторно использованный val, {diagnostic_count} групп — не независимый test')]:
        for method,m in report[scope].items():
            table+=f'<tr><td>{esc(title)}</td><td>{esc(method)}</td><td>{m["accuracy"]:.3f}</td><td>{m["balanced_accuracy"]:.3f}</td><td>{m["f1"]:.3f}</td><td>{m["auroc"]:.3f}</td><td>{m["log_loss"]:.3f}</td><td>{esc(m["confusion_matrix"])}</td></tr>'
    delta=report['paired_accuracy_comparison'];delta_text=f'{delta["estimate"]:+.3f}'
    report['conclusion']=('No improvement in grouped OOF accuracy.' if delta['estimate']<=0 else
                         'Higher grouped OOF accuracy in this pilot; not independent evidence of generalization.')+' No production replacement.'
    report['constant_positive_accuracy_reference']=sum(b['video_target'] for b in bags if b['split']=='train')/max(1,train_count)
    write_json(experiment/'report.json',report)
    cards=''
    for r in queue:
        video=rel(root/'data/raw/open'/r['video'])
        source=rel(root/r['source'])
        cards+=f'''<details><summary>{esc(r['pair_id'])} · {esc(r['video'])} · {r['start_sec']:g}–{r['end_sec']:g} с · {esc(r['action_cardinality'])}</summary>
        <p><b>Исходное слабое описание:</b> {esc(r['caption'])}</p>
        <p>Контекст {r['context_start_sec']:g}–{r['context_end_sec']:g} с; <a href="{source}#L{r['source_lines'][0]}">источник, строки {esc(r['source_lines'])}</a>.
        Действия-кандидаты: {esc(', '.join(r['action_candidates']) or 'неясно')}. Предметы-кандидаты: {esc(', '.join(r['object_candidates']) or 'не определены')}. Видимость не подтверждена.</p>
        <video controls preload="none" muted src="{video}#t={r['context_start_sec']},{r['context_end_sec']}"></video>
        <ol>{''.join('<li>'+esc(q)+'</li>' for q in r['human_questions'])}</ol>
        <p>Ответы следует сохранять через существующий human review; эта страница не меняет разметку.</p></details>'''
    human_exports=[]
    for metadata in sorted((root/'data/human_review/exports').glob('*/metadata.json')):
        d=json.loads(metadata.read_text())
        human_exports.append(dict(path=str(metadata.relative_to(root)),sha256=digest(metadata),
           confirmed_actions=d.get('training_actions',d.get('count')),schema_version=d.get('schema_version'),taxonomy_version=d.get('taxonomy_version')))
    write_json(experiment/'human_source_status.json',dict(import_enabled=False,schema_agreement='pending',
      observed_exports=human_exports,confirmed_actions_consumed=0,contract='ml/narration_training/human_contract.proposed.json'))
    final=report['final_data_usage']['narration']
    resource_text=(f'Отдельный GPU-пилот: {pilot["visual_clips"]} интервала, {pilot["elapsed_seconds"]:.1f} с. ' if pilot else 'Отдельный GPU-пилот для этого кеша не зафиксирован. ')
    epochs=report['protocol']['epochs']
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>ForteBank: обучение с описаниями</title>
    <style>body{{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#172432;background:#fafbfc}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd4dd;padding:8px;text-align:left}}details{{background:white;border:1px solid #ccd4dd;padding:12px;margin:12px 0}}summary{{cursor:pointer;font-weight:600}}video{{width:100%;max-height:480px;background:#111}}code{{background:#e9edf2;padding:2px 5px}}.notice{{background:#fff1cf;padding:16px}}a{{color:#1754ac}}</style>
    <h1>Обучение на видео и пользовательских описаниях</h1>
    <p class="notice">Экспериментальная модель. Production не заменена. Слабые описания, предложения действий и предполагаемые нарушения не являются подтверждённой разметкой.</p>
    <h2>Что фактически использовано</h2>
    <p>Проверены все {audit['input_pairs']} пар из {audit['pair_groups']} train-групп: видео, SHA-256, group ID, исходное наблюдение, текст, строки расшифровки, привязка раздела и время. Ошибок происхождения: {audit['excluded_pairs']}. Все {audit['approximate']} границы приблизительны. Три карантинных наблюдения исключены ранее и в эти 99 не входят.</p>
    <p>По исходным кодам и словам описания: {audit['action_cardinality']['single_candidate']} кандидатов на одно действие, {audit['action_cardinality']['multiple_candidates']} составных, {audit['action_cardinality']['possibly_multiple']} потенциально составных, {audit['action_cardinality']['unclear']} неясных. Это автоматический разбор текста, не проверка кадров человеком. Неоднозначность отмечена у {audit['ambiguous']} пар.</p>
    <p>Финальное обучение: {final['video_bags']} целых train-видео, {final['unique_text_pairs']} уникальных слабых пар; {final['pair_epoch_exposures']} предъявлений за {final['epochs']} эпох. Исключений из текстовой задачи: {sum(final['excluded_pairs'].values())}. Вспомогательная задача действий использовала {final['action_positive_pairs']} пар; четыре неоднозначные пары участвуют только в текстовой задаче с весом 0.25. Подтверждённых действий использовано: 0.</p>
    <h2>Метод и протокол</h2>
    <p>Замороженный мультиязычный <a href="https://huggingface.co/google/siglip2-so400m-patch14-384">SigLIP 2 So400m</a>, лицензия Apache-2.0, {extraction['parameters']:,} параметров, revision <code>{esc(extraction['model_source']['revision'])}</code>. Восемь упорядоченных кадров на интервал. Небольшой временной адаптер использует среднее, направление изменения во времени и разницу последнего/первого кадров.</p>
    <p>Baseline и narration используют одинаковые видео, архитектуру, начальные seed, {epochs} эпох и бюджет подбора. В narration добавлены косинусное сопоставление с исходным русским текстом и положительные действия из описания; неназванные действия не становятся отрицательными. Бинарная цель применяется только к объединению окон целого исходного видео (MIL). Связь градиентов с текстом проверена тестом подмены текста.</p>
    <p>Четыре внешних групповых fold; внутри каждого — три групповых fold для выбора learning rate из 0.001 и 0.003 по log loss. Внутренняя выборка не содержит даже описаний внешних групп. Порог 0.5 фиксирован заранее. Все версии одного исходника остаются в одной группе. Финальные параметры выбраны внутри {train_count} train-групп до обращения к {diagnostic_count} диагностическим группам. Версия разбиения: {esc(audit.get('split_version','original_32_8'))}.</p>
    <h2>Сравнение</h2><p class="notice">{'Улучшения accuracy нет. Этот эксперимент не даёт оснований заменять основной классификатор.' if delta['estimate']<=0 else 'В этом пилоте accuracy выросла, но независимого подтверждения улучшения пока нет.'}</p><table><tr><th>Протокол</th><th>Метод</th><th>Accuracy</th><th>Balanced acc.</th><th>F1</th><th>AUROC</th><th>Log loss</th><th>[[TN,FP],[FN,TP]]</th></tr>{table}</table>
    <p>Изменение OOF accuracy: {delta_text}. Описательный парный bootstrap-интервал: {esc(delta['descriptive_95_percentile_interval'])}. Он не учитывает неопределённость обучения и предыдущие эксперименты на этом датасете; это не доказательство улучшения.</p>
    <p>Для масштаба: постоянный положительный ответ даёт на этих train-группах accuracy {report['constant_positive_accuracy_reference']:.3f}. Итоговая baseline training BCE: {report['final_data_usage']['baseline']['trace'][-1]['video_loss']:.6f}. Для оценки обобщения сравнивайте OOF-метрики двух методов в таблице; низкая ошибка на train сама по себе не доказывает качество.</p>
    <p>Метрики типов и границ событий не рассчитаны: согласованного набора подтверждённых событий нет. Все {diagnostic_count} диагностических видео входили в ранее использованный val и не являются независимым test. Оценки модели не откалиброваны.</p>
    <h2>Ресурсы и ограничения</h2><p>{resource_text}Полное извлечение: {extraction['visual_clips']} интервалов в кеше, {extraction['elapsed_seconds']:.1f} с. GPU-работа ограничена timeout и общим flock. Обучение адаптера выполнено на CPU. Кадры, аудио и описания не отправлялись внешним AI API. Загружены локальные веса; все признаки рассчитаны локально на выделенном сервере.</p>
    <p>Текстовый предел энкодера: {tokens['max_tokens']} токенов; усечённых описаний: {tokens['truncated_pairs']}. Модель исходно кодирует изображения: временной адаптер очень мал и не заменяет подтверждённые границы событий. Мелкие предметы могут быть неразличимы. Возможны смещения камеры и групп; 99 коррелированных пар из 13 групп — маленькая выборка. Отсутствие предмета или проверки в описании не означает нарушения.</p>
    <p>Human review: подготовлен отдельный предложенный контракт, импорт выключен до согласования. Существующий экспорт содержит 0 подтверждённых действий. Ни reject, ни needs_context, ни normal_action не преобразуются в бинарный вердикт окна или видео.</p>
    <h2>Артефакты</h2><ul>
    <li><a href="{rel(data/'pairs.jsonl')}">Все 99 аудированных пар и вопросы</a>, <a href="{rel(data/'quarantined_sources.jsonl')}">карантин</a></li>
    <li><a href="report.json">Машиночитаемые метрики</a>, <a href="data_usage.jsonl">фактическое использование данных во всех fit</a>, <a href="selection.json">подбор только внутри train</a></li>
    <li><a href="baseline.pt">Baseline checkpoint</a>, <a href="narration.pt">Narration checkpoint</a>, <a href="verification.json">проверка сохранности и утечек</a></li>
    <li><a href="action_proposals_train.jsonl">Кандидаты действий на train — не GT и не метрики качества</a></li></ul>
    <h2>Короткая очередь вопросов: {len(queue)} интервалов</h2><p>Видео открываются только с локального диска; кодек браузера может потребовать существующий preview. Все вопросы для остальных пар сохранены в pairs.jsonl. Точные границы и различимость объектов необходимо проверить вручную.</p>{cards}
    </html>'''
    (experiment/'index.html').write_text(text)
    print(dict(report=str(experiment/'index.html'),questions=len(queue),human_actions_consumed=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();build(a.root.resolve(),a.dataset.resolve(),a.features.resolve(),a.experiment.resolve())
