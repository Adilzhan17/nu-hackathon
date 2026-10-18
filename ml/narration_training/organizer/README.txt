Контекстная разметка ForteBank: перенос и воспроизводимое продолжение

Источники остаются неизменными. Эта ветка использует исходный manifest.csv
(32 train / 8 val групп), зафиксированный в organizer_v1/blind/plan.json.
Другие версии разделения и активные настройки обучения автоматически не принимаются.
Организаторы раскрыли 5 исходных val-групп: №4 (также дубль №47), 20, 24, 43, 44.
Эти пояснения не являются независимым тестом и не входят в обучающий экспорт.

Слои:
  source/organizer_claims_v1.jsonl — 22 исходных пояснения organizer_provided,
      дословный текст, исходная строка, SHA256, подтверждение нумерации «да».
  number_mapping.* — соответствие реальным файлам, SHA256, group_id, split.
  blind/video_*/survey.json — независимый обзор всей длительности, 1 кадр/с.
  blind/video_*/focus_*.json — переходы, 4 кадра/с, контекст до/после, память.
  blind/video_*/memory_trace.jsonl — только предшествующие наблюдения;
      last known не становится current. Роли и связи денег — гипотезы.
  blind/video_*/rule_recheck.json — отдельный проход общих правил после наблюдений.
  report/scenarios.jsonl — пять слоёв, включая отдельное сравнение с организаторами
      и снимок подтверждений человека. Все автонаблюдения остаются proposed.
  report/train_export.jsonl — 17 слабых сценариев целого train-видео; это НЕ
      интервальные положительные метки и НЕ подтверждённые нарушения.
  report/train_human_actions.jsonl — отдельно подтверждённые train-действия.

Модель: локальная Qwen/Qwen3-VL-2B-Instruct, 2 127 532 032 параметра,
revision 89644892e4d85e24eaac8bacfd4f463576704203, Apache-2.0.
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
Сетевые AI API не используются. Веса проверяются по SHA256.
Аудио не декодируется. Личности не распознаются. Юридические статьи не выводятся.
Рамки объектов здесь не создаются: текстовое наблюдение не является детекцией.

Новый GPU: outdoor-orange-whippet, T4 16 GB, SSH порт 15171.
Старый GPU не используется. Не запускать старый ml/migration/sync_t4_progress.sh:
он рассчитан на оба сервера и больше не соответствует условиям переноса.
Ключ /root/.ssh/nuhack_brev используется только SSH; не читать и не копировать его.
Общий GPU-lock /home/ubuntu/nuhack-training/.gpu-work.lock обязателен.

1. Проверить существующий процесс и журнал:
ssh -i /root/.ssh/nuhack_brev -p 15171 -o BatchMode=yes ubuntu@global.prd.ga.run.brev.nvidia.com \
 'cd /home/ubuntu/nuhack-training && ps -eo pid,etime,args | grep -E "[p]ython.*(t4_organizer|organizer.resume)" && tail -10 runs/narration_training/organizer_t4/resume.log'

2. Во время работы копировать только завершённые цепочки в НОВЫЙ снимок:
.venv/bin/python -m ml.narration_training.organizer.snapshot_t4 \
 --output runs/narration_training/t4_snapshots/UNIQUE_NAME
После выхода процесса добавить --include-partial. Скрипт сверяет хеши до и после
передачи. Другой run задаётся через --run runs/narration_training/.../blind.
Никакие файлы на T4 этим скриптом не изменяются.

3. Сверить / объединить после остановки прежнего процесса:
PYTHONPATH=runs/narration_training/organizer_audit_env .venv/bin/python \
 -m ml.narration_training.organizer.reconcile \
 --source runs/narration_training/organizer_v1 \
 --input snapshot=backups/migration_20261017/snapshot_20261017T215328Z/organizer_v1/blind \
 --input t4=runs/narration_training/t4_snapshots/FINAL/runs/narration_training/organizer_t4/blind \
 --output runs/narration_training/organizer_reconciled_v1

Для только проверки добавить --audit-only и использовать новый output.
Проверяется точное воспроизведение JPEG/PTS из исходников (PyAV 16.0.1,
Pillow 12.3.0); PyAV 19 генерирует иные JPEG, поэтому для аудита установлен
изолированный av==16.0.1 под runs/narration_training/organizer_audit_env.
Не ослаблять сравнение SHA256 ради несовпадающего декодера.

Выбирается целая проверенная причинная цепочка; при равенстве — порядок --input.
Согласие с организатором / ручной меткой НЕ является критерием выбора.
Частично завершённая цепочка сохраняет только проверенный последовательный префикс.
Остальное остаётся в alternatives/, различия — в response_variants.jsonl.
Код, источник, хеши ответов и точных входов — в chain_audit.jsonl,
source_integrity.json, response_provenance.jsonl, reconciled_origin.json.
Runtime T4 на уровне run НЕ означает, что старые кэшированные ответы были FP16.
BF16 и FP16 могут дать разные ответы; численная эквивалентность не заявляется.

4. Перенести НОВУЮ согласованную версию и новые модули на T4, затем запускать
только после завершения прежнего процесса, из /home/ubuntu/nuhack-training:
env OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 timeout 1500 .venv-scenario/bin/python -u -m ml.narration_training.organizer.resume \
 --run runs/narration_training/organizer_reconciled_v1/blind \
 --model artifacts/assets/scenario_qwen3 \
 --lock /home/ubuntu/nuhack-training/.gpu-work.lock --max-seconds 1440

Модуль resume не меняет analyze.py или t4_organizer.py. Он применяет строго
проверенную FP16-поправку в памяти, сверяет весь промпт, историю, кадры и revision
перед использованием кэша и пишет runtime_ref для новых ответов.
Код 75 при достижении лимита означает сохранённое продолжение, а не успех всей
задачи. Прежде чем повторять команду, проверить выход прежнего процесса.
Не оставлять бессрочный daemon или бесконечный цикл повторов.

5. Отчёт на основном сервере после копирования результатов и аудита:
.venv/bin/python -m ml.narration_training.organizer.finalize \
 --run runs/narration_training/organizer_reconciled_v1 \
 --output runs/narration_training/organizer_reconciled_v1/report
.venv/bin/python -m ml.narration_training.organizer.verify \
 --run runs/narration_training/organizer_reconciled_v1 \
 --report runs/narration_training/organizer_reconciled_v1/report

finalize сохраняет исходный слой правил и добавляет проверку связи участников,
денег и операций. Неподтверждённая связь, неопределённый предмет или смена клиента
оставляют insufficient_data. Это проверка предложений, не получение новой GT.
Отдельные worked_scenarios.jsonl и continuity_guard_reviews.jsonl сохраняют
таймкоды, реальные кадры, пропуски и основания понижения вердикта.

6. Проверки:
.venv/bin/python -m unittest ml.narration_training.organizer.test_organizer \
 ml.narration_training.organizer.test_reconcile ml.narration_training.organizer.test_continuity -q

Потребитель слабых сценариев:
from ml.narration_training.organizer.dataset import scenario_loader
for batch in scenario_loader('/root/nuhack', '.../report/train_export.jsonl'):
    # video_path + scenario_text + expected_sequence + trust;
    # никаких binary_window_target / event_targets из пояснения.
    pass

context_ablation.json сравнивает одинаковые кадры с историей и без неё на четырёх
train-видео. Число восстановленных ручных записей — диагностическое покрытие
неполной разметки, НЕ точность, НЕ полный recall, НЕ независимая test-метрика.
Непроверенные дополнительные предложения не считаются ложными срабатываниями.
Повторные наблюдения одного действия в survey/focus не объявляются уникальными
событиями. Полный временной обзор 1 кадр/с не означает просмотр каждого сырого
кадра. coarse_only_intervals перечисляет интервалы без детального прохода.

Никакой сервер, интерфейс разметки или production-модель этим кодом не меняются.
