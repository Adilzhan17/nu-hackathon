РАЗВЁРТЫВАНИЕ И СОВМЕСТИМОСТЬ
============================

Шаблоны в systemd/ предназначены для новой установки /opt/forte-vision.
Не включать их рядом с уже запущенными nuhack-* на тех же портах.
Существующие имена сервисов и публичные URL не переименовываются автоматически.

Роли:
  forte-vision-analysis  — Python coordinator, loopback8771
  forte-vision-api       — отдельный API жюри, loopback8765
  web                   — Next.js с теми же proxy-маршрутами
  inference worker      — запускается coordinator на собственном GPU, не daemon

В текущем развёртывании nuhack-annotator уже обслуживает API жюри вместе
с разметкой. Не запускать второй jury API на8765. Каноническая команда
python -m forte_vision.interfaces.review_http сохраняет этот режим.

Конфигурация GPU централизована в forte_vision/config.py. Новая установка
использует FORTE_GPU_HOST, FORTE_GPU_PORT, FORTE_GPU_KEY_FILE, FORTE_GPU_ROOT.
Для существующего сервера пока сохранены прежние значения по умолчанию.
Проверка значений выполняется до запуска SSH; содержимое ключей не читается
конфигурационным модулем. Пример переменных — .env.example.

Исходный checkout требуется целиком: frontend/analysis/index.html,
forte_vision/ и legacy-модули API. Это не автономный Python-wheel.
Минимальные CPU-зависимости Gemma coordinator — requirements-coordinator.txt.
Дополнительные зависимости старых X3D/исследований — ml/requirements-runtime.txt;
GPU-runtime и веса подготавливаются существующим bootstrap, не скачиваются
при обработке видео. Node зависимости — frontend/package-lock.json.

Порядок выкладки рефакторинга:
1. Проверить отсутствие активного анализа/обучения, не прерывать чужой GPU job.
2. Выполнить make check и make check-web в проверяемой версии.
3. Скопировать forte_vision/ на GPU-хост до перезапуска coordinator: новый
   транспорт запускает python -m forte_vision.inference.gemma.
4. Убедиться, что новая команда и старая ml.gemma_lab.runner дают --help.
5. Обновить main-host исходники и перезапустить существующий analysis-service
   в спокойном состоянии. Новый HTML берётся из frontend/analysis/index.html.
6. Проверить публичный экран, auth API и полный контрольный анализ.

Не переносить/не стирать runs/, artifacts/, data/, .secrets/ или GPU lock.
Не запускать тренировку ради проверки структуры. Не создавать новый платный GPU.

Откат:
Исходная версия сохранена в release/forte-vision-mvp (a113284). В согласованное
окно вернуть её код на main-host и прежние точки запуска, затем перезапустить
analysis-service. Вернуть старый runner на GPU только если его исходники менялись.
Наличие нового forte_vision/ на GPU не мешает старому runner, когда старые
ml/gemma_lab файлы сохранены. Не использовать git reset --hard и не менять
сохранённые результаты. Рабочие шаблоны systemd сами по себе ничего не включают.
