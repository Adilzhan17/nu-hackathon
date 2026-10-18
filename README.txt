FORTE VISION
============
Модульный монорепозиторий с выделенным GPU-worker для анализа видео.

Точка входа в устройство проекта: docs/ARCHITECTURE.txt.
Назначение веток и результаты их проверки: docs/BRANCHES.txt.
Развёртывание и совместимость: deploy/README.txt.
Контракт API жюри: contracts/jury.openapi.json.

Основные каталоги:
  frontend/       веб-интерфейс, Next.js proxy, экран анализа в analysis/
  forte_vision/   рабочее Python-ядро и сервис анализа Gemma
  contracts/      внешний HTTP/JSON-контракт
  deploy/         примеры конфигурации и systemd, без секретов
  tests/          проверки архитектурных границ и конфигурации
  scripts/        инструменты сопровождения репозитория
  ml/             исследования, разметка, legacy API и совместимость

data/, runs/, artifacts/, .secrets/ — локальные данные, не исходный код.
Их не переносили, не переименовывали и не добавляли в Git.

Проверки из корня (GPU и обучение не запускаются):
  make PYTHON=.venv/bin/python check
  make NODE=node check-web

На текущем сервере Node расположен в /opt/agent-studio-runtime/node:
  make NODE=/opt/agent-studio-runtime/node check-web

Явные точки запуска, каждая выполняет только свою роль:
  python -m forte_vision.interfaces.analysis_http --port 8771
  NUHACK_JURY_ENGINE=gemma python -m forte_vision.interfaces.jury_http --port 8765
  python -m forte_vision.interfaces.review_http --port 8765

jury_http и review_http — альтернативы на порту8765, не два сервиса одновременно.
В текущем размещении review_http/старый annotator включает API жюри.
Интерфейс /demo/gemma и внешние /api/v1 URL сохраняются совместимыми.

Рефакторинг не меняет веса модели, правила принятия решения или наборы обучения.
Не переносите в эту ветку backend/ из main без отдельной интеграционной задачи:
там другой Go-сервис с неподключённым Vision-процессором.
