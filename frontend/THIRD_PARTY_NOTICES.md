# Компоненты и лицензии

Основные прямые зависимости используют разрешительные open-source лицензии. Точные установленные версии зафиксированы в `package-lock.json`; тексты лицензий поставляются в соответствующих npm-пакетах.

| Компонент                          | Лицензия                                 |
| ---------------------------------- | ---------------------------------------- |
| Next.js, React, React DOM          | MIT                                      |
| Tailwind CSS, tailwind-merge, clsx | MIT                                      |
| Radix UI, shadcn/ui                | MIT                                      |
| class-variance-authority           | Apache-2.0                               |
| Lucide React                       | ISC; часть исходных Feather-иконок — MIT |
| Zod                                | MIT                                      |
| Inter                              | SIL Open Font License 1.1                |
| TypeScript, Playwright             | Apache-2.0                               |
| tsx, sharp                         | MIT / Apache-2.0 соответственно          |

Button и Dialog основаны на подходе shadcn/ui: доступные Radix primitives и локально изменяемый код. Атрибуция: shadcn/ui, Copyright (c) 2023 shadcn, MIT. Полный текст MIT приведён в `licenses/shadcn-MIT.txt`.

Дизайн-рекомендации использованы как reference, исходники этих репозиториев в приложение не включены:

- https://github.com/pbakaus/impeccable — Apache-2.0.
- https://github.com/nextlevelbuilder/ui-ux-pro-max-skill — MIT.
- https://github.com/Leonxlnx/taste-skill — MIT.

[Perricheno](https://github.com/Perricheno/perricheno) использован как визуальный референс по просьбе команды: нейтральная палитра, светлая навигация и отдельная белая рабочая область. Исходный код, логотипы и графические материалы Perricheno не включены в проект. Файл лицензии в изученной версии репозитория не найден.

`public/demo/scene.svg`, `evidence.jpg` и `cashier-preview.mp4` созданы специально для данного прототипа. Это схематичные учебные материалы без внешней съёмки. FFmpeg используется только внешним инструментом генерации; бинарник FFmpeg не распространяется с приложением. Моделей ML и их весов в frontend нет.

Для production-дистрибуции команда должна сохранять уведомления из установленных пакетов. Лицензии backend, ML-моделей и реального датасета находятся вне frontend-модуля.
