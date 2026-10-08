---
layout: home
hero:
  name: 4DA → 3DGS
  text: Пайплайн реконструкции
  tagline: От видео и синтетических ракурсов до статических Gaussian splats, Rerun и сохранённых результатов в S3.
  actions:
    - theme: brand
      text: Начать запуск
      link: /guide/quickstart
    - theme: alt
      text: Архитектура
      link: /guide/architecture
features:
  - title: Явные стадии
    details: Worker собирает последовательность пасов, утилиты выполняют отдельные операции.
  - title: Переносимый конфиг
    details: Параметры эксперимента в JSON, пути и версии инструментов — в окружении машины.
  - title: Восстановление
    details: Checkpoint каждого вычислительного паса и публикация проверенных файлов в S3.
---

## Как читать документацию

1. [Компоненты системы](./guide/architecture.md) и [поток данных](./guide/data-flow.md) показывают границы ответственности и моменты сохранения.
2. [Окружение](./guide/environment.md), [быстрый старт](./guide/quickstart.md) и [worker](./guide/worker.md) описывают подготовку и эксплуатацию.
3. [Конфиг](./reference/config.md), [утилиты](./reference/utilities.md) и [пасы](./reference/passes.md) служат справочниками.

Документация описывает реализацию schema v7 в `main`. Подготовленная в [PR #21](https://github.com/vinneyto/4da_3dgs_pipeline/pull/21) [постобработка](./reference/postprocessing.md) вынесена отдельно: её ещё нет в документируемой версии кода.
