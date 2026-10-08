# Сборка документации и GitHub Pages

Сайт на VitePress хранится в `docs/`. Markdown содержит ссылки на код и Mermaid-схемы. Локальный поиск, навигация и темы доступны без внешнего сервиса.

## Локальная работа

Из корня репозитория (Node.js 22+, Python 3.11+):

```bash
npm ci
npm run docs:check
npm run docs:dev
```

Откройте адрес dev server с base path `/4da_3dgs_pipeline/`. Для production build и preview:

```bash
npm run docs:build
npm run docs:preview
```

VitePress проверяет внутренние ссылки при сборке. Output — `docs/.vitepress/dist/`; cache/dist/node_modules не коммитятся. Для сборки не нужны GPU, Conda, weights, AWS credentials или Python ML dependencies.

## Обновлять вместе с кодом

При изменении CLI, env settings или training profile:

```bash
npm run docs:reference
npm run docs:check
npm run docs:build
```

`generate_docs_reference.py` импортирует CLI без выполнения операций и формирует таблицы всех аргументов, runtime env variables и training defaults. `--check` сравнивает output с tracked страницами и завершается ошибкой при расхождении. После изменения семантики также обновите ручные пояснения, схемы и примеры; генератор не определяет смысл параметра автоматически.

Страницы `reference/cli.md`, `reference/environment.md`, `reference/training-defaults.md` генерируются. Остальные страницы редактируются вручную. API/план и CLI examples сверяются с `src/recon_pipeline/`.

## Публикация

Workflow `.github/workflows/docs.yml`:

- В PR проверяет generated reference и собирает сайт с проверкой ссылок.
- После push в `main` или ручного запуска на `main` собирает и публикует Pages artifact.
- Деплой использует `pages:write` и OIDC (`id-token:write`); personal access token не нужен.

Перед первым deploy владелец репозитория выбирает **Settings → Pages → Build and deployment → Source → GitHub Actions**. Ветка с документацией должна попасть в `main`. До этого ссылка не является опубликованным сайтом.

Ожидаемый адрес: [vinneyto.github.io/4da_3dgs_pipeline/](https://vinneyto.github.io/4da_3dgs_pipeline/).

`base` в `.vitepress/config.mts` соответствует имени репозитория. Если репозиторий переименовать или перейти на custom domain, обновите `base` и адрес. Mermaid обрабатывается `vitepress-plugin-mermaid`; обычный fenced source также читается в GitHub Markdown.

Официальные руководства: [VitePress deployment](https://vitepress.dev/guide/deploy), [Mermaid plugin](https://github.com/emersonbottero/vitepress-plugin-mermaid).
