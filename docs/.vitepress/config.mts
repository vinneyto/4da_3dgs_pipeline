import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  lang: 'ru-RU',
  title: '4DA → 3DGS',
  description: 'Архитектура, запуск и API пайплайна реконструкции',
  base: '/4da_3dgs_pipeline/',
  lastUpdated: true,
  mermaid: {
    htmlLabels: false,
    flowchart: { useMaxWidth: false },
    sequence: { useMaxWidth: false, wrap: true, width: 130, actorMargin: 15 }
  },
  themeConfig: {
    nav: [
      { text: 'Руководство', link: '/guide/quickstart' },
      { text: 'Конфигурация', link: '/reference/config' },
      { text: 'CLI и API', link: '/reference/utilities' }
    ],
    sidebar: [
      { text: 'Система', items: [
        { text: 'Обзор', link: '/' },
        { text: 'Компоненты', link: '/guide/architecture' },
        { text: 'Поток данных и хранение', link: '/guide/data-flow' }
      ] },
      { text: 'Запуск', items: [
        { text: 'Быстрый старт', link: '/guide/quickstart' },
        { text: 'Окружение и проверки', link: '/guide/environment' },
        { text: 'Worker, очередь и восстановление', link: '/guide/worker' }
      ] },
      { text: 'Справочник', items: [
        { text: 'Конфиг реконструкции', link: '/reference/config' },
        { text: 'Утилиты и примеры CLI', link: '/reference/utilities' },
        { text: 'Все аргументы CLI', link: '/reference/cli' },
        { text: 'Пасы и Python API', link: '/reference/passes' },
        { text: 'Переменные окружения', link: '/reference/environment' },
        { text: 'Постобработка — PR #21', link: '/reference/postprocessing' }
      ] },
      { text: 'Документация', items: [
        { text: 'Сборка и GitHub Pages', link: '/contributing' }
      ] }
    ],
    search: { provider: 'local' },
    outline: { level: [2, 3], label: 'На этой странице' },
    docFooter: { prev: 'Предыдущая страница', next: 'Следующая страница' },
    lastUpdated: { text: 'Обновлено' },
    editLink: {
      pattern: 'https://github.com/vinneyto/4da_3dgs_pipeline/edit/main/docs/:path',
      text: 'Редактировать на GitHub'
    },
    socialLinks: [{ icon: 'github', link: 'https://github.com/vinneyto/4da_3dgs_pipeline' }]
  }
}))
