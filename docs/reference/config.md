# Конфиг реконструкции

Run document schema v7 описывает **эксперимент**, а окружение машины задаётся через `RECON_*`. В JSON нельзя помещать `environment`, `aws_worker.local`, workspace paths, `nerfstudio_bin` или Python interpreter. В коде нет отдельного JSON Schema файла: правила реализованы loaders, dataclasses и validators.

## Корневая структура

| Поле | Обязательность | Смысл |
| --- | --- | --- |
| `schema_version` | Обязательно при загрузке | Для новых конфигов `7`; loaders принимают исторические версии 1–7 |
| `experiment_name` | Нужно для нового переносимого документа | Имя каталога `runs/name` и S3 experiment prefix; не содержит `/`, `\`, `..` |
| `pipeline` | Обязательно | Typed stages `dataset` и `reconstruction` |
| `artifacts` | Опционально | Независимые exports/recordings. Отсутствующие задачи выключены |
| `aws_worker` | Обязательно для AWS worker, не нужно для local loader | S3, region, job identity, notifications, shutdown |

Для новых JSON используйте корневой `experiment_name`. Legacy значение `pipeline.experiment_name` ещё читается; если dataset config также содержит имя, оно должно совпадать. Не все неизвестные ключи корня строго отклоняются: успешное чтение опечатки не означает поддержку поля.

Все defaults ниже относятся к **runtime loader**. CLI generator может явно записывать иные значения: например `recon-config` по умолчанию выбирает `layer_pitches=[0]`, тогда как `FourDAnyoneConfig` — `[-15,0,15]`. Поэтому лучше сохранять параметры эксперимента явно.

## Полный пример из репозитория

Пример ниже экспортирует кадр 60 из уже существующей генерации, обучает Splatfacto и создаёт Rerun сцены. Замените placeholders ресурсов перед запуском.

<<< ../../config/splatfacto-run.example.json

Для генерации нового датасета смотрите [quickstart](../guide/quickstart.md). Для полного разбора полей продолжайте ниже.

## `pipeline.dataset`

| Поле | Обязательность / default | Смысл |
| --- | --- | --- |
| `enabled` | Опционально, `true` | Выполнить новый 4DAnyone inference. `false` позволяет экспортировать из существующего source experiment |
| `type` | Нужно при `enabled=true` | Только `"4danyone"` |
| `config` | Опциональный object | Параметры 4DAnyone, см. таблицу |

### `pipeline.dataset.config`

| Поле | Default / обязательность | Смысл и ограничения |
| --- | --- | --- |
| `video` | Обязательно для local loader; AWS берёт вход из bucket config | Путь относительно `RECON_DATA_ROOT/input`, без абсолютного пути и `..` |
| `views_per_layer` | `24` | Число синтетических ракурсов на слой; положительное |
| `layer_pitches` | `[-15,0,15]` | Углы вертикальных слоёв в градусах; непустой список, каждый угол в `[-15,45]` |
| `start_yaw` | `0` | Начальный горизонтальный угол в градусах |
| `yaw_span` | `360` | Горизонтальный охват в градусах, `1..360` |
| `target_fps` | `30.0` | Целевая временная частота подготовки входа; положительная |
| `seed` | `42` | Seed генерации |
| `turbo` / `enable_turbo` | `true` | Режим Turbo 4DAnyone. Используйте одно имя; AWS loader отвергает одновременное присутствие |
| `attention_backend` | `"auto"` | Передаётся upstream как выбор attention backend; строгого enum в этом loader нет |
| `resume` | `false` | Поле совместимости. Текущий inference pass не передаёт resume в утилиту и использует `--replace-existing`; восстановление worker задают checkpoints |

Общее число ракурсов: `views_per_layer × len(layer_pitches)`; оно должно делиться на 6. Например `24 × 3 = 72`. Даже export-only config материализуется в `FourDAnyoneConfig`, поэтому базовые ограничения параметров остаются в силе.

В AWS `bucket.video` определяет реальный S3 input и локальный basename независимо от `dataset.config.video`. Для понятного конфига не задавайте противоречащие друг другу входы.

## `pipeline.reconstruction`

| Поле | Обязательность / default | Смысл |
| --- | --- | --- |
| `enabled` | Если stage отсутствует — выключен; если object непустой без enabled — включён | Для ясности всегда задавайте `true/false` |
| `type` | Нужно при enabled | Только `"nerfstudio_splatfacto"` |
| `config.frames` | `null` | `null` или отсутствие = все кадры из `artifacts.dataset.nerfstudio.frames`; список = выбранное подмножество |
| `config.*` | Опциональные поля профиля | Defaults и смысл ниже |

Кадры уникальны, `0..120`, непустой список при явном выборе. Включённая reconstruction требует включённого `artifacts.dataset.nerfstudio`. Каждый training frame должен быть экспортирован. `frames=null` не означает все кадры исходного видео — это все **выбранные для экспорта** кадры.

### Профиль Splatfacto

Все поля опциональны. Их значения напрямую формируют `ns-train splatfacto`: `max_num_iterations` становится top-level training option; model settings — `--pipeline.model.*`. Mask settings используются до обучения при создании private RGBA copy.

| Поле | Назначение |
| --- | --- |
| `max_num_iterations` | Число training iterations |
| `stop_split_at` | Итерация, после которой останавливается split/densification |
| `cull_alpha_thresh` | Порог opacity для удаления слабых Gaussian |
| `densify_grad_thresh` | Порог градиента изображения для densification |
| `densify_size_thresh` | Пространственный size threshold, разделяющий split/duplicate поведение densification |
| `split_screen_size` | Порог размера на экране для splitting |
| `num_downscales` | Число уровней начального уменьшения разрешения |
| `resolution_schedule` | Интервал увеличения разрешения обучения |
| `cull_scale_thresh` | Порог масштаба для culling слишком больших Gaussian |
| `stop_screen_size_at` | Итерация остановки screen-size criterion |
| `use_scale_regularization` | Включить регуляризацию соотношения Gaussian scales |
| `max_gauss_ratio` | Максимальное соотношение scales для регуляризации |
| `sh_degree` | Степень сферических гармоник `0..3`; определяет число SH-коэффициентов в полной PLY |
| `mask_threshold` | Threshold `0..255` для бинарной alpha по маске, сравнение `alpha >= threshold` |
| `mask_erosion_pixels` | Радиус erosion; kernel `(2r+1) × (2r+1)`; `0` отключает erosion |

[Точные defaults профиля](./training-defaults.md) формируются из `TrainingProfile` при обновлении документации. Числовые поля должны быть finite и неотрицательными; iterations и resolution_schedule положительные. Integer/bool поля проверяются по типу. Размеры screen thresholds передаются в Nerfstudio без пересчёта в пиксели; точная интерпретация — model API pinned Nerfstudio.

Wrapper фиксирует random background, classic rasterization, TensorBoard visualization и `nerfstudio-data --eval-mode all`. Эти значения сейчас не являются полями run JSON.

Private RGBA copy не меняет исходный экспорт: RGB + threshold/erosion mask превращаются в RGBA PNG, ссылки `mask_path` удаляются из training transforms. Текущая утилита требует минимум 4 камеры и непустой foreground. Отдельного режима `use_mask=false` пока нет; `mask_erosion_pixels=0` отключает только erosion, не маску.

Экспорт всегда создаёт **полную Gaussian PLY с SH**, а не RGB point cloud. Ограничение `max_splats` из Rerun не сокращает эту PLY.

## `artifacts.dataset.nerfstudio`

Отсутствие/`null`/`false` выключает экспорт. `true` включает defaults. Object без `enabled` включает экспорт (`NerfstudioArtifactConfig.enabled=true`).

| Поле | Default | Смысл |
| --- | --- | --- |
| `enabled` | `true` внутри object | Создать статические datasets |
| `source_experiment_name` | `null` → текущий experiment | Откуда читать `4danyone/`; простое имя каталога |
| `frames` | `[60]` | Временные indices, уникальные, `0..120`, непустой список |
| `device` | `"cuda:0"` | Устройство upstream exporter |
| `replace_existing` | `false` | Совместимое поле конфигурации; текущий pass всегда передаёт `--replace-existing` при фактическом повторном выполнении |

Если source отличается от текущего experiment либо dataset disabled, AWS worker добавляет `s3-restore-experiment:<source>`. Экспорт всё равно сохраняется в **новый** experiment. Если source отсутствует в конфиге export-only run, источником считается текущее имя — соответствующая генерация уже должна существовать.

## `artifacts.dataset.rerun`

Отсутствие/`null`/`false` выключает задачу; `true` включает defaults. У object `enabled` по умолчанию `false` — отличие от Nerfstudio artifact.

| Поле | Default | Смысл |
| --- | --- | --- |
| `enabled` | `false` | Создать запись датасета |
| `view_count` | `4` | Число камер для записи; положительное |
| `device` | `"auto"` | Устройство реконструкции данных скелета; непустая строка |
| `source_experiment_name` | `null` → текущий experiment | Source 4DAnyone generation; простое имя каталога |
| `replace_existing` | `false` | Текущий pass при исполнении всегда передаёт `--replace-existing` |

Это запись анимации/камер датасета, не запись обученной статической Gaussian-сцены.

## `artifacts.reconstruction.rerun`

Отсутствие выключает. `true` — enabled с defaults. Для object нужен `enabled=true`; default `false`.

| Поле | Default | Смысл |
| --- | --- | --- |
| `enabled` | `false` | Создать отдельную запись для каждого обучаемого кадра |
| `max_splats` | `100000` | Положительный лимит Gaussian в Rerun-визуализации; full PLY остаётся полной |
| `view_count` | `4` | Положительное число камер в записи |
| `source_experiment_name` | `null` | Если указан, должен совпадать с текущим experiment; другие trained experiments здесь не поддержаны |
| `device` | `"auto"` | Допустимы `auto`, `cpu`; импорт PLY выполняется на CPU, pass не передаёт отдельный device flag |
| `replace_existing` | `false` | Текущий pass всегда передаёт `--replace-existing` при исполнении |

Требует включённую reconstruction. Python для этого exporter задаётся `RECON_SPLATFACTO_RERUN_PYTHON`, а не полем JSON.

## `aws_worker`

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `job_id` | Обязательно | Простое имя каталога `jobs/`; generator по умолчанию использует experiment name |
| `region` | Обязательно | AWS region для SDK. Generator default — `us-east-1` |
| `bucket` | Обязательно | Object из следующей таблицы |
| `sync_models` | `true` | Синхронизировать веса из S3; false использует локальный cache |
| `upload_results` | `true` | Добавить per-producer S3 publication, cloud recovery и diagnostics |
| `shutdown_on` | `"never"` | `never`, `success`, `failure`, `always` |
| `notifications` | Отсутствует → нет каналов | Email/Telegram настройки |
| `sagemaker` | Обязательно для современного AWS документа даже при `shutdown_on=never` | Идентификаторы App |
| `cloudwatch` | Отсутствует/`null` → выключен | Настройка streaming логов |

### `aws_worker.bucket`

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `name` | Обязательно, непустое | Bucket name, без `s3://` |
| `video` | Обязательно | Относительный key внутри input_prefix, например `people/demo.MOV`; без `..` и полного S3 URI |
| `input_prefix` | `"input"` | Prefix входов |
| `models_prefix` | `"models"` | Prefix общего cache моделей |
| `runs_prefix` | `"runs"` | Prefix результатов |

Leading/trailing `/` нормализуются; `..` запрещён. Например bucket `demo-bucket`, input_prefix `input`, video `people/demo.MOV` → `s3://demo-bucket/input/people/demo.MOV`, локально `D/input/demo.MOV`. Даже при dataset disabled поле video нужно для валидного AWS config и identity.

### `aws_worker.sagemaker`

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `domain_id` | Обязательно | SageMaker Studio domain |
| `space_name` | Обязательно | Space |
| `app_name` | `"default"` | Имя App типа JupyterLab |

Shutdown удаляет указанную App, а не весь domain/space. [Worker](../guide/worker.md) объясняет deferral при проблемах сохранения.

### `aws_worker.cloudwatch`

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `log_group` | Обязательно при наличии object | Существующая группа; 1–512 символов, допустимы буквы/цифры и `. - _ / #` |
| `enabled` | `true` | Включить streaming |

`recon-aws-worker start --cloudwatch-log-group ...` переопределяет настройку для запуска/очереди. Без секции и override streaming не запускается.

### `aws_worker.notifications.email`

Отсутствие/`null` — нет email. Для object:

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `topic_name` | Обязательно | SNS topic |
| `email` | Обязательно | Адрес подписки |
| `enabled` | `true` | Включить email notifications |

Создание и подтверждение подписки: `recon-aws-worker configure-email --config ...`.

### `aws_worker.notifications.telegram`

Отсутствие/`null` — нет Telegram. Для object:

| Поле | Default / обязательность | Смысл |
| --- | --- | --- |
| `enabled` | `true` | Включить канал |
| `chat_id` | Нужно в object, непустое при enabled | Получатель |
| `bot_token_env` | Один из двух token sources при enabled | Имя переменной с token; рекомендуемый способ, например `CP_4DA_TELEGRAM_BOT_TOKEN` |
| `bot_token` | Альтернатива bot_token_env | Прямой token; не помещайте секрет в tracked run config |
| `shutdown_command` | `true` | Разрешить `/shutdown` авторизованному пользователю |
| `allowed_user_id` | `null` → chat_id | Telegram user ID; явно задавайте для группового чата |

При enabled должен быть ровно один из `bot_token`, `bot_token_env`. Переменная с token должна существовать при выполнении. Defaults генератора CLI указаны в [CLI reference](./cli.md); отсутствие аргумента token_env в JSON не означает автоматический выбор имени.

## Проверка и миграция

`recon-config` проверяет settings без загрузки окружения и без AWS вызовов. Старый документ обновляется через template-copy, machine settings удаляются, а источник не меняется:

```bash
recon-config --template config/old.json --output config/new.json \
  --experiment-name new_experiment --video video.MOV
```

При template-copy pipeline/artifact/AWS settings сохраняются. Явные stage/frame/source flags могут их переопределить. Не все generator flags применяются как overrides шаблона: в текущей реализации training profile flags не заменяют сохранённый профиль; редактируйте его в JSON. Источник артефактов, ссылавшийся на имя template experiment, переносится на новое имя автоматически.

`recon-aws-worker plan` дополнительно материализует окружение и проверяет artifact contract, но не выполняет preflight AWS или inference. Хотя loaders принимают v1–v6, новые документы надо создавать v7.

Root `postprocessing` в текущем `main` ещё не реализован. Подготовленная схема — [постобработка PR #21](./postprocessing.md).

Источники: [config.py](https://github.com/vinneyto/4da_3dgs_pipeline/blob/main/src/recon_pipeline/datasets/fourdanyone/config.py), [AWS config](https://github.com/vinneyto/4da_3dgs_pipeline/blob/main/src/recon_pipeline/workers/aws/config.py), [TrainingProfile](https://github.com/vinneyto/4da_3dgs_pipeline/blob/main/src/recon_pipeline/utilities/reconstructions/nerfstudio/_profile.py).
