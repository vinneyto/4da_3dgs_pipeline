# Утилиты и примеры CLI

Утилита выполняет одну операцию и не требует run JSON или worker. Все входы, output paths и tool paths передаются явно. Console entrypoints объявлены в `pyproject.toml`; полный список обязательных/опциональных флагов и defaults — [аргументы CLI](./cli.md).

## Общий протокол

```json
{"event":"progress","fraction":0.5,"message":"Exported synchronized frame 60"}
{"event":"result","data":{"path":"/data/output"}}
```

Каждый event — отдельная строка stdout. Успешная операция выдаёт один итоговый `result` с object в `data`. Диагностика сторонних библиотек перенаправляется в stderr. Ошибка даёт ненулевой exit code; consumers не должны считать последний progress успешным result. В `run_utility` повторный result, не-object data или отсутствие result — ошибка.

`core.utility.run_utility` запускает `python -u -m MODULE`, читает result и преобразует progress в pipeline events. Старый `FOURDA_PROGRESS {...}` также распознаётся адаптером. Utility не отправляет уведомления напрямую.

Можно запускать одну и ту же операцию тремя способами:

```bash
recon-4danyone --help
python -m recon_pipeline.utilities.datasets.fourdanyone.inference --help
python -m pdb -m recon_pipeline.utilities.datasets.fourdanyone.inference --help
```

## Общие пути для примеров

Подставьте существующие абсолютные пути и реальный bucket. Shell variables здесь — удобство примеров, утилиты сами их не читают:

```bash
export DEMO_ROOT=/data/recon
export DEMO_RUN="$DEMO_ROOT/runs/demo_01"
export DEMO_SOURCE="$DEMO_ROOT/runs/source_01"
export DEMO_FOURDA=/opt/4DAnyone
export DEMO_NS_BIN=/opt/conda/envs/splatfacto/bin
export DEMO_RERUN_PYTHON=/opt/conda/envs/splatfacto-rerun/bin/python
export DEMO_BUCKET=YOUR_BUCKET
export DEMO_REGION=us-east-1
```

Инференс и dataset export запускайте из Python env 4DAnyone; Splatfacto — из Nerfstudio env, reconstruction Rerun — из отдельного Rerun env. Storage/cloud utilities требуют `[aws]`, dataset Rerun — `[rerun]`, reconstruction Rerun — `[splatfacto-rerun]`. Базовый пакет не устанавливает ML-системы вместо environment installer.

## Датасеты

### `recon-prepare-experiment`

Создаёт workspace и сохраняет произвольный settings snapshot. Не интерпретирует параметры реконструкции.

```bash
recon-prepare-experiment --directory "$DEMO_RUN" \
  --settings-output "$DEMO_RUN/pipeline-config.json" \
  --settings '{"note":"standalone experiment"}'
```

Result: `path`, `settings_path`. При отсутствии `--settings-output` используется `directory/config.json`.

### `recon-4danyone`

Запускает pinned upstream inference с явными video/model/output paths. Результат — generation directory, необходимый exporters.

```bash
recon-4danyone --fourdanyone-root "$DEMO_FOURDA" \
  --video "$DEMO_ROOT/input/video.MOV" --output "$DEMO_RUN/4danyone" \
  --model-dir "$DEMO_ROOT/models" --views-per-layer 24 \
  --layer-pitches -15 0 15 --start-yaw 0 --yaw-span 360 \
  --target-fps 30 --seed 42 --turbo --attention-backend auto
```

Result: `path`. Требуются inference.py, GVHMR, weights и корректное число ракурсов. `--replace-existing` разрешает замену output; без него существующий output — ошибка.

## Nerfstudio и реконструкция

### `recon-nerfstudio-export`

Экспортирует один или несколько временных срезов completed generation в multicamera datasets.

```bash
recon-nerfstudio-export --fourdanyone-root "$DEMO_FOURDA" \
  --generation "$DEMO_RUN/4danyone" --output "$DEMO_RUN/nerfstudio" \
  --model-dir "$DEMO_ROOT/models" --frames 0 60 120 --device cuda:0
```

Result: `datasets=[{frame,dataset_dir}, ...]`, `frames`, `count`. Передаёт управление upstream `scripts/export_nerfstudio.py` и проверяет `transforms.json`. `--replace-existing` заменяет выбранные frame directories, а не весь список невыбранных кадров. Standalone parser допускает distinct nonnegative indices; portable config дополнительно ограничивает их `0..120`.

### `recon-prepare-rgba`

Создаёт private RGBA training copy. Минимум 4 камеры; нужны RGB images и mask каждой камеры. Source и output должны быть отдельными, не вложенными друг в друга каталогами. Output должен отсутствовать.

```bash
"$DEMO_NS_BIN/python" -m recon_pipeline.utilities.reconstructions.nerfstudio.prepare_rgba \
  --dataset "$DEMO_RUN/nerfstudio/frame_060" \
  --output "$DEMO_RUN/rgba-preview/frame_060" \
  --mask-threshold 127 --mask-erosion-pixels 1
```

Result: `dataset_dir`, `camera_count`, `foreground_min`, `foreground_max`. Alpha строится по threshold и erosion; пустой foreground — ошибка. Нельзя отключить маски только через erosion=0.

### `recon-splatfacto`

Выполняет RGBA preparation → training → full SH PLY export → validation → experiment manifest. RGBA preparation уже включена: не подавайте `rgba-preview` из предыдущего примера вместо исходного multicamera dataset с masks.

```bash
"$DEMO_NS_BIN/python" -m recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto \
  --dataset "$DEMO_RUN/nerfstudio/frame_060" \
  --output "$DEMO_RUN/splatfacto/frame_060" \
  --nerfstudio-bin "$DEMO_NS_BIN" --experiment-name demo_01_frame_060 \
  --max-num-iterations 60000 --sh-degree 2 \
  --mask-threshold 127 --mask-erosion-pixels 1
```

Все [параметры профиля](./config.md#профиль-splatfacto) доступны как kebab-case flags. `--replace-existing` удаляет старый reconstruction output перед расчётом. Возвращаются:

| Поля result | Содержание |
| --- | --- |
| `dataset_dir`, `source_dataset`, `output_dir` | Training copy, исходный dataset, output |
| `config`, `run_root`, `dataparser_transform` | Nerfstudio `config.yml`, run directory, transformations |
| `splat_ply`, `full_splat_count` | Проверенная full SH Gaussian PLY и число Gaussian |
| `train_log`, `camera_count` | Training log и число камер |
| `run_timestamp`, `experiment_name` | Identity training run |
| `training_profile`, `mask_mode` | Фактический профиль; `rgba_alpha_random_background` |

Тот же result сохраняется в `experiment_manifest.json`. Полный training output/checkpoints находится в `nerfstudio_outputs/`.

## Rerun

### `recon-rerun`

Записывает камеры, видео и скелет из generation 4DAnyone.

```bash
recon-rerun --generation "$DEMO_RUN/4danyone" \
  --output "$DEMO_RUN/rerun/demo_01.rrd" --experiment demo_01 \
  --fourdanyone-root "$DEMO_FOURDA" --model-dir "$DEMO_ROOT/models" \
  --view-count 4 --device auto
```

Result: `path`. Нужны metadata/cameras и данные upstream generation. `--replace-existing` заменяет запись.

### `recon-splatfacto-rerun`

Записывает статическую Gaussian-сцену из PLY вместе с training cameras/metrics. Выполняется в отдельном CPU Rerun env:

```bash
# Прочитайте run_root/run_timestamp из experiment_manifest.json.
export DEMO_TRAIN_RUN=/data/recon/runs/demo_01/splatfacto/frame_060/nerfstudio_outputs/demo_01_frame_060/splatfacto/YOUR_TIMESTAMP
"$DEMO_RERUN_PYTHON" -m recon_pipeline.utilities.artefacts.rerun.splatfacto \
  --splat "$DEMO_RUN/splatfacto/frame_060/exports/splat.ply" \
  --dataset "$DEMO_RUN/splatfacto/frame_060/dataset" \
  --run-root "$DEMO_TRAIN_RUN" \
  --output "$DEMO_RUN/splatfacto/frame_060/rerun/reconstruction.rrd" \
  --experiment-name demo_01_frame_060 --run-timestamp YOUR_TIMESTAMP \
  --max-splats 100000 --view-count 4
```

Result содержит `rerun_path`, `visualized_splats`, `full_splat_count`, `logged_metrics`, `selected_camera_indices` и статистику сцены. Лимит Gaussian применяется только к visualization; PLY на диске не сокращается.

## AWS и S3

AWS credentials выбирает boto3. `--region` default — `us-east-1`; `--bucket` обязателен у всех перечисленных storage utilities.

### `recon-aws-preflight`

Проверяет identity, чтение входа, LIST моделей и write probe.

```bash
recon-aws-preflight --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --input-key input/video.MOV --models-prefix models --write-prefix runs
```

`--no-check-input` полезен для export-only работы. Для проверки App дополнительно задайте `--sagemaker-domain-id`, `--sagemaker-space-name`, `--sagemaker-app-name`. Result: account/caller ARN, bucket, writable prefixes, App status, video URI, model object count. Write probe действительно делает PUT и пытается DELETE.

### `recon-s3-download-input`

```bash
recon-s3-download-input --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --key input/video.MOV --output "$DEMO_ROOT/input/video.MOV"
```

Result: `path`, `s3_uri`. `--replace-existing` разрешает заменить локальный input.

### `recon-s3-sync-models`

```bash
recon-s3-sync-models --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix models --destination "$DEMO_ROOT/models" --sync
```

Result: `path`, `objects`, `downloaded`. Cache сравнивается по размеру; это не SHA-256 commit recovery. `--no-sync` использует существующий destination без S3 sync.

### `recon-s3-restore-experiment`

Восстанавливает **generation источника**, не целиком новый training run. Сначала использует inference commit; для legacy experiments без marker — только `prefix/4danyone/`.

```bash
recon-s3-restore-experiment --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix runs/source_01 --destination "$DEMO_SOURCE"
```

Result: generation `path`, `objects`, `downloaded`. `--replace-existing` удаляет локальную generation перед restore. В конце обязательны metadata/cameras.

### `recon-s3-upload-results`

Legacy/manual whole-directory upload. Современный worker использует **publish-artifacts после каждого producer**, а не этот финальный full-run upload.

```bash
recon-s3-upload-results --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix runs/manual-copy --source "$DEMO_RUN"
```

Result: `s3_uri`. `--replace-existing` у этой операции удаляет существующий destination prefix перед upload. Этот флаг имеет более широкую область действия, чем replace у per-frame computation.

### `recon-s3-publish-artifacts`

Низкоуровневая публикация выбранного producer bundle. Для штатного запуска её параметры собирает worker. Для повторной диагностики используйте **реальные** `pass-state.json` и upload signature из того же плана; произвольная строка signature не соответствует recovery.

```bash
# DEMO_UPLOAD_SIGNATURE — fingerprint upload-паса из checkpoint/плана запуска.
recon-s3-publish-artifacts --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix runs/demo_01 --root "$DEMO_RUN" --data-root "$DEMO_ROOT" \
  --paths splatfacto/frame_060 \
  --checkpoint "$DEMO_RUN/.recon-pipeline/pass-state.json" \
  --pass-id splatfacto:frame_060 --upload-id s3-upload:splatfacto:frame_060 \
  --signature "$DEMO_UPLOAD_SIGNATURE"
```

`--paths` задаёт существующие пути относительно `--root`; выход за root запрещён. Marker публикуется последним. Result: `s3_uri`, `marker_key`, `files`.

### `recon-s3-recover-run`

Восстановление текущего плана из committed bundles. `--plan` — JSON array; каждый element содержит `id`, `producer_signature`, `upload_id`, `signature`, соответствующие composition root. Worker формирует его автоматически.

```bash
# DEMO_RECOVERY_PLAN_JSON — фактический plan этого эксперимента.
recon-s3-recover-run --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix runs/demo_01 --root "$DEMO_RUN" --data-root "$DEMO_ROOT" \
  --checkpoint "$DEMO_RUN/.recon-pipeline/pass-state.json" \
  --plan "$DEMO_RECOVERY_PLAN_JSON"
```

Пример формы одного элемента, не готовая signature:

```json
[{"id":"splatfacto:frame_060","producer_signature":"<producer SHA-256>","upload_id":"s3-upload:splatfacto:frame_060","signature":"<upload SHA-256>"}]
```

Result: `restored`, `downloaded`. `--reset-commits` инвалидирует markers указанного плана и удаляет локальный checkpoint; возвращает `invalidated`. Не удаляет весь artifact prefix.

### `recon-s3-upload-diagnostics`

```bash
# Используйте attempt-report.json, созданный worker для этой попытки.
recon-s3-upload-diagnostics --region "$DEMO_REGION" --bucket "$DEMO_BUCKET" \
  --prefix runs/demo_01/.recon-pipeline/attempts/manual-diagnostics \
  --root "$DEMO_RUN" --job-dir "$DEMO_ROOT/jobs/demo_01" \
  --report "$DEMO_ROOT/jobs/demo_01/attempt-report.json" --full
```

Report JSON должен содержать `state`, `pass_id`, `attempt_id`, `run_prefix`. Сохраняет report/status/request и logs. Без `--full` берутся хвосты до 1 MiB последних трёх experiment logs плюс pipeline log. `--skip-logs` сохраняет только reports/status/request. Result: `s3_uri`, `full_logs`.

## Manifest

### `recon-write-run-manifest`

Собирает итоговую metadata из явных аргументов; не ищет datasets/reconstructions автоматически. В примере ниже training result ещё не добавлен — для полного manifest передайте result objects через соответствующие JSON flags.

```bash
recon-write-run-manifest --output "$DEMO_RUN/pipeline-result.json" \
  --experiment-name demo_01 --experiment-dir "$DEMO_RUN" \
  --inference-dir "$DEMO_RUN/4danyone" --num-views 72 \
  --datasets '[{"frame":60,"dataset_dir":"/data/recon/runs/demo_01/nerfstudio/frame_060"}]' \
  --durations '{}' --reconstructions '[]' --reconstruction-recordings '[]'
```

`--rerun-file` добавляет dataset recording. Result: `manifest`, `path`. Manifest включает experiment paths, datasets, reconstructions, reconstruction_recordings, rerun_file, num_views, pass durations, elapsed sum и finished_at.

## Orchestration CLI

`recon-config` создаёт/копирует portable run JSON; это не самостоятельная вычислительная утилита. Примеры — [quickstart](../guide/quickstart.md) и [config](./config.md). `recon-aws-worker` управляет detached job/очередью — [руководство](../guide/worker.md). Их аргументы также входят в [полный CLI справочник](./cli.md).

Системные команды `scripts/setup_environment.sh`, `scripts/check_environment.sh`, `scripts/download_4danyone_models.sh` описаны в [окружении](../guide/environment.md). Compatibility-модуль `datasets.fourdanyone.runner --request` оставлен для старых внутренних вызовов; новые integrations используйте через явные utility flags.
