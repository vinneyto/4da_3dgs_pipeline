# Быстрый старт

## 1. Подготовить машину

Следуйте [настройке окружения](./environment.md). После установки в текущем терминале:

```bash
source "$HOME/.config/recon-pipeline/environment.sh"
./scripts/check_environment.sh --require-cuda
```

## 2. Подготовить входы

Поместите видео в `s3://YOUR_BUCKET/input/video.MOV`, веса — в `models/`. Выберите существующий SageMaker domain/space/app. Пример хранит только имена ресурсов; credentials берутся из AWS SDK, а machine paths — из `RECON_*`.

## 3. Создать конфиг

```bash
recon-config \
  --output config/run.json \
  --experiment-name demo_01 \
  --bucket YOUR_BUCKET --video video.MOV --region us-east-1 \
  --dataset --views-per-layer 24 --layer-pitches -15 0 15 \
  --nerfstudio --nerfstudio-frames 60 \
  --reconstruction --reconstruction-frames 60 \
  --no-rerun --no-email --no-telegram \
  --shutdown-on never \
  --sagemaker-domain-id d-YOUR_DOMAIN \
  --sagemaker-space-name YOUR_SPACE --sagemaker-app-name default
```

Команда создаёт и валидирует schema v7 без GPU, Conda и AWS API вызовов. Это **72 синтетических ракурса**, статический экспорт момента `60` и одна Splatfacto-сцена. Перед запуском нужны реальные bucket/domain/space значения. Все поля объяснены в [справочнике конфигурации](../reference/config.md).

Для нового эксперимента из рабочего конфига:

```bash
recon-config --template config/run.json --output config/demo_02.json \
  --experiment-name demo_02 --video another-video.MOV
```

## 4. Проверить план и запустить

```bash
recon-aws-worker plan --config config/run.json
recon-aws-worker start --config config/run.json
recon-aws-worker status --config config/run.json --json
recon-aws-worker logs --config config/run.json --lines 100 --follow
```

`plan` требует настроенные `RECON_*`, но не обращается к AWS и не запускает inference. `start` запускает detached процесс: закрытие терминала не останавливает job.

## 5. Найти результаты

Локально: `$RECON_DATA_ROOT/runs/demo_01/`. PLY: `splatfacto/frame_060/exports/splat.ply`. Итоговый индекс: `pipeline-result.json`. При `upload_results=true` результаты публикуются в `s3://YOUR_BUCKET/runs/demo_01/` после каждого вычислительного паса.

Полные схемы сохранения и восстановления: [поток данных](./data-flow.md). Ошибка и повторный запуск: [worker](./worker.md).
