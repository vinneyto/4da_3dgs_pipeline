# Worker, очередь и восстановление

## Управление одним заданием

```bash
recon-aws-worker plan --config config/run.json
recon-aws-worker start --config config/run.json
recon-aws-worker status --config config/run.json --json
recon-aws-worker logs --config config/run.json --lines 200 --follow
recon-aws-worker stop --config config/run.json
```

`stop` отправляет SIGTERM группе процессов worker. Это не отдельная команда удаления SageMaker App. Терминальный статус может быть `succeeded`, `failed`, `cancelled`; прочитайте `status.json` и логи для причины ошибки.

Worker сохраняет snapshot запроса, PID и durable status в `jobs/job_id/`. Повторный запуск уже активного задания не создаёт второй процесс. Для повторной попытки после завершения используйте тот же конфиг.

## Очередь из конфигов

```bash
recon-aws-worker start \
  --config config/first.json --config config/second.json \
  --queue-id demo-queue --shutdown-on always
recon-aws-worker status --queue-id demo-queue --json
recon-aws-worker logs --queue-id demo-queue --follow
```

Конфиги исполняются последовательно. Ошибка одного задания фиксируется в `queue-result.json`, после чего очередь продолжает следующие задания; итоговая очередь получает статус failed, если было хотя бы одно неуспешное задание. По умолчанию queue ID — `queue-<first job ID>`. Политика shutdown применяется после очереди, а не между заданиями. Конфиги очереди должны ссылаться на совместимые region и SageMaker App. Подробные параметры и проверки — [CLI](../reference/cli.md#recon-aws-worker).

## CloudWatch

Можно хранить настройку в `aws_worker.cloudwatch`:

```json
{
  "cloudwatch": {
    "enabled": true,
    "log_group": "/recon-pipeline/YOUR_DEPLOYMENT"
  }
}
```

Это fragment секции `aws_worker`. Группа должна существовать. Параметр запуска переопределяет её для всех конфигов очереди, сохраняя исходные файлы:

```bash
recon-aws-worker start --config config/run.json \
  --cloudwatch-log-group /recon-pipeline/YOUR_DEPLOYMENT
```

Без секции и CLI override CloudWatch отключён. `enabled=false` также отключает секцию. Локальный лог остаётся доступен независимо от облачного stream.

## Восстановление

Повторите `start` с тем же конфигом **без `--force`**. AWS pipeline использует `resume_by_id=True`: checkpoint проверяется по ID, набору артефактов, fingerprint релевантных настроек и наличию нужных файлов. Включение дополнительного артефакта не означает автоматическое повторение всех предыдущих вычислений.

При `upload_results=true` recovery сначала читает commit markers текущего плана в S3. Файлы восстанавливаются с проверкой размеров/SHA-256, сохранённые пути материализуются относительно текущего `RECON_DATA_ROOT`. Завершённые upload-пасы подтверждаются только remote marker, а не одним локальным status.

Примеры поведения:

| Изменение | Что происходит |
| --- | --- |
| Новый запуск того же конфига с целыми файлами | Вычислительные пасы переиспользуют валидные checkpoints |
| Другая training profile | Соответствующие Splatfacto и связанные recordings теряют валидность |
| Добавление кадра в реконструкцию | Новый кадр обучается отдельно; существующие кадры сохраняют checkpoints при прежнем профиле/источнике |
| Изменение списка экспортируемых кадров | `nerfstudio-export` проверяет полный набор и при несовпадении выполняется снова |
| Upload завершился ошибкой после вычисления | Локальный producer checkpoint может сохраниться; публикация повторяется |
| Диск новой машины пуст | Можно восстановить только опубликованные в S3 bundles с валидными commits |
| `upload_results=false` | Только локальные checkpoints; потеря локального диска не восстанавливается из новых S3 bundles |

Fingerprint входа использует bucket/prefix/video и параметры генерации, **не хеш содержимого исходного видео**. Перезапись другого видео под тем же S3 key не является надёжной сменой identity. Используйте новый key/experiment либо сознательно повторите вычисления с `--force`.

`--force` очищает локальные checkpoints и инвалидирует remote commits выбранного плана; вычисления и загрузки выполняются заново. Он не очищает целиком S3 run prefix. Пасы вызывают утилиты с `--replace-existing`, поэтому новый расчёт может заменить локальные файлы стадии. `resume` в dataset config не управляет этим механизмом.

## Уведомления и finalizers

SNS email требует предварительной подписки:

```bash
recon-aws-worker configure-email --config config/run.json
```

После команды подтвердите AWS Subscription Confirmation email. Telegram отправляет стартовый план, начало/окончание/ошибки пасов с snapshot CPU/RAM/disk/GPU. Периодического polling ресурсов и live log streaming через Telegram нет. При включённом `shutdown_command` команда `/shutdown` ограничена настроенным chat/user.

Finalizers запускаются и после успеха, и после ошибки: S3 diagnostics → CloudWatch flush (если включён) → SageMaker shutdown. `shutdown_on` принимает `never`, `success`, `failure`, `always`. При проблемах сохранения S3 или CloudWatch shutdown откладывается; при отмене через `KeyboardInterrupt` штатный shutdown finalizer не удаляет App. Ошибки finalizer видны как ошибки финализации, даже если вычисления завершились.

## Локальное выполнение

В проекте нет общего console command `recon-local-worker`. Для диагностики локально запускайте [утилиты](../reference/utilities.md) с явными путями. `config/local-run.example.json` можно загрузить через `load_pipeline_config(Path(...))`, затем составить `Pipeline` из пасов и начальных артефактов вручную — [Python API](../reference/passes.md).

## Граница worker request

При чтении `request.json` worker вызывает `parse_worker_request` и проверяет все вложенные настройки до запуска операций. Через `isinstance(request, QueueRequest)` выбирается очередь; оставшаяся ветка принимает `ExperimentRequest`. `run_queue` получает только `QueueRequest`, `_run_experiment` — только `ExperimentRequest`.

`build_start_request` также возвращает эти объекты. CLI overrides и отключение shutdown у дочерних экспериментов выполняются через `dataclasses.replace`, без изменения исходного конфига. `AwsBackgroundJob.prepare/start` сохраняют запрос через `request.to_dict()`; дочерний процесс снова разбирает его на границе своего JSON-файла. [Объявления моделей и правила совместимости](../reference/config.md#типизированные-объекты-в-python).
