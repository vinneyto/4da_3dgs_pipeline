# Пасы и Python API

Пас адаптирует runtime config/context в вызов утилиты и возвращает артефакты. Обработка тяжёлых данных остаётся в `utilities/`; S3 persistence и notifications — ответственность worker. Публичные типы ядра экспортируются из `recon_pipeline.core` и `recon_pipeline.core.pipeline`.

## Контракт

```python
from typing import Protocol
from recon_pipeline.core import PassResult, PipelineContext

class PipelinePass(Protocol):
    id: str
    name: str
    requires: frozenset[str]
    provides: frozenset[str]

    def run(self, context: PipelineContext) -> PassResult: ...
    def cleanup(self, context: PipelineContext) -> None: ...
```

`requires`/`provides` — имена артефактов, не пути. `Pipeline.prepare(initial_artifacts=...)` проверяет уникальные pass IDs, наличие всех dependencies в заданном порядке и отсутствие повторной записи существующих artifact keys. Он не строит порядок автоматически.

`run()` должен вернуть **точно** объявленный набор `provides`; пропущенные или лишние keys — ошибка. `cleanup` проверяется через `getattr`: реализация может его не определять. Если определён, он вызывается **перед** `run`, а не в finally после паса.

### `PipelineContext`

| API / поле | Поведение |
| --- | --- |
| `artifacts: dict[str, Any]` | Доступные значения; runner обновляет результатами completed/skipped пасов |
| `require(key)` | Вернуть артефакт; KeyError с объяснением при отсутствии |
| `values: dict[str, Any]` | Runtime metadata: длительности, failed pass, persistence state и т.п. |
| `pass_results: dict[str, PassResult]` | Результаты по pass ID |
| `report_progress(fraction, message)` | Только внутри выполняемого паса; core ограничивает fraction в `[0,1]` и публикует event |

`PassResult(artifacts={}, details={})` — dataclass. `details` попадает в lifecycle event/checkpoint; в checkpoints нужны JSON-serializable значения (Path/dataclass/list/dict тоже поддержаны сериализатором).

## Каталог вычислительных пасов

В constructor таблицах `config` — `FourDAnyoneConfig`, `worker` — `AwsWorkerConfig`, `frame` — integer временного кадра. `runner` — injectable `CommandRunner`, если поддержан; default запускает subprocess. Пути к модулям ниже относительно `recon_pipeline`.

| Класс / модуль | Constructor | ID | Requires → provides |
| --- | --- | --- | --- |
| `PrepareExperimentPass`, `datasets.fourdanyone.passes.prepare_experiment` | `(config, *, runner=None)` | `prepare-experiment` | ∅ → `experiment.workspace` |
| `FourDAnyoneInferencePass`, `datasets.fourdanyone.passes.inference` | `(config, *, runner=None)` | `fourdanyone-inference` | workspace + `input.video` + `models.cache` → `experiment.4danyone:E` |
| `NerfstudioExportPass`, `reconstructions.nerfstudio.passes.export` | `(config, *, runner=None)` | `nerfstudio-export` | workspace + models + `experiment.4danyone:source` → `dataset.nerfstudio` |
| `SplatfactoPass`, `reconstructions.nerfstudio.passes.splatfacto` | `(config, frame, *, runner=None)` | `splatfacto:frame_NNN` | workspace + `dataset.nerfstudio` → `reconstruction.splatfacto:frame_NNN` |
| `SplatfactoRerunPass`, `artifacts.rerun.passes.splatfacto` | `(config, frame, *, runner=None)` | `splatfacto-rerun:frame_NNN` | trained frame → `recording.splatfacto:frame_NNN` |
| `RerunExportPass`, `artifacts.rerun.passes.export` | `(config, *, runner=None)` | `rerun-export` | workspace + models + source generation → `dataset.rerun` |

| Пас | Значение возвращаемого артефакта | Побочные эффекты / utility |
| --- | --- | --- |
| Prepare | Path experiment directory | Создать workspace и `pipeline-config.json`; `utilities.datasets.fourdanyone.prepare_experiment` |
| Inference | Path generation | Создать `4danyone/`; `utilities.datasets.fourdanyone.inference` |
| Nerfstudio export | List `{frame,dataset_dir}` | Создать выбранные `nerfstudio/frame_NNN/`; `utilities.reconstructions.nerfstudio.export` |
| Splatfacto | Result dict training + добавленное `frame` | RGBA/training/export/manifest; `utilities.reconstructions.nerfstudio.splatfacto` |
| Splatfacto Rerun | Result dict recording + `frame` | Создать `splatfacto/frame_NNN/rerun/reconstruction.rrd`; `utilities.artefacts.rerun.splatfacto` |
| Dataset Rerun | Path `.rrd` | Создать `rerun/E.rrd`; `utilities.artefacts.rerun.export` |

`FourDAnyoneInferencePass.arguments()` и `NerfstudioExportPass.arguments()` возвращают собранный список CLI flags. Current passes используют `--replace-existing`; валидный checkpoint предотвращает повторное выполнение.

## AWS-пасы

Пакет — `recon_pipeline.workers.aws.passes` (конкретные файлы указаны в таблице).

| Класс / файл | Constructor | ID | Requires → provides |
| --- | --- | --- | --- |
| `AwsPreflightPass`, `preflight.py` | `(worker, pipeline_config, *, runner=None)` | `aws-preflight` | ∅ → `aws.health` |
| `S3DownloadInputPass`, `download_input.py` | `(worker, *, runner=None)` | `s3-download-input` | `aws.health` → `input.video` |
| `S3SyncModelsPass`, `sync_models.py` | `(worker, *, runner=None)` | `s3-sync-models` | `aws.health` → `models.cache` |
| `S3RestoreExperimentPass`, `restore_experiment.py` | `(worker, experiment_name, destination, *, runner=None)` | `s3-restore-experiment:E` | `aws.health` → `experiment.4danyone:E` |
| `WriteRunManifestPass`, `write_run_manifest.py` | `(config, *, runner=None)` | `write-run-manifest` | workspace + все включённые trained/recording frames → `run.result` |
| `S3UploadArtifactsPass`, `upload_artifacts.py` | `(worker, config, producer)` | `s3-upload:<producer.id>` | producer.provides → artifact key с ID upload-паса |
| `S3UploadResultsPass`, `upload_results.py` | `(worker, config, *, runner=None)` | `s3-upload-results` | `run.result` → `aws.s3.result` |

| Пас | Результат / особенности |
| --- | --- |
| Preflight | Health dict; после standalone AWS access check добавляет проверку notification channels |
| Download input | Path локального видео |
| Sync models | Path кеша. Пас остаётся в плане при `sync_models=false`, но передаёт `--no-sync` |
| Restore source | Path `source/4danyone`; добавляется по source dependencies |
| Write manifest | Manifest dict; пути datasets/dataset Rerun берёт из context, если они там есть |
| Upload artifacts | S3 URI bundle; последним публикует commit marker, не весь experiment сразу |
| Upload results | S3 URI whole-directory upload; оставлен для совместимости/ручного использования, современный `build_aws_pipeline` его не добавляет |

Для точного набора enabled passes конкретного конфига:

```bash
recon-aws-worker plan --config config/run.json
```

## Recovery adapters

`workers.aws.recovery.RecoverablePass(wrapped, config, signature, legacy_signature=None)` делегирует вычисление wrapped pass и добавляет `checkpoint_signature` / `validate_checkpoint(checkpoint, context)`.

Validator сверяет релевантные settings и реальные выходы. Для Nerfstudio проверяются список frames и `transforms.json`, inference — metadata/cameras, training — PLY и config. Это не универсальная глубокая валидация всей генерации или корректности модели.

`S3UploadArtifactsPass` имеет собственную upload signature (producer signature + destination). Его checkpoint нельзя переиспользовать, если producer выполнялся заново в текущем context.

`RecoveryCheckpointStore(worker, config, upload_passes)` расширяет `JsonPassCheckpointStore`: перед первой загрузкой восстанавливает commits текущего плана из S3; `clear()` при force инвалидирует выбранные remote markers. Подробная [схема публикации](../guide/data-flow.md).

## API ядра и checkpoints

```python
Pipeline(
    passes,
    finalizers=None,
    observers=None,
    checkpoint_store=None,
    force=False,
    resume_by_id=False,
)
```

Опции после `passes` — keyword-only. `prepare(initial_artifacts=frozenset()) -> ExecutionPlan`; `run(context=None) -> PipelineOutcome`. Ошибка паса повторно выбрасывается после finalizers; ошибка только finalizers — `PipelineFinalizationError`. Events/observer ошибки best-effort и не ломают computation сами по себе.

Default `resume_by_id=False` восстанавливает completed prefix по порядку/контракту. В AWS включён `True`: восстановление по ID требует `validate_checkpoint`; пасы без валидатора выполняются заново. `force=True` сбрасывает checkpoints.

`JsonPassCheckpointStore(path, experiment_name)` атомарно сохраняет `.tmp` → replace. API: `load()`, `retain(pass_ids)`, `complete(pass_id,result,duration_seconds)`, `clear()`. Checkpoint включает ID, PassResult, duration и timestamp; не является checkpoint оптимизатора Nerfstudio. Незавершённый training pass при рестарте wrapper запускается заново.

## Пример нового паса

Пример проверяет только API ядра и не требует GPU/AWS:

```python
from recon_pipeline.core import Pipeline, PipelineContext, PassResult

class LabelPass:
    id = "label"
    name = "Label input"
    requires = frozenset({"input.name"})
    provides = frozenset({"output.label"})

    def run(self, context):
        name = context.require("input.name")
        context.report_progress(1.0, "Label ready")
        return PassResult(artifacts={"output.label": f"Run: {name}"})

context = PipelineContext(artifacts={"input.name": "demo"})
pipeline = Pipeline([LabelPass()])
pipeline.prepare(set(context.artifacts))
outcome = pipeline.run(context)
assert outcome.succeeded
assert context.require("output.label") == "Run: demo"
```

Для нового utility-backed паса вызывайте `core.utility.run_utility(module, arguments, context, *, runner=None, env=None, python=None)`, проверьте структуру result и верните стабильные keys. `env` передаётся subprocess runner как overrides; выбранный Python получает source package в PYTHONPATH.

## Локальная композиция существующих пасов

Loader local config использует настроенное `PipelineEnvironment`, но не AWS worker. Видео и модели должны уже быть на диске. Пример запускает inference + static export из local example, не включает S3/notifications/shutdown:

```python
from pathlib import Path
from recon_pipeline.core import Pipeline, PipelineContext
from recon_pipeline.datasets.fourdanyone.config import load_pipeline_config
from recon_pipeline.datasets.fourdanyone.passes import (
    PrepareExperimentPass, FourDAnyoneInferencePass,
)
from recon_pipeline.reconstructions.nerfstudio.passes import NerfstudioExportPass

config = load_pipeline_config(Path("config/local-run.example.json"))
config.validate_paths()
context = PipelineContext(artifacts={
    "input.video": config.video_path,
    "models.cache": config.model_dir,
})
pipeline = Pipeline([
    PrepareExperimentPass(config),
    FourDAnyoneInferencePass(config),
    NerfstudioExportPass(config),
])
pipeline.run(context)
```

Пример явно задаёт стадии, не выбирает их автоматически по flags. Для export-only состава вместо inference добавьте готовый `experiment.4danyone:<source>` в initial artifacts. Для production AWS используйте `build_aws_pipeline` и worker CLI.

## Observers и finalizers

`PipelineObserver` имеет `start(context)`, `handle(event,context)`, `flush()`, `close()`. `QueuedPipelineObserver(thread_name=...)` обрабатывает effects через ordered background queue. Lifecycle: pipeline start → pass started/progress/completed/skipped/failed → pipeline success/failure → finalizer events → pipeline finalized. Перед finalizers и после финализации event queue flush выполняется явно.

Finalizer API: `id`, `name`, `run(context, outcome) -> None`. В штатном AWS составе:

| Finalizer | Constructor | ID | Действие |
| --- | --- | --- | --- |
| `S3DiagnosticsFinalizer` в `workers.aws.persistence` | `(persistence)` | `s3-save-diagnostics` | Завершить observer, сохранить report/status/logs до shutdown |
| `CloudWatchFlushFinalizer` в `workers.aws.cloudwatch` | `(session)` | `cloudwatch-flush` | Flush queued/spooled logs |
| `SageMakerShutdownFinalizer` в `workers.aws.finalizers.sagemaker_shutdown` | `(worker)` | `sagemaker-shutdown` | Применить policy, учесть cancellation и persistence failure |

Первые два добавляются только при включённых функциях. Finalizers — часть lifecycle, а не producer passes с artifacts.

Исходники: [core](https://github.com/vinneyto/4da_3dgs_pipeline/tree/main/src/recon_pipeline/core), [AWS composition](https://github.com/vinneyto/4da_3dgs_pipeline/blob/main/src/recon_pipeline/workers/aws/pipeline.py).
