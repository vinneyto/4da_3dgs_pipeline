# 4DAnyone → 3DGS pipeline

Reproducible [4DAnyone](https://github.com/ant-research/4DAnyone) inference and synchronized-frame export to the Nerfstudio/3DGS format.

## Architecture

The repository separates the synchronous pipeline core, domain pass providers,
and AWS composition root.

| Component | Responsibility | AWS dependency |
|---|---|---|
| `fourda_pipeline.Pipeline` | Ordered pass execution, artifact validation, events, observers, and guaranteed finalizers | None |
| `fourda_4danyone` | 4DAnyone workspace and inference passes | None |
| `fourda_nerfstudio` / `fourda_rerun` | Independent artifact passes | None |
| `fourda-aws-worker` | AWS-specific detached execution, durable job status, S3 staging/result upload, optional email/Telegram notifications, SageMaker App shutdown | Optional `[aws]` extra |

The core never imports `boto3`, reads environment variables, uploads files, or
calls an AWS API. Passes run synchronously in an explicit order. Observers react
to lifecycle events for console output, durable status, Telegram/email, and
resource monitoring, but they never schedule the next pass. Finalizers always
run after success or failure.

The source tree mirrors this boundary:

```text
src/fourda_pipeline/   # cloud-independent Pipeline, pass contracts, events
src/fourda_4danyone/   # 4DAnyone pass provider
src/fourda_nerfstudio/ # synchronized-frame artifact pass
src/fourda_rerun/      # interactive recording artifact pass
src/fourda_aws_worker/ # AWS passes, observers, builder, and detached jobs
```

`fourda-aws-worker` is currently a console-based worker that simulates a future managed SageMaker Job. It runs the core pipeline as a detached job inside a JupyterLab App and owns every AWS-side effect. A future RunPod or local worker can be added as another package without modifying `fourda_pipeline`.

## One JSON run document

All paths and run parameters are explicit in one JSON document. Neither CLI discovers paths through environment variables.

```bash
cp config/run.example.json config/run.json
```

Edit `config/run.json` before running anything. It has four sections:

- `environment`: installation paths and pinned dependency versions used by setup scripts;
- `pipeline`: cloud-independent processing stages;
- `artifacts`: independently enabled Nerfstudio and Rerun outputs;
- `aws_worker`: S3 input, local persistent roots, background job identity, shutdown policy, output upload, SNS, and SageMaker App identity.

For an AWS run, `pipeline` deliberately contains no filesystem paths. The worker derives
`video_path` from `aws_worker.bucket` and the remaining local paths from `aws_worker.local`,
downloads the required S3 data, and constructs the complete
`FourDAnyoneConfig` immediately before execution.

The run document is currently assembled only by `fourda-aws-worker`. A local or
RunPod composition root can later reuse the same core and domain passes without
bringing in AWS dependencies.

All filesystem paths must be absolute. This makes a run document self-contained and prevents hidden differences between shells, `.bashrc` files, notebooks, and background workers.

The checked-in [`config/run.example.json`](config/run.example.json) reproduces the validated camera setup:

| Parameter | Value |
|---|---:|
| `views_per_layer` | 24 |
| `layer_pitches` | `[0]` |
| Total cameras | 24 |
| `start_yaw` | 0° |
| `yaw_span` | 360° |
| `target_fps` | 30 |
| `seed` | 42 |
| Model | turbo |
| Nerfstudio artifact frames | `[60]` |

`artifacts.dataset.nerfstudio.frames` is always an array. It defaults to the
single synchronized frame `[60]`, but already supports batch export such as
`[30, 60, 90]`. The future 3DGS reconstruction stage will consume these
per-frame datasets as a batch instead of introducing a separate singular-frame
setting.

`RERUN_VIEW_COUNT=4` from the Colab notebook belongs to downstream visualization. It is not a 4DAnyone inference or export parameter.

## Persistent SageMaker Space layout

The example configuration uses:

```text
/home/sagemaker-user/work/4DAnyone/              upstream repository
/home/sagemaker-user/work/4da_3dgs_pipeline/     this repository
/home/sagemaker-user/4danyone-data/
├── input/
├── models/
├── runs/
└── jobs/                                        job requests, status, and logs
```

These directories reside on the Space's persistent EBS volume and survive JupyterLab App restarts. GPU instance charges apply only while the App is running; EBS storage charges continue while it is stopped.

## Environment setup

```bash
cd /home/sagemaker-user/work
git clone https://github.com/vinneyto/4da_3dgs_pipeline.git
cd 4da_3dgs_pipeline

cp config/run.example.json config/run.json
# Edit every REPLACE_* value and verify all absolute paths.

./scripts/setup_4danyone_env.sh config/run.json
```

The setup script reads every path and version from the JSON document. It creates the conda environment, clones the pinned upstream revision when needed, installs a compatible CUDA PyTorch/Torchvision pair, switches to headless OpenCV, installs FFmpeg, and installs this project with AWS job support.

It is safe to run on a CPU instance. CUDA-enabled packages are installed there and validated later when a GPU is present.

For a local installation that does not need AWS:

```bash
python -m pip install -e .
```

## Model download

Place the licensed SMPL-X archive at the bucket and prefix specified in the JSON document. With the example layout, the object is:

```text
s3://<aws_worker.bucket>/<aws_worker.models_prefix>/smplx/models_smplx_v1_1.zip
```

Then run:

```bash
./scripts/download_4danyone_models.sh config/run.json
```

This CPU-safe script reads the conda, repository, model, region, bucket, and prefix values from JSON. It synchronizes existing model assets from S3, installs SMPL-X, downloads missing upstream assets, and validates the final model set.

This remains the one-time model installation step. At job startup the worker also syncs
objects under `aws_worker.models_prefix`, but it reuses matching files on the persistent
EBS volume and does not reinstall Python dependencies or recreate external checkpoints.

With `artifacts.dataset.nerfstudio.enabled=true`, the Nerfstudio pass writes each
selected synchronized moment to:

```text
<pipeline.runs_dir>/<pipeline.experiment_name>/nerfstudio/frame_060/
├── transforms.json
├── sparse_pcd.ply
├── images/
└── masks/
```

Set `pipeline.dataset.config.resume` to `true` to reuse completed 4DAnyone inference.
Nerfstudio export is controlled separately by `artifacts.dataset.nerfstudio`; set
`replace_existing=true` only when an existing static dataset should be rebuilt.

## Generate an AWS run document

`fourda-worker-config` turns CLI arguments into the validated JSON document consumed by
the detached worker. The JSON remains the durable, inspectable handoff: generating it does
not call AWS or start a job.

Generate the one-layer, 24-view smoke run:

```bash
fourda-worker-config \
  --output config/run.json \
  --experiment-name leo_one_layer_24views_01 \
  --s3-video-path s3://cp-4da-d9f856354df8/input/leo.MOV \
  --views-per-layer 24 \
  --layer-pitches 0 \
  --nerfstudio-frames 60 \
  --telegram-chat-id YOUR_TELEGRAM_CHAT_ID \
  --telegram-bot-token-env CP_4DA_TELEGRAM_BOT_TOKEN \
  --sagemaker-domain-id d-x1ij0jwvo44o \
  --sagemaker-space-name cp-4da-jupyter-d9f856354df8

python -m json.tool config/run.json
```

The bucket is inferred from a full `s3://` video URI. `job_id` defaults to the experiment
name, while region, persistent SageMaker paths, S3 prefixes, artifact frames, FPS, seed, turbo mode,
and shutdown policy have the defaults shown by `fourda-worker-config --help`. Pass
`--force` to intentionally replace an existing document.

For a later three-layer run, choose a new experiment name and pass:

```bash
--layer-pitches -15 0 15
```

Nerfstudio export is not part of the expensive 4DAnyone stage. To create new
temporal slices from a completed experiment without rerunning inference:

```bash
fourda-worker-config \
  --output config/export-more-frames.json \
  --experiment-name leo_static_exports_v2 \
  --no-dataset \
  --nerfstudio \
  --nerfstudio-source-experiment-name leo_three_layers_72views_01 \
  --nerfstudio-frames 30 60 90 \
  --nerfstudio-device cpu \
  --bucket cp-4da-d9f856354df8 \
  --video leo.MOV \
  --sagemaker-domain-id d-x1ij0jwvo44o \
  --sagemaker-space-name cp-4da-jupyter-d9f856354df8
```

The AWS worker restores the source experiment from S3 only when it is absent from
the persistent volume. Rerun and Nerfstudio artifacts can share that restored source.

## AWS-aware background job

Start the detached worker:

```bash
fourda-aws-worker start --config config/run.json
```

After it starts, the terminal, VS Code, and browser tab may be closed. The process continues inside the running JupyterLab App.

```bash
fourda-aws-worker status --config config/run.json
fourda-aws-worker status --config config/run.json --json
fourda-aws-worker logs --config config/run.json --lines 200
fourda-aws-worker logs --config config/run.json --follow
fourda-aws-worker stop --config config/run.json
```

Inspect the exact ordered plan and its artifact contracts without calling AWS:

```bash
fourda-aws-worker plan --config config/run.json
```

The AWS composition root builds synchronous passes for:

1. running a fail-closed AWS health check before expensive GPU work;
2. downloading the configured bucket video into `<data_root>/input/` when the dataset stage is enabled, reusing a matching local file;
3. synchronizing S3 model objects into `<data_root>/models/`, reusing the persistent cache;
4. restoring any prior source experiment required by artifact-only work;
5. preparing the experiment workspace;
6. running 4DAnyone inference when enabled;
7. exporting enabled Nerfstudio and Rerun artifacts;
8. writing the run manifest and optionally uploading results;
9. applying the SageMaker shutdown policy in a guaranteed finalizer.

Observers record pass progress in `<data_root>/jobs/<job_id>/status.json` and send
optional email/Telegram messages when the preflight, each pass, or the pipeline
completes or fails. Observers never schedule passes. The core Pipeline invokes the
next pass directly and runs all finalizers after either success or failure.

The startup health check validates the current STS identity, bucket access, the exact S3
video object, model-prefix listing, result-prefix `PutObject`, optional notification channels,
and the configured SageMaker App when automatic shutdown is enabled. Notification failures
are recorded as warnings and never block inference. Any mandatory AWS check, staging
operation, or path validation failure prevents inference and marks the durable job failed.

Progress reflects native 4DAnyone stages, not an exact remaining-time estimate.

### Background execution limitation

The worker still runs inside the JupyterLab App. Stopping the App manually or through Idle Shutdown terminates the worker. Configure the idle timeout with sufficient margin or use `"shutdown_on": "success"`; the worker then stops the App after local output, S3 upload, status persistence, and notification.

Valid shutdown policies are:

- `never` — never stop the App automatically;
- `success` — stop only after a successful pipeline and upload;
- `failure` — stop only after a failed pipeline;
- `always` — stop after either success or failure.

Shutdown uses SageMaker `DeleteApp`. It stops compute without deleting the Space or its persistent EBS volume.

## Notifications

Both notification channels are optional and independent. If neither is configured, the
worker still runs normally. A missing, deleted, or pending email subscription and a broken
Telegram configuration are reported in the job log but do not block GPU work.

### Telegram

Create a bot with `@BotFather`, send the bot one message, and obtain the destination
`chat_id`. Keep the token out of JSON and Git:

```bash
export CP_4DA_TELEGRAM_BOT_TOKEN="REPLACE_WITH_BOT_TOKEN"
```

Persist that export in the SageMaker Space's `~/.bashrc`, then configure:

```json
"notifications": {
  "email": null,
  "telegram": {
    "enabled": true,
    "chat_id": "REPLACE_WITH_CHAT_ID",
    "bot_token_env": "CP_4DA_TELEGRAM_BOT_TOKEN"
  }
}
```

The worker validates the bot and chat with `getChat` during its health check and uses
`sendMessage` for start, success, and failure messages. A literal `bot_token` is also
accepted instead of `bot_token_env`, but it writes the secret into the run document and
detached job request and is therefore not recommended.

### Amazon SNS email

The execution role must allow `sns:CreateTopic`, `sns:Subscribe`, `sns:Publish`, and
`sns:ListSubscriptionsByTopic`. Configure the optional email channel, then run:

```json
"email": {
  "enabled": true,
  "topic_name": "cp-4da-pipeline-DEPLOYMENT_ID",
  "email": "you@example.com"
}
```

```bash
fourda-aws-worker configure-email --config config/run.json
```

Confirm the AWS `Subscription Confirmation` email to receive messages. The worker continues
when that subscription is missing, deleted, or pending. No second AWS configuration file is
created; the run JSON remains the single source of truth.

Automatic shutdown additionally requires `sagemaker:DeleteApp`. Domain ID, Space name,
and App name are read from the `aws_worker.sagemaker_*` fields in the same JSON document.

## Tests

Tests do not run model inference and do not require a GPU:

```bash
python -m pip install -e '.[dev]'
pytest
```
