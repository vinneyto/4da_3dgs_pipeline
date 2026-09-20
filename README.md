# 4DAnyone → 3DGS pipeline

Reproducible [4DAnyone](https://github.com/ant-research/4DAnyone) inference and synchronized-frame export to the Nerfstudio/3DGS format.

## Architecture

The repository deliberately separates the local pipeline from AWS orchestration.

| Component | Responsibility | AWS dependency |
|---|---|---|
| `FourDAnyonePipeline` | 4DAnyone inference and local Nerfstudio/3DGS export | None |
| `fourda-pipeline` | Blocking local CLI around `FourDAnyonePipeline` | None |
| `fourda-aws-worker` | AWS-specific detached execution, durable job status, S3 staging/result upload, SNS email, SageMaker App shutdown | Optional `[aws]` extra |

`fourda-pipeline` never imports `boto3`, reads environment variables, uploads files, or calls an AWS API. The same core can run on a workstation, another cloud provider, or inside a future managed SageMaker Job.

The source tree mirrors this boundary:

```text
src/fourda_pipeline/   # cloud-independent pipeline and blocking CLI
src/fourda_aws_worker/ # AWS worker, background jobs, S3, SNS, and SageMaker
```

`fourda-aws-worker` is currently a console-based worker that simulates a future managed SageMaker Job. It runs the core pipeline as a detached job inside a JupyterLab App and owns every AWS-side effect. A future RunPod or local worker can be added as another package without modifying `fourda_pipeline`.

## One JSON run document

All paths and run parameters are explicit in one JSON document. Neither CLI discovers paths through environment variables.

```bash
cp config/run.example.json config/run.json
```

Edit `config/run.json` before running anything. It has four sections:

- `environment`: installation paths and pinned dependency versions used by setup scripts;
- `pipeline`: cloud-independent camera, inference, and export parameters;
- `aws_worker`: S3 input, local persistent roots, background job identity, shutdown policy, output upload, SNS, and SageMaker App identity.

For an AWS run, `pipeline` deliberately contains no filesystem paths. The worker derives
`video_path`, `model_dir`, `runs_dir`, and `fourdanyone_root` from `aws_worker.s3_video_path`
and `aws_worker.local`, downloads the required S3 data, and constructs the complete
`FourDAnyoneConfig` immediately before execution.

The AWS example is not directly accepted by `fourda-pipeline`, because that blocking,
cloud-independent CLI does not download S3 inputs. A completely local run document may
omit `environment` and `aws_worker`, but its `pipeline` object must explicitly provide
`video_path`, `fourdanyone_root`, `model_dir`, and `runs_dir`.

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
| Exported frame | 60 |

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

## Blocking local run

```bash
cp config/local-run.example.json config/local-run.json
# Edit all absolute paths.
fourda-pipeline --config config/local-run.json
```

`config/local-run.json` must contain the complete local paths described above. This command
performs no AWS operations. It writes the selected synchronized moment to:

```text
<pipeline.runs_dir>/<pipeline.experiment_name>/nerfstudio/frame_060/
├── transforms.json
├── sparse_pcd.ply
├── images/
└── masks/
```

Set `pipeline.resume` to `true` to reuse successfully completed intermediate outputs. When it is `false`, existing output directories are protected against accidental overwrite.

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
  --frame 60 \
  --sns-topic-name cp-4da-pipeline-d9f856354df8 \
  --notification-email YOUR_EMAIL \
  --sagemaker-domain-id d-x1ij0jwvo44o \
  --sagemaker-space-name cp-4da-jupyter-d9f856354df8

python -m json.tool config/run.json
```

The bucket is inferred from a full `s3://` video URI. `job_id` defaults to the experiment
name, while region, persistent SageMaker paths, S3 prefixes, frame, FPS, seed, turbo mode,
and shutdown policy have the defaults shown by `fourda-worker-config --help`. Pass
`--force` to intentionally replace an existing document.

For a later three-layer run, choose a new experiment name and pass:

```bash
--layer-pitches -15 0 15
```

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

The job layer:

1. runs a fail-closed AWS health check before expensive GPU work;
2. downloads `aws_worker.s3_video_path` into `<data_root>/input/`, reusing a matching local file;
3. synchronizes S3 model objects into `<data_root>/models/`, reusing the persistent cache;
4. derives the full local pipeline configuration and validates its paths;
5. sends an SNS `job started` email only after staging succeeds;
6. starts the AWS-independent core pipeline;
7. records stage-based progress in `<data_root>/jobs/<job_id>/status.json`;
8. uploads the completed experiment to `s3://<bucket>/<runs_prefix>/<experiment_name>/` when `aws_worker.upload_results` is `true`;
9. sends an SNS success or failure email;
10. applies `aws_worker.shutdown_on` and optionally stops the SageMaker JupyterLab App.

The startup health check validates the current STS identity, bucket access, the exact S3
video object, model-prefix listing, result-prefix `PutObject`, a confirmed SNS subscription
for the configured email address, and the configured SageMaker App when automatic shutdown
is enabled. If any mandatory check, staging operation, path validation, or startup SNS
publish fails, inference does not start and the durable job status becomes `failed`.

Progress reflects native 4DAnyone stages, not an exact remaining-time estimate.

### Background execution limitation

The worker still runs inside the JupyterLab App. Stopping the App manually or through Idle Shutdown terminates the worker. Configure the idle timeout with sufficient margin or use `"shutdown_on": "success"`; the worker then stops the App after local output, S3 upload, status persistence, and notification.

Valid shutdown policies are:

- `never` — never stop the App automatically;
- `success` — stop only after a successful pipeline and upload;
- `always` — stop after either success or failure.

Shutdown uses SageMaker `DeleteApp`. It stops compute without deleting the Space or its persistent EBS volume.

## Amazon SNS email

The execution role must allow `sns:CreateTopic`, `sns:Subscribe`, `sns:Publish`, and
`sns:ListSubscriptionsByTopic`. Configure `aws_worker.sns_topic_name` and
`aws_worker.notification_email`, then run:

```bash
fourda-aws-worker configure-email --config config/run.json
```

Confirm the AWS `Subscription Confirmation` email before starting a job. The worker refuses
to start the pipeline while the configured subscription is missing or pending. No second
AWS configuration file is created; the run JSON remains the single source of truth.

Automatic shutdown additionally requires `sagemaker:DeleteApp`. Domain ID, Space name,
and App name are read from the `aws_worker.sagemaker_*` fields in the same JSON document.

## Tests

Tests do not run model inference and do not require a GPU:

```bash
python -m pip install -e '.[dev]'
pytest
```
