# 4DAnyone → 3DGS pipeline

Python orchestration for reproducible [4DAnyone](https://github.com/ant-research/4DAnyone) runs in AWS SageMaker Studio and export of one synchronized moment to the Nerfstudio/3DGS format.

The current pipeline:

1. accepts one monocular input video;
2. runs 4DAnyone with a configurable virtual-camera layout;
3. exports a selected frame to `transforms.json`, images, masks, and a point cloud;
4. optionally synchronizes the result to S3;
5. supports detached execution with durable status, logs, SNS email notifications, and optional JupyterLab App shutdown.

> This version prepares a **static** 3DGS dataset for a selected moment in time. Exporting all 121 temporal frames and training a dynamic 4DGS model are outside the current scope.

## Default camera configuration

The defaults reproduce the validated Colab run:

| Parameter | Value |
|---|---:|
| `views_per_layer` | 24 |
| `layer_pitches` | `-15, 0, 15` |
| Total cameras | 72 |
| `start_yaw` | 0° |
| `yaw_span` | 360° |
| `target_fps` | 30 |
| `seed` | 42 |
| Model | turbo |
| Exported frame | 60 |

`RERUN_VIEW_COUNT=4` from the Colab notebook is not a 4DAnyone inference or export parameter. It belongs to downstream visualization and is therefore not part of this CLI.

## Persistent Space layout

```text
~/work/4DAnyone/              upstream 4DAnyone repository
~/work/4da_3dgs_pipeline/     this repository
~/4danyone-data/
├── input/
├── models/
├── runs/
└── jobs/                     background requests, status files, and logs
```

These directories reside on the Space's persistent 50 GB EBS volume and survive JupyterLab App restarts. GPU instance charges apply only while the App is running; EBS storage charges continue while the App is stopped.

## Environment setup

```bash
cd "$HOME/work"
git clone https://github.com/vinneyto/4da_3dgs_pipeline.git
cd 4da_3dgs_pipeline

chmod +x scripts/*.sh
./scripts/setup_4danyone_env.sh
```

The script creates an isolated `$HOME/.conda/envs/4danyone` environment, installs a compatible CUDA PyTorch/Torchvision pair, replaces GUI OpenCV with its headless build, installs FFmpeg, and installs this CLI. It is safe to run on a CPU instance: the CUDA build is installed there and validated later on a GPU instance. On a GPU instance, the script also performs a real CUDA operation.

Recommended `~/.bashrc` settings:

```bash
export CP_4DA_ENV="$HOME/.conda/envs/4danyone"
export CP_4DA_REPO_ROOT="$HOME/work/4DAnyone"
export CP_4DA_DATA_ROOT="$HOME/4danyone-data"
export CP_4DA_MODEL_DIR="$CP_4DA_DATA_ROOT/models"
export CP_4DA_INPUT_DIR="$CP_4DA_DATA_ROOT/input"
export CP_4DA_JOBS_DIR="$CP_4DA_DATA_ROOT/jobs"
export PYTHONNOUSERSITE=1

source /opt/conda/etc/profile.d/conda.sh
conda activate "$CP_4DA_ENV"
```

## Model download

First, place the licensed SMPL-X archive at:

```text
s3://${CP_4DA_BUCKET}/models/smplx/models_smplx_v1_1.zip
```

Then run:

```bash
cd "$HOME/work/4da_3dgs_pipeline"
./scripts/download_4danyone_models.sh
```

This stage is CPU-safe as well. The script synchronizes existing files from S3, installs SMPL-X, and downloads any missing 4DAnyone, GVHMR, VGG-19, and BiRefNet assets.

## Blocking run

Use this mode first to validate the configuration. The terminal remains attached until the run completes.

```bash
fourda-pipeline \
  --video "$CP_4DA_INPUT_DIR/leo.MOV" \
  --experiment-name leon_video_72views_01 \
  --views-per-layer 24 \
  --layer-pitches=-15,0,15 \
  --start-yaw 0 \
  --yaw-span 360 \
  --target-fps 30 \
  --seed 42 \
  --frame-indices 60
```

When `CP_4DA_BUCKET` is set, the result is automatically uploaded to:

```text
s3://${CP_4DA_BUCKET}/runs/leon_video_72views_01/
```

Use `--no-s3-upload` to disable the upload or `--s3-output-uri` to provide another destination.

The exported moment is written to:

```text
~/4danyone-data/runs/leon_video_72views_01/nerfstudio/frame_060/
├── transforms.json
├── sparse_pcd.ply
├── images/
└── masks/
```

`--resume` reuses successfully completed intermediate outputs. Without it, an existing output directory is protected against accidental overwrite.

## Background run and status

```bash
fourda-job start \
  --video "$CP_4DA_INPUT_DIR/leo.MOV" \
  --experiment-name leon_video_72views_01 \
  --views-per-layer 24 \
  --layer-pitches=-15,0,15 \
  --start-yaw 0 \
  --yaw-span 360 \
  --target-fps 30 \
  --seed 42 \
  --frame-indices 60 \
  --shutdown-on success
```

After `start`, the terminal, VS Code, and browser tab may be closed. A detached worker continues to run inside the JupyterLab App.

```bash
fourda-job status leon_video_72views_01
fourda-job status leon_video_72views_01 --json
fourda-job logs leon_video_72views_01 --lines 200
fourda-job logs leon_video_72views_01 --follow
fourda-job stop leon_video_72views_01
```

Progress is derived from native 4DAnyone stages. It indicates the current stage rather than an exact estimate of remaining time. Status is written atomically to `~/4danyone-data/jobs/<job-id>/status.json`.

### Background execution limitation

The worker runs **inside the SageMaker JupyterLab App**. If the App is stopped manually or by Idle Shutdown before completion, the process terminates. For a long run, configure an idle timeout with sufficient margin or use `--shutdown-on success`; the App will then stop itself immediately after the result, S3 upload, and email notification are complete.

Shutdown policies:

- `never` — never stop the App automatically;
- `success` — stop only after a successful run;
- `always` — stop after either success or failure.

Shutdown uses SageMaker `DeleteApp`. It stops GPU compute without deleting the Space or its persistent EBS volume.

## Amazon SNS email notifications

The execution role must allow `sns:CreateTopic`, `sns:Subscribe`, and `sns:Publish`. After granting those permissions, configure email from inside the Space:

```bash
fourda-job configure-email --email you@example.com
```

AWS sends a `Subscription Confirmation` email. Confirm it before relying on notifications. Configuration is stored in `~/.config/4da-3dgs-pipeline/aws.json`. Background jobs then send an email after success or failure and before any automatic App shutdown.

`--shutdown-on` also requires `sagemaker:DeleteApp` and the following environment variables:

```bash
export CP_SM_DOMAIN_ID="d-..."
export CP_SM_SPACE_NAME="cp-4da-jupyter-${CP_DEPLOYMENT_ID}"
export CP_SM_JUPYTER_APP_NAME="default"
export CP_AWS_REGION="us-east-1"
```

## Tests

The tests do not run model inference and do not require a GPU:

```bash
python -m pip install -e '.[dev]'
pytest
```
