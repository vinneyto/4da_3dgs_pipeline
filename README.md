# Reconstruction pipeline CLI

## Check an existing machine first

After updating the repository, run this without changing your environment:

```bash
./scripts/check_environment.sh --require-cuda 2>&1 | tee environment-check.txt
# Or a structured report; exit 1 means at least one failed check.
./scripts/check_environment.sh --json --require-cuda > environment-check.json
```

The checker needs only system Python. It does not install packages, change env
variables, create workspace directories, call AWS, download weights or compile
CUDA extensions. It reports every missing/invalid `RECON_*` variable, repository
and GVHMR submodule, workspace folder, package version, executable, model asset,
CUDA compiler and PyTorch GPU access. No credentials or unrelated env variables
are included in the report. Without `--require-cuda`, absence of a GPU is a
warning, allowing preparation on a CPU instance; broken imports still fail.

## Install or update the machine environment

```bash
./scripts/setup_environment.sh --config config/run.example.json
source "$HOME/.config/recon-pipeline/environment.sh"
```

Edit the example's bucket and SageMaker identifiers before model download. This
script installs Conda if needed, checks out the pinned 4DAnyone repository and
GVHMR, creates input/models/runs/jobs/environment folders and installs three
separate persistent environments: 4DAnyone + AWS worker, Nerfstudio/Splatfacto +
CUDA toolkit/compiler, and Splatfacto Rerun. It also downloads model weights when
`--config` is supplied. Licensed SMPL-X assets must already be provided in your
bucket; the installer cannot obtain that license for you.

Existing exported `RECON_*` values take precedence over the saved env file and
installation defaults. Variables are saved to
`~/.config/recon-pipeline/environment.sh` and loaded from Bash startup files.
Use `--env-file FILE` (or `RECON_ENV_FILE`) to choose another file and source it
in the current terminal. Defaults are based on the current user's home and the
repository location, without a hard-coded SageMaker username.

Configure-only preserves the existing `CUDA_HOME`, or detects a toolkit already
on `PATH`. The checker prints the detected CUDA version even before
`RECON_CUDA_VERSION` has been exported.

To fill/persist variables on an existing machine without reinstalling anything:

```bash
./scripts/setup_environment.sh --configure-only
source "$HOME/.config/recon-pipeline/environment.sh"
./scripts/check_environment.sh --require-cuda
```

Inspect missing settings before using configure-only: it does not discover old
custom paths; export those explicitly first. Full setup is for Linux x86_64.
Tool installation can run without `--config`; in that case model download is a
separate step and the final checker reports any missing assets.

If the checker confirms that 4DAnyone already runs with CUDA and model assets
are present, reuse that environment while installing the missing tools:

```bash
./scripts/setup_environment.sh --reuse-4danyone
source "$HOME/.config/recon-pipeline/environment.sh"
./scripts/check_environment.sh --require-cuda
```

This mode installs the pipeline's AWS/Rerun worker extras (including `telebot`),
writes the package lock and installs Splatfacto and its Rerun environment. It
skips the 4DAnyone repository setup, requirements and PyTorch installation. Full
installation selects and persists the Splatfacto toolkit in `CUDA_HOME`.

Installation writes its complete output to a timestamped `environment-setup-*.log`
in the pipeline checkout. Each stage is labeled; a failure prints the stage,
exit status and log path. Use `--log-file FILE` to choose a path. To diagnose a
partially installed machine, rerun setup and share the end of that log:

```bash
./scripts/setup_environment.sh --reuse-4danyone --log-file environment-setup.log
tail -n 100 environment-setup.log
```

Existing Conda prefixes are reused. The checker reports missing packages even
when an environment contains only Python; its output does not contain the
installer's failure reason. Configure-only does not create an installation log.

Setup installs EGL/OpenGL and libusb libraries inside the Splatfacto Conda prefix. The
checker imports Open3D and creates an empty point cloud to catch missing native
libraries before training (including `libEGL.so.1` on headless SageMaker images).
To repair an existing installation without rerunning the full setup:

```bash
conda install --prefix "$(dirname "$RECON_NERFSTUDIO_BIN")" \
  -c conda-forge libegl libgl libusb -y
./scripts/check_environment.sh --require-cuda
```

Restart a failed worker with the same config and without `--force` to reuse
completed pass checkpoints, including the exported Nerfstudio datasets.

## Environment boundary

`PipelineEnvironment` (`src/recon_pipeline/environment.py`) reads machine
settings from environment variables at the orchestration boundary. Runtime
loading has no path or version defaults and lists missing variables together.
Only installation supplies defaults. `RECON_NS_BIN`, used in the previous
standalone instructions, is accepted as an alias for `RECON_NERFSTUDIO_BIN`.

| Machine setting | Environment variable |
| --- | --- |
| Conda initialization | `RECON_CONDA_BOOTSTRAP` |
| 4DAnyone/worker Conda prefix | `RECON_CONDA_ENV` |
| Pipeline checkout | `RECON_PIPELINE_ROOT` |
| 4DAnyone checkout | `RECON_FOURDANYONE_ROOT` |
| Workspace root | `RECON_DATA_ROOT` |
| 4DAnyone repository URL and revision | `RECON_FOURDANYONE_GIT_URL`, `RECON_FOURDANYONE_GIT_REF` |
| Python version | `RECON_PYTHON_VERSION` |
| 4DAnyone PyTorch and Torchvision | `RECON_TORCH_VERSION`, `RECON_TORCHVISION_VERSION` |
| 4DAnyone wheel index | `RECON_TORCH_INDEX_URL` |
| OpenCV fallback version | `RECON_OPENCV_FALLBACK_VERSION` |
| 4DAnyone package lock | `RECON_LOCK_FILE` |
| Splatfacto executables | `RECON_NERFSTUDIO_BIN` |
| Splatfacto Rerun interpreter | `RECON_SPLATFACTO_RERUN_PYTHON` |
| Splatfacto PyTorch and Torchvision | `RECON_SPLATFACTO_TORCH_VERSION`, `RECON_SPLATFACTO_TORCHVISION_VERSION` |
| Splatfacto wheel index | `RECON_SPLATFACTO_TORCH_INDEX_URL` |
| Nerfstudio, gsplat, CUDA toolkit versions | `RECON_NERFSTUDIO_VERSION`, `RECON_GSPLAT_VERSION`, `RECON_CUDA_VERSION` |

The setup file also sets `PATH`, `CUDA_HOME`, `CXX` and `LD_LIBRARY_PATH` for the
installed tools. When Conda CUDA headers exist under
`$CUDA_HOME/targets/x86_64-linux/include`, it also prepends that directory to
`CPATH` for gsplat's C++ compilation, preserving existing entries without
duplicating the directory on repeated sourcing. Use `--configure-only` followed
by sourcing the env file to refresh these settings without reinstalling.
Run JSON schema v7 contains only pipeline/artifact settings and
high-level AWS settings (bucket, region, prefixes, notifications, SageMaker).
It rejects `environment`, `aws_worker.local`, `nerfstudio_bin`, interpreter and
workspace paths. Local inputs use `pipeline.dataset.config.video`, relative to
`RECON_DATA_ROOT/input`. Older run documents can still be loaded, but their
machine settings are ignored in favor of env. Cloning a v6 template removes them
and writes v7; the source JSON remains intact.

Utilities remain independent: paths, tool binaries and operation parameters
are explicit CLI arguments. They do not import `PipelineEnvironment` or read
`RECON_*` settings. Passes supply those arguments and select the env interpreter.
Generating/validating run JSON with `recon-config` needs no configured machine,
Conda installation or GPU.

## Clone an existing run configuration

Use a validated JSON document as a template with a new run identity. Omit
`--video` to preserve the input. Pipeline, artifact, AWS, notification and shutdown
settings are preserved unless explicitly overridden. Artifact source references that pointed at the template
experiment are updated to the new experiment automatically.

```bash
export RECON_TEMPLATE_CONFIG="$RECON_PIPELINE_ROOT/config/leo-three-layers-rerun.json"
export RECON_RUN_CONFIG="$RECON_PIPELINE_ROOT/config/alex-three-layers-rerun.json"
export RECON_EXPERIMENT_NAME="alex_three_layers_72views_01"
export RECON_VIDEO="alex.MOV"

recon-config \
  --template "$RECON_TEMPLATE_CONFIG" \
  --output "$RECON_RUN_CONFIG" \
  --experiment-name "$RECON_EXPERIMENT_NAME" \
  --video "$RECON_VIDEO"
```

Pass `--force` only when the output file may be replaced deliberately. Use
`--job-id` to override the default new job ID or `--bucket` to move the cloned
run to another bucket.

Template copies accept `--dataset`/`--no-dataset`,
`--reconstruction`/`--no-reconstruction`, `--reconstruction-frames`,
`--nerfstudio`/`--no-nerfstudio`, `--nerfstudio-frames`,
`--nerfstudio-source-experiment-name`, and `--rerun`/`--no-rerun`.
To reconstruct every frame from an existing 4DAnyone experiment:

```bash
recon-config \
  --template config/leo-three-layers-rerun.json \
  --output config/leo-three-layers-splatfacto-all.json \
  --experiment-name leo_three_layers_72views_splatfacto_all_01 \
  --no-dataset --reconstruction --nerfstudio \
  --nerfstudio-frames $(seq 0 120) \
  --nerfstudio-source-experiment-name leo_three_layers_72views_01 \
  --no-rerun
```

`--reconstruction-frames` preserves an existing selection when omitted. A null
selection in the source config trains all exported frames. Other generator
options (such as training profile flags) do not override template settings.

## Configure a three-layer AWS run

Define every value used to generate the run document:

```bash
# Run identity
export RECON_EXPERIMENT_NAME="leo_three_layers_72views_01"
export RECON_JOB_ID="$RECON_EXPERIMENT_NAME"
export RECON_RUN_CONFIG="$RECON_PIPELINE_ROOT/config/leo-three-layers-rerun.json"

# 4DAnyone dataset stage
export RECON_VIDEO="leo.MOV"
export RECON_VIEWS_PER_LAYER="24"
export RECON_LAYER_PITCHES=(-15 0 15)
export RECON_START_YAW="0"
export RECON_YAW_SPAN="360"
export RECON_TARGET_FPS="30"
export RECON_SEED="42"
export RECON_ATTENTION_BACKEND="auto"

# Dataset artifacts
export RECON_NERFSTUDIO_FRAMES=(60)
export RECON_NERFSTUDIO_DEVICE="cuda:0"
export RECON_RERUN_VIEW_COUNT="4"
export RECON_RERUN_DEVICE="auto"

# S3 layout
export RECON_INPUT_PREFIX="input"
export RECON_MODELS_PREFIX="models"
export RECON_RUNS_PREFIX="runs"

# Notifications and shutdown
export RECON_TELEGRAM_BOT_TOKEN_ENV="CP_4DA_TELEGRAM_BOT_TOKEN"
export RECON_SHUTDOWN_ON="always"
: "${CP_4DA_TELEGRAM_CHAT_ID:?CP_4DA_TELEGRAM_CHAT_ID is required}"
export CP_4DA_TELEGRAM_ALLOWED_USER_ID="$CP_4DA_TELEGRAM_CHAT_ID"
```

The existing deployment variables must also be available:

```bash
: "${CP_4DA_BUCKET:?CP_4DA_BUCKET is required}"
: "${CP_AWS_REGION:?CP_AWS_REGION is required}"
: "${CP_SM_DOMAIN_ID:?CP_SM_DOMAIN_ID is required}"
: "${CP_SM_SPACE_NAME:?CP_SM_SPACE_NAME is required}"
: "${CP_SM_JUPYTER_APP_NAME:?CP_SM_JUPYTER_APP_NAME is required}"
: "${CP_4DA_TELEGRAM_BOT_TOKEN:?CP_4DA_TELEGRAM_BOT_TOKEN is required}"
```

Generate the complete JSON document without relying on configurable CLI defaults:

```bash
recon-config \
  --output "$RECON_RUN_CONFIG" \
  --experiment-name "$RECON_EXPERIMENT_NAME" \
  --job-id "$RECON_JOB_ID" \
  --dataset \
  --views-per-layer "$RECON_VIEWS_PER_LAYER" \
  --layer-pitches "${RECON_LAYER_PITCHES[@]}" \
  --start-yaw "$RECON_START_YAW" \
  --yaw-span "$RECON_YAW_SPAN" \
  --target-fps "$RECON_TARGET_FPS" \
  --seed "$RECON_SEED" \
  --turbo \
  --attention-backend "$RECON_ATTENTION_BACKEND" \
  --no-resume \
  --nerfstudio \
  --nerfstudio-frames "${RECON_NERFSTUDIO_FRAMES[@]}" \
  --nerfstudio-device "$RECON_NERFSTUDIO_DEVICE" \
  --nerfstudio-source-experiment-name "$RECON_EXPERIMENT_NAME" \
  --no-nerfstudio-replace-existing \
  --rerun \
  --rerun-view-count "$RECON_RERUN_VIEW_COUNT" \
  --rerun-device "$RECON_RERUN_DEVICE" \
  --rerun-source-experiment-name "$RECON_EXPERIMENT_NAME" \
  --no-rerun-replace-existing \
  --bucket "$CP_4DA_BUCKET" \
  --video "$RECON_VIDEO" \
  --region "$CP_AWS_REGION" \
  --input-prefix "$RECON_INPUT_PREFIX" \
  --models-prefix "$RECON_MODELS_PREFIX" \
  --runs-prefix "$RECON_RUNS_PREFIX" \
  --sync-models \
  --upload-results \
  --shutdown-on "$RECON_SHUTDOWN_ON" \
  --no-email \
  --telegram \
  --telegram-chat-id "$CP_4DA_TELEGRAM_CHAT_ID" \
  --telegram-bot-token-env "$RECON_TELEGRAM_BOT_TOKEN_ENV" \
  --telegram-shutdown-command \
  --telegram-allowed-user-id "$CP_4DA_TELEGRAM_ALLOWED_USER_ID" \
  --sagemaker-domain-id "$CP_SM_DOMAIN_ID" \
  --sagemaker-space-name "$CP_SM_SPACE_NAME" \
  --sagemaker-app-name "$CP_SM_JUPYTER_APP_NAME"
```

Telegram notifications are event-driven: one startup plan, then pass start,
completion, or failure messages with a compact CPU/RAM/disk/GPU snapshot. There
is no periodic resource polling or live log streaming.

## Debug one operation at a time

All standalone utilities and their operation helpers live under
`src/recon_pipeline/utilities/`, grouped into `artefacts`, `datasets`,
`reconstructions`, `storage`, and `cloud`. Passes and worker code live outside
this directory.

Each utility is an independent Python program. It accepts only the parameters
of its operation, emits progress and a final result on stdout, writes diagnostics
to stderr, and returns a process exit code. It does not import worker or pipeline
configuration. Passes assemble arguments and interpret stdout; the worker owns
sequencing, checkpoints and notifications.

| Operation | Console command | Module under `recon_pipeline` |
| --- | --- | --- |
| Prepare workspace | `recon-prepare-experiment` | `utilities.datasets.fourdanyone.prepare_experiment` |
| Generate dataset | `recon-4danyone` | `utilities.datasets.fourdanyone.inference` |
| Prepare RGBA training copy | `recon-prepare-rgba` | `utilities.reconstructions.nerfstudio.prepare_rgba` |
| Train and export one frame | `recon-splatfacto` | `utilities.reconstructions.nerfstudio.splatfacto` |
| Record trained Gaussian scene | `recon-splatfacto-rerun` | `utilities.artefacts.rerun.splatfacto` |
| Export synchronized frames | `recon-nerfstudio-export` | `utilities.reconstructions.nerfstudio.export` |
| Export recording | `recon-rerun` | `utilities.artefacts.rerun.export` |
| Check resource access | `recon-aws-preflight` | `utilities.cloud.aws.preflight` |
| Download video | `recon-s3-download-input` | `utilities.storage.s3.download_input` |
| Sync model cache | `recon-s3-sync-models` | `utilities.storage.s3.sync_models` |
| Restore source experiment | `recon-s3-restore-experiment` | `utilities.storage.s3.restore_experiment` |
| Write manifest | `recon-write-run-manifest` | `utilities.artefacts.manifest.write` |
| Upload results | `recon-s3-upload-results` | `utilities.storage.s3.upload_results` |
| Commit one pass's artifacts | `recon-s3-publish-artifacts` | `utilities.storage.s3.publish_artifacts` |
| Restore committed pass results | `recon-s3-recover-run` | `utilities.storage.s3.recover_run` |
| Save status and diagnostic logs | `recon-s3-upload-diagnostics` | `utilities.storage.s3.upload_diagnostics` |

Each utility has its own file and can be run with a console command, `python -m`,
or the Python debugger:

```bash
recon-4danyone --help
python -m recon_pipeline.utilities.datasets.fourdanyone.inference --help
python -m pdb -m recon_pipeline.utilities.datasets.fourdanyone.inference --help
```

Run individual operations with explicit parameters:

```bash
export RECON_EXPERIMENT_DIR="$RECON_DATA_ROOT/runs/$RECON_EXPERIMENT_NAME"
export RECON_GENERATION_DIR="$RECON_EXPERIMENT_DIR/4danyone"
export RECON_VIDEO_PATH="$RECON_DATA_ROOT/input/$RECON_VIDEO"
export RECON_MODEL_DIR="$RECON_DATA_ROOT/models"
export RECON_RERUN_PATH="$RECON_EXPERIMENT_DIR/rerun/$RECON_EXPERIMENT_NAME.rrd"

recon-aws-preflight \
  --region "$CP_AWS_REGION" \
  --bucket "$CP_4DA_BUCKET" \
  --input-key "$RECON_INPUT_PREFIX/$RECON_VIDEO" \
  --models-prefix "$RECON_MODELS_PREFIX" \
  --write-prefix "$RECON_RUNS_PREFIX"

recon-s3-download-input \
  --region "$CP_AWS_REGION" \
  --bucket "$CP_4DA_BUCKET" \
  --key "$RECON_INPUT_PREFIX/$RECON_VIDEO" \
  --output "$RECON_VIDEO_PATH"

recon-s3-sync-models \
  --region "$CP_AWS_REGION" \
  --bucket "$CP_4DA_BUCKET" \
  --prefix "$RECON_MODELS_PREFIX" \
  --destination "$RECON_MODEL_DIR"

# Settings are an arbitrary JSON snapshot; preparation does not interpret them.
recon-prepare-experiment \
  --directory "$RECON_EXPERIMENT_DIR" \
  --settings '{"note": "standalone debugging"}'

recon-4danyone \
  --fourdanyone-root "$RECON_FOURDANYONE_ROOT" \
  --video "$RECON_VIDEO_PATH" \
  --output "$RECON_GENERATION_DIR" \
  --model-dir "$RECON_MODEL_DIR" \
  --views-per-layer "$RECON_VIEWS_PER_LAYER" \
  --layer-pitches "${RECON_LAYER_PITCHES[@]}" \
  --start-yaw "$RECON_START_YAW" \
  --yaw-span "$RECON_YAW_SPAN" \
  --target-fps "$RECON_TARGET_FPS" \
  --seed "$RECON_SEED" \
  --turbo \
  --attention-backend "$RECON_ATTENTION_BACKEND"

recon-nerfstudio-export \
  --fourdanyone-root "$RECON_FOURDANYONE_ROOT" \
  --generation "$RECON_GENERATION_DIR" \
  --output "$RECON_EXPERIMENT_DIR/nerfstudio" \
  --model-dir "$RECON_MODEL_DIR" \
  --frames "${RECON_NERFSTUDIO_FRAMES[@]}" \
  --device "$RECON_NERFSTUDIO_DEVICE"

recon-rerun \
  --generation "$RECON_GENERATION_DIR" \
  --output "$RECON_RERUN_PATH" \
  --experiment "$RECON_EXPERIMENT_NAME" \
  --fourdanyone-root "$RECON_FOURDANYONE_ROOT" \
  --model-dir "$RECON_MODEL_DIR" \
  --view-count "$RECON_RERUN_VIEW_COUNT" \
  --device "$RECON_RERUN_DEVICE"

# Restore a prior experiment instead of generating a new dataset.
export RECON_SOURCE_EXPERIMENT="leo_three_layers_72views_01"
recon-s3-restore-experiment \
  --region "$CP_AWS_REGION" \
  --bucket "$CP_4DA_BUCKET" \
  --prefix "$RECON_RUNS_PREFIX/$RECON_SOURCE_EXPERIMENT" \
  --destination "$RECON_DATA_ROOT/runs/$RECON_SOURCE_EXPERIMENT"

export RECON_TOTAL_VIEWS="$((RECON_VIEWS_PER_LAYER * ${#RECON_LAYER_PITCHES[@]}))"
recon-write-run-manifest \
  --output "$RECON_EXPERIMENT_DIR/pipeline-result.json" \
  --experiment-name "$RECON_EXPERIMENT_NAME" \
  --experiment-dir "$RECON_EXPERIMENT_DIR" \
  --inference-dir "$RECON_GENERATION_DIR" \
  --num-views "$RECON_TOTAL_VIEWS" \
  --rerun-file "$RECON_RERUN_PATH"

recon-s3-upload-results \
  --region "$CP_AWS_REGION" \
  --bucket "$CP_4DA_BUCKET" \
  --prefix "$RECON_RUNS_PREFIX/$RECON_EXPERIMENT_NAME" \
  --source "$RECON_EXPERIMENT_DIR"
```

Use `--replace-existing` to rebuild an existing inference directory, frame
dataset or recording. Downloads/restores using this flag discard their owned
local output. The standalone whole-directory `recon-s3-upload-results` command
deletes the selected S3 prefix when given this flag. The AWS worker instead
publishes individual committed artifact bundles and does not delete the run
prefix. Shared model caches are retained.

Stdout is newline-delimited JSON with `event: "progress"` and one final
`event: "result"`. Diagnostics from upstream libraries/exporters go to stderr.
Results describe the created artifacts; datasets and recordings remain ordinary
files. There are no temporary request/response files for communication with the
worker.

```json
{"event": "progress", "fraction": 0.5, "message": "Exported synchronized frame 60"}
{"event": "result", "data": {"path": "/absolute/path/to/artifact"}}
```

```bash
recon-s3-download-input \
  --bucket "$CP_4DA_BUCKET" \
  --region "$CP_AWS_REGION" \
  --key "$RECON_INPUT_PREFIX/$RECON_VIDEO" \
  --output "$RECON_VIDEO_PATH" \
  > download-events.jsonl 2> download.log
```

For a manifest with frame datasets and timings, pass `--datasets` with a JSON
list of `frame`/`dataset_dir` objects and `--durations` with a JSON object mapping
operation names to seconds. Both default to empty collections.

The access-check utility can inspect SageMaker resources using
`--sagemaker-domain-id`/`--sagemaker-space-name`/`--sagemaker-app-name`.
AWS credentials use the standard boto3 environment/role chain.
Utilities do not check or send notifications and accept no notification settings
or tokens. The preflight pass checks optional SNS email and Telegram channels
in the worker process after the AWS utility succeeds. Worker observers send
notifications based on pass lifecycle events and progress parsed from stdout.
Telegram access uses the synchronous `pyTelegramBotAPI` client, included in the
`aws` extra; tokens are resolved only inside the worker.

## Reconstruct one or more static frames

This stage follows the working [Splatfacto Colab](https://colab.research.google.com/drive/1yJ7sgxn7TxiW0m5CASOdGU5bD04tzzLq): private RGB+mask-to-RGBA copies, stock Splatfacto with random backgrounds and all input cameras, then a full Gaussian PLY with SH coefficients. The default profile is 60,000 iterations, SH degree 2, threshold 127 and 1px mask erosion. Source datasets are preserved.

`scripts/setup_environment.sh` installs the tools in a separate environment using the notebook profile: Python 3.11, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, Nerfstudio 1.1.5 and gsplat 1.4.0. `RECON_NERFSTUDIO_BIN` points to its `bin` directory containing `python`, `ns-train` and `ns-export`. Running a pipeline never installs packages or patches Nerfstudio.

The ready-to-edit `config/splatfacto-run.example.json` trains frame 60 from a saved 4DAnyone experiment and exports its reconstruction recording. Set its bucket, SageMaker identifiers and source experiment before running.

Enable `pipeline.reconstruction.enabled` in the JSON. Leave `config.frames` as `null` to train every frame listed in `artifacts.dataset.nerfstudio.frames`, or supply a subset. The Nerfstudio dataset artifact must be enabled. To use an existing 4DAnyone run, disable `pipeline.dataset.enabled` and select its name in `artifacts.dataset.nerfstudio.source_experiment_name`.

The generator also accepts `--reconstruction`, `--nerfstudio-bin`, `--reconstruction-frames`, and `--splatfacto-*` profile options. JSON examples contain the complete profile even when reconstruction is disabled. Each selected frame gets its own training pass and durable checkpoint. Restart skips completed frames; `--force` reruns them.

Run one frame independently, using the same environment as the pipeline:

```bash
export RECON_NS_BIN="$RECON_NERFSTUDIO_BIN"
export RECON_STATIC_DATASET="$RECON_DATA_ROOT/runs/$RECON_EXPERIMENT_NAME/nerfstudio/frame_060"
export RECON_STATIC_OUTPUT="$RECON_DATA_ROOT/runs/$RECON_EXPERIMENT_NAME/splatfacto/frame_060"
export RECON_TRAINING_NAME="${RECON_EXPERIMENT_NAME}_frame_060"
export RECON_ITERATIONS="60000"
export RECON_STOP_SPLIT="32000"
export RECON_CULL_ALPHA="0.02"
export RECON_DENSIFY_GRAD="0.0003"
export RECON_DENSIFY_SIZE="0.0075"
export RECON_SPLIT_SCREEN="0.03"
export RECON_DOWNSCALES="1"
export RECON_RESOLUTION_SCHEDULE="2000"
export RECON_CULL_SCALE="0.10"
export RECON_STOP_SCREEN="32000"
export RECON_MAX_GAUSS_RATIO="10.0"
export RECON_SH_DEGREE="2"
export RECON_MASK_THRESHOLD="127"
export RECON_MASK_EROSION="1"

PYTHONPATH="$RECON_PIPELINE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$RECON_NS_BIN/python" -m recon_pipeline.utilities.reconstructions.nerfstudio.splatfacto \
  --dataset "$RECON_STATIC_DATASET" \
  --output "$RECON_STATIC_OUTPUT" \
  --nerfstudio-bin "$RECON_NS_BIN" \
  --experiment-name "$RECON_TRAINING_NAME" \
  --max-num-iterations "$RECON_ITERATIONS" \
  --stop-split-at "$RECON_STOP_SPLIT" \
  --cull-alpha-thresh "$RECON_CULL_ALPHA" \
  --densify-grad-thresh "$RECON_DENSIFY_GRAD" \
  --densify-size-thresh "$RECON_DENSIFY_SIZE" \
  --split-screen-size "$RECON_SPLIT_SCREEN" \
  --num-downscales "$RECON_DOWNSCALES" \
  --resolution-schedule "$RECON_RESOLUTION_SCHEDULE" \
  --cull-scale-thresh "$RECON_CULL_SCALE" \
  --stop-screen-size-at "$RECON_STOP_SCREEN" \
  --use-scale-regularization \
  --max-gauss-ratio "$RECON_MAX_GAUSS_RATIO" \
  --sh-degree "$RECON_SH_DEGREE" \
  --mask-threshold "$RECON_MASK_THRESHOLD" \
  --mask-erosion-pixels "$RECON_MASK_EROSION" \
  > splatfacto-events.jsonl 2> splatfacto.log
```

For preparation alone, run `recon-prepare-rgba --dataset ... --output ... --mask-threshold ... --mask-erosion-pixels ...` in an environment with NumPy, Pillow and OpenCV. Every utility also supports `--help` without GPU imports. Existing standalone output directories require an explicit `--replace-existing`.

Outputs for each frame live in `splatfacto/frame_NNN/`: the RGBA training dataset, Nerfstudio config/checkpoints, `dataparser_transforms.json`, TensorBoard events, training/export logs, `exports/splat.ply`, and `experiment_manifest.json`. The Gaussian PLY is checked for positions, scales, rotations, opacity, DC and higher SH coefficients. `pipeline-result.json` lists all reconstructed frames. With `upload_results: true`, each complete frame folder is uploaded immediately after its reconstruction pass.

For the notebook's optional reconstruction recording, enable `artifacts.reconstruction.rerun.enabled`. Its interpreter comes from `RECON_SPLATFACTO_RERUN_PYTHON`. It needs Rerun 0.36, NumPy, Pillow, plyfile and TensorBoard (the `splatfacto-rerun` extra describes those dependencies). The generator accepts `--reconstruction-rerun`, `--reconstruction-rerun-max-splats` and `--reconstruction-rerun-view-count`.

A separate recording pass per frame writes `rerun/reconstruction.rrd`, with aligned cameras, four selected views, metrics and a Gaussian-center debug view. `max_splats` limits only the Rerun sample; the full `exports/splat.ply` stays intact. Reconstruction recordings currently use the current run's freshly trained models; cross-experiment recording-only runs are rejected explicitly.

## Validate and inspect

```bash
test -f "$RECON_RUN_CONFIG"
python -m json.tool "$RECON_RUN_CONFIG" >/dev/null
recon-aws-worker plan --config "$RECON_RUN_CONFIG"
```

## Run in the background

For a sequential queue, repeat `--config` in execution order:

```bash
recon-aws-worker start \
  --config config/experiment-1.json \
  --config config/experiment-2.json \
  --config config/experiment-3.json \
  --queue-id splatfacto-batch \
  --shutdown-on always

recon-aws-worker logs --queue-id splatfacto-batch --follow
recon-aws-worker status --queue-id splatfacto-batch --json
recon-aws-worker stop --queue-id splatfacto-batch
```

Each experiment finishes before the next starts. Failed experiments are recorded
and the queue continues. Per-experiment shutdown is disabled; the queue's policy
is applied only after all experiments finish. `always` stops the App regardless
of failures; `never` leaves it running; `success` requires all runs to succeed;
`failure` stops it if any run fails. If omitted, the queue uses the final config's
policy. For a single config, `--shutdown-on` overrides that run's policy.
Input JSON files are never modified.

Queue configs must have distinct experiment/job IDs and target the same
SageMaker App and region. An omitted queue ID defaults to `queue-<first job ID>`.
Each experiment retains its own job status, log and reconstruction checkpoints.
The queue has a combined live log and a `queue-result.json` summary under
`$RECON_DATA_ROOT/jobs/<queue-id>/`. Any failed experiment makes the final queue
status `failed`, even though later experiments still run. Restart the same queue
without `--force` to reuse completed pass checkpoints. `stop --queue-id` cancels
the current experiment and prevents remaining experiments from starting; it
does not apply the automatic shutdown policy.

```bash
recon-aws-worker start --config "$RECON_RUN_CONFIG"
recon-aws-worker status --config "$RECON_RUN_CONFIG"
recon-aws-worker status --config "$RECON_RUN_CONFIG" --json
recon-aws-worker logs --config "$RECON_RUN_CONFIG" --lines 200
recon-aws-worker logs --config "$RECON_RUN_CONFIG" --lines 200 --follow
```

The AWS worker checkpoints successful passes locally in the experiment's
`.recon-pipeline/pass-state.json`. With the existing `aws_worker.upload_results`
set to `true`, it also adds an upload pass immediately after workspace creation,
4DAnyone inference, Nerfstudio dataset export, each Splatfacto frame, each Rerun
export, and the final run manifest. No config schema change is required.

Each upload transfers only its producer's outputs to the existing
`s3://<bucket>/<runs_prefix>/<experiment>/` layout. A completion record is written
last under `.recon-pipeline/commits/`, with the producer's result, settings
signature, relative paths, file sizes and SHA-256 hashes. Artifacts must be
successfully uploaded before their completion is committed. Publishing does
not delete the experiment prefix or other experiments.

Running the same `start` command again, including on an empty disk:

- archives any previous local terminal job attempt and log;
- restores matching committed outputs from S3 and verifies their hashes;
- rebuilds local checkpoints, rebasing artifact paths to `RECON_DATA_ROOT`;
- rechecks AWS access and synchronizes input/model/source dependencies;
- skips completed computations whose settings and required outputs still match;
- retries an unfinished upload without repeating its completed computation when
  that computation is still present locally;
- executes incomplete or changed passes and uploads their outputs.

Passes are matched by ID rather than by their position. Adding upload passes
does not discard completed reconstructions. Expanding an export's frame list
invalidates that export, while retaining already completed reconstruction
frames whose training settings still match. Existing local checkpoints from
before cloud recovery are adopted using their saved `pipeline-config.json`
and output checks, then uploaded; legacy S3-only experiments have no pass
completion records and cannot automatically skip their own computations.
References to legacy source experiments remain supported: only their
`4danyone/` data is downloaded, not unrelated reconstruction folders.

Infrastructure checks still run even when all computational passes are done.
The worker never starts a second process while the same local job is running.
Use a single worker per experiment; cross-machine execution locks are not yet
provided. Keep experiment names and input object keys immutable. If their
contents are replaced without changing settings, use `--force`.

Ignore all checkpoints and rebuild the experiment from its first pass:

```bash
recon-aws-worker start --config "$RECON_RUN_CONFIG" --force
```

Changes to dataset, export and training settings automatically invalidate the
corresponding computations. `--force` ignores local checkpoints and removes
the selected plan's S3 completion records before recomputing. It preserves
artifact objects until they are overwritten by successful new uploads.

No manual deletion of `runs/<experiment>` or `jobs/<job>` is required between
attempts. `--force` does not delete shared model caches; cleanup is limited to
outputs owned by the pass being rerun.

### Cloud status and failure reports

The worker uploads status, its input config snapshot, and recent log tails at
startup and every 60 seconds. Log tails are limited to 1 MiB per file: the
pipeline log and up to three recent operation logs. Small status updates do
not retransmit artifact bundles. Literal Telegram tokens are redacted in
config snapshots; environment variable names are preserved.

The latest status is at
`runs/<experiment>/.recon-pipeline/status.json` (using your configured runs
prefix). It points to an attempt directory containing `report.json`,
`status.json`, `request.json`, `job/pipeline.log`, and operation logs under
`logs/`. Attempt paths are unique, so a restart preserves earlier reports.
The final report includes completed passes, durations, the failed pass ID,
exception type and message, full traceback, command, exit code, and captured
output tail where available. Empty exception messages use their representation.

Before applying either a single-run or queue shutdown policy, the worker saves
the final report and full logs. If an artifact upload or final diagnostics
upload fails, managed shutdown is deferred to preserve the local copy. A queue
still continues to its next experiment, then defers shutdown if any experiment
has unsaved data. Fix S3 access and restart the affected experiment to retry.
Setting `upload_results: false` disables cloud recovery, reports and these
shutdown protections; local checkpoints remain available.

Completed committed frames survive disk/VM loss. A forced termination can
interrupt the current computation or upload; that unfinished frame is retrained
after restoration, rather than resumed from an intermediate training iteration.
A hard kill cannot guarantee a final failure report, but the latest heartbeat
and previously committed outputs remain in S3. Container deployment itself is
outside this change.

## Stop

```bash
recon-aws-worker stop --config "$RECON_RUN_CONFIG"
```

The configured Telegram bot also accepts:

```text
/shutdown
/shutdown leo_three_layers_72views_01
```

## One-time environment and model setup

```bash
./scripts/setup_environment.sh --config "$RECON_RUN_CONFIG"
# To download/validate weights separately after installation:
./scripts/download_4danyone_models.sh "$RECON_RUN_CONFIG"
```

## Tests

```bash
python -m pip install --editable '.[dev]'
pytest
```
