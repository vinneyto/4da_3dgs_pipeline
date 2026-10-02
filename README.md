# Reconstruction pipeline CLI

## Install or update

```bash
cd "$HOME/work/4da_3dgs_pipeline"
git switch main
git pull --ff-only origin main

python -m pip uninstall --yes fourda-3dgs-pipeline
python -m pip install --editable '.[aws,rerun]'

recon-config --help
recon-aws-worker --help
```

## Clone an existing run configuration

Use a validated JSON document as a template when only the input and run identity
change. All pipeline, artifact, environment, AWS, notification, and shutdown
settings are preserved. Artifact source references that pointed at the template
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

## Configure a three-layer AWS run

Define every value used to generate the run document:

```bash
# Persistent environment
export RECON_CONDA_BOOTSTRAP="/opt/conda/etc/profile.d/conda.sh"
export RECON_CONDA_ENV="$HOME/.conda/envs/4danyone"
export RECON_PIPELINE_ROOT="$HOME/work/4da_3dgs_pipeline"
export RECON_FOURDANYONE_ROOT="$HOME/work/4DAnyone"
export RECON_DATA_ROOT="$HOME/4danyone-data"
export RECON_LOCK_FILE="$RECON_DATA_ROOT/environment/requirements-lock.txt"

# Pinned upstream and Python packages
export RECON_FOURDANYONE_GIT_URL="https://github.com/ant-research/4DAnyone.git"
export RECON_FOURDANYONE_GIT_REF="e38f210827f7b3effbe5b573ea07cfcf17e72dca"
export RECON_PYTHON_VERSION="3.11"
export RECON_TORCH_VERSION="2.8.0"
export RECON_TORCHVISION_VERSION="0.23.0"
export RECON_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
export RECON_OPENCV_FALLBACK_VERSION="4.14.0.94"

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
  --conda-bootstrap "$RECON_CONDA_BOOTSTRAP" \
  --conda-env "$RECON_CONDA_ENV" \
  --pipeline-repo-root "$RECON_PIPELINE_ROOT" \
  --fourdanyone-git-url "$RECON_FOURDANYONE_GIT_URL" \
  --fourdanyone-git-ref "$RECON_FOURDANYONE_GIT_REF" \
  --python-version "$RECON_PYTHON_VERSION" \
  --torch-version "$RECON_TORCH_VERSION" \
  --torchvision-version "$RECON_TORCHVISION_VERSION" \
  --torch-index-url "$RECON_TORCH_INDEX_URL" \
  --opencv-fallback-version "$RECON_OPENCV_FALLBACK_VERSION" \
  --lock-file "$RECON_LOCK_FILE" \
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
  --sagemaker-app-name "$CP_SM_JUPYTER_APP_NAME" \
  --data-root "$RECON_DATA_ROOT" \
  --fourdanyone-root "$RECON_FOURDANYONE_ROOT"
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
dataset or recording. Downloads/restores discard their owned local output;
uploads delete only the selected S3 prefix before uploading. The pipeline passes
this flag when rerunning a pass; checkpoint and `--force` behavior is preserved.
Shared model caches are retained.

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

Install the upstream tools manually in a separate environment. The notebook uses Python 3.11, PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, Nerfstudio 1.1.5 and gsplat 1.4.0. That environment also needs NumPy, Pillow, OpenCV, ninja and a working CUDA toolkit. Set `pipeline.reconstruction.config.nerfstudio_bin` to its `bin` directory containing `python`, `ns-train` and `ns-export`. The pipeline never installs packages or patches Nerfstudio.

The ready-to-edit `config/splatfacto-run.example.json` trains frame 60 from a saved 4DAnyone experiment and exports its reconstruction recording. Set its bucket, source experiment, local paths and environment paths before running.

Enable `pipeline.reconstruction.enabled` in the JSON. Leave `config.frames` as `null` to train every frame listed in `artifacts.dataset.nerfstudio.frames`, or supply a subset. The Nerfstudio dataset artifact must be enabled. To use an existing 4DAnyone run, disable `pipeline.dataset.enabled` and select its name in `artifacts.dataset.nerfstudio.source_experiment_name`.

The generator also accepts `--reconstruction`, `--nerfstudio-bin`, `--reconstruction-frames`, and `--splatfacto-*` profile options. JSON examples contain the complete profile even when reconstruction is disabled. Each selected frame gets its own training pass and durable checkpoint. Restart skips completed frames; `--force` reruns them.

Run one frame independently, using the same environment as the pipeline:

```bash
export RECON_NS_BIN="$HOME/.conda/envs/splatfacto/bin"
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

Outputs for each frame live in `splatfacto/frame_NNN/`: the RGBA training dataset, Nerfstudio config/checkpoints, `dataparser_transforms.json`, TensorBoard events, training/export logs, `exports/splat.ply`, and `experiment_manifest.json`. The Gaussian PLY is checked for positions, scales, rotations, opacity, DC and higher SH coefficients. `pipeline-result.json` lists all reconstructed frames; the regular result upload includes their outputs.

For the notebook's optional reconstruction recording, enable `artifacts.reconstruction.rerun.enabled` and set its `python` to a separately installed Rerun environment. It needs Rerun 0.36, NumPy, Pillow, plyfile and TensorBoard (the `splatfacto-rerun` extra describes those dependencies). The generator accepts `--reconstruction-rerun`, `--reconstruction-rerun-python`, `--reconstruction-rerun-max-splats` and `--reconstruction-rerun-view-count`.

A separate recording pass per frame writes `rerun/reconstruction.rrd`, with aligned cameras, four selected views, metrics and a Gaussian-center debug view. `max_splats` limits only the Rerun sample; the full `exports/splat.ply` stays intact. Reconstruction recordings currently use the current run's freshly trained models; cross-experiment recording-only runs are rejected explicitly.

## Validate and inspect

```bash
test -f "$RECON_RUN_CONFIG"
python -m json.tool "$RECON_RUN_CONFIG" >/dev/null
recon-aws-worker plan --config "$RECON_RUN_CONFIG"
```

## Run in the background

```bash
recon-aws-worker start --config "$RECON_RUN_CONFIG"
recon-aws-worker status --config "$RECON_RUN_CONFIG"
recon-aws-worker status --config "$RECON_RUN_CONFIG" --json
recon-aws-worker logs --config "$RECON_RUN_CONFIG" --lines 200
recon-aws-worker logs --config "$RECON_RUN_CONFIG" --lines 200 --follow
```

The worker checkpoints every successfully completed pass in the experiment's
`.recon-pipeline/pass-state.json`. Running the same `start` command again:

- archives the previous terminal job attempt and its log;
- restores the longest unchanged prefix of completed passes;
- starts at the first incomplete or newly inserted pass;
- invalidates and reruns every pass after that boundary;
- removes each rerun pass's owned output immediately before the pass starts.

If every configured pass is complete, the worker skips all regular passes. It
never starts a second process while the same job is still running.

Ignore all checkpoints and rebuild the experiment from its first pass:

```bash
recon-aws-worker start --config "$RECON_RUN_CONFIG" --force
```

Use `--force` after changing parameters of an existing pass. Pass insertion,
removal, or reordering is detected automatically from the ordered checkpoint
sequence and invalidates that pass position and every later pass.

No manual deletion of `runs/<experiment>` or `jobs/<job>` is required between
attempts. `--force` does not delete shared model caches; cleanup is limited to
outputs owned by the pass being rerun.

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
./scripts/setup_4danyone_env.sh "$RECON_RUN_CONFIG"
./scripts/download_4danyone_models.sh "$RECON_RUN_CONFIG"
```

## Tests

```bash
python -m pip install --editable '.[dev]'
pytest
```
