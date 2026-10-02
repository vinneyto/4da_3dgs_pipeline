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

Every pass launches one Python utility in a separate process. Each utility has
its own file under the owning package's `utilities/` directory, its own console
command, and a `python -m` entry point. The pass supplies arguments and maps JSON
results and progress back to the pipeline; the operation and output cleanup live
in the utility.

| Operation | Console command | Module under `recon_pipeline` |
| --- | --- | --- |
| Prepare workspace | `recon-prepare-experiment` | `datasets.fourdanyone.utilities.prepare_experiment` |
| Generate dataset | `recon-4danyone` | `datasets.fourdanyone.utilities.inference` |
| Export synchronized frames | `recon-nerfstudio-export` | `reconstructions.nerfstudio.utilities.export` |
| Export recording | `recon-rerun` | `artifacts.rerun.utilities.export` |
| Check AWS dependencies | `recon-aws-preflight` | `workers.aws.utilities.preflight` |
| Download video | `recon-s3-download-input` | `workers.aws.utilities.download_input` |
| Sync model cache | `recon-s3-sync-models` | `workers.aws.utilities.sync_models` |
| Restore source experiment | `recon-s3-restore-experiment` | `workers.aws.utilities.restore_experiment` |
| Write manifest | `recon-write-run-manifest` | `workers.aws.utilities.write_run_manifest` |
| Upload results | `recon-s3-upload-results` | `workers.aws.utilities.upload_results` |

Inspect or debug a single utility without starting the worker:

```bash
recon-4danyone --help
python -m recon_pipeline.datasets.fourdanyone.utilities.inference --help
python -m pdb -m recon_pipeline.datasets.fourdanyone.utilities.inference --help
```

AWS utilities accept `--worker-config`: either the existing run JSON or a JSON
object containing just the AWS worker settings. Local generation and export
commands use explicit paths and do not need AWS settings.

```bash
export RECON_EXPERIMENT_DIR="$RECON_DATA_ROOT/runs/$RECON_EXPERIMENT_NAME"
export RECON_GENERATION_DIR="$RECON_EXPERIMENT_DIR/4danyone"
export RECON_VIDEO_PATH="$RECON_DATA_ROOT/input/$RECON_VIDEO"
export RECON_MODEL_DIR="$RECON_DATA_ROOT/models"
export RECON_MATERIALIZED_CONFIG="$RECON_EXPERIMENT_DIR/pipeline-config.json"
export RECON_RERUN_PATH="$RECON_EXPERIMENT_DIR/rerun/$RECON_EXPERIMENT_NAME.rrd"

recon-aws-preflight \
  --worker-config "$RECON_RUN_CONFIG" \
  --require-input-video

recon-s3-download-input --worker-config "$RECON_RUN_CONFIG"
recon-s3-sync-models --worker-config "$RECON_RUN_CONFIG"

# Prepare a new workspace from a local run config with absolute paths.
# Alternatively, use a pipeline-config.json saved by an earlier run.
export RECON_LOCAL_CONFIG="$RECON_PIPELINE_ROOT/config/local-run.example.json"
recon-prepare-experiment --pipeline-config "$RECON_LOCAL_CONFIG"

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
  --worker-config "$RECON_RUN_CONFIG" \
  --experiment "$RECON_SOURCE_EXPERIMENT" \
  --destination "$RECON_DATA_ROOT/runs/$RECON_SOURCE_EXPERIMENT"

recon-write-run-manifest \
  --pipeline-config "$RECON_MATERIALIZED_CONFIG" \
  --rerun-file "$RECON_RERUN_PATH"

recon-s3-upload-results \
  --worker-config "$RECON_RUN_CONFIG" \
  --experiment "$RECON_EXPERIMENT_NAME" \
  --source "$RECON_EXPERIMENT_DIR"
```

Edit the example local config's absolute paths, experiment name and parameters
before preparing a new workspace. The materialized `pipeline-config.json` is
also produced automatically by the pipeline's preparation pass.

For an existing inference directory, frame dataset or recording, explicitly
pass `--replace-existing` to rebuild it. On downloads/restores the flag discards
the operation's owned local output; on uploads it deletes only the selected
experiment's S3 result prefix before uploading. The pipeline supplies this flag
for passes it reruns, preserving checkpoint/`--force` behavior. Shared model
caches are retained.

Every utility prints its JSON result after its logs, or writes it to
`--result-file PATH`. Progress is streamed as `RECON_PROGRESS` JSON lines.
For a manifest with frame datasets and timings, supply `--datasets PATH` (a JSON
list of `frame`/`dataset_dir` objects) and `--durations PATH` (a JSON object of pass
IDs and seconds). Both are optional for manual debugging.

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
