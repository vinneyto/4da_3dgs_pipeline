#!/usr/bin/env bash

# Download and validate model assets using paths and AWS settings from JSON.
# This stage is CPU-safe.

set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 /absolute/path/to/run.json" >&2; exit 2; }
CONFIG_PATH="$1"
[[ -f "$CONFIG_PATH" ]] || { echo "config not found: $CONFIG_PATH" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
READ_CONFIG="$SCRIPT_DIR/read_config.py"
get() { /usr/bin/python3 "$READ_CONFIG" "$CONFIG_PATH" "$1"; }

CONDA_BOOTSTRAP="$(get environment.conda_bootstrap)"
CONDA_ENV="$(get environment.conda_env)"
FOURDANYONE_ROOT="$(get aws_worker.local.fourdanyone_root)"
DATA_ROOT="$(get aws_worker.local.data_root)"
MODEL_DIR="$DATA_ROOT/models"
AWS_REGION="$(get aws_worker.region)"
S3_BUCKET="$(get aws_worker.bucket)"
MODELS_PREFIX="$(get aws_worker.models_prefix)"

SMPLX_ARCHIVE="$MODEL_DIR/smplx/models_smplx_v1_1.zip"
SMPLX_MODEL="$MODEL_DIR/body_models/smplx/SMPLX_NEUTRAL.npz"
GVHMR_ROOT="$FOURDANYONE_ROOT/third_party/GVHMR"

log() { printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONDA_BOOTSTRAP" ]] || fail "Conda bootstrap not found: $CONDA_BOOTSTRAP"
[[ -d "$FOURDANYONE_ROOT/.git" ]] || fail "4DAnyone repository not found: $FOURDANYONE_ROOT"
[[ -f "$GVHMR_ROOT/hmr4d/__init__.py" ]] || fail "GVHMR submodule is not initialized"
[[ -x "$CONDA_ENV/bin/python" ]] || fail "Conda environment not found: $CONDA_ENV"

# shellcheck disable=SC1090
source "$CONDA_BOOTSTRAP"
conda activate "$CONDA_ENV"
mkdir -p "$MODEL_DIR"
cd "$FOURDANYONE_ROOT"

log "Synchronizing existing models from S3"
aws s3 sync "s3://${S3_BUCKET}/${MODELS_PREFIX}/" "$MODEL_DIR/" \
  --region "$AWS_REGION" --no-cli-pager

if [[ ! -f "$SMPLX_MODEL" ]]; then
  [[ -f "$SMPLX_ARCHIVE" ]] || fail "Licensed SMPL-X archive not found: $SMPLX_ARCHIVE"
  log "Installing SMPL-X"
  python scripts/download_smplx.py \
    --archive_path "$SMPLX_ARCHIVE" \
    --model_dir "$MODEL_DIR" \
    --gvhmr_root "$GVHMR_ROOT"
fi

log "Downloading missing 4DAnyone, GVHMR, VGG-19, and BiRefNet assets"
python scripts/download_model.py --model_dir "$MODEL_DIR" --gvhmr_root "$GVHMR_ROOT"

log "Validating model files"
python - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
from fdanyone.assets import BIREFNET_DIR, BIREFNET_FILES, MODEL_FILES, SMPLX_MODEL

root = Path(sys.argv[1])
required = [*(root / p for p in MODEL_FILES), *(root / BIREFNET_DIR / p for p in BIREFNET_FILES), root / SMPLX_MODEL]
missing = [p for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing files:\n" + "\n".join(f" - {p}" for p in missing))
print(f"Models: OK — {len(required)} required files found")
PY

find "$GVHMR_ROOT/inputs/checkpoints" -type l -printf '%p -> %l\n' | sort
du -sh "$MODEL_DIR"
df -h "$(dirname "$MODEL_DIR")"
log "Model setup completed"
