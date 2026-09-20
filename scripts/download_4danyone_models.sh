#!/usr/bin/env bash

# Download and validate model assets. This step is CPU-safe.

set -Eeuo pipefail

CP_4DA_ENV="${CP_4DA_ENV:-$HOME/.conda/envs/4danyone}"
CP_4DA_REPO_ROOT="${CP_4DA_REPO_ROOT:-$HOME/work/4DAnyone}"
CP_4DA_DATA_ROOT="${CP_4DA_DATA_ROOT:-$HOME/4danyone-data}"
CP_4DA_MODEL_DIR="${CP_4DA_MODEL_DIR:-$CP_4DA_DATA_ROOT/models}"
CP_AWS_REGION="${CP_AWS_REGION:-us-east-1}"
export CP_4DA_ENV CP_4DA_REPO_ROOT CP_4DA_DATA_ROOT CP_4DA_MODEL_DIR CP_AWS_REGION

SMPLX_ARCHIVE="$CP_4DA_MODEL_DIR/smplx/models_smplx_v1_1.zip"
SMPLX_MODEL="$CP_4DA_MODEL_DIR/body_models/smplx/SMPLX_NEUTRAL.npz"
GVHMR_ROOT="$CP_4DA_REPO_ROOT/third_party/GVHMR"

log() { printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -d "$CP_4DA_REPO_ROOT/.git" ]] || fail "4DAnyone repository not found: $CP_4DA_REPO_ROOT"
[[ -f "$GVHMR_ROOT/hmr4d/__init__.py" ]] || fail "GVHMR submodule is not initialized"
[[ -x "$CP_4DA_ENV/bin/python" ]] || fail "Conda environment not found: $CP_4DA_ENV"

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CP_4DA_ENV"
export PYTHONNOUSERSITE=1
mkdir -p "$CP_4DA_MODEL_DIR"
cd "$CP_4DA_REPO_ROOT"

if [[ -n "${CP_4DA_BUCKET:-}" ]]; then
  log "Synchronizing existing models from S3"
  aws s3 sync "s3://${CP_4DA_BUCKET}/models/" "$CP_4DA_MODEL_DIR/" \
    --region "$CP_AWS_REGION" --no-cli-pager
fi

if [[ ! -f "$SMPLX_MODEL" ]]; then
  [[ -f "$SMPLX_ARCHIVE" ]] || fail "Licensed SMPL-X archive not found: $SMPLX_ARCHIVE"
  log "Installing SMPL-X"
  python scripts/download_smplx.py \
    --archive_path "$SMPLX_ARCHIVE" \
    --model_dir "$CP_4DA_MODEL_DIR" \
    --gvhmr_root "$GVHMR_ROOT"
fi

log "Downloading missing 4DAnyone, GVHMR, VGG-19 and BiRefNet assets"
python scripts/download_model.py --model_dir "$CP_4DA_MODEL_DIR" --gvhmr_root "$GVHMR_ROOT"

log "Validating model files"
python - <<'PY'
import os
from pathlib import Path
from fdanyone.assets import BIREFNET_DIR, BIREFNET_FILES, MODEL_FILES, SMPLX_MODEL

root = Path(os.environ["CP_4DA_MODEL_DIR"])
required = [*(root / p for p in MODEL_FILES), *(root / BIREFNET_DIR / p for p in BIREFNET_FILES), root / SMPLX_MODEL]
missing = [p for p in required if not p.is_file()]
if missing:
    raise SystemExit("Missing files:\n" + "\n".join(f" - {p}" for p in missing))
print(f"Models: OK — {len(required)} required files found")
PY

find "$GVHMR_ROOT/inputs/checkpoints" -type l -printf '%p -> %l\n' | sort
du -sh "$CP_4DA_MODEL_DIR"
df -h "$HOME"
log "Model setup completed"
