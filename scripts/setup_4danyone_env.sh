#!/usr/bin/env bash

# Build the persistent 4DAnyone environment from RECON_* variables.
# This script is CPU-safe; CUDA execution is validated when a GPU is present.

set -Eeuo pipefail

[[ $# -eq 0 ]] || { echo "usage: $0 (settings come from RECON_* env)" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
get() { PYTHONPATH="$SCRIPT_DIR/../src${PYTHONPATH:+:$PYTHONPATH}" python3 -m recon_pipeline.environment_cli value "$1"; }

CONDA_BOOTSTRAP="$(get conda_bootstrap)"
CONDA_ENV="$(get conda_env)"
PIPELINE_REPO_ROOT="$(get pipeline_repo_root)"
FOURDANYONE_ROOT="$(get fourdanyone_root)"
FOURDANYONE_GIT_URL="$(get fourdanyone_git_url)"
FOURDANYONE_GIT_REF="$(get fourdanyone_git_ref)"
PYTHON_VERSION="$(get python_version)"
TORCH_VERSION="$(get torch_version)"
TORCHVISION_VERSION="$(get torchvision_version)"
TORCH_INDEX_URL="$(get torch_index_url)"
OPENCV_FALLBACK_VERSION="$(get opencv_fallback_version)"
LOCK_FILE="$(get lock_file)"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "This script expects Linux."
[[ "$(uname -m)" == "x86_64" ]] || fail "This script expects x86_64."
[[ -f "$CONDA_BOOTSTRAP" ]] || fail "Conda bootstrap not found: $CONDA_BOOTSTRAP"
[[ -d "$PIPELINE_REPO_ROOT" ]] || fail "Pipeline repository not found: $PIPELINE_REPO_ROOT"

if [[ ! -d "$FOURDANYONE_ROOT/.git" ]]; then
  log "Cloning 4DAnyone"
  mkdir -p "$(dirname "$FOURDANYONE_ROOT")"
  git clone "$FOURDANYONE_GIT_URL" "$FOURDANYONE_ROOT"
  git -C "$FOURDANYONE_ROOT" checkout --detach "$FOURDANYONE_GIT_REF"
fi
git -C "$FOURDANYONE_ROOT" submodule update --init third_party/GVHMR
current_ref="$(git -C "$FOURDANYONE_ROOT" rev-parse HEAD)"
if [[ "$current_ref" != "$FOURDANYONE_GIT_REF" ]]; then
  echo "WARNING: expected 4DAnyone $FOURDANYONE_GIT_REF, found $current_ref" >&2
fi

# shellcheck disable=SC1090
# Conda and its activation hooks access optional unset variables.
set +u
source "$CONDA_BOOTSTRAP"
set -u
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  EXPECT_CUDA=1
  log "GPU instance detected"
  nvidia-smi -L
else
  EXPECT_CUDA=0
  log "CPU preparation mode; GPU validation is deferred"
fi

if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  log "Creating persistent conda environment: $CONDA_ENV"
  conda create --prefix "$CONDA_ENV" "python=$PYTHON_VERSION" pip -y
fi
set +u
conda activate "$CONDA_ENV"
set -u
PYTHON="$CONDA_ENV/bin/python"

log "Installing packaging tools and CUDA-enabled PyTorch"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --no-cache-dir \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  --index-url "$TORCH_INDEX_URL"

log "Installing 4DAnyone requirements"
"$PYTHON" -m pip install --no-cache-dir -r "$FOURDANYONE_ROOT/requirements.txt"
"$PYTHON" -m pip install --no-cache-dir \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  --index-url "$TORCH_INDEX_URL"

opencv_version="$("$PYTHON" -m pip show opencv-python 2>/dev/null | awk '/^Version:/ {print $2}' || true)"
opencv_version="${opencv_version:-$OPENCV_FALLBACK_VERSION}"
log "Replacing GUI OpenCV with headless $opencv_version"
"$PYTHON" -m pip uninstall --yes opencv-python opencv-python-headless || true
"$PYTHON" -m pip install --no-cache-dir "opencv-python-headless==$opencv_version"

if [[ ! -x "$CONDA_ENV/bin/ffmpeg" ]]; then
  conda install --prefix "$CONDA_ENV" --channel conda-forge ffmpeg -y
fi

log "Installing this pipeline with AWS job and Rerun support"
"$PYTHON" -m pip install --editable "$PIPELINE_REPO_ROOT[aws,rerun]"
"$PYTHON" -m pip install ipykernel
"$PYTHON" -m ipykernel install --user --name 4danyone --display-name "Python (4DAnyone)"

log "Validating imports and Torchvision NMS"
"$PYTHON" - "$EXPECT_CUDA" <<'PY'
import sys
import av, cv2, einops, kornia, numpy, smplx, timm, torch, torchvision, transformers, ultralytics
from torchvision.ops import nms

expect_cuda = sys.argv[1] == "1"
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]])
scores = torch.tensor([0.9, 0.8, 0.7])
cpu_result = nms(boxes, scores, 0.5)
assert cpu_result.tolist() == [0, 2]
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
if expect_cuda:
    assert torch.cuda.is_available(), "GPU detected but PyTorch cannot use CUDA"
    print("GPU:", torch.cuda.get_device_name(0))
    assert torch.equal(cpu_result, nms(boxes.cuda(), scores.cuda(), 0.5).cpu())
print("4DAnyone imports: OK")
PY

"$CONDA_ENV/bin/ffmpeg" -version | head -n 1
"$PYTHON" -m pip check || echo "NOTE: review any dependency warnings above."
mkdir -p "$(dirname "$LOCK_FILE")"
"$PYTHON" -m pip freeze > "$LOCK_FILE"

log "Environment setup completed"
du -sh "$CONDA_ENV" "$FOURDANYONE_ROOT"
df -h "$(dirname "$CONDA_ENV")"
