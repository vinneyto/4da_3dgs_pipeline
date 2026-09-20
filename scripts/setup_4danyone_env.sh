#!/usr/bin/env bash

# Build the persistent 4DAnyone environment in SageMaker Studio. This is safe
# on a CPU instance: CUDA-enabled Torch is installed now and validated later
# when the same persistent volume is attached to a GPU instance.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CP_PIPELINE_REPO_ROOT="${CP_PIPELINE_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}" 
export CP_4DA_ENV="${CP_4DA_ENV:-$HOME/.conda/envs/4danyone}"
export CP_4DA_REPO_ROOT="${CP_4DA_REPO_ROOT:-$HOME/work/4DAnyone}"
export PYTHONNOUSERSITE=1

CP_4DA_GIT_URL="${CP_4DA_GIT_URL:-https://github.com/ant-research/4DAnyone.git}"
CP_4DA_GIT_REF="${CP_4DA_GIT_REF:-e38f210827f7b3effbe5b573ea07cfcf17e72dca}"
CP_4DA_PYTHON_VERSION="${CP_4DA_PYTHON_VERSION:-3.11}"
CP_4DA_TORCH_VERSION="${CP_4DA_TORCH_VERSION:-2.8.0}"
CP_4DA_TORCHVISION_VERSION="${CP_4DA_TORCHVISION_VERSION:-0.23.0}"
CP_4DA_TORCH_INDEX_URL="${CP_4DA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
CP_4DA_OPENCV_FALLBACK_VERSION="${CP_4DA_OPENCV_FALLBACK_VERSION:-4.14.0.94}"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || fail "This script expects Linux."
[[ "$(uname -m)" == "x86_64" ]] || fail "This script expects x86_64."
[[ -f /opt/conda/etc/profile.d/conda.sh ]] || fail "SageMaker conda bootstrap was not found."

if [[ ! -d "$CP_4DA_REPO_ROOT/.git" ]]; then
  log "Cloning 4DAnyone"
  mkdir -p "$(dirname "$CP_4DA_REPO_ROOT")"
  git clone "$CP_4DA_GIT_URL" "$CP_4DA_REPO_ROOT"
  git -C "$CP_4DA_REPO_ROOT" checkout --detach "$CP_4DA_GIT_REF"
fi
git -C "$CP_4DA_REPO_ROOT" submodule update --init third_party/GVHMR
current_ref="$(git -C "$CP_4DA_REPO_ROOT" rev-parse HEAD)"
if [[ "$current_ref" != "$CP_4DA_GIT_REF" ]]; then
  echo "WARNING: expected 4DAnyone $CP_4DA_GIT_REF, found $current_ref" >&2
fi

# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export CP_4DA_EXPECT_CUDA=1
  log "GPU instance detected"
  nvidia-smi -L
else
  export CP_4DA_EXPECT_CUDA=0
  log "CPU preparation mode; GPU validation is deferred"
fi

if [[ ! -x "$CP_4DA_ENV/bin/python" ]]; then
  log "Creating persistent conda environment: $CP_4DA_ENV"
  conda create --prefix "$CP_4DA_ENV" "python=$CP_4DA_PYTHON_VERSION" pip -y
fi
conda activate "$CP_4DA_ENV"
PYTHON="$CP_4DA_ENV/bin/python"

log "Installing packaging tools and CUDA-enabled PyTorch"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install --no-cache-dir \
  "torch==$CP_4DA_TORCH_VERSION" \
  "torchvision==$CP_4DA_TORCHVISION_VERSION" \
  --index-url "$CP_4DA_TORCH_INDEX_URL"

log "Installing 4DAnyone requirements"
"$PYTHON" -m pip install --no-cache-dir -r "$CP_4DA_REPO_ROOT/requirements.txt"
"$PYTHON" -m pip install --no-cache-dir \
  "torch==$CP_4DA_TORCH_VERSION" \
  "torchvision==$CP_4DA_TORCHVISION_VERSION" \
  --index-url "$CP_4DA_TORCH_INDEX_URL"

opencv_version="$("$PYTHON" -m pip show opencv-python 2>/dev/null | awk '/^Version:/ {print $2}')"
opencv_version="${opencv_version:-$CP_4DA_OPENCV_FALLBACK_VERSION}"
log "Replacing GUI OpenCV with headless $opencv_version"
"$PYTHON" -m pip uninstall --yes opencv-python opencv-python-headless || true
"$PYTHON" -m pip install --no-cache-dir "opencv-python-headless==$opencv_version"

if [[ ! -x "$CP_4DA_ENV/bin/ffmpeg" ]]; then
  conda install --prefix "$CP_4DA_ENV" --channel conda-forge ffmpeg -y
fi

log "Installing this pipeline and registering its Jupyter kernel"
"$PYTHON" -m pip install --editable "$CP_PIPELINE_REPO_ROOT"
"$PYTHON" -m pip install ipykernel
"$PYTHON" -m ipykernel install --user --name 4danyone --display-name "Python (4DAnyone)"

log "Validating imports and Torchvision NMS"
"$PYTHON" - <<'PY'
import os
import av, cv2, einops, kornia, numpy, smplx, timm, torch, torchvision, transformers, ultralytics
from torchvision.ops import nms

boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]])
scores = torch.tensor([0.9, 0.8, 0.7])
cpu_result = nms(boxes, scores, 0.5)
assert cpu_result.tolist() == [0, 2]
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
if os.environ["CP_4DA_EXPECT_CUDA"] == "1":
    assert torch.cuda.is_available(), "GPU detected but PyTorch cannot use CUDA"
    print("GPU:", torch.cuda.get_device_name(0))
    assert torch.equal(cpu_result, nms(boxes.cuda(), scores.cuda(), 0.5).cpu())
print("4DAnyone imports: OK")
PY

"$CP_4DA_ENV/bin/ffmpeg" -version | head -n 1
"$PYTHON" -m pip check || echo "NOTE: SageMaker base-package conflicts do not affect this isolated environment."

lock_file="${CP_4DA_LOCK_FILE:-$HOME/4danyone-requirements-lock.txt}"
"$PYTHON" -m pip freeze > "$lock_file"
if [[ "${CP_UPLOAD_LOCK_TO_S3:-0}" == "1" ]]; then
  [[ -n "${CP_4DA_BUCKET:-}" ]] || fail "CP_4DA_BUCKET is required."
  aws s3 cp "$lock_file" "s3://${CP_4DA_BUCKET}/environment/4danyone-requirements-lock.txt" \
    --region "${CP_AWS_REGION:-us-east-1}" --no-cli-pager
fi

log "Environment setup completed"
du -sh "$CP_4DA_ENV" "$CP_4DA_REPO_ROOT"
df -h "$HOME"
