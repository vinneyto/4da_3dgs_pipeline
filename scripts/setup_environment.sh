#!/usr/bin/env bash
# Install the persistent tools; no run-specific paths or versions come from JSON.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${RECON_ENV_FILE:-$HOME/.config/recon-pipeline/environment.sh}"
CONFIGURE_ONLY=0
REUSE_FOURDANYONE=0
RUN_CONFIG=""
SETUP_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure-only) CONFIGURE_ONLY=1; shift ;;
    --reuse-4danyone) REUSE_FOURDANYONE=1; shift ;;
    --config) RUN_CONFIG="${2:?--config requires a run JSON}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file requires a path}"; shift 2 ;;
    --log-file) SETUP_LOG="${2:?--log-file requires a path}"; shift 2 ;;
    --help|-h) echo "usage: $0 [--configure-only] [--reuse-4danyone] [--config RUN.json] [--env-file FILE] [--log-file FILE]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo "Installation expects Linux x86_64 (SageMaker/CUDA)." >&2; exit 1;
}
SETUP_STAGE="environment configuration"
INSTALLER=""
if [[ "$CONFIGURE_ONLY" -eq 0 ]]; then
  SETUP_LOG="${SETUP_LOG:-$REPO_ROOT/environment-setup-$(date -u +%Y%m%dT%H%M%SZ)-$$.log}"
  SETUP_LOG="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$SETUP_LOG")"
  mkdir -p "$(dirname "$SETUP_LOG")"
  printf 'Setup log: %s\n' "$SETUP_LOG"
  exec > >(tee -a "$SETUP_LOG") 2>&1
  setup_exit() {
    local setup_status=$?
    if [[ "$setup_status" -ne 0 ]]; then
      printf '\nERROR: setup stopped at "%s" (exit %s). Log: %s\n' "$SETUP_STAGE" "$setup_status" "$SETUP_LOG" >&2
    fi
    [[ -z "$INSTALLER" ]] || rm -f "$INSTALLER"
    return "$setup_status"
  }
  trap setup_exit EXIT
fi
stage() { SETUP_STAGE="$1"; printf '\n==> %s\n' "$SETUP_STAGE"; }
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ENV_FILE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$ENV_FILE")"
# The configurator merges previous settings with the current exported variables.
python3 -m recon_pipeline.environment_cli configure --env-file "$ENV_FILE" --repo "$REPO_ROOT"
source "$ENV_FILE"
[[ "$CONFIGURE_ONLY" -eq 0 ]] || exit 0

if [[ ! -f "$RECON_CONDA_BOOTSTRAP" ]]; then
  stage "Conda bootstrap installation"
  command -v curl >/dev/null || { echo "curl is required to install Conda" >&2; exit 1; }
  CONDA_PREFIX_ROOT="${RECON_CONDA_BOOTSTRAP%/etc/profile.d/conda.sh}"
  [[ "$CONDA_PREFIX_ROOT" != "$RECON_CONDA_BOOTSTRAP" ]] || {
    echo "RECON_CONDA_BOOTSTRAP must end with /etc/profile.d/conda.sh" >&2; exit 1;
  }
  INSTALLER="$(mktemp)"
  curl --fail --location --silent --show-error \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$INSTALLER"
  bash "$INSTALLER" -b -p "$CONDA_PREFIX_ROOT"
  rm -f "$INSTALLER"
  INSTALLER=""
fi
stage "Conda initialization"
source "$RECON_CONDA_BOOTSTRAP"
stage "Workspace directories"
for directory in input models runs jobs environment; do
  mkdir -p "$RECON_DATA_ROOT/$directory"
done
if [[ "$REUSE_FOURDANYONE" -eq 1 ]]; then
  stage "4DAnyone worker dependencies (reuse existing environment)"
  WORKER_PYTHON="$RECON_CONDA_ENV/bin/python"
  [[ -x "$WORKER_PYTHON" ]] || {
    echo "Cannot reuse 4DAnyone: Python not found at $WORKER_PYTHON" >&2; exit 1;
  }
  echo "Reusing existing 4DAnyone; installing worker dependencies only"
  "$WORKER_PYTHON" -m pip install --editable "$RECON_PIPELINE_ROOT[aws,rerun]"
  mkdir -p "$(dirname "$RECON_LOCK_FILE")"
  "$WORKER_PYTHON" -m pip freeze > "$RECON_LOCK_FILE"
else
  stage "4DAnyone environment installation"
  "$SCRIPT_DIR/setup_4danyone_env.sh"
fi

SPLATFACTO_ENV="$(dirname "$RECON_NERFSTUDIO_BIN")"
if [[ ! -x "$RECON_NERFSTUDIO_BIN/python" ]]; then
  stage "Splatfacto Conda environment creation"
  conda create --prefix "$SPLATFACTO_ENV" "python=$RECON_PYTHON_VERSION" pip -y
fi
# gsplat JIT needs a compiler/toolkit even when PyTorch wheels contain CUDA runtime.
stage "Splatfacto CUDA toolkit and C++ compiler"
conda install --prefix "$SPLATFACTO_ENV" -y -c conda-forge -c nvidia \
  "cuda-toolkit=$RECON_CUDA_VERSION" gxx_linux-64=11 ninja
stage "Splatfacto Conda activation"
conda activate "$SPLATFACTO_ENV"
export CUDA_HOME="$SPLATFACTO_ENV"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
NS_PYTHON="$RECON_NERFSTUDIO_BIN/python"
stage "Splatfacto Python packaging tools"
"$NS_PYTHON" -m pip install --upgrade pip setuptools wheel
stage "Splatfacto PyTorch CUDA wheels"
"$NS_PYTHON" -m pip install \
  "torch==$RECON_SPLATFACTO_TORCH_VERSION" "torchvision==$RECON_SPLATFACTO_TORCHVISION_VERSION" \
  --index-url "$RECON_SPLATFACTO_TORCH_INDEX_URL"
stage "Nerfstudio and gsplat packages"
"$NS_PYTHON" -m pip install "nerfstudio==$RECON_NERFSTUDIO_VERSION" "gsplat==$RECON_GSPLAT_VERSION" \
  'numpy>=1.26,<2' ninja
# Nerfstudio installs GUI OpenCV; this worker renders headlessly.
stage "Splatfacto headless OpenCV and pipeline package"
"$NS_PYTHON" -m pip uninstall --yes opencv-python opencv-python-headless
"$NS_PYTHON" -m pip install 'opencv-python-headless>=4.8,<4.12' 'numpy>=1.26,<2'
"$NS_PYTHON" -m pip install --editable "$RECON_PIPELINE_ROOT"
"$NS_PYTHON" -m pip freeze > "$RECON_DATA_ROOT/environment/splatfacto-requirements-lock.txt"

RERUN_ENV="$(dirname "$(dirname "$RECON_SPLATFACTO_RERUN_PYTHON")")"
stage "Splatfacto Rerun environment installation"
if [[ ! -x "$RECON_SPLATFACTO_RERUN_PYTHON" ]]; then
  conda create --prefix "$RERUN_ENV" "python=$RECON_PYTHON_VERSION" pip -y
fi
"$RECON_SPLATFACTO_RERUN_PYTHON" -m pip install --editable "$RECON_PIPELINE_ROOT[splatfacto-rerun]"
"$RECON_SPLATFACTO_RERUN_PYTHON" -m pip freeze > "$RECON_DATA_ROOT/environment/splatfacto-rerun-requirements-lock.txt"

# Full installation selects the newly installed toolkit. Configure-only keeps
# the existing CUDA_HOME (or the toolkit found on PATH).
stage "Persist installed toolkit settings"
python3 -m recon_pipeline.environment_cli configure --env-file "$ENV_FILE" --repo "$REPO_ROOT"
source "$ENV_FILE"

if [[ -n "$RUN_CONFIG" ]]; then
  stage "Model weights"
  "$SCRIPT_DIR/download_4danyone_models.sh" "$RUN_CONFIG"
else
  echo "Model weights: run ./scripts/download_4danyone_models.sh RUN.json (licensed SMPL-X archive must be in S3)."
fi
echo "Tools installed. In your calling terminal: source $ENV_FILE"
stage "Final environment check"
"$SCRIPT_DIR/check_environment.sh"
