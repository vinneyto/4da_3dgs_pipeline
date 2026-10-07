#!/usr/bin/env bash
# Install the persistent tools; no run-specific paths or versions come from JSON.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${RECON_ENV_FILE:-$HOME/.config/recon-pipeline/environment.sh}"
CONFIGURE_ONLY=0
RUN_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configure-only) CONFIGURE_ONLY=1; shift ;;
    --config) RUN_CONFIG="${2:?--config requires a run JSON}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file requires a path}"; shift 2 ;;
    --help|-h) echo "usage: $0 [--configure-only] [--config RUN.json] [--env-file FILE]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo "Installation expects Linux x86_64 (SageMaker/CUDA)." >&2; exit 1;
}
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ENV_FILE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$ENV_FILE")"
# The configurator merges previous settings with the current exported variables.
python3 -m recon_pipeline.environment_cli configure --env-file "$ENV_FILE" --repo "$REPO_ROOT"
source "$ENV_FILE"
[[ "$CONFIGURE_ONLY" -eq 0 ]] || exit 0

if [[ ! -f "$RECON_CONDA_BOOTSTRAP" ]]; then
  command -v curl >/dev/null || { echo "curl is required to install Conda" >&2; exit 1; }
  CONDA_PREFIX_ROOT="${RECON_CONDA_BOOTSTRAP%/etc/profile.d/conda.sh}"
  [[ "$CONDA_PREFIX_ROOT" != "$RECON_CONDA_BOOTSTRAP" ]] || {
    echo "RECON_CONDA_BOOTSTRAP must end with /etc/profile.d/conda.sh" >&2; exit 1;
  }
  INSTALLER="$(mktemp)"
  trap 'rm -f "$INSTALLER"' EXIT
  curl --fail --location --silent --show-error \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$INSTALLER"
  bash "$INSTALLER" -b -p "$CONDA_PREFIX_ROOT"
  rm -f "$INSTALLER"
  trap - EXIT
fi
source "$RECON_CONDA_BOOTSTRAP"
for directory in input models runs jobs environment; do
  mkdir -p "$RECON_DATA_ROOT/$directory"
done
"$SCRIPT_DIR/setup_4danyone_env.sh"

SPLATFACTO_ENV="$(dirname "$RECON_NERFSTUDIO_BIN")"
if [[ ! -x "$RECON_NERFSTUDIO_BIN/python" ]]; then
  conda create --prefix "$SPLATFACTO_ENV" "python=$RECON_PYTHON_VERSION" pip -y
fi
# gsplat JIT needs a compiler/toolkit even when PyTorch wheels contain CUDA runtime.
conda install --prefix "$SPLATFACTO_ENV" -y -c conda-forge -c nvidia \
  "cuda-toolkit=$RECON_CUDA_VERSION" gxx_linux-64=11 ninja
conda activate "$SPLATFACTO_ENV"
export CUDA_HOME="$SPLATFACTO_ENV"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
NS_PYTHON="$RECON_NERFSTUDIO_BIN/python"
"$NS_PYTHON" -m pip install --upgrade pip setuptools wheel
"$NS_PYTHON" -m pip install \
  "torch==$RECON_SPLATFACTO_TORCH_VERSION" "torchvision==$RECON_SPLATFACTO_TORCHVISION_VERSION" \
  --index-url "$RECON_SPLATFACTO_TORCH_INDEX_URL"
"$NS_PYTHON" -m pip install "nerfstudio==$RECON_NERFSTUDIO_VERSION" "gsplat==$RECON_GSPLAT_VERSION" \
  'numpy>=1.26,<2' ninja
# Nerfstudio installs GUI OpenCV; this worker renders headlessly.
"$NS_PYTHON" -m pip uninstall --yes opencv-python opencv-python-headless
"$NS_PYTHON" -m pip install 'opencv-python-headless>=4.8,<4.12' 'numpy>=1.26,<2'
"$NS_PYTHON" -m pip install --editable "$RECON_PIPELINE_ROOT"
"$NS_PYTHON" -m pip freeze > "$RECON_DATA_ROOT/environment/splatfacto-requirements-lock.txt"

RERUN_ENV="$(dirname "$(dirname "$RECON_SPLATFACTO_RERUN_PYTHON")")"
if [[ ! -x "$RECON_SPLATFACTO_RERUN_PYTHON" ]]; then
  conda create --prefix "$RERUN_ENV" "python=$RECON_PYTHON_VERSION" pip -y
fi
"$RECON_SPLATFACTO_RERUN_PYTHON" -m pip install --editable "$RECON_PIPELINE_ROOT[splatfacto-rerun]"
"$RECON_SPLATFACTO_RERUN_PYTHON" -m pip freeze > "$RECON_DATA_ROOT/environment/splatfacto-rerun-requirements-lock.txt"

# Full installation selects the newly installed toolkit. Configure-only keeps
# the existing CUDA_HOME (or the toolkit found on PATH).
python3 -m recon_pipeline.environment_cli configure --env-file "$ENV_FILE" --repo "$REPO_ROOT"
source "$ENV_FILE"

if [[ -n "$RUN_CONFIG" ]]; then
  "$SCRIPT_DIR/download_4danyone_models.sh" "$RUN_CONFIG"
else
  echo "Model weights: run ./scripts/download_4danyone_models.sh RUN.json (licensed SMPL-X archive must be in S3)."
fi
echo "Tools installed. In your calling terminal: source $ENV_FILE"
"$SCRIPT_DIR/check_environment.sh"
