#!/usr/bin/env bash
# One-time setup on the persistent GPU instance. Never modify the 4DAnyone env.
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 /absolute/path/to/run.json" >&2; exit 2; }
CONFIG_PATH="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
get() { /usr/bin/python3 "$SCRIPT_DIR/read_config.py" "$CONFIG_PATH" "$1"; }

CONDA_BOOTSTRAP="$(get environment.conda_bootstrap)"
SPLATFACTO_ENV="$(get environment.splatfacto_env)"
[[ -f "$CONDA_BOOTSTRAP" ]] || { echo "Conda bootstrap missing: $CONDA_BOOTSTRAP" >&2; exit 1; }
source "$CONDA_BOOTSTRAP"

if [[ ! -x "$SPLATFACTO_ENV/bin/python" ]]; then
  conda create --prefix "$SPLATFACTO_ENV" python=3.11 pip -y
fi
PYTHON="$SPLATFACTO_ENV/bin/python"
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.2.2+cu121' 'torchvision==0.17.2+cu121'
"$PYTHON" -m pip install 'numpy==1.26.4'
"$PYTHON" -m pip install 'numpy==1.26.4' 'setuptools==75.8.0' wheel ninja \
  'nerfstudio==1.1.5' 'gsplat==1.4.0' 'plyfile>=1.1' tensorboard
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.2.2+cu121' 'torchvision==0.17.2+cu121'

export PATH="$SPLATFACTO_ENV/bin:$PATH"
"$PYTHON" - <<'PY'
import torch
import gsplat
from importlib.metadata import version
from gsplat.cuda._backend import _C

assert torch.cuda.is_available(), "Splatfacto requires a CUDA GPU"
assert _C is not None, "gsplat CUDA backend is unavailable"
assert torch.__version__.startswith("2.2.2+cu121"), torch.__version__
assert version("nerfstudio") == "1.1.5"
assert gsplat.__version__ == "1.4.0"
print("Splatfacto ready:", torch.__version__, torch.cuda.get_device_name(0))
PY
test -x "$SPLATFACTO_ENV/bin/ns-train"
test -x "$SPLATFACTO_ENV/bin/ns-export"
