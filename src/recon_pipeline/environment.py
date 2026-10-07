"""Machine settings read once at the orchestration boundary, never by utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

# Defaults belong to installation only. Runtime loading never guesses machine paths.
INSTALL_DEFAULTS = {
    "RECON_CONDA_BOOTSTRAP": "/opt/conda/etc/profile.d/conda.sh",
    "RECON_CONDA_ENV": "{home}/.conda/envs/4danyone",
    "RECON_PIPELINE_ROOT": "{repo}",
    "RECON_FOURDANYONE_ROOT": "{home}/work/4DAnyone",
    "RECON_DATA_ROOT": "{home}/4danyone-data",
    "RECON_FOURDANYONE_GIT_URL": "https://github.com/ant-research/4DAnyone.git",
    "RECON_FOURDANYONE_GIT_REF": "e38f210827f7b3effbe5b573ea07cfcf17e72dca",
    "RECON_PYTHON_VERSION": "3.11",
    "RECON_TORCH_VERSION": "2.8.0",
    "RECON_TORCHVISION_VERSION": "0.23.0",
    "RECON_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu126",
    "RECON_OPENCV_FALLBACK_VERSION": "4.14.0.94",
    "RECON_LOCK_FILE": "{data}/environment/requirements-lock.txt",
    "RECON_NERFSTUDIO_BIN": "{home}/.conda/envs/splatfacto/bin",
    "RECON_SPLATFACTO_RERUN_PYTHON": "{home}/.conda/envs/splatfacto-rerun/bin/python",
    "RECON_SPLATFACTO_TORCH_VERSION": "2.2.2",
    "RECON_SPLATFACTO_TORCHVISION_VERSION": "0.17.2",
    "RECON_SPLATFACTO_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu121",
    "RECON_NERFSTUDIO_VERSION": "1.1.5",
    "RECON_GSPLAT_VERSION": "1.4.0",
    "RECON_CUDA_VERSION": "12.1.1",
}

ENV_NAMES = {
    "conda_bootstrap": "RECON_CONDA_BOOTSTRAP",
    "conda_env": "RECON_CONDA_ENV",
    "pipeline_repo_root": "RECON_PIPELINE_ROOT",
    "fourdanyone_root": "RECON_FOURDANYONE_ROOT",
    "data_root": "RECON_DATA_ROOT",
    "fourdanyone_git_url": "RECON_FOURDANYONE_GIT_URL",
    "fourdanyone_git_ref": "RECON_FOURDANYONE_GIT_REF",
    "python_version": "RECON_PYTHON_VERSION",
    "torch_version": "RECON_TORCH_VERSION",
    "torchvision_version": "RECON_TORCHVISION_VERSION",
    "torch_index_url": "RECON_TORCH_INDEX_URL",
    "opencv_fallback_version": "RECON_OPENCV_FALLBACK_VERSION",
    "lock_file": "RECON_LOCK_FILE",
    "nerfstudio_bin": "RECON_NERFSTUDIO_BIN",
    "splatfacto_rerun_python": "RECON_SPLATFACTO_RERUN_PYTHON",
    "splatfacto_torch_version": "RECON_SPLATFACTO_TORCH_VERSION",
    "splatfacto_torchvision_version": "RECON_SPLATFACTO_TORCHVISION_VERSION",
    "splatfacto_torch_index_url": "RECON_SPLATFACTO_TORCH_INDEX_URL",
    "nerfstudio_version": "RECON_NERFSTUDIO_VERSION",
    "gsplat_version": "RECON_GSPLAT_VERSION",
    "cuda_version": "RECON_CUDA_VERSION",
}
PATH_FIELDS = frozenset(
    {
        "conda_bootstrap",
        "conda_env",
        "pipeline_repo_root",
        "fourdanyone_root",
        "data_root",
        "lock_file",
        "nerfstudio_bin",
        "splatfacto_rerun_python",
    }
)


def environment_values(environ: Mapping[str, str]) -> dict[str, str]:
    values = dict(environ)
    # Name used in the existing standalone Splatfacto instructions.
    if not values.get("RECON_NERFSTUDIO_BIN") and values.get("RECON_NS_BIN"):
        values["RECON_NERFSTUDIO_BIN"] = values["RECON_NS_BIN"]
    return values


@dataclass(frozen=True, slots=True)
class PipelineEnvironment:
    conda_bootstrap: Path
    conda_env: Path
    pipeline_repo_root: Path
    fourdanyone_root: Path
    data_root: Path
    fourdanyone_git_url: str
    fourdanyone_git_ref: str
    python_version: str
    torch_version: str
    torchvision_version: str
    torch_index_url: str
    opencv_fallback_version: str
    lock_file: Path
    nerfstudio_bin: Path
    splatfacto_rerun_python: Path
    splatfacto_torch_version: str
    splatfacto_torchvision_version: str
    splatfacto_torch_index_url: str
    nerfstudio_version: str
    gsplat_version: str
    cuda_version: str

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> PipelineEnvironment:
        values = environment_values(os.environ if environ is None else environ)
        missing = [
            name for name in ENV_NAMES.values() if not values.get(name, "").strip()
        ]
        if missing:
            raise ValueError(
                "Missing environment variables:\n"
                + "\n".join(f" - {name}" for name in missing)
                + "\nRun ./scripts/check_environment.sh; configure with ./scripts/setup_environment.sh --configure-only"
            )
        kwargs = {}
        for field in fields(cls):
            value = values[ENV_NAMES[field.name]]
            if field.name in PATH_FIELDS:
                value = Path(value).expanduser()
                if not value.is_absolute():
                    raise ValueError(
                        f"{ENV_NAMES[field.name]} must be absolute: {value}"
                    )
            kwargs[field.name] = value
        return cls(**kwargs)

    @property
    def python(self) -> Path:
        return self.conda_env / "bin/python"

    def materialize(self, payload: dict[str, Any], video: Path) -> dict[str, Any]:
        values = dict(payload)
        values.pop("video", None)
        values.update(
            video_path=video,
            fourdanyone_root=self.fourdanyone_root,
            model_dir=self.data_root / "models",
            runs_dir=self.data_root / "runs",
            python=str(self.python),
        )
        values["reconstruction"] = {
            **values.get("reconstruction", {}),
            "nerfstudio_bin": str(self.nerfstudio_bin),
        }
        rerun = values.get("reconstruction_rerun") or {}
        if isinstance(rerun, bool):
            rerun = {"enabled": rerun}
        values["reconstruction_rerun"] = {
            **rerun,
            "python": str(self.splatfacto_rerun_python),
        }
        return values
