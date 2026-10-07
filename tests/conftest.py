from pathlib import Path

import pytest

from recon_pipeline.environment import INSTALL_DEFAULTS, PipelineEnvironment


@pytest.fixture
def pipeline_environment(monkeypatch, tmp_path):
    values = {}
    for name, default in INSTALL_DEFAULTS.items():
        values[name] = default.format(
            home=tmp_path,
            repo=Path(__file__).parents[1],
            data=tmp_path / "data",
        )
    values["RECON_DATA_ROOT"] = str(tmp_path / "data")
    values["RECON_FOURDANYONE_ROOT"] = str(tmp_path / "4DAnyone")
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return PipelineEnvironment.from_environ(values)
