"""Request boundaries and real worker dispatch, without AWS/GPU execution."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from recon_pipeline.datasets.fourdanyone.config import (
    DatasetConfiguration,
    extract_4danyone_dataset_config,
    load_pipeline_config,
)
from recon_pipeline.settings import DatasetSettings, PipelineSettings, ArtifactSettings
from recon_pipeline.workers.aws.requests import (
    ExperimentRequest,
    QueueRequest,
    parse_worker_request,
)
from recon_pipeline.workers.aws import worker, queue

ROOT = Path(__file__).parents[1]


def run_document():
    return json.loads((ROOT / "config/splatfacto-run.example.json").read_text())


@pytest.mark.parametrize(
    "filename", ["run.example.json", "splatfacto-run.example.json"]
)
def test_current_json_roundtrip_keeps_typed_nested_settings(filename):
    document = json.loads((ROOT / "config" / filename).read_text())
    parsed = parse_worker_request(document)
    assert isinstance(parsed, ExperimentRequest)
    assert isinstance(parsed.pipeline.dataset.config, DatasetSettings)
    assert parsed.artifacts.dataset.nerfstudio.frames == tuple(
        document["artifacts"]["dataset"]["nerfstudio"]["frames"]
    )
    assert parse_worker_request(json.loads(json.dumps(parsed.to_dict()))) == parsed
    assert (
        parsed.pipeline.reconstruction.config.max_num_iterations
        == document["pipeline"]["reconstruction"]["config"]["max_num_iterations"]
    )
    assert "local" not in parsed.to_dict()["aws_worker"]


@pytest.mark.parametrize("kind", ["experiment", "unknown", None, 5])
def test_unknown_kind_has_clear_error(kind):
    with pytest.raises(ValueError, match="unknown worker request kind"):
        parse_worker_request({**run_document(), "kind": kind})


@pytest.mark.parametrize(
    "field,value",
    [("frame", 60), ("frame_indices", [30, 60]), ("export_device", "cpu")],
)
@pytest.mark.parametrize("flat", [False, True])
def test_legacy_export_fields_are_rejected_with_new_location(field, value, flat):
    document = run_document()
    dataset = document["pipeline"]["dataset"]["config"]
    dataset[field] = value
    if flat:
        document["pipeline"] = dataset
    with pytest.raises(ValueError, match="artifacts.dataset.nerfstudio"):
        parse_worker_request(document)


def test_flat_dataset_no_longer_implicitly_enables_export():
    settings = extract_4danyone_dataset_config(
        PipelineSettings.from_dict({"experiment_name": "flat"})
    )
    assert isinstance(settings, DatasetConfiguration)
    assert settings.dataset_enabled
    assert not settings.nerfstudio.enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_root_nerfstudio_settings_control_export(enabled):
    document = run_document()
    document["pipeline"]["dataset"]["enabled"] = True
    document["pipeline"]["reconstruction"]["enabled"] = False
    document["artifacts"]["reconstruction"]["rerun"]["enabled"] = False
    document["postprocessing"]["splat_conversion"]["enabled"] = False
    document["artifacts"]["dataset"]["nerfstudio"] = {
        "enabled": enabled,
        "frames": [30, 90],
        "device": "cpu",
    }
    parsed = ExperimentRequest.from_dict(document)
    settings = extract_4danyone_dataset_config(
        parsed.pipeline,
        experiment_name=parsed.experiment_name,
        artifacts=parsed.artifacts,
    )
    assert settings.nerfstudio.enabled is enabled
    assert settings.nerfstudio.frames == (30, 90)
    assert settings.nerfstudio.device == "cpu"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("dataset", "seed", True),
        ("dataset", "target_fps", "30"),
        ("dataset", "resume", "false"),
        ("dataset", "layer_pitches", [0.5]),
        ("nerfstudio", "frames", ["60"]),
        ("nerfstudio", "enabled", "false"),
        ("worker", "sync_models", "false"),
        ("cloudwatch", "enabled", "false"),
    ],
)
def test_invalid_nested_types_fail_before_execution(section, key, value):
    document = run_document()
    targets = {
        "dataset": document["pipeline"]["dataset"]["config"],
        "nerfstudio": document["artifacts"]["dataset"]["nerfstudio"],
        "worker": document["aws_worker"],
        "cloudwatch": document["aws_worker"].setdefault(
            "cloudwatch", {"log_group": "/test"}
        ),
    }
    targets[section][key] = value
    with pytest.raises(TypeError, match=key):
        parse_worker_request(document)


def test_queue_contains_typed_experiments_and_roundtrips():
    first = ExperimentRequest.from_dict(run_document())
    second = replace(
        first,
        experiment_name="second",
        aws_worker=replace(first.aws_worker, job_id="second"),
        artifacts=replace(
            first.artifacts,
            reconstruction=replace(
                first.artifacts.reconstruction,
                rerun=replace(
                    first.artifacts.reconstruction.rerun,
                    source_experiment_name="second",
                ),
            ),
        ),
    )
    request = QueueRequest(
        (first, second), "always", first.aws_worker.region, first.aws_worker.sagemaker
    )
    restored = parse_worker_request(json.loads(json.dumps(request.to_dict())))
    assert restored == request
    assert all(isinstance(c, ExperimentRequest) for c in restored.configs)
    assert first.aws_worker.shutdown_on == request.configs[0].aws_worker.shutdown_on


@pytest.mark.parametrize("configs", [[], [{}], [{"kind": "queue"}], "bad"])
def test_invalid_queue_configs_are_rejected(configs):
    first = ExperimentRequest.from_dict(run_document())
    values = QueueRequest(
        (first,), "never", first.aws_worker.region, first.aws_worker.sagemaker
    ).to_dict()
    values["configs"] = configs
    with pytest.raises((ValueError, TypeError)):
        parse_worker_request(values)


def test_worker_dispatches_parsed_queue_before_loading_environment(
    tmp_path, monkeypatch
):
    first = ExperimentRequest.from_dict(run_document())
    request = QueueRequest(
        (first,), "never", first.aws_worker.region, first.aws_worker.sagemaker
    )
    (tmp_path / "request.json").write_text(json.dumps(request.to_dict()))

    def dispatch(path, parsed):
        assert path == tmp_path
        assert isinstance(parsed, QueueRequest)
        assert isinstance(parsed.configs[0], ExperimentRequest)
        return 17

    monkeypatch.setattr(queue, "run_queue", dispatch)
    monkeypatch.setattr(
        worker.PipelineEnvironment,
        "from_environ",
        lambda: pytest.fail("queue must not load machine paths"),
    )
    assert worker.run_worker(tmp_path) == 17


def test_worker_passes_typed_experiment_to_execution(
    tmp_path, pipeline_environment, monkeypatch
):
    document = run_document()
    document["force"] = True
    (tmp_path / "request.json").write_text(json.dumps(document))

    def execute(path, parsed, aws, config, log_session):
        assert isinstance(parsed, ExperimentRequest)
        assert parsed.force is True
        assert config.nerfstudio.frames == parsed.artifacts.dataset.nerfstudio.frames
        assert config.python == str(pipeline_environment.python)
        return 23

    monkeypatch.setattr(worker, "_run_experiment", execute)
    assert worker.run_worker(tmp_path) == 23


def test_local_example_uses_typed_materialization(pipeline_environment):
    config = load_pipeline_config(ROOT / "config/local-run.example.json")
    assert config.video_path == pipeline_environment.data_root / "input/leo.MOV"
    assert config.nerfstudio.enabled


def test_null_optional_sections_use_disabled_defaults():
    document = json.loads((ROOT / "config/run.example.json").read_text())
    document["artifacts"] = None
    document["postprocessing"] = {"splat_conversion": None}
    parsed = ExperimentRequest.from_dict(document)
    assert not parsed.artifacts.dataset.nerfstudio.enabled
    assert not parsed.postprocessing.splat_conversion.enabled


def test_disabled_reconstruction_keeps_known_profile_settings():
    document = json.loads((ROOT / "config/run.example.json").read_text())
    document["pipeline"]["reconstruction"]["config"].update(
        max_num_iterations=1234, unknown_future_setting=True
    )
    parsed = ExperimentRequest.from_dict(document)
    assert parsed.pipeline.reconstruction.config.max_num_iterations == 1234
    assert (
        parsed.to_dict()["pipeline"]["reconstruction"]["config"]["max_num_iterations"]
        == 1234
    )
