import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import replace

import pytest

from recon_pipeline.postprocessing.config import (
    PostprocessingConfig,
    SplatConversionConfig,
)


def source_ply(tmp_path):
    source = tmp_path / "source.ply"
    source.write_bytes(b"fixture source PLY")
    return source


def conversion_command(tmp_path, formats, converter):
    return [
        sys.executable,
        "-m",
        "recon_pipeline.utilities.artefacts.splats.convert",
        "--input",
        str(source_ply(tmp_path)),
        "--output",
        str(tmp_path / "result"),
        "--splat-transform",
        str(converter),
        "--formats",
        *formats,
    ]


@pytest.fixture
def converter(tmp_path):
    executable = tmp_path / "converter tools/splat-transform"
    executable.parent.mkdir()
    executable.write_text(f"#!{sys.executable}\n" + """
import os, sys
from pathlib import Path
if os.environ.get("FAKE_CONVERT_FAIL"):
    print("conversion failed", file=sys.stderr)
    sys.exit(3)
assert sys.argv[1:4] == ["--quiet", "--gpu", "cpu"]
assert Path(sys.argv[4]).is_file()
if not os.environ.get("FAKE_CONVERT_NO_OUTPUT"):
    Path(sys.argv[5]).write_bytes(b"converted fixture")
""")
    executable.chmod(0o755)
    return executable


def test_multiple_delivery_formats_retain_source_and_manifest(tmp_path, converter):
    command = conversion_command(tmp_path, ["compressed_ply", "spz", "sog"], converter)
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])["data"]
    assert result["formats"] == ["compressed_ply", "spz", "sog"]
    assert set(result["exported_artifacts"]) == set(result["formats"])
    assert {Path(p).name for p in result["exported_artifacts"].values()} == {
        "splat.compressed.ply",
        "splat.spz",
        "splat.sog",
    }
    assert all(Path(p).stat().st_size for p in result["exported_artifacts"].values())
    assert Path(result["source_ply"]).read_bytes() == b"fixture source PLY"
    assert json.loads((tmp_path / "result/manifest.json").read_text()) == result


@pytest.mark.parametrize("flag", ["FAKE_CONVERT_FAIL", "FAKE_CONVERT_NO_OUTPUT"])
def test_failed_conversion_does_not_complete_frame(tmp_path, converter, flag):
    command = conversion_command(tmp_path, ["ply", "spz"], converter)
    completed = subprocess.run(
        command, capture_output=True, text=True, env={**os.environ, flag: "1"}
    )
    assert completed.returncode != 0
    assert not (tmp_path / "result/manifest.json").exists()
    assert not any(
        json.loads(line).get("event") == "result"
        for line in completed.stdout.splitlines()
    )


def test_missing_converter_fails_before_postprocessing(tmp_path):
    command = conversion_command(tmp_path, ["spz"], tmp_path / "missing")
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0 and "Install splat-transform" in completed.stderr
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("formats", [[], ["ply", "ply"], ["splat"], "spz", [None]])
def test_export_formats_are_validated(formats):
    with pytest.raises(ValueError, match="formats"):
        SplatConversionConfig(formats=formats)


# Use real subprocess-backed training fixtures to test resuming across the stage boundary.
from test_splatfacto import binaries, dataset


def test_failed_conversion_and_format_changes_resume_without_retraining(
    tmp_path, binaries, converter, monkeypatch
):
    from recon_pipeline.core import Pipeline, PipelineContext, JsonPassCheckpointStore
    from recon_pipeline.core.command import CommandError
    from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
    from recon_pipeline.reconstructions.nerfstudio.passes import NERFSTUDIO_DATASETS
    from recon_pipeline.reconstructions.nerfstudio.passes.splatfacto import (
        SplatfactoPass,
        splatfacto_artifact,
    )
    from recon_pipeline.reconstructions.nerfstudio.splatfacto_config import (
        SplatfactoConfig,
    )
    from recon_pipeline.postprocessing.splat_conversion import (
        SplatConversionPass,
        converted_splat_artifact,
    )
    from recon_pipeline.workers.aws.recovery import RecoverablePass, fingerprint
    from recon_pipeline.workers.aws.passes import WriteRunManifestPass
    from test_aws_pipeline_plan import make_pipeline_config

    source = dataset(tmp_path / "dataset")
    config = make_pipeline_config(
        tmp_path,
        reconstruction=SplatfactoConfig(enabled=True, nerfstudio_bin=str(binaries)),
        postprocessing=PostprocessingConfig(
            SplatConversionConfig(
                enabled=True, formats=("spz",), splat_transform=str(converter)
            )
        ),
    )
    store = JsonPassCheckpointStore(
        config.experiment_dir / ".recon-pipeline/pass-state.json",
        config.experiment_name,
    )

    def run(config):
        passes = [
            SplatfactoPass(config, 60),
            SplatConversionPass(config, 60),
            WriteRunManifestPass(config),
        ]
        wrapped = [
            RecoverablePass(
                p,
                config,
                fingerprint(p.id, config.settings_dict(), ("bucket", "input", "video")),
            )
            for p in passes
        ]
        context = PipelineContext(
            artifacts={
                EXPERIMENT_WORKSPACE: config.experiment_dir,
                NERFSTUDIO_DATASETS: [{"frame": 60, "dataset_dir": str(source)}],
            }
        )
        Pipeline(wrapped, checkpoint_store=store, resume_by_id=True).run(context)
        return context

    monkeypatch.setenv("FAKE_CONVERT_FAIL", "1")
    with pytest.raises(CommandError, match="exit code"):
        run(config)
    assert list(store.load()) == ["splatfacto:frame_060"]
    trained = store.load()["splatfacto:frame_060"].result.artifacts[
        splatfacto_artifact(60)
    ]
    original_ply = Path(trained["splat_ply"]).read_bytes()
    monkeypatch.delenv("FAKE_CONVERT_FAIL")
    run(config)
    changed = replace(
        config,
        postprocessing=PostprocessingConfig(
            replace(
                config.postprocessing.splat_conversion,
                formats=("compressed_ply", "sog"),
            )
        ),
    )
    context = run(changed)
    result = context.require(converted_splat_artifact(60))
    assert set(result["exported_artifacts"]) == {"compressed_ply", "sog"}
    assert Path(trained["splat_ply"]).read_bytes() == original_ply
    assert not (Path(result["output_dir"]) / "splat.spz").exists()
    calls = [
        json.loads(line) for line in (binaries / "calls.jsonl").read_text().splitlines()
    ]
    assert sum(c["tool"] == "ns-train" for c in calls) == 1
    manifest = json.loads((config.experiment_dir / "pipeline-result.json").read_text())
    assert manifest["postprocessing"]["splat_conversion"] == [result]
    assert manifest["reconstructions"][0]["splat_ply"] == trained["splat_ply"]
    # A missing processed output invalidates only the conversion checkpoint.
    Path(result["exported_artifacts"]["sog"]).unlink()
    run(changed)
    assert Path(result["exported_artifacts"]["sog"]).is_file()
    calls = [
        json.loads(line) for line in (binaries / "calls.jsonl").read_text().splitlines()
    ]
    assert sum(c["tool"] == "ns-train" for c in calls) == 1


def test_postprocessing_requires_reconstructed_artifacts(tmp_path):
    from test_aws_pipeline_plan import make_pipeline_config

    with pytest.raises(ValueError, match="requires an enabled reconstruction"):
        make_pipeline_config(
            tmp_path, postprocessing={"splat_conversion": {"enabled": True}}
        )


def test_postprocessing_preserves_source_when_output_overlaps(tmp_path, converter):
    command = conversion_command(tmp_path, ["spz"], converter)
    command[command.index("--output") + 1] = str(tmp_path)
    completed = subprocess.run(
        [*command, "--replace-existing"], capture_output=True, text=True
    )
    assert completed.returncode != 0 and "must not contain" in completed.stderr
    assert (tmp_path / "source.ply").read_bytes() == b"fixture source PLY"
