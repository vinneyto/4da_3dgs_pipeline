"""AWS recovery policy; computation and storage remain separate CLI operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from recon_pipeline.core import JsonPassCheckpointStore, PipelineContext
from recon_pipeline.core.utility import run_utility


def fingerprint(pass_id, settings, input_identity):
    inference = {
        key: settings[key]
        for key in (
            "video",
            "views_per_layer",
            "layer_pitches",
            "start_yaw",
            "yaw_span",
            "target_fps",
            "seed",
            "enable_turbo",
            "attention_backend",
        )
    }
    base = {"input": input_identity, "dataset": inference}
    if pass_id == "prepare-experiment" or pass_id == "write-run-manifest":
        base["settings"] = settings
    elif pass_id == "nerfstudio-export":
        base["export"] = settings["nerfstudio"]
    elif pass_id.startswith(("splatfacto:", "splatfacto-rerun:", "splat-convert:")):
        export = dict(settings["nerfstudio"])
        export.pop("frames", None)
        export.pop("replace_existing", None)
        base["export"] = export
        training = dict(settings["reconstruction"])
        training.pop("frames", None)
        training.pop("enabled", None)
        base["training"] = training
        base["frame"] = pass_id.rsplit("_", 1)[-1]
        if pass_id.startswith("splat-convert:"):
            base["conversion"] = settings.get("postprocessing", {}).get(
                "splat_conversion", {}
            )
        if pass_id.startswith("splatfacto-rerun:"):
            base["recording"] = settings["reconstruction_rerun"]
    elif pass_id == "rerun-export":
        base["recording"] = settings["rerun"]
    return hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()


def artifact_paths(pipeline_pass, config):
    if pipeline_pass.id == "prepare-experiment":
        return ["pipeline-config.json"]
    if pipeline_pass.id == "fourdanyone-inference":
        return ["4danyone"]
    if pipeline_pass.id == "nerfstudio-export":
        return [f"nerfstudio/frame_{frame:03d}" for frame in config.nerfstudio.frames]
    if pipeline_pass.id.startswith("splatfacto:"):
        return [f"splatfacto/frame_{pipeline_pass.frame:03d}"]
    if pipeline_pass.id.startswith("splatfacto-rerun:"):
        return [f"splatfacto/frame_{pipeline_pass.frame:03d}/rerun"]
    if pipeline_pass.id.startswith("splat-convert:"):
        return [f"postprocessing/splat_conversion/frame_{pipeline_pass.frame:03d}"]
    if pipeline_pass.id == "rerun-export":
        return [config.rerun_path.relative_to(config.experiment_dir).as_posix()]
    if pipeline_pass.id == "write-run-manifest":
        return ["pipeline-result.json"]
    return []


class RecoverablePass:
    def __init__(self, wrapped, config, signature, legacy_signature=None):
        self.wrapped = wrapped
        self.config = config
        self.checkpoint_signature = signature
        self.legacy_signature = legacy_signature
        for name in ("id", "name", "requires", "provides"):
            setattr(self, name, getattr(wrapped, name))

    def run(self, context):
        return self.wrapped.run(context)

    def validate_checkpoint(self, checkpoint, context):
        signature = checkpoint.result.details.get(
            "checkpoint_signature", self.legacy_signature
        )
        if signature != self.checkpoint_signature:
            return False
        result = checkpoint.result
        # A larger export selection must never reuse a smaller completed selection.
        if self.id == "nerfstudio-export":
            datasets = result.artifacts.get("dataset.nerfstudio", [])
            if {item["frame"] for item in datasets} != set(
                self.config.nerfstudio.frames
            ):
                return False
            return all(
                (Path(item["dataset_dir"]) / "transforms.json").is_file()
                for item in datasets
            )
        if self.id == "fourdanyone-inference":
            return all(
                (self.config.inference_dir / file).is_file()
                for file in ("metadata.json", "cameras.json")
            )
        if self.id.startswith("splatfacto:"):
            trained = next(iter(result.artifacts.values()))
            return (
                Path(trained["splat_ply"]).is_file()
                and Path(trained["config"]).is_file()
            )
        if self.id.startswith("splat-convert:"):
            converted = next(iter(result.artifacts.values()))
            return (
                set(converted.get("exported_artifacts", {}))
                == set(self.config.postprocessing.splat_conversion.formats)
                and (Path(converted["output_dir"]) / "manifest.json").is_file()
                and all(
                    Path(path).is_file() and Path(path).stat().st_size > 0
                    for path in converted["exported_artifacts"].values()
                )
            )
        return all(
            (self.config.experiment_dir / name).exists()
            for name in artifact_paths(self.wrapped, self.config)
        )


class RecoveryCheckpointStore(JsonPassCheckpointStore):
    def __init__(self, worker, config, upload_passes):
        super().__init__(
            config.experiment_dir / ".recon-pipeline/pass-state.json",
            config.experiment_name,
        )
        self.worker, self.config, self.upload_passes = worker, config, upload_passes
        self.restored = False
        self.forced = False

    def clear(self):
        self.forced = True
        self._recover(reset=True)
        super().clear()

    def _recover(self, reset=False):
        context = PipelineContext(_progress_callback=lambda fraction, message: None)
        return run_utility(
            "recon_pipeline.utilities.storage.s3.recover_run",
            [
                "--bucket",
                self.worker.bucket_name,
                "--region",
                self.worker.region,
                "--prefix",
                "/".join(
                    filter(None, (self.worker.runs_prefix, self.config.experiment_name))
                ),
                "--root",
                str(self.config.experiment_dir),
                "--data-root",
                str(self.worker.local.data_root),
                "--checkpoint",
                str(self.path),
                "--plan",
                json.dumps(
                    [
                        {
                            "id": item.producer.id,
                            "producer_signature": item.producer.checkpoint_signature,
                            "upload_id": item.id,
                            "signature": item.checkpoint_signature,
                        }
                        for item in self.upload_passes
                    ]
                ),
                *(["--reset-commits"] if reset else []),
            ],
            context,
        )

    def load(self):
        if not self.restored and not self.forced:
            self.restored = True
            self._recover()
        return super().load()
