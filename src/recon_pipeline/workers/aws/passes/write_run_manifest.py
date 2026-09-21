"""Persist the final local manifest for a pipeline run."""

import json
from datetime import UTC, datetime

from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE
from recon_pipeline.datasets.fourdanyone.config import FourDAnyoneConfig
from recon_pipeline.reconstructions.nerfstudio.passes import NERFSTUDIO_DATASETS
from recon_pipeline.core import PassResult, PipelineContext
from recon_pipeline.artifacts.rerun.passes import RERUN_RECORDING

from ..artifacts import RUN_RESULT


class WriteRunManifestPass:
    id = "write-run-manifest"
    name = "Write run manifest"
    requires = frozenset({EXPERIMENT_WORKSPACE})
    provides = frozenset({RUN_RESULT})

    def __init__(self, config: FourDAnyoneConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PassResult:
        durations = dict(context.values.get("pass_durations", {}))
        result = {
            "experiment_name": self.config.experiment_name,
            "experiment_dir": str(self.config.experiment_dir),
            "inference_dir": str(self.config.inference_dir),
            "datasets": context.artifacts.get(NERFSTUDIO_DATASETS, []),
            "rerun_file": (
                str(context.artifacts[RERUN_RECORDING])
                if RERUN_RECORDING in context.artifacts
                else None
            ),
            "num_views": self.config.num_views,
            "pass_durations_seconds": durations,
            "elapsed_seconds": sum(durations.values()),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        path = self.config.experiment_dir / "pipeline-result.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        context.report_progress(1.0, "Run manifest written")
        return PassResult(
            artifacts={RUN_RESULT: result},
            details={"path": str(path)},
        )
