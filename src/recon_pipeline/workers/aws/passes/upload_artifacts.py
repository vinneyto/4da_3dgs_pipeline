"""Publish one producer's explicit outputs before starting the next computation."""

import hashlib
from recon_pipeline.core import PassResult
from recon_pipeline.core.utility import run_utility
from ..recovery import artifact_paths


class S3UploadArtifactsPass:
    def __init__(self, worker, config, producer):
        self.worker, self.config, self.producer = worker, config, producer
        self.id = "s3-upload:" + producer.id
        self.name = "Upload artifacts: " + producer.name
        self.requires = producer.provides
        self.provides = frozenset({self.id})
        self.checkpoint_signature = hashlib.sha256(
            (
                producer.checkpoint_signature
                + worker.bucket_name
                + "/"
                + worker.runs_prefix
            ).encode()
        ).hexdigest()

    def validate_checkpoint(self, checkpoint, context):
        return (
            self.producer.id not in context.values.get("executed_passes", set())
            and checkpoint.result.details.get("checkpoint_signature")
            == self.checkpoint_signature
        )

    def run(self, context):
        result = run_utility(
            "recon_pipeline.utilities.storage.s3.publish_artifacts",
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
                "--paths",
                *artifact_paths(self.producer.wrapped, self.config),
                "--checkpoint",
                str(self.config.experiment_dir / ".recon-pipeline/pass-state.json"),
                "--pass-id",
                self.producer.id,
                "--upload-id",
                self.id,
                "--signature",
                self.checkpoint_signature,
            ],
            context,
        )
        return PassResult(artifacts={self.id: result["s3_uri"]}, details=result)
