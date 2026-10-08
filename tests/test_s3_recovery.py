import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recon_pipeline.core import (
    JsonPassCheckpointStore,
    PassResult,
    Pipeline,
    PipelineContext,
    PipelineOutcome,
)
from recon_pipeline.utilities.storage.s3 import bundles
from recon_pipeline.workers.aws.persistence import (
    S3PersistenceObserver,
    exception_report,
)
from recon_pipeline.workers.aws.recovery import RecoverablePass, fingerprint
from test_aws_pipeline_plan import make_pipeline_config
from test_aws_health import make_config as make_worker


class Missing(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.events = []
        self.fail_upload = None
        self.fail_commit = False

    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Missing()
        return {"Body": io.BytesIO(self.objects[Bucket, Key])}

    def put_object(self, *, Bucket, Key, Body, **kw):
        if self.fail_commit and "/commits/" in Key:
            raise OSError("commit unavailable")
        self.events.append(("put", Key))
        self.objects[Bucket, Key] = Body

    def delete_object(self, *, Bucket, Key):
        self.events.append(("delete", Key))
        self.objects.pop((Bucket, Key), None)

    def upload_file(self, file, bucket, key):
        if key.endswith(self.fail_upload or "<never>"):
            raise OSError("upload unavailable")
        self.events.append(("upload", key))
        self.objects[bucket, key] = Path(file).read_bytes()

    def download_file(self, bucket, key, file):
        self.events.append(("download", key))
        Path(file).write_bytes(self.objects[bucket, key])

    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[Bucket, Key])}


def fixture_bundle(tmp_path, client=None):
    data = tmp_path / "old-machine"
    root = data / "runs/example"
    (root / "splatfacto/frame_000").mkdir(parents=True)
    (root / "splatfacto/frame_000/splat.ply").write_bytes(b"gaussians")
    (root / "splatfacto/frame_000/model.ckpt").write_bytes(b"checkpoint")
    checkpoint = {
        "id": "splatfacto:frame_000",
        "completed_at": "now",
        "duration_seconds": 4,
        "artifacts": {
            "trained": {"splat_ply": str(root / "splatfacto/frame_000/splat.ply")}
        },
        "details": {"checkpoint_signature": "producer"},
    }
    options = dict(
        bucket="bucket",
        prefix="runs/example",
        root=root,
        data_root=data,
        paths=["splatfacto/frame_000"],
        checkpoint=checkpoint,
        upload_id="s3-upload:splatfacto:frame_000",
        signature="upload",
        region="region",
        client=client or MemoryS3(),
    )
    plan = [
        dict(
            id=checkpoint["id"],
            producer_signature="producer",
            upload_id=options["upload_id"],
            signature="upload",
        )
    ]
    return options, plan


def test_commit_is_last_and_recovery_rebases_paths_on_empty_disk(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    client = options["client"]
    bundles.publish(**options)
    assert client.events[-1] == (
        "put",
        bundles.marker_key("runs/example", plan[0]["id"]),
    )
    marker = bundles.read_json(client, "bucket", client.events[-1][1])
    assert "old-machine" not in json.dumps(marker)
    new_data = tmp_path / "new-machine"
    root = new_data / "runs/example"
    checkpoint_path = root / ".recon-pipeline/pass-state.json"
    result = bundles.recover(
        bucket="bucket",
        prefix="runs/example",
        root=root,
        data_root=new_data,
        plan=plan,
        checkpoint_path=checkpoint_path,
        region="region",
        client=client,
    )
    assert result == {"restored": 1, "downloaded": 2}
    checkpoint = json.loads(checkpoint_path.read_text())["passes"]
    assert checkpoint[0]["artifacts"]["trained"]["splat_ply"] == str(
        root / "splatfacto/frame_000/splat.ply"
    )
    assert checkpoint[1]["id"] == options["upload_id"]
    result = bundles.recover(
        bucket="bucket",
        prefix="runs/example",
        root=root,
        data_root=new_data,
        plan=plan,
        checkpoint_path=checkpoint_path,
        region="region",
        client=client,
    )
    assert result["downloaded"] == 0


@pytest.mark.parametrize("fail", ["file", "commit"])
def test_incomplete_upload_never_restores_a_completed_stage(tmp_path, fail):
    options, plan = fixture_bundle(tmp_path)
    client = options["client"]
    if fail == "file":
        client.fail_upload = "splat.ply"
    else:
        client.fail_commit = True
    with pytest.raises(OSError):
        bundles.publish(**options)
    assert (
        bundles.read_json(
            client, "bucket", bundles.marker_key("runs/example", plan[0]["id"])
        )
        is None
    )


def test_changed_signature_is_not_restored(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    bundles.publish(**options)
    plan[0]["signature"] = "changed"
    path = tmp_path / "new/runs/example/.recon-pipeline/pass-state.json"
    result = bundles.recover(
        bucket="bucket",
        prefix="runs/example",
        root=path.parents[1],
        data_root=tmp_path / "new",
        plan=plan,
        checkpoint_path=path,
        region="region",
        client=options["client"],
    )
    assert result["restored"] == 0
    assert json.loads(path.read_text())["passes"] == []


def test_corrupt_remote_artifact_does_not_restore_completion(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    bundles.publish(**options)
    options["client"].objects[
        "bucket", "runs/example/splatfacto/frame_000/splat.ply"
    ] = b"corrupted"
    path = tmp_path / "new/runs/example/.recon-pipeline/pass-state.json"
    with pytest.raises(ValueError, match="integrity"):
        bundles.recover(
            bucket="bucket",
            prefix="runs/example",
            root=path.parents[1],
            data_root=tmp_path / "new",
            plan=plan,
            checkpoint_path=path,
            region="region",
            client=options["client"],
        )
    assert not path.exists()


@pytest.mark.parametrize("relative", ["../escape", "/absolute"])
def test_unsafe_paths_are_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        bundles.local_path(tmp_path, relative)


def test_symlink_escape_is_rejected(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "root/link").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        bundles.local_path(tmp_path / "root", "link/file")


class Stage:
    requires = frozenset()

    def __init__(self, id, calls, fail=False):
        self.id = self.name = id
        self.provides = frozenset({id})
        self.checkpoint_signature = "settings"
        self.calls, self.fail = calls, fail

    def run(self, ctx):
        self.calls.append(self.id)
        if self.fail:
            raise OSError("failed upload")
        return PassResult(artifacts={self.id: "result"})

    def validate_checkpoint(self, checkpoint, context):
        return (
            checkpoint.result.details["checkpoint_signature"]
            == self.checkpoint_signature
        )


def test_failed_upload_retries_without_repeating_computation(tmp_path):
    calls = []
    store = JsonPassCheckpointStore(tmp_path / "state.json", "example")
    with pytest.raises(OSError):
        Pipeline(
            [Stage("compute", calls), Stage("upload", calls, fail=True)],
            checkpoint_store=store,
            resume_by_id=True,
        ).run()
    Pipeline(
        [Stage("compute", calls), Stage("upload", calls), Stage("next", calls)],
        checkpoint_store=store,
        resume_by_id=True,
    ).run()
    assert calls == ["compute", "upload", "upload", "next"]


def test_inserting_uploads_preserves_later_computation_checkpoints(tmp_path):
    calls = []
    store = JsonPassCheckpointStore(tmp_path / "state.json", "example")
    Pipeline(
        [Stage("a", calls), Stage("b", calls)],
        checkpoint_store=store,
        resume_by_id=True,
    ).run()
    calls.clear()
    Pipeline(
        [
            Stage("a", calls),
            Stage("upload-a", calls),
            Stage("b", calls),
            Stage("upload-b", calls),
        ],
        checkpoint_store=store,
        resume_by_id=True,
    ).run()
    assert calls == ["upload-a", "upload-b"]


def test_expanding_export_selection_rejects_old_three_frame_result(tmp_path):
    from recon_pipeline.reconstructions.nerfstudio.passes.export import (
        NerfstudioExportPass,
    )
    from recon_pipeline.reconstructions.nerfstudio.config import (
        NerfstudioArtifactConfig,
    )

    config = make_pipeline_config(
        tmp_path, nerfstudio=NerfstudioArtifactConfig(frames=(0, 1, 60, 120))
    )
    wrapped = RecoverablePass(NerfstudioExportPass(config), config, "same", "same")
    checkpoint = SimpleNamespace(
        result=PassResult(
            artifacts={
                "dataset.nerfstudio": [
                    {"frame": f, "dataset_dir": str(tmp_path / str(f))}
                    for f in (0, 60, 120)
                ]
            }
        )
    )
    assert not wrapped.validate_checkpoint(checkpoint, PipelineContext())


def test_frame_fingerprint_survives_expanding_selection_but_changes_with_profile(
    tmp_path,
):
    config = make_pipeline_config(tmp_path)
    settings = config.settings_dict()
    original = fingerprint(
        "splatfacto:frame_000", settings, ("bucket", "input", "video")
    )
    settings["nerfstudio"]["frames"] = list(range(121))
    settings["reconstruction"]["frames"] = list(range(121))
    assert (
        fingerprint("splatfacto:frame_000", settings, ("bucket", "input", "video"))
        == original
    )
    settings["reconstruction"]["max_num_iterations"] = 5
    assert (
        fingerprint("splatfacto:frame_000", settings, ("bucket", "input", "video"))
        != original
    )


def test_blank_error_has_type_and_traceback():
    try:
        raise StopIteration()
    except StopIteration as error:
        report = exception_report(error)
    assert report["message"] == "StopIteration()"
    assert "raise StopIteration()" in report["traceback"]


def test_failure_report_uploaded_before_shutdown(monkeypatch, tmp_path):
    from recon_pipeline.workers.aws.persistence import S3DiagnosticsFinalizer
    from recon_pipeline.workers.aws.finalizers.sagemaker_shutdown import (
        SageMakerShutdownFinalizer,
    )

    calls = []
    worker = make_worker(tmp_path, shutdown_on="always")
    persistence = S3PersistenceObserver(
        worker, make_pipeline_config(tmp_path), tmp_path / "job"
    )
    monkeypatch.setattr(
        persistence, "upload", lambda full=False: calls.append("report")
    )
    monkeypatch.setattr(
        "recon_pipeline.workers.aws.finalizers.sagemaker_shutdown.stop_sagemaker_app",
        lambda *a: calls.append("shutdown"),
    )
    with pytest.raises(OSError):
        Pipeline(
            [Stage("failed", [], fail=True)],
            finalizers=[
                S3DiagnosticsFinalizer(persistence),
                SageMakerShutdownFinalizer(worker),
            ],
        ).run()
    assert calls == ["report", "shutdown"]
    assert persistence.document["error"]["type"] == "OSError"


def test_report_upload_failure_defers_shutdown(monkeypatch, tmp_path):
    from recon_pipeline.workers.aws.persistence import S3DiagnosticsFinalizer
    from recon_pipeline.workers.aws.finalizers.sagemaker_shutdown import (
        SageMakerShutdownFinalizer,
    )

    calls = []
    worker = make_worker(tmp_path, shutdown_on="always")
    persistence = S3PersistenceObserver(
        worker, make_pipeline_config(tmp_path), tmp_path / "job"
    )

    def failed(full=False):
        raise OSError("S3 unavailable")

    monkeypatch.setattr(persistence, "upload", failed)
    monkeypatch.setattr(
        "recon_pipeline.workers.aws.finalizers.sagemaker_shutdown.stop_sagemaker_app",
        lambda *a: calls.append("shutdown"),
    )
    with pytest.raises(Exception):
        Pipeline(
            [],
            finalizers=[
                S3DiagnosticsFinalizer(persistence),
                SageMakerShutdownFinalizer(worker),
            ],
        ).run()
    assert calls == []
    assert (tmp_path / "job/s3-persistence-failed").exists()


def test_periodic_tail_and_full_failure_logs_are_available_in_bucket(tmp_path):
    from recon_pipeline.utilities.storage.s3.upload_diagnostics import snapshot

    client = MemoryS3()
    job = tmp_path / "job"
    job.mkdir()
    root = tmp_path / "run"
    root.mkdir()
    (job / "pipeline.log").write_text("pipeline output")
    (root / "train.log").write_text("traceback")
    (job / "status.json").write_text(json.dumps({"state": "finalizing"}))
    report = job / "report.json"
    report.write_text(
        json.dumps(
            {
                "run_prefix": "runs/example",
                "state": "failed",
                "pass_id": "frame_1",
                "attempt_id": "attempt",
            }
        )
    )
    snapshot(
        bucket="bucket",
        prefix="runs/example/attempt",
        job_dir=job,
        root=root,
        report=report,
        region="region",
        client=client,
    )
    assert (
        client.objects["bucket", "runs/example/attempt/logs/train.log.tail"]
        == b"traceback"
    )
    snapshot(
        bucket="bucket",
        prefix="runs/example/attempt",
        job_dir=job,
        root=root,
        report=report,
        region="region",
        client=client,
        full=True,
    )
    assert (
        client.objects["bucket", "runs/example/attempt/job/pipeline.log"]
        == b"pipeline output"
    )
    assert (
        json.loads(
            client.objects["bucket", "runs/example/.recon-pipeline/status.json"]
        )["state"]
        == "failed"
    )


def test_missing_remote_commit_retries_upload_and_retains_local_computation(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    path = options["root"] / ".recon-pipeline/pass-state.json"
    path.parent.mkdir()
    upload = dict(options["checkpoint"], id=options["upload_id"])
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": "example",
                "passes": [options["checkpoint"], upload],
            }
        )
    )
    bundles.recover(
        bucket=options["bucket"],
        prefix=options["prefix"],
        root=options["root"],
        data_root=options["data_root"],
        plan=plan,
        checkpoint_path=path,
        region="region",
        client=options["client"],
    )
    assert [item["id"] for item in json.loads(path.read_text())["passes"]] == [
        options["checkpoint"]["id"]
    ]


def test_force_invalidates_only_plan_commits_and_preserves_artifact_objects(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    bundles.publish(**options)
    path = options["root"] / ".recon-pipeline/pass-state.json"
    options["client"].objects["bucket", "runs/other/result.ply"] = b"unrelated"
    bundles.recover(
        bucket=options["bucket"],
        prefix=options["prefix"],
        root=options["root"],
        data_root=options["data_root"],
        plan=plan,
        checkpoint_path=path,
        region="region",
        client=options["client"],
        reset_commits=True,
    )
    assert (
        bundles.read_json(
            options["client"],
            "bucket",
            bundles.marker_key(options["prefix"], plan[0]["id"]),
        )
        is None
    )
    assert (
        options["client"].objects[
            "bucket", "runs/example/splatfacto/frame_000/splat.ply"
        ]
        == b"gaussians"
    )
    assert options["client"].objects["bucket", "runs/other/result.ply"] == b"unrelated"


def test_access_denied_is_not_treated_as_an_empty_experiment():
    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    client = MemoryS3()

    def fail(**kwargs):
        raise Denied()

    client.get_object = fail
    with pytest.raises(Denied):
        bundles.read_json(client, "bucket", "key")


def test_legacy_completion_receives_signature_before_first_publication(
    tmp_path, monkeypatch
):
    from recon_pipeline.datasets.fourdanyone.passes.prepare_experiment import (
        PrepareExperimentPass,
    )
    from recon_pipeline.datasets.fourdanyone.artifacts import EXPERIMENT_WORKSPACE

    config = make_pipeline_config(tmp_path)
    config.experiment_dir.mkdir(parents=True)
    (config.experiment_dir / "pipeline-config.json").write_text(
        json.dumps(config.settings_dict())
    )
    store = JsonPassCheckpointStore(
        config.experiment_dir / ".recon-pipeline/pass-state.json",
        config.experiment_name,
    )
    store.complete(
        "prepare-experiment",
        PassResult(artifacts={EXPERIMENT_WORKSPACE: config.experiment_dir}),
        1,
    )
    producer = PrepareExperimentPass(config)
    monkeypatch.setattr(
        producer, "run", lambda *args: pytest.fail("Legacy computation was repeated")
    )
    wrapped = RecoverablePass(producer, config, "signature", "signature")
    Pipeline([wrapped], checkpoint_store=store, resume_by_id=True).run()
    assert (
        store.load()["prepare-experiment"].result.details["checkpoint_signature"]
        == "signature"
    )


def test_artifact_upload_failure_defers_shutdown_even_when_report_is_saved(
    tmp_path, monkeypatch
):
    from recon_pipeline.workers.aws.persistence import S3DiagnosticsFinalizer
    from recon_pipeline.workers.aws.finalizers.sagemaker_shutdown import (
        SageMakerShutdownFinalizer,
    )

    worker = make_worker(tmp_path, shutdown_on="always")
    observer = S3PersistenceObserver(
        worker, make_pipeline_config(tmp_path), tmp_path / "job"
    )
    monkeypatch.setattr(observer, "upload", lambda full=False: None)
    monkeypatch.setattr(
        "recon_pipeline.workers.aws.finalizers.sagemaker_shutdown.stop_sagemaker_app",
        lambda *args: pytest.fail("Unsaved artifact must prevent shutdown"),
    )
    with pytest.raises(OSError):
        Pipeline(
            [Stage("s3-upload:compute", [], fail=True)],
            finalizers=[
                S3DiagnosticsFinalizer(observer),
                SageMakerShutdownFinalizer(worker),
            ],
        ).run()
    assert (tmp_path / "job/s3-persistence-failed").exists()


def test_config_snapshot_is_saved_at_first_heartbeat_and_redacts_literal_tokens(
    tmp_path,
):
    from recon_pipeline.utilities.storage.s3.upload_diagnostics import snapshot

    client = MemoryS3()
    job = tmp_path / "job"
    job.mkdir()
    (job / "request.json").write_text(
        json.dumps(
            {
                "aws_worker": {
                    "notifications": {
                        "telegram": {
                            "bot_token": "secret",
                            "bot_token_env": "TOKEN_VARIABLE",
                        }
                    }
                }
            }
        )
    )
    report = job / "report.json"
    report.write_text(
        json.dumps(
            {
                "run_prefix": "runs/example",
                "state": "running",
                "pass_id": "inference",
                "attempt_id": "first",
            }
        )
    )
    snapshot(
        bucket="bucket",
        prefix="runs/example/attempt",
        job_dir=job,
        root=tmp_path / "run",
        report=report,
        region="region",
        client=client,
    )
    saved = json.loads(client.objects["bucket", "runs/example/attempt/request.json"])
    assert saved["aws_worker"]["notifications"]["telegram"]["bot_token"] == "[REDACTED]"
    assert (
        saved["aws_worker"]["notifications"]["telegram"]["bot_token_env"]
        == "TOKEN_VARIABLE"
    )
    assert ("bucket", "runs/example/.recon-pipeline/status.json") in client.objects


def test_source_experiment_restores_verified_generation_bundle(tmp_path, monkeypatch):
    from recon_pipeline.utilities.storage.s3 import restore_experiment

    client = MemoryS3()
    data = tmp_path / "source-disk"
    root = data / "runs/source"
    generation = root / "4danyone"
    generation.mkdir(parents=True)
    for name in ("metadata.json", "cameras.json", "view_000.mp4"):
        (generation / name).write_bytes(name.encode())
    (root / "unrelated.ply").write_bytes(b"unrelated")
    checkpoint = {
        "id": "fourdanyone-inference",
        "completed_at": "now",
        "details": {"checkpoint_signature": "producer"},
        "artifacts": {},
    }
    bundles.publish(
        bucket="bucket",
        prefix="runs/source",
        root=root,
        data_root=data,
        paths=["4danyone"],
        checkpoint=checkpoint,
        upload_id="upload",
        signature="signature",
        region="region",
        client=client,
    )
    monkeypatch.setattr(restore_experiment, "_client", lambda region: client)
    destination = tmp_path / "new-disk/runs/source"
    result = restore_experiment.restore(
        SimpleNamespace(
            destination=destination,
            replace_existing=False,
            prefix="runs/source",
            region="region",
            bucket="bucket",
        )
    )
    assert result["downloaded"] == 3
    assert (destination / "4danyone/view_000.mp4").read_bytes() == b"view_000.mp4"
    assert not (destination / "unrelated.ply").exists()


def test_postprocessing_changes_do_not_invalidate_reconstruction(tmp_path):
    settings = make_pipeline_config(tmp_path).settings_dict()
    training = fingerprint(
        "splatfacto:frame_000", settings, ("bucket", "input", "video")
    )
    conversion = fingerprint(
        "splat-convert:frame_000", settings, ("bucket", "input", "video")
    )
    settings["postprocessing"]["splat_conversion"] = {
        "enabled": True,
        "formats": ["spz"],
    }
    assert (
        fingerprint("splatfacto:frame_000", settings, ("bucket", "input", "video"))
        == training
    )
    assert (
        fingerprint("splat-convert:frame_000", settings, ("bucket", "input", "video"))
        != conversion
    )
    settings.pop("postprocessing")
    assert (
        fingerprint("splatfacto:frame_000", settings, ("bucket", "input", "video"))
        == training
    )


def test_converted_artifacts_publish_and_recover_with_rebased_manifest_paths(tmp_path):
    options, plan = fixture_bundle(tmp_path)
    exports = options["root"] / "postprocessing/splat_conversion/frame_000"
    exports.mkdir(parents=True)
    artifacts = {}
    for fmt, name in [
        ("ply", "splat.ply"),
        ("compressed_ply", "splat.compressed.ply"),
        ("spz", "splat.spz"),
        ("sog", "splat.sog"),
    ]:
        path = exports / name
        path.write_bytes(fmt.encode())
        artifacts[fmt] = str(path)
    options["paths"] = ["postprocessing/splat_conversion/frame_000"]
    options["checkpoint"]["id"] = "splat-convert:frame_000"
    options["checkpoint"]["artifacts"] = {
        "converted": {"exported_artifacts": artifacts}
    }
    options["upload_id"] = "s3-upload:splat-convert:frame_000"
    plan[0].update(id=options["checkpoint"]["id"], upload_id=options["upload_id"])
    bundles.publish(**options)
    new_data = tmp_path / "other-machine"
    root = new_data / "runs/example"
    checkpoint_path = root / ".recon-pipeline/pass-state.json"
    result = bundles.recover(
        bucket="bucket",
        prefix="runs/example",
        root=root,
        data_root=new_data,
        plan=plan,
        checkpoint_path=checkpoint_path,
        region="region",
        client=options["client"],
    )
    assert result == {"restored": 1, "downloaded": 4}
    recovered = json.loads(checkpoint_path.read_text())["passes"][0]["artifacts"][
        "converted"
    ]["exported_artifacts"]
    for fmt, original in artifacts.items():
        expected = root / Path(original).relative_to(options["root"])
        assert recovered[fmt] == str(expected)
        assert expected.read_bytes() == fmt.encode()
