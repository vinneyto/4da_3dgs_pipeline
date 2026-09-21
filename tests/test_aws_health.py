from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from recon_pipeline.workers.aws import aws
from recon_pipeline.workers.aws.config import (
    AwsWorkerConfig,
    BucketConfig,
    LocalWorkspaceConfig,
    SageMakerAppConfig,
    SnsConfig,
    TelegramConfig,
)


class FakeClient:
    def __init__(
        self,
        service: str,
        *,
        subscription_arn: str = "arn:aws:sns:test:subscription",
    ):
        self.service = service
        self.subscription_arn = subscription_arn
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, arguments: dict) -> None:
        self.calls.append((name, arguments))

    def get_caller_identity(self):
        return {"Account": "123456789012", "Arn": "arn:aws:sts::123:assumed-role/test/session"}

    def head_bucket(self, **kwargs):
        self._record("head_bucket", kwargs)

    def head_object(self, **kwargs):
        self._record("head_object", kwargs)
        return {"ContentLength": 100}

    def list_objects_v2(self, **kwargs):
        self._record("list_objects_v2", kwargs)
        return {"KeyCount": 1}

    def put_object(self, **kwargs):
        self._record("put_object", kwargs)

    def delete_object(self, **kwargs):
        self._record("delete_object", kwargs)

    def delete_objects(self, **kwargs):
        self._record("delete_objects", kwargs)

    def create_topic(self, **kwargs):
        self._record("create_topic", kwargs)
        return {"TopicArn": "arn:aws:sns:us-east-1:123:fourda"}

    def list_subscriptions_by_topic(self, **kwargs):
        self._record("list_subscriptions_by_topic", kwargs)
        return {
            "Subscriptions": [
                {
                    "Protocol": "email",
                    "Endpoint": "owner@example.com",
                    "SubscriptionArn": self.subscription_arn,
                }
            ]
        }

    def publish(self, **kwargs):
        self._record("publish", kwargs)

    def describe_app(self, **kwargs):
        self._record("describe_app", kwargs)
        return {"Status": "InService"}


class FakeBoto3:
    def __init__(self, *, subscription_arn: str = "arn:aws:sns:test:subscription"):
        self.clients = {
            name: FakeClient(name, subscription_arn=subscription_arn)
            for name in ("sts", "s3", "sns", "sagemaker")
        }

    def client(self, name: str, **_kwargs):
        return self.clients[name]


class FakePaginator:
    def paginate(self, **_kwargs):
        return [
            {
                "Contents": [
                    {"Key": "models/checkpoint.bin", "Size": 4},
                    {"Key": "models/", "Size": 0},
                ]
            }
        ]


class FakeExperimentPaginator:
    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        return [
            {
                "Contents": [
                    {"Key": f"{prefix}4danyone/metadata.json", "Size": 4},
                    {"Key": f"{prefix}4danyone/cameras.json", "Size": 4},
                ]
            }
        ]


class FakeDownloadS3(FakeClient):
    def head_object(self, **kwargs):
        self._record("head_object", kwargs)
        return {"ContentLength": 5}

    def download_file(self, bucket, key, filename):
        self._record("download_file", {"Bucket": bucket, "Key": key, "Filename": filename})
        Path(filename).write_bytes(b"video" if key.endswith(".MOV") else b"data")

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()


class FakeExperimentS3(FakeDownloadS3):
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakeExperimentPaginator()


def make_config(
    tmp_path: Path,
    *,
    shutdown_on: str = "success",
    telegram: TelegramConfig | None = None,
) -> AwsWorkerConfig:
    return AwsWorkerConfig(
        job_id="job-01",
        shutdown_on=shutdown_on,
        region="us-east-1",
        bucket=BucketConfig(
            name="fourda-test",
            video="leo.MOV",
            input_prefix="input",
            runs_prefix="runs",
            models_prefix="models",
        ),
        sync_models=True,
        upload_results=True,
        local=LocalWorkspaceConfig(
            data_root=tmp_path / "data",
            fourdanyone_root=tmp_path / "4DAnyone",
        ),
        sns=SnsConfig(topic_name="fourda", email="owner@example.com"),
        sagemaker=SageMakerAppConfig(
            domain_id="d-test", space_name="space-test", app_name="default"
        ),
        telegram=telegram,
    )


def test_health_check_validates_aws_without_sending_notifications(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path), "job-01")

    assert result.account == "123456789012"
    assert result.writable_prefixes == ("runs",)
    assert result.sagemaker_app_status == "InService"
    assert result.video_s3_uri == "s3://fourda-test/input/leo.MOV"
    s3_calls = fake.clients["s3"].calls
    assert [name for name, _ in s3_calls].count("put_object") == 1
    assert [call for call in fake.clients["sns"].calls if call[0] == "publish"] == []


def test_health_check_reports_pending_email_without_blocking(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3(subscription_arn="PendingConfirmation")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path), "job-01")

    assert "PendingConfirmation" in result.email_status
    assert not [call for call in fake.clients["sns"].calls if call[0] == "publish"]


def test_health_check_treats_deleted_email_as_unavailable(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3(subscription_arn="Deleted")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path), "job-01")

    assert "Deleted" in result.email_status


def test_health_check_validates_optional_telegram(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    monkeypatch.setattr(aws, "_check_telegram", lambda _config: "123456")
    config = make_config(
        tmp_path,
        telegram=TelegramConfig(chat_id="123456", bot_token="test-token"),
    )

    result = aws.run_health_check(config, "job-01")

    assert result.telegram_status == "ready (chat 123456)"


def test_health_check_supports_no_notification_channels(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    config = replace(make_config(tmp_path), sns=None)

    result = aws.run_health_check(config, "job-01")

    assert result.email_status == "disabled"
    assert result.telegram_status == "disabled"
    assert fake.clients["sns"].calls == []


def test_notification_failure_is_returned_instead_of_raised(monkeypatch, tmp_path) -> None:
    config = replace(make_config(tmp_path), sns=None)
    config = replace(
        config,
        telegram=TelegramConfig(chat_id="123456", bot_token="invalid-token"),
    )
    monkeypatch.setattr(
        aws,
        "_publish_telegram",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("denied")),
    )

    result = aws.publish_notification(config, "subject", "message")

    assert result == {"telegram": "skipped: denied"}


def test_health_check_skips_sagemaker_when_shutdown_is_disabled(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path, shutdown_on="never"), "job-01")

    assert result.sagemaker_app_status is None
    assert fake.clients["sagemaker"].calls == []


def test_worker_downloads_video_and_reuses_matching_local_copy(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    fake.clients["s3"] = FakeDownloadS3("s3")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    config = make_config(tmp_path)

    first = aws.download_input_video(config)
    second = aws.download_input_video(config)

    assert first == second == tmp_path / "data/input/leo.MOV"
    downloads = [call for call in fake.clients["s3"].calls if call[0] == "download_file"]
    assert len(downloads) == 1


def test_worker_syncs_changed_model_objects(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    fake.clients["s3"] = FakeDownloadS3("s3")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    config = make_config(tmp_path)

    found, downloaded = aws.sync_model_objects(config)

    assert (found, downloaded) == (1, 1)
    assert (tmp_path / "data/models/checkpoint.bin").read_bytes() == b"data"


def test_worker_restores_prior_experiment_for_artifact_only_run(
    monkeypatch, tmp_path
) -> None:
    fake = FakeBoto3()
    fake.clients["s3"] = FakeExperimentS3("s3")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    config = make_config(tmp_path)
    destination = tmp_path / "data/runs/leo-original"

    found, downloaded = aws.sync_experiment_results(
        config, "leo-original", destination
    )

    assert (found, downloaded) == (2, 2)
    assert (destination / "4danyone/metadata.json").read_bytes() == b"data"
    assert (destination / "4danyone/cameras.json").read_bytes() == b"data"


def test_worker_deletes_only_one_experiment_result_prefix(
    monkeypatch, tmp_path
) -> None:
    fake = FakeBoto3()
    fake.clients["s3"] = FakeExperimentS3("s3")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)
    config = make_config(tmp_path)

    deleted = aws.delete_experiment_results(config, "leo")

    assert deleted == 2
    delete_call = [
        arguments
        for name, arguments in fake.clients["s3"].calls
        if name == "delete_objects"
    ][0]
    assert delete_call["Bucket"] == "fourda-test"
    assert delete_call["Delete"]["Objects"] == [
        {"Key": "runs/leo/4danyone/metadata.json"},
        {"Key": "runs/leo/4danyone/cameras.json"},
    ]
