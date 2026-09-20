from __future__ import annotations

from pathlib import Path

import pytest

from fourda_aws_worker import aws
from fourda_aws_worker.config import AwsWorkerConfig, SageMakerAppConfig, SnsConfig


class FakeClient:
    def __init__(self, service: str, *, subscription_arn: str = "confirmed-subscription"):
        self.service = service
        self.subscription_arn = subscription_arn
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, arguments: dict) -> None:
        self.calls.append((name, arguments))

    def get_caller_identity(self):
        return {"Account": "123456789012", "Arn": "arn:aws:sts::123:assumed-role/test/session"}

    def head_bucket(self, **kwargs):
        self._record("head_bucket", kwargs)

    def put_object(self, **kwargs):
        self._record("put_object", kwargs)

    def delete_object(self, **kwargs):
        self._record("delete_object", kwargs)

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
    def __init__(self, *, subscription_arn: str = "confirmed-subscription"):
        self.clients = {
            name: FakeClient(name, subscription_arn=subscription_arn)
            for name in ("sts", "s3", "sns", "sagemaker")
        }

    def client(self, name: str, **_kwargs):
        return self.clients[name]


def make_config(tmp_path: Path, *, shutdown_on: str = "success") -> AwsWorkerConfig:
    return AwsWorkerConfig(
        job_id="job-01",
        jobs_dir=tmp_path,
        shutdown_on=shutdown_on,
        region="us-east-1",
        bucket="fourda-test",
        input_prefix="input",
        runs_prefix="runs",
        models_prefix="models",
        upload_input=True,
        upload_results=True,
        sns=SnsConfig(topic_name="fourda", email="owner@example.com"),
        sagemaker=SageMakerAppConfig(
            domain_id="d-test", space_name="space-test", app_name="default"
        ),
    )


def test_health_check_validates_aws_and_sends_start_notification(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path), "job-01")

    assert result.account == "123456789012"
    assert result.writable_prefixes == ("input", "runs")
    assert result.sagemaker_app_status == "InService"
    s3_calls = fake.clients["s3"].calls
    assert [name for name, _ in s3_calls].count("put_object") == 2
    publish = [call for call in fake.clients["sns"].calls if call[0] == "publish"]
    assert len(publish) == 1
    assert publish[0][1]["Subject"] == "4DAnyone AWS job started: job-01"


def test_health_check_rejects_pending_email_subscription(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3(subscription_arn="PendingConfirmation")
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    with pytest.raises(RuntimeError, match="pending confirmation"):
        aws.run_health_check(make_config(tmp_path), "job-01")

    assert not [call for call in fake.clients["sns"].calls if call[0] == "publish"]


def test_health_check_skips_sagemaker_when_shutdown_is_disabled(monkeypatch, tmp_path) -> None:
    fake = FakeBoto3()
    monkeypatch.setattr(aws, "_boto3", lambda: fake)

    result = aws.run_health_check(make_config(tmp_path, shutdown_on="never"), "job-01")

    assert result.sagemaker_app_status is None
    assert fake.clients["sagemaker"].calls == []
