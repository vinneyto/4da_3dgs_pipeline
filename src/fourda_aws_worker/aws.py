"""AWS API adapter for S3, SNS, and SageMaker operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AwsWorkerConfig, SageMakerAppConfig


def _boto3():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("AWS support requires: pip install 'fourda-3dgs-pipeline[aws]'") from error
    return boto3


def configure_email(config: AwsWorkerConfig) -> dict[str, str]:
    sns = _boto3().client("sns", region_name=config.region)
    topic_arn = sns.create_topic(
        Name=config.sns.topic_name,
        Tags=[
            {"Key": "Project", "Value": "camera-path"},
            {"Key": "Component", "Value": "4danyone"},
        ],
    )["TopicArn"]
    response = sns.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=config.sns.email,
        ReturnSubscriptionArn=True,
    )
    return {
        "region": config.region,
        "topic_arn": topic_arn,
        "email": config.sns.email,
        "subscription_arn": response["SubscriptionArn"],
    }


def topic_arn(config: AwsWorkerConfig) -> str:
    return _boto3().client("sns", region_name=config.region).create_topic(
        Name=config.sns.topic_name
    )["TopicArn"]


def publish_notification(config: AwsWorkerConfig, subject: str, message: str) -> None:
    _boto3().client("sns", region_name=config.region).publish(
        TopicArn=topic_arn(config),
        Subject=subject[:100],
        Message=message,
    )


def publish_completion(config: AwsWorkerConfig, subject: str, message: str) -> None:
    """Backward-compatible name for completion/failure notifications."""
    publish_notification(config, subject, message)


@dataclass(frozen=True, slots=True)
class AwsHealthCheckResult:
    account: str
    caller_arn: str
    bucket: str
    writable_prefixes: tuple[str, ...]
    topic_arn: str
    subscription_arn: str
    sagemaker_app_status: str | None


def _confirmed_email_subscription(sns: Any, arn: str, email: str) -> str:
    next_token: str | None = None
    pending = False
    while True:
        arguments = {"TopicArn": arn}
        if next_token:
            arguments["NextToken"] = next_token
        response = sns.list_subscriptions_by_topic(**arguments)
        for subscription in response.get("Subscriptions", []):
            if (
                subscription.get("Protocol") == "email"
                and subscription.get("Endpoint", "").casefold() == email.casefold()
            ):
                subscription_arn = subscription.get("SubscriptionArn", "")
                if subscription_arn and subscription_arn != "PendingConfirmation":
                    return subscription_arn
                pending = True
        next_token = response.get("NextToken")
        if not next_token:
            break

    state = "pending confirmation" if pending else "missing"
    raise RuntimeError(
        f"SNS email subscription for {email} is {state}. Run "
        "'fourda-aws-worker configure-email --config <path>' and confirm the email "
        "before starting the worker."
    )


def _probe_s3_prefix(s3: Any, config: AwsWorkerConfig, prefix: str, job_id: str) -> str:
    key = "/".join(
        part for part in (prefix.strip("/"), ".worker-health", f"{job_id}.json") if part
    )
    body = json.dumps({"job_id": job_id, "purpose": "fourda-aws-worker health check"})
    s3.put_object(
        Bucket=config.bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    # Delete is not required by the worker itself. Clean up when the execution
    # role allows it, but do not turn an optional permission into a prerequisite.
    try:
        s3.delete_object(Bucket=config.bucket, Key=key)
    except Exception:
        pass
    return prefix


def run_health_check(config: AwsWorkerConfig, job_id: str) -> AwsHealthCheckResult:
    """Validate every AWS dependency before expensive pipeline work begins."""
    boto3 = _boto3()

    identity = boto3.client("sts", region_name=config.region).get_caller_identity()

    s3 = boto3.client("s3", region_name=config.region)
    s3.head_bucket(Bucket=config.bucket)
    prefixes: list[str] = []
    if config.upload_input:
        prefixes.append(_probe_s3_prefix(s3, config, config.input_prefix, job_id))
    if config.upload_results and config.runs_prefix not in prefixes:
        prefixes.append(_probe_s3_prefix(s3, config, config.runs_prefix, job_id))

    sns = boto3.client("sns", region_name=config.region)
    arn = sns.create_topic(Name=config.sns.topic_name)["TopicArn"]
    subscription_arn = _confirmed_email_subscription(sns, arn, config.sns.email)

    app_status: str | None = None
    if config.shutdown_on != "never":
        app = boto3.client("sagemaker", region_name=config.region).describe_app(
            DomainId=config.sagemaker.domain_id,
            SpaceName=config.sagemaker.space_name,
            AppType="JupyterLab",
            AppName=config.sagemaker.app_name,
        )
        app_status = app["Status"]
        if app_status != "InService":
            raise RuntimeError(
                f"SageMaker JupyterLab App must be InService, got {app_status!r}"
            )

    result = AwsHealthCheckResult(
        account=identity["Account"],
        caller_arn=identity["Arn"],
        bucket=config.bucket,
        writable_prefixes=tuple(prefixes),
        topic_arn=arn,
        subscription_arn=subscription_arn,
        sagemaker_app_status=app_status,
    )
    publish_notification(
        config,
        f"4DAnyone AWS job started: {job_id}",
        "\n".join(
            [
                f"4DAnyone AWS worker health check passed for job {job_id}.",
                f"Account: {result.account}",
                f"Caller: {result.caller_arn}",
                f"S3 bucket: {result.bucket}",
                f"Writable prefixes: {', '.join(result.writable_prefixes) or 'none required'}",
                f"SNS topic: {result.topic_arn}",
                f"SageMaker App: {result.sagemaker_app_status or 'shutdown disabled'}",
                "The 4DAnyone pipeline is starting now.",
            ]
        ),
    )
    return result


def experiment_s3_uri(config: AwsWorkerConfig, experiment_name: str) -> str:
    key = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return f"s3://{config.bucket}/{key}/"


def upload_input_video(config: AwsWorkerConfig, video_path: Path) -> str:
    key = "/".join(part for part in (config.input_prefix, video_path.name) if part)
    _boto3().client("s3", region_name=config.region).upload_file(
        str(video_path),
        config.bucket,
        key,
    )
    return f"s3://{config.bucket}/{key}"


def upload_directory(config: AwsWorkerConfig, source: Path, experiment_name: str) -> str:
    client = _boto3().client("s3", region_name=config.region)
    prefix = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    for path in source.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(source).as_posix()}"
            client.upload_file(str(path), config.bucket, key)
    return experiment_s3_uri(config, experiment_name)


def stop_sagemaker_app(region: str, app: SageMakerAppConfig) -> None:
    _boto3().client("sagemaker", region_name=region).delete_app(
        DomainId=app.domain_id,
        SpaceName=app.space_name,
        AppType="JupyterLab",
        AppName=app.app_name,
    )
