"""Service adapters for S3, SNS, Telegram, and SageMaker operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import AwsWorkerConfig, SageMakerAppConfig
from recon_pipeline.utilities.storage.s3 import operations as storage
from recon_pipeline.utilities.cloud.aws.sdk import boto3 as _boto3
from recon_pipeline.utilities.cloud.aws.access import (
    AwsAccessCheckResult,
    check_access,
)

from .telegram import TelegramClient


@dataclass(frozen=True, slots=True)
class AwsHealthCheckResult(AwsAccessCheckResult):
    topic_arn: str | None
    subscription_arn: str | None
    email_status: str
    telegram_status: str


def _confirmed_email_subscription(sns: Any, arn: str, email: str) -> str:
    next_token: str | None = None
    observed_state: str | None = None
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
                if subscription_arn.startswith("arn:"):
                    return subscription_arn
                observed_state = subscription_arn or "unknown"
        next_token = response.get("NextToken")
        if not next_token:
            break

    state = observed_state or "missing"
    raise RuntimeError(
        f"SNS email subscription for {email} is {state}; confirm the subscription before using notifications"
    )


def configure_email(config: AwsWorkerConfig) -> dict[str, str]:
    if config.sns is None:
        raise ValueError("email notifications are not configured")
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
    if config.sns is None:
        raise ValueError("email notifications are not configured")
    return (
        _boto3()
        .client("sns", region_name=config.region)
        .create_topic(Name=config.sns.topic_name)["TopicArn"]
    )


def _check_telegram(config: AwsWorkerConfig) -> str:
    assert config.telegram is not None
    return TelegramClient(config.telegram).check_chat()


def _telegram_text(subject: str, message: str) -> str:
    return f"{subject}\n\n{message}"[:4096]


def _publish_telegram(config: AwsWorkerConfig, subject: str, message: str) -> int:
    assert config.telegram is not None
    return TelegramClient(config.telegram).send_message(
        _telegram_text(subject, message)
    )


def publish_telegram(config: AwsWorkerConfig, subject: str, message: str) -> int:
    """Publish one Telegram message and return its message ID."""
    if config.telegram is None or not config.telegram.enabled:
        raise ValueError("Telegram notifications are not enabled")
    return _publish_telegram(config, subject, message)


def edit_telegram(
    config: AwsWorkerConfig,
    message_id: int,
    subject: str,
    message: str,
) -> None:
    """Replace an existing Telegram message with its terminal status."""
    if config.telegram is None or not config.telegram.enabled:
        raise ValueError("Telegram notifications are not enabled")
    TelegramClient(config.telegram).edit_message(
        message_id, _telegram_text(subject, message)
    )


def publish_telegram_photos(
    config: AwsWorkerConfig,
    paths: Sequence[Path],
    caption: str,
) -> tuple[int, ...]:
    """Publish two to ten local photos as one Telegram album."""
    if config.telegram is None or not config.telegram.enabled:
        raise ValueError("Telegram notifications are not enabled")
    return TelegramClient(config.telegram).send_photos(paths, caption)


def publish_email(config: AwsWorkerConfig, subject: str, message: str) -> str:
    """Publish one email notification and return its delivery status."""
    if config.sns is None or not config.sns.enabled:
        return "disabled"
    try:
        sns = _boto3().client("sns", region_name=config.region)
        arn = topic_arn(config)
        _confirmed_email_subscription(sns, arn, config.sns.email)
        sns.publish(TopicArn=arn, Subject=subject[:100], Message=message)
        return "sent"
    except Exception as error:
        return f"skipped: {error}"


def get_telegram_updates(
    config: AwsWorkerConfig,
    offset: int | None,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Long-poll bot commands without exposing the bot token to callers."""
    if config.telegram is None or not config.telegram.enabled:
        raise ValueError("Telegram notifications are not enabled")
    return TelegramClient(config.telegram).get_updates(offset, timeout_seconds)


def publish_notification(
    config: AwsWorkerConfig, subject: str, message: str
) -> dict[str, str]:
    """Publish to every enabled channel without making notifications job-critical."""
    outcomes: dict[str, str] = {}
    if config.sns is not None and config.sns.enabled:
        outcomes["email"] = publish_email(config, subject, message)
    if config.telegram is not None and config.telegram.enabled:
        try:
            _publish_telegram(config, subject, message)
            outcomes["telegram"] = "sent"
        except Exception as error:
            outcomes["telegram"] = f"skipped: {error}"
    return outcomes


def publish_completion(
    config: AwsWorkerConfig, subject: str, message: str
) -> dict[str, str]:
    """Backward-compatible name for completion/failure notifications."""
    return publish_notification(config, subject, message)


def run_health_check(
    config: AwsWorkerConfig, job_id: str, *, require_input_video: bool = True
) -> AwsHealthCheckResult:
    app = config.sagemaker if config.shutdown_on != "never" else None
    access = check_access(
        region=config.region,
        bucket=config.bucket_name,
        input_key=config.bucket.video_key,
        check_input=require_input_video,
        models_prefix=config.models_prefix if config.sync_models else None,
        write_prefix=config.runs_prefix if config.upload_results else None,
        sagemaker_domain_id=app.domain_id if app else None,
        sagemaker_space_name=app.space_name if app else None,
        sagemaker_app_name=app.app_name if app else "default",
        sdk=_boto3(),
    )

    return AwsHealthCheckResult(**asdict(access), **check_notification_channels(config))


def check_notification_channels(config: AwsWorkerConfig) -> dict[str, Any]:
    """Check optional worker channels without sending messages or blocking a run."""
    arn = subscription_arn = None
    email_status = telegram_status = "disabled"
    if config.sns is not None and config.sns.enabled:
        try:
            sns = _boto3().client("sns", region_name=config.region)
            arn = sns.create_topic(Name=config.sns.topic_name)["TopicArn"]
            subscription_arn = _confirmed_email_subscription(sns, arn, config.sns.email)
            email_status = "ready"
        except Exception as error:
            email_status = f"unavailable: {error}"
    if config.telegram is not None and config.telegram.enabled:
        try:
            telegram_status = f"ready (chat {_check_telegram(config)})"
        except Exception as error:
            telegram_status = f"unavailable: {error}"
    return {
        "topic_arn": arn,
        "subscription_arn": subscription_arn,
        "email_status": email_status,
        "telegram_status": telegram_status,
    }


def download_input_video(config: AwsWorkerConfig) -> Path:
    bucket, key = config.video_s3_location
    return storage.download_file(
        bucket=bucket,
        key=key,
        destination=config.local_video_path,
        region=config.region,
        client=_boto3().client("s3", region_name=config.region),
    )


def sync_model_objects(config: AwsWorkerConfig) -> tuple[int, int]:
    if not config.sync_models:
        return 0, 0
    return storage.sync_prefix(
        bucket=config.bucket_name,
        prefix=config.models_prefix,
        destination=config.local.model_dir,
        region=config.region,
        client=_boto3().client("s3", region_name=config.region),
    )


def sync_experiment_results(
    config: AwsWorkerConfig, experiment_name: str, destination: Path
) -> tuple[int, int]:
    prefix = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return storage.sync_prefix(
        bucket=config.bucket_name,
        prefix=prefix,
        destination=destination,
        region=config.region,
        require_objects=True,
        client=_boto3().client("s3", region_name=config.region),
    )


def experiment_s3_uri(config: AwsWorkerConfig, experiment_name: str) -> str:
    key = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return f"s3://{config.bucket_name}/{key}/"


def delete_experiment_results(config: AwsWorkerConfig, experiment_name: str) -> int:
    prefix = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return storage.delete_prefix(
        bucket=config.bucket_name,
        prefix=prefix,
        region=config.region,
        client=_boto3().client("s3", region_name=config.region),
    )


def upload_directory(
    config: AwsWorkerConfig, source: Path, experiment_name: str
) -> str:
    prefix = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return storage.upload_directory(
        bucket=config.bucket_name,
        prefix=prefix,
        source=source,
        region=config.region,
        client=_boto3().client("s3", region_name=config.region),
    )


def stop_sagemaker_app(region: str, app: SageMakerAppConfig) -> None:
    _boto3().client("sagemaker", region_name=region).delete_app(
        DomainId=app.domain_id,
        SpaceName=app.space_name,
        AppType="JupyterLab",
        AppName=app.app_name,
    )
