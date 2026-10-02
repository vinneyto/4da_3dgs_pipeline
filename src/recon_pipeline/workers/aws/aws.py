"""Service adapters for S3, SNS, Telegram, and SageMaker operations."""

from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AwsWorkerConfig, SageMakerAppConfig
from recon_pipeline.utilities.storage.s3 import operations as storage
from recon_pipeline.utilities.cloud.aws.sdk import boto3 as _boto3
from recon_pipeline.utilities.cloud.aws.access import (
    AwsHealthCheckResult,
    check_access,
    confirmed_email_subscription as _confirmed_email_subscription,
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


def _telegram_request(
    config: AwsWorkerConfig,
    method: str,
    payload: dict[str, Any],
    *,
    request_timeout: float = 15,
) -> Any:
    if config.telegram is None:
        raise ValueError("Telegram notifications are not configured")
    token = config.telegram.resolve_bot_token()
    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=request_timeout) as response:
            result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        # Do not chain urllib exceptions: their URL contains the secret bot token.
        raise RuntimeError(
            f"Telegram API {method} request failed: {type(error).__name__}"
        ) from None
    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API {method} rejected the request: {result.get('description', 'unknown error')}"
        )
    return result.get("result")


def _telegram_multipart_request(
    config: AwsWorkerConfig,
    method: str,
    fields: dict[str, str],
    files: dict[str, Path],
    *,
    request_timeout: float = 30,
) -> Any:
    if config.telegram is None:
        raise ValueError("Telegram notifications are not configured")
    token = config.telegram.resolve_bot_token()
    boundary = f"recon-pipeline-{uuid.uuid4().hex}"
    body = bytearray()

    def append(value: bytes) -> None:
        body.extend(value)
        body.extend(b"\r\n")

    for name, value in fields.items():
        append(f"--{boundary}".encode())
        append(f'Content-Disposition: form-data; name="{name}"'.encode())
        append(b"")
        append(value.encode("utf-8"))
    for name, path in files.items():
        append(f"--{boundary}".encode())
        append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"'
            ).encode()
        )
        append(
            (
                "Content-Type: "
                f"{mimetypes.guess_type(path.name)[0] or 'application/octet-stream'}"
            ).encode()
        )
        append(b"")
        append(path.read_bytes())
    body.extend(f"--{boundary}--\r\n".encode())

    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=request_timeout) as response:
            result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        # Do not chain urllib exceptions: their URL contains the secret bot token.
        raise RuntimeError(
            f"Telegram API {method} request failed: {type(error).__name__}"
        ) from None
    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram API {method} rejected the request: "
            f"{result.get('description', 'unknown error')}"
        )
    return result.get("result")


def _check_telegram(config: AwsWorkerConfig) -> str:
    assert config.telegram is not None
    chat = _telegram_request(config, "getChat", {"chat_id": config.telegram.chat_id})
    return str(chat.get("id", config.telegram.chat_id))


def _telegram_text(subject: str, message: str) -> str:
    return f"{subject}\n\n{message}"[:4096]


def _publish_telegram(config: AwsWorkerConfig, subject: str, message: str) -> int:
    assert config.telegram is not None
    result = _telegram_request(
        config,
        "sendMessage",
        {
            "chat_id": config.telegram.chat_id,
            "text": _telegram_text(subject, message),
        },
    )
    return int(result["message_id"])


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
    _telegram_request(
        config,
        "editMessageText",
        {
            "chat_id": config.telegram.chat_id,
            "message_id": str(message_id),
            "text": _telegram_text(subject, message),
        },
    )


def publish_telegram_photos(
    config: AwsWorkerConfig,
    paths: Sequence[Path],
    caption: str,
) -> tuple[int, ...]:
    """Publish two to ten local photos as one Telegram album."""
    if config.telegram is None or not config.telegram.enabled:
        raise ValueError("Telegram notifications are not enabled")
    photos = tuple(Path(path) for path in paths)
    if not 2 <= len(photos) <= 10:
        raise ValueError("Telegram media groups require between 2 and 10 photos")
    missing = [str(path) for path in photos if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Telegram photo files are missing: " + ", ".join(missing)
        )

    media: list[dict[str, str]] = []
    files: dict[str, Path] = {}
    for index, path in enumerate(photos):
        field = f"photo{index}"
        item = {"type": "photo", "media": f"attach://{field}"}
        if index == 0:
            item["caption"] = caption[:1024]
        media.append(item)
        files[field] = path
    result = _telegram_multipart_request(
        config,
        "sendMediaGroup",
        {
            "chat_id": config.telegram.chat_id,
            "media": json.dumps(media),
        },
        files,
    )
    return tuple(int(message["message_id"]) for message in result)


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
    payload: dict[str, Any] = {
        "timeout": str(timeout_seconds),
        "allowed_updates": json.dumps(["message"]),
    }
    if offset is not None:
        payload["offset"] = str(offset)
    result = _telegram_request(
        config,
        "getUpdates",
        payload,
        request_timeout=max(timeout_seconds + 5, 15),
    )
    return list(result or [])


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
    email = config.sns if config.sns is not None and config.sns.enabled else None
    telegram = (
        config.telegram
        if config.telegram is not None and config.telegram.enabled
        else None
    )
    app = config.sagemaker if config.shutdown_on != "never" else None
    return check_access(
        region=config.region,
        bucket=config.bucket_name,
        input_key=config.bucket.video_key,
        check_input=require_input_video,
        models_prefix=config.models_prefix if config.sync_models else None,
        write_prefix=config.runs_prefix if config.upload_results else None,
        sns_topic_name=email.topic_name if email else None,
        notification_email=email.email if email else None,
        telegram_chat_id=telegram.chat_id if telegram else None,
        telegram_check=(lambda: _check_telegram(config)) if telegram else None,
        sagemaker_domain_id=app.domain_id if app else None,
        sagemaker_space_name=app.space_name if app else None,
        sagemaker_app_name=app.app_name if app else "default",
        sdk=_boto3(),
    )


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
