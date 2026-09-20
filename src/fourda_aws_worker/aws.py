"""Service adapters for S3, SNS, Telegram, and SageMaker operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AwsWorkerConfig, SageMakerAppConfig


def _boto3():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("AWS support requires: pip install 'fourda-3dgs-pipeline[aws]'") from error
    return boto3


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
    return _boto3().client("sns", region_name=config.region).create_topic(
        Name=config.sns.topic_name
    )["TopicArn"]


def _telegram_request(config: AwsWorkerConfig, method: str, payload: dict[str, str]) -> Any:
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
        with urlopen(request, timeout=15) as response:
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


def _check_telegram(config: AwsWorkerConfig) -> str:
    assert config.telegram is not None
    chat = _telegram_request(config, "getChat", {"chat_id": config.telegram.chat_id})
    return str(chat.get("id", config.telegram.chat_id))


def _publish_telegram(config: AwsWorkerConfig, subject: str, message: str) -> None:
    assert config.telegram is not None
    text = f"{subject}\n\n{message}"
    _telegram_request(
        config,
        "sendMessage",
        {"chat_id": config.telegram.chat_id, "text": text[:4096]},
    )


def publish_notification(config: AwsWorkerConfig, subject: str, message: str) -> dict[str, str]:
    """Publish to every enabled channel without making notifications job-critical."""
    outcomes: dict[str, str] = {}
    if config.sns is not None and config.sns.enabled:
        try:
            sns = _boto3().client("sns", region_name=config.region)
            arn = topic_arn(config)
            _confirmed_email_subscription(sns, arn, config.sns.email)
            sns.publish(TopicArn=arn, Subject=subject[:100], Message=message)
            outcomes["email"] = "sent"
        except Exception as error:
            outcomes["email"] = f"skipped: {error}"
    if config.telegram is not None and config.telegram.enabled:
        try:
            _publish_telegram(config, subject, message)
            outcomes["telegram"] = "sent"
        except Exception as error:
            outcomes["telegram"] = f"skipped: {error}"
    return outcomes


def publish_completion(config: AwsWorkerConfig, subject: str, message: str) -> dict[str, str]:
    """Backward-compatible name for completion/failure notifications."""
    return publish_notification(config, subject, message)


@dataclass(frozen=True, slots=True)
class AwsHealthCheckResult:
    account: str
    caller_arn: str
    bucket: str
    writable_prefixes: tuple[str, ...]
    topic_arn: str | None
    subscription_arn: str | None
    email_status: str
    telegram_status: str
    sagemaker_app_status: str | None
    video_s3_uri: str
    model_object_count: int


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
    video_bucket, video_key = config.video_s3_location
    s3.head_object(Bucket=video_bucket, Key=video_key)

    model_object_count = 0
    if config.sync_models:
        response = s3.list_objects_v2(
            Bucket=config.bucket,
            Prefix=f"{config.models_prefix}/" if config.models_prefix else "",
            MaxKeys=1,
        )
        model_object_count = int(response.get("KeyCount", 0))

    prefixes: list[str] = []
    if config.upload_results:
        prefixes.append(_probe_s3_prefix(s3, config, config.runs_prefix, job_id))

    arn: str | None = None
    subscription_arn: str | None = None
    email_status = "disabled"
    if config.sns is not None and config.sns.enabled:
        try:
            sns = boto3.client("sns", region_name=config.region)
            arn = sns.create_topic(Name=config.sns.topic_name)["TopicArn"]
            subscription_arn = _confirmed_email_subscription(sns, arn, config.sns.email)
            email_status = "ready"
        except Exception as error:
            email_status = f"unavailable: {error}"

    telegram_status = "disabled"
    if config.telegram is not None and config.telegram.enabled:
        try:
            chat_id = _check_telegram(config)
            telegram_status = f"ready (chat {chat_id})"
        except Exception as error:
            telegram_status = f"unavailable: {error}"

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
        email_status=email_status,
        telegram_status=telegram_status,
        sagemaker_app_status=app_status,
        video_s3_uri=config.video_s3_uri,
        model_object_count=model_object_count,
    )
    return result


def publish_started(
    config: AwsWorkerConfig, result: AwsHealthCheckResult, experiment_name: str
) -> dict[str, str]:
    return publish_notification(
        config,
        f"4DAnyone AWS job started: {config.job_id}",
        "\n".join(
            [
                f"4DAnyone AWS worker health check passed for job {config.job_id}.",
                f"Experiment: {experiment_name}",
                f"Account: {result.account}",
                f"Caller: {result.caller_arn}",
                f"S3 input: {result.video_s3_uri}",
                f"Local input: {config.local_video_path}",
                f"Model objects visible in S3: {result.model_object_count}",
                f"Email notifications: {result.email_status}",
                f"Telegram notifications: {result.telegram_status}",
                f"SageMaker App: {result.sagemaker_app_status or 'shutdown disabled'}",
                "Input staging completed. The 4DAnyone pipeline is starting now.",
            ]
        ),
    )


def download_input_video(config: AwsWorkerConfig) -> Path:
    client = _boto3().client("s3", region_name=config.region)
    bucket, key = config.video_s3_location
    metadata = client.head_object(Bucket=bucket, Key=key)
    destination = config.local_video_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == metadata["ContentLength"]:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    client.download_file(bucket, key, str(temporary))
    temporary.replace(destination)
    return destination


def sync_model_objects(config: AwsWorkerConfig) -> tuple[int, int]:
    """Download changed S3 model objects, preserving the persistent EBS cache."""
    if not config.sync_models:
        return 0, 0
    client = _boto3().client("s3", region_name=config.region)
    prefix = f"{config.models_prefix}/" if config.models_prefix else ""
    paginator = client.get_paginator("list_objects_v2")
    found = 0
    downloaded = 0
    config.local.model_dir.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=config.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative = key[len(prefix) :]
            if not relative or relative.endswith("/"):
                continue
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe model object key: {key}")
            found += 1
            destination = config.local.model_dir.joinpath(*relative_path.parts)
            if destination.is_file() and destination.stat().st_size == item["Size"]:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".download")
            client.download_file(config.bucket, key, str(temporary))
            temporary.replace(destination)
            downloaded += 1
    return found, downloaded


def experiment_s3_uri(config: AwsWorkerConfig, experiment_name: str) -> str:
    key = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return f"s3://{config.bucket}/{key}/"


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
