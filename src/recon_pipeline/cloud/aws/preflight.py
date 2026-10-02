"""Check explicitly selected cloud dependencies, without execution configuration."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .sdk import boto3


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


def confirmed_email_subscription(sns: Any, arn: str, email: str) -> str:
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


def check_telegram(chat_id: str, token_env: str) -> str:
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Telegram bot token environment variable {token_env!r} is not set"
        )
    request = Request(
        f"https://api.telegram.org/bot{token}/getChat",
        data=urlencode({"chat_id": chat_id}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Telegram getChat request failed: {type(error).__name__}"
        ) from None
    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram getChat rejected the request: {result.get('description', 'unknown error')}"
        )
    return str(result["result"].get("id", chat_id))


def check_access(
    *,
    region: str,
    bucket: str,
    input_key: str | None = None,
    check_input: bool = True,
    models_prefix: str | None = None,
    write_prefix: str | None = None,
    sns_topic_name: str | None = None,
    notification_email: str | None = None,
    telegram_chat_id: str | None = None,
    telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN",
    sagemaker_domain_id: str | None = None,
    sagemaker_space_name: str | None = None,
    sagemaker_app_name: str = "default",
    sdk: Any = None,
    telegram_check: Callable[[], str] | None = None,
) -> AwsHealthCheckResult:
    if bool(sns_topic_name) != bool(notification_email):
        raise ValueError(
            "SNS topic name and notification email must be supplied together"
        )
    if bool(sagemaker_domain_id) != bool(sagemaker_space_name):
        raise ValueError("SageMaker domain and space must be supplied together")
    sdk = sdk if sdk is not None else boto3()
    identity = sdk.client("sts", region_name=region).get_caller_identity()
    s3 = sdk.client("s3", region_name=region)
    s3.head_bucket(Bucket=bucket)
    if input_key and check_input:
        s3.head_object(Bucket=bucket, Key=input_key)
    model_count = 0
    if models_prefix is not None:
        prefix = models_prefix.strip("/")
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix + "/" if prefix else "", MaxKeys=1
        )
        model_count = int(response.get("KeyCount", 0))
    writable = []
    if write_prefix is not None:
        prefix = write_prefix.strip("/")
        key = "/".join(
            part
            for part in (prefix, ".access-check", uuid.uuid4().hex + ".json")
            if part
        )
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=b'{"purpose":"storage access check"}',
            ContentType="application/json",
        )
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass
        writable.append(prefix)
    arn = subscription_arn = None
    email_status = "disabled"
    if sns_topic_name:
        try:
            sns = sdk.client("sns", region_name=region)
            arn = sns.create_topic(Name=sns_topic_name)["TopicArn"]
            subscription_arn = confirmed_email_subscription(
                sns, arn, notification_email
            )
            email_status = "ready"
        except Exception as error:
            email_status = f"unavailable: {error}"
    telegram_status = "disabled"
    if telegram_chat_id:
        try:
            chat_id = (
                telegram_check()
                if telegram_check
                else check_telegram(telegram_chat_id, telegram_bot_token_env)
            )
            telegram_status = f"ready (chat {chat_id})"
        except Exception as error:
            telegram_status = f"unavailable: {error}"
    app_status = None
    if sagemaker_domain_id:
        app = sdk.client("sagemaker", region_name=region).describe_app(
            DomainId=sagemaker_domain_id,
            SpaceName=sagemaker_space_name,
            AppType="JupyterLab",
            AppName=sagemaker_app_name,
        )
        app_status = app["Status"]
        if app_status != "InService":
            raise RuntimeError(
                f"SageMaker JupyterLab App must be InService, got {app_status!r}"
            )
    return AwsHealthCheckResult(
        account=identity["Account"],
        caller_arn=identity["Arn"],
        bucket=bucket,
        writable_prefixes=tuple(writable),
        topic_arn=arn,
        subscription_arn=subscription_arn,
        email_status=email_status,
        telegram_status=telegram_status,
        sagemaker_app_status=app_status,
        video_s3_uri=f"s3://{bucket}/{input_key or ''}",
        model_object_count=model_count,
    )
