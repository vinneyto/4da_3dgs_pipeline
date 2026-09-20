"""Small AWS integrations used after a background job finishes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3


def aws_config_path() -> Path:
    return Path(os.environ.get("CP_4DA_AWS_CONFIG", "~/.config/4da-3dgs-pipeline/aws.json")).expanduser()


def load_aws_config() -> dict[str, Any]:
    path = aws_config_path()
    return json.loads(path.read_text()) if path.is_file() else {}


def save_aws_config(payload: dict[str, Any]) -> Path:
    path = aws_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def configure_email(email: str, topic_name: str, region: str) -> dict[str, str]:
    sns = boto3.client("sns", region_name=region)
    topic_arn = sns.create_topic(
        Name=topic_name,
        Tags=[
            {"Key": "Project", "Value": "camera-path"},
            {"Key": "Component", "Value": "4danyone"},
        ],
    )["TopicArn"]
    response = sns.subscribe(
        TopicArn=topic_arn,
        Protocol="email",
        Endpoint=email,
        ReturnSubscriptionArn=True,
    )
    payload = {
        "region": region,
        "sns_topic_arn": topic_arn,
        "email": email,
        "subscription_arn": response["SubscriptionArn"],
    }
    save_aws_config(payload)
    return payload


def publish_completion(topic_arn: str, subject: str, message: str, region: str) -> None:
    boto3.client("sns", region_name=region).publish(
        TopicArn=topic_arn,
        Subject=subject[:100],
        Message=message,
    )


def stop_current_sagemaker_app(region: str) -> None:
    required = {
        "DomainId": os.environ.get("CP_SM_DOMAIN_ID"),
        "SpaceName": os.environ.get("CP_SM_SPACE_NAME"),
        "AppName": os.environ.get("CP_SM_JUPYTER_APP_NAME", "default"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("cannot stop SageMaker App; missing environment variables: " + ", ".join(missing))
    boto3.client("sagemaker", region_name=region).delete_app(
        DomainId=required["DomainId"],
        SpaceName=required["SpaceName"],
        AppType="JupyterLab",
        AppName=required["AppName"],
    )
