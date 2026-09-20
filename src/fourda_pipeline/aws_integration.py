"""AWS adapter used exclusively by the background-job layer."""

from __future__ import annotations

from pathlib import Path

from .job_config import AwsConfig, SageMakerAppConfig


def _boto3():
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("AWS support requires: pip install 'fourda-3dgs-pipeline[aws]'") from error
    return boto3


def configure_email(config: AwsConfig) -> dict[str, str]:
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


def topic_arn(config: AwsConfig) -> str:
    return _boto3().client("sns", region_name=config.region).create_topic(
        Name=config.sns.topic_name
    )["TopicArn"]


def publish_completion(config: AwsConfig, subject: str, message: str) -> None:
    _boto3().client("sns", region_name=config.region).publish(
        TopicArn=topic_arn(config),
        Subject=subject[:100],
        Message=message,
    )


def experiment_s3_uri(config: AwsConfig, experiment_name: str) -> str:
    key = "/".join(part for part in (config.runs_prefix, experiment_name) if part)
    return f"s3://{config.bucket}/{key}/"


def upload_directory(config: AwsConfig, source: Path, experiment_name: str) -> str:
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
