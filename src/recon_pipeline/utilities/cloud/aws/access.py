"""Check explicitly selected cloud dependencies, without execution configuration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .sdk import boto3


@dataclass(frozen=True, slots=True)
class AwsAccessCheckResult:
    account: str
    caller_arn: str
    bucket: str
    writable_prefixes: tuple[str, ...]
    sagemaker_app_status: str | None
    video_s3_uri: str
    model_object_count: int


def check_access(
    *,
    region: str,
    bucket: str,
    input_key: str | None = None,
    check_input: bool = True,
    models_prefix: str | None = None,
    write_prefix: str | None = None,
    sagemaker_domain_id: str | None = None,
    sagemaker_space_name: str | None = None,
    sagemaker_app_name: str = "default",
    sdk: Any = None,
) -> AwsAccessCheckResult:
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
    return AwsAccessCheckResult(
        account=identity["Account"],
        caller_arn=identity["Arn"],
        bucket=bucket,
        writable_prefixes=tuple(writable),
        sagemaker_app_status=app_status,
        video_s3_uri=f"s3://{bucket}/{input_key or ''}",
        model_object_count=model_count,
    )
