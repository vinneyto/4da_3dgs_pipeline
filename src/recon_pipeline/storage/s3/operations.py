"""S3 operations that accept storage parameters, not execution configuration."""

from pathlib import Path, PurePosixPath
from typing import Any

from recon_pipeline.cloud.aws.sdk import boto3


def _client(region: str, client: Any = None):
    return client if client is not None else boto3().client("s3", region_name=region)


def normalize_prefix(value: str) -> str:
    value = value.strip("/")
    if ".." in PurePosixPath(value).parts:
        raise ValueError("S3 prefix must not contain parent directory components")
    return value + "/" if value else ""


def download_file(
    *, bucket: str, key: str, destination: Path, region: str, client: Any = None
) -> Path:
    client = _client(region, client)
    metadata = client.head_object(Bucket=bucket, Key=key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == metadata["ContentLength"]
    ):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".download")
    client.download_file(bucket, key, str(temporary))
    temporary.replace(destination)
    return destination


def sync_prefix(
    *,
    bucket: str,
    prefix: str,
    destination: Path,
    region: str,
    require_objects: bool = False,
    client: Any = None,
) -> tuple[int, int]:
    prefix = normalize_prefix(prefix)
    client = _client(region, client)
    found = downloaded = 0
    destination.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.startswith(prefix):
                raise ValueError(f"object is outside requested prefix: {key}")
            relative = key[len(prefix) :]
            if not relative or relative.endswith("/"):
                continue
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe object key: {key}")
            found += 1
            target = destination.joinpath(*path.parts)
            if target.is_file() and target.stat().st_size == item["Size"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download")
            client.download_file(bucket, key, str(temporary))
            temporary.replace(target)
            downloaded += 1
    if require_objects and not found:
        raise FileNotFoundError(
            f"No source experiment objects found at s3://{bucket}/{prefix}"
        )
    return found, downloaded


def delete_prefix(*, bucket: str, prefix: str, region: str, client: Any = None) -> int:
    prefix = normalize_prefix(prefix)
    if not prefix:
        raise ValueError("a nonempty prefix is required for replacement")
    client = _client(region, client)
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if any(not item["Key"].startswith(prefix) for item in objects):
            raise ValueError("listed object is outside the replacement prefix")
        for offset in range(0, len(objects), 1000):
            batch = objects[offset : offset + 1000]
            if batch:
                client.delete_objects(
                    Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
                )
                deleted += len(batch)
    return deleted


def upload_directory(
    *, bucket: str, prefix: str, source: Path, region: str, client: Any = None
) -> str:
    prefix = normalize_prefix(prefix)
    client = _client(region, client)
    for path in source.rglob("*"):
        if path.is_file():
            client.upload_file(
                str(path), bucket, prefix + path.relative_to(source).as_posix()
            )
    return f"s3://{bucket}/{prefix}"
