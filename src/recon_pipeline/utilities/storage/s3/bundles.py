"""Publish verified files before a portable completion record; restore committed files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .operations import _client, normalize_prefix

SCHEMA = 1


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    target = Path(root).joinpath(*path.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return target


def portable(value, data_root):
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.startswith(str(data_root).rstrip("/") + "/"):
        return {"$data_path": Path(value).relative_to(data_root).as_posix()}
    if isinstance(value, dict):
        return {k: portable(v, data_root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v, data_root) for v in value]
    return value


def materialize(value, data_root):
    if isinstance(value, dict):
        if set(value) == {"$data_path"}:
            return str(local_path(data_root, value["$data_path"]))
        return {k: materialize(v, data_root) for k, v in value.items()}
    if isinstance(value, list):
        return [materialize(v, data_root) for v in value]
    return value


def marker_key(prefix, pass_id):
    # Hash IDs: callers cannot turn pass IDs into object paths.
    return (
        normalize_prefix(prefix)
        + ".recon-pipeline/commits/"
        + hashlib.sha256(pass_id.encode()).hexdigest()
        + ".json"
    )


def read_json(client, bucket, key):
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    return json.loads(response["Body"].read())


def put_json(client, bucket, key, value):
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(value, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
    )


def publish(
    *,
    bucket,
    prefix,
    root,
    data_root,
    paths,
    checkpoint,
    upload_id,
    signature,
    region,
    client=None,
):
    client = _client(region, client)
    root, data_root = Path(root), Path(data_root)
    files = {}
    for relative in paths:
        selected = local_path(root, relative)
        if not selected.exists():
            raise FileNotFoundError(f"Artifact missing: {selected}")
        for path in ([selected] if selected.is_file() else sorted(selected.rglob("*"))):
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            local_path(root, name)
            files[name] = {
                "path": name,
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
    if not files:
        raise ValueError("Artifact bundle contains no files")
    key = marker_key(prefix, checkpoint["id"])
    # Invalidate just this commit before changing its files, never the run prefix.
    client.delete_object(Bucket=bucket, Key=key)
    for name, metadata in files.items():
        path = local_path(root, name)
        client.upload_file(str(path), bucket, normalize_prefix(prefix) + name)
        remote = client.head_object(Bucket=bucket, Key=normalize_prefix(prefix) + name)
        if (
            remote["ContentLength"] != metadata["size"]
            or digest(path) != metadata["sha256"]
        ):
            raise RuntimeError(f"Artifact changed or upload is incomplete: {name}")
    record = {
        "schema_version": SCHEMA,
        "experiment_name": root.name,
        "signature": signature,
        "upload_id": upload_id,
        "checkpoint": portable(checkpoint, data_root),
        "files": list(files.values()),
    }
    put_json(client, bucket, key, record)
    return {
        "s3_uri": f"s3://{bucket}/{normalize_prefix(prefix)}",
        "marker_key": key,
        "files": len(files),
    }


def restore_files(client, bucket, prefix, root, record):
    if record.get("schema_version") != SCHEMA or not record.get("files"):
        raise ValueError("Unsupported or empty artifact commit")
    downloaded = 0
    for metadata in record["files"]:
        target = local_path(root, metadata["path"])
        if (
            target.is_file()
            and target.stat().st_size == metadata["size"]
            and digest(target) == metadata["sha256"]
        ):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        try:
            client.download_file(
                bucket, normalize_prefix(prefix) + metadata["path"], str(temporary)
            )
            if (
                temporary.stat().st_size != metadata["size"]
                or digest(temporary) != metadata["sha256"]
            ):
                raise ValueError(f'Artifact integrity check failed: {metadata["path"]}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        downloaded += 1
    return downloaded


def recover(
    *,
    bucket,
    prefix,
    root,
    data_root,
    plan,
    checkpoint_path,
    region,
    client=None,
    reset_commits=False,
):
    client = _client(region, client)
    root = Path(root)
    checkpoint_path = Path(checkpoint_path)
    if reset_commits:
        # --force invalidates the selected plan's completions, never its artifact prefix.
        for item in plan:
            client.delete_object(Bucket=bucket, Key=marker_key(prefix, item["id"]))
        checkpoint_path.unlink(missing_ok=True)
        return {"invalidated": len(plan)}
    document = (
        json.loads(checkpoint_path.read_text())
        if checkpoint_path.is_file()
        else {"schema_version": 1, "experiment_name": root.name, "passes": []}
    )
    if (
        document.get("schema_version") != 1
        or document.get("experiment_name") != root.name
    ):
        raise ValueError("Local checkpoint belongs to a different experiment or schema")
    by_id = {item["id"]: item for item in document["passes"]}
    downloaded = restored = 0
    for item in plan:
        # Only a verified remote commit can confirm the upload, even on a warm disk.
        by_id.pop(item["upload_id"], None)
        record = read_json(client, bucket, marker_key(prefix, item["id"]))
        if record is None or record.get("signature") != item["signature"]:
            continue
        if (
            record.get("experiment_name") != root.name
            or record.get("upload_id") != item["upload_id"]
        ):
            raise ValueError(
                "Artifact commit belongs to a different experiment or pass"
            )
        producer = materialize(record["checkpoint"], data_root)
        if (
            producer.get("id") != item["id"]
            or producer.get("details", {}).get("checkpoint_signature")
            != item["producer_signature"]
        ):
            raise ValueError("Artifact commit has an inconsistent producer signature")
        # All file integrity checks must finish before any completion is restored.
        downloaded += restore_files(client, bucket, prefix, root, record)
        by_id[item["id"]] = producer
        by_id[item["upload_id"]] = {
            "id": item["upload_id"],
            "completed_at": producer["completed_at"],
            "duration_seconds": 0,
            "artifacts": {
                item["upload_id"]: f"s3://{bucket}/{normalize_prefix(prefix)}"
            },
            "details": {"checkpoint_signature": item["signature"]},
        }
        restored += 1
    document["passes"] = list(by_id.values())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(checkpoint_path)
    return {"restored": restored, "downloaded": downloaded}
