"""Save explicit status, reports and logs without replacing artifact objects."""

from pathlib import Path
from ._arguments import parser
from .bundles import put_json
from .operations import _client, normalize_prefix
from recon_pipeline.utilities._output import run_operation
import json


def redact(value):
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key in ("bot_token", "aws_secret_access_key", "aws_session_token")
                and item
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def snapshot(*, bucket, prefix, job_dir, root, report, region, full=False, client=None):
    client = _client(region, client)
    prefix = normalize_prefix(prefix)
    report = Path(report)
    status = Path(job_dir) / "status.json"
    document = json.loads(report.read_text())
    # report.json is uploaded even if a log or status file subsequently fails.
    client.upload_file(str(report), bucket, prefix + "report.json")
    current = json.loads(status.read_text()) if status.is_file() else {}
    current.update(
        state=document["state"],
        active_pass=document["pass_id"],
        attempt_id=document["attempt_id"],
        report_prefix=prefix,
    )
    if document.get("error"):
        current["error"] = document["error"]
    put_json(client, bucket, prefix + "status.json", current)
    put_json(
        client, bucket, document["run_prefix"] + "/.recon-pipeline/status.json", current
    )
    request = Path(job_dir) / "request.json"
    if request.is_file():
        put_json(
            client,
            bucket,
            prefix + "request.json",
            redact(json.loads(request.read_text())),
        )
    logs = list(Path(root).rglob("*.log")) if Path(root).exists() else []
    if not full:
        logs = sorted(logs, key=lambda p: p.stat().st_mtime)[-3:]
    logs += [Path(job_dir) / "pipeline.log"]
    for path in logs:
        if not path.is_file():
            continue
        relative = (
            "job/pipeline.log"
            if path == Path(job_dir) / "pipeline.log"
            else "logs/" + path.relative_to(root).as_posix()
        )
        if full:
            client.upload_file(str(path), bucket, prefix + relative)
        else:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 1024 * 1024))
                client.put_object(
                    Bucket=bucket, Key=prefix + relative + ".tail", Body=stream.read()
                )
    return {"s3_uri": f"s3://{bucket}/{prefix}", "full_logs": full}


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--job-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--full", action="store_true")
    a = p.parse_args(argv)
    run_operation(
        lambda: snapshot(
            bucket=a.bucket,
            prefix=a.prefix,
            job_dir=a.job_dir,
            root=a.root,
            report=a.report,
            region=a.region,
            full=a.full,
        )
    )


if __name__ == "__main__":
    main()
