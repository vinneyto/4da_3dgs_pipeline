from pathlib import Path

from recon_pipeline.workers.aws.status import JobStatus


def test_status_round_trip_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    status = JobStatus(job_id="leo")
    status.update(state="running", stage="inference", progress=0.45, pid=123)
    status.write(path)

    loaded = JobStatus.read(path)
    assert loaded.job_id == "leo"
    assert loaded.state == "running"
    assert loaded.progress == 0.45
    assert loaded.pid == 123
    assert not loaded.terminal


def test_success_is_terminal() -> None:
    status = JobStatus(job_id="leo", state="succeeded")
    assert status.terminal
