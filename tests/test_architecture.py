from pathlib import Path


def test_core_package_has_no_aws_dependency() -> None:
    core = Path(__file__).parents[1] / "src/fourda_pipeline"
    forbidden = ("boto3", "fourda_aws_worker", "sagemaker", "sns", "s3://")
    for path in core.glob("*.py"):
        source = path.read_text().lower()
        for token in forbidden:
            assert token not in source, f"{path.name} crosses the AWS boundary via {token!r}"


def test_aws_worker_is_a_separate_package() -> None:
    source_root = Path(__file__).parents[1] / "src"
    assert (source_root / "fourda_pipeline/pipeline.py").is_file()
    assert (source_root / "fourda_aws_worker/worker.py").is_file()
    assert not (source_root / "fourda_pipeline/worker.py").exists()
    assert (source_root / "fourda_rerun/exporter.py").is_file()
    for path in (source_root / "fourda_rerun").glob("*.py"):
        assert "boto3" not in path.read_text().lower()
