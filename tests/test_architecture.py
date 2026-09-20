from pathlib import Path


def test_core_package_has_no_aws_dependency() -> None:
    core = Path(__file__).parents[1] / "src/fourda_pipeline"
    forbidden = ("boto3", "fourda_aws_job", "sagemaker", "sns", "s3://")
    for path in core.glob("*.py"):
        source = path.read_text().lower()
        for token in forbidden:
            assert token not in source, f"{path.name} crosses the AWS boundary via {token!r}"


def test_aws_job_is_a_separate_package() -> None:
    source_root = Path(__file__).parents[1] / "src"
    assert (source_root / "fourda_pipeline/pipeline.py").is_file()
    assert (source_root / "fourda_aws_job/worker.py").is_file()
    assert not (source_root / "fourda_pipeline/worker.py").exists()
