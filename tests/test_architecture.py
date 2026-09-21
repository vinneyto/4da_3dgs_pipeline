import ast
from pathlib import Path


def test_core_package_has_no_aws_dependency() -> None:
    core = Path(__file__).parents[1] / "src/fourda_pipeline"
    forbidden = (
        "boto3",
        "fourda_aws_worker",
        "fourda_4danyone",
        "fourda_nerfstudio",
        "fourda_rerun",
        "sagemaker",
        "sns",
        "s3://",
    )
    for path in core.glob("*.py"):
        source = path.read_text().lower()
        for token in forbidden:
            assert token not in source, f"{path.name} crosses the AWS boundary via {token!r}"


def test_aws_worker_is_a_separate_package() -> None:
    source_root = Path(__file__).parents[1] / "src"
    assert (source_root / "fourda_pipeline/pipeline.py").is_file()
    assert (source_root / "fourda_pipeline/events.py").is_file()
    assert (source_root / "fourda_4danyone/passes/inference.py").is_file()
    assert (source_root / "fourda_4danyone/passes/prepare_experiment.py").is_file()
    assert (source_root / "fourda_aws_worker/worker.py").is_file()
    assert not (source_root / "fourda_pipeline/worker.py").exists()
    assert (source_root / "fourda_nerfstudio/config.py").is_file()
    assert (source_root / "fourda_nerfstudio/passes/export.py").is_file()
    assert (source_root / "fourda_rerun/exporter.py").is_file()
    assert (source_root / "fourda_rerun/passes/export.py").is_file()
    for package in ("fourda_4danyone", "fourda_nerfstudio", "fourda_rerun"):
        for path in (source_root / package).glob("*.py"):
            assert "boto3" not in path.read_text().lower()


def test_every_pass_has_its_own_module() -> None:
    source_root = Path(__file__).parents[1] / "src"
    pass_roots = (
        source_root / "fourda_4danyone/passes",
        source_root / "fourda_nerfstudio/passes",
        source_root / "fourda_rerun/passes",
        source_root / "fourda_aws_worker/passes",
    )
    for pass_root in pass_roots:
        for path in pass_root.glob("*.py"):
            if path.name == "__init__.py":
                continue
            classes = [
                node.name
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name.endswith("Pass")
            ]
            assert len(classes) == 1, f"{path} must define exactly one pass: {classes}"


def test_only_aws_worker_exposes_a_pipeline_cli() -> None:
    project = Path(__file__).parents[1]
    pyproject = (project / "pyproject.toml").read_text()
    assert "fourda-aws-worker" in pyproject
    assert 'fourda-pipeline = ' not in pyproject
