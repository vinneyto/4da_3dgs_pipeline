import ast
from pathlib import Path


def test_core_package_has_no_aws_dependency() -> None:
    core = Path(__file__).parents[1] / "src/recon_pipeline/core"
    forbidden = (
        "boto3",
        "recon_pipeline.workers",
        "recon_pipeline.datasets",
        "recon_pipeline.reconstructions",
        "recon_pipeline.artifacts",
        "sagemaker",
        "sns",
        "s3://",
    )
    for path in core.glob("*.py"):
        source = path.read_text().lower()
        for token in forbidden:
            assert (
                token not in source
            ), f"{path.name} crosses the AWS boundary via {token!r}"


def test_aws_worker_is_a_separate_package() -> None:
    source_root = Path(__file__).parents[1] / "src"
    namespace = source_root / "recon_pipeline"
    assert (namespace / "core/pipeline.py").is_file()
    assert (namespace / "core/events.py").is_file()
    assert (namespace / "datasets/fourdanyone/passes/inference.py").is_file()
    assert (namespace / "datasets/fourdanyone/passes/prepare_experiment.py").is_file()
    assert (namespace / "workers/aws/worker.py").is_file()
    assert not (namespace / "core/worker.py").exists()
    assert (namespace / "reconstructions/nerfstudio/config.py").is_file()
    assert (namespace / "reconstructions/nerfstudio/passes/export.py").is_file()
    assert (namespace / "artifacts/rerun/exporter.py").is_file()
    assert (namespace / "artifacts/rerun/passes/export.py").is_file()
    for package in (
        namespace / "datasets/fourdanyone",
        namespace / "reconstructions/nerfstudio",
        namespace / "artifacts/rerun",
    ):
        for path in package.glob("*.py"):
            assert "boto3" not in path.read_text().lower()


def test_every_pass_has_its_own_module() -> None:
    source_root = Path(__file__).parents[1] / "src/recon_pipeline"
    pass_roots = (
        source_root / "datasets/fourdanyone/passes",
        source_root / "reconstructions/nerfstudio/passes",
        source_root / "artifacts/rerun/passes",
        source_root / "workers/aws/passes",
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
    assert "recon-aws-worker" in pyproject
    assert "recon-pipeline = " not in pyproject


def test_passes_only_launch_utilities_and_map_results() -> None:
    root = Path(__file__).parents[1] / "src/recon_pipeline"
    for path in root.glob("**/passes/*.py"):
        if path.name == "__init__.py":
            continue
        source = path.read_text()
        assert "run_utility(" in source, f"{path} must launch a standalone utility"
        for forbidden in (
            "import shutil",
            "import boto3",
            "RerunExporter",
            ".write_text(",
            ".unlink(",
            ".mkdir(",
            "from ..aws import",
        ):
            assert forbidden not in source, f"{path} embeds an operation: {forbidden}"


def test_operation_utilities_have_no_execution_config_dependencies() -> None:
    root = Path(__file__).parents[1] / "src/recon_pipeline"
    sources = [root / "cli.py", *root.glob("**/utilities/*.py")]
    for path in sources:
        source = path.read_text()
        for forbidden in (
            "AwsWorkerConfig",
            "FourDAnyoneConfig",
            "PipelineContext",
            "PassResult",
            "recon_pipeline.core",
            "recon_pipeline.workers",
            "--worker-config",
            "--pipeline-config",
            "--result-file",
        ):
            assert (
                forbidden not in source
            ), f"{path} depends on orchestration: {forbidden}"
