from pathlib import Path

import pytest

from recon_pipeline.core.command import CommandError, CommandRunner


def test_command_error_retains_combined_output_tail(tmp_path: Path) -> None:
    with pytest.raises(CommandError) as raised:
        CommandRunner().run(
            [
                "python",
                "-c",
                "import sys; print('stdout'); print('stderr', file=sys.stderr); raise SystemExit(2)",
            ],
            cwd=tmp_path,
        )

    assert raised.value.return_code == 2
    assert "stdout" in raised.value.output_tail
    assert "stderr" in raised.value.output_tail


def test_only_stdout_reaches_the_output_callback(tmp_path: Path, capsys) -> None:
    import sys

    lines = []
    CommandRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; print('result'); print('diagnostic', file=sys.stderr)",
        ],
        cwd=tmp_path,
        on_line=lines.append,
    )
    captured = capsys.readouterr()
    assert lines == ["result"]
    assert "result" in captured.out and "diagnostic" not in captured.out
    assert "diagnostic" in captured.err


def test_large_stderr_output_is_drained_while_reading_stdout(tmp_path: Path) -> None:
    import sys

    lines = []
    CommandRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('x' * 200000 + '\\n'); print('finished')",
        ],
        cwd=tmp_path,
        on_line=lines.append,
    )
    assert lines == ["finished"]
