from pathlib import Path

import pytest

from recon_pipeline.core.command import CommandError, CommandRunner


def test_command_error_retains_combined_output_tail(tmp_path: Path) -> None:
    with pytest.raises(CommandError) as raised:
        CommandRunner().run(
            ["python", "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr); raise SystemExit(2)"],
            cwd=tmp_path,
        )

    assert raised.value.return_code == 2
    assert "stdout" in raised.value.output_tail
    assert "stderr" in raised.value.output_tail
