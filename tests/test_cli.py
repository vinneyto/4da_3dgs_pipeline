import argparse

import pytest

from fourda_pipeline.cli_common import comma_separated_ints


def test_comma_separated_ints() -> None:
    assert comma_separated_ints("-15, 0,15") == (-15, 0, 15)


def test_comma_separated_ints_rejects_text() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        comma_separated_ints("0,nope")
