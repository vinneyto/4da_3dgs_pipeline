import pytest

from fourda_rerun.exporter import select_camera_views


def test_selects_evenly_spaced_views_from_first_layer() -> None:
    cameras = [
        {"camera_id": index, "layer_index": 0, "yaw": index * 15}
        for index in range(24)
    ]
    cameras += [
        {"camera_id": 24 + index, "layer_index": 1, "yaw": index * 15}
        for index in range(24)
    ]

    selected = select_camera_views(cameras, 4)

    assert [camera["yaw"] for camera in selected] == [0, 90, 180, 270]


def test_requires_enough_views_in_first_layer() -> None:
    with pytest.raises(ValueError, match="layer 0"):
        select_camera_views([{"camera_id": 0, "layer_index": 0, "yaw": 0}], 4)
