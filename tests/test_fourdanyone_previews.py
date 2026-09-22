from recon_pipeline.datasets.fourdanyone.previews import select_cardinal_camera_ids


def test_cardinal_previews_use_the_layer_closest_to_zero_pitch() -> None:
    assert select_cardinal_camera_ids(24, (-15, 0, 15)) == (24, 30, 36, 42)


def test_cardinal_previews_support_a_single_layer() -> None:
    assert select_cardinal_camera_ids(24, (0,)) == (0, 6, 12, 18)
