"""Create lightweight still previews from a completed 4DAnyone result."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def select_cardinal_camera_ids(
    views_per_layer: int,
    layer_pitches: Sequence[int],
    *,
    count: int = 4,
) -> tuple[int, ...]:
    """Select evenly spaced yaw views from the layer closest to zero pitch."""
    if views_per_layer < count:
        raise ValueError(f"at least {count} views per layer are required")
    if not layer_pitches:
        raise ValueError("at least one camera layer is required")
    layer_index = min(
        range(len(layer_pitches)),
        key=lambda index: (abs(layer_pitches[index]), index),
    )
    offset = layer_index * views_per_layer
    return tuple(offset + (index * views_per_layer // count) for index in range(count))


def extract_middle_frame(video_path: Path, output_path: Path) -> Path:
    """Decode one representative frame without retaining the full video."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError("4DAnyone previews require the av package") from error

    video_path = Path(video_path)
    output_path = Path(output_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        target = max((int(stream.frames) - 1) // 2, 0) if stream.frames else 60
        selected = None
        for index, frame in enumerate(container.decode(stream)):
            selected = frame
            if index >= target:
                break
        if selected is None:
            raise RuntimeError(f"video contains no decodable frames: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_image().save(output_path, format="JPEG", quality=88)
    return output_path


def create_cardinal_previews(
    result_dir: Path,
    output_dir: Path,
    views_per_layer: int,
    layer_pitches: Sequence[int],
) -> tuple[Path, ...]:
    """Extract four cardinal-yaw stills from the result's near-horizontal layer."""
    camera_ids = select_cardinal_camera_ids(views_per_layer, layer_pitches)
    result_dir = Path(result_dir)
    return tuple(
        extract_middle_frame(
            result_dir / "videos" / f"{camera_id:02d}.mp4",
            Path(output_dir) / f"camera-{camera_id:02d}.jpg",
        )
        for camera_id in camera_ids
    )
