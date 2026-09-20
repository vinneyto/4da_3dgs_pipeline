"""Build a Rerun recording from one completed 4DAnyone generation."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any


ProgressCallback = Callable[[int, int, str], None]


def select_camera_views(cameras: list[dict[str, Any]], view_count: int) -> list[dict[str, Any]]:
    """Select evenly spaced cameras from the first pitch layer."""
    import numpy as np

    first_layer = sorted(
        (camera for camera in cameras if int(camera["layer_index"]) == 0),
        key=lambda camera: float(camera["yaw"]),
    )
    if len(first_layer) < view_count:
        raise ValueError(
            f"Need at least {view_count} cameras in layer 0, found {len(first_layer)}"
        )
    indices = np.linspace(0, len(first_layer), view_count, endpoint=False, dtype=int)
    return [first_layer[int(index)] for index in indices]


def _read_video_frames(path: Path, expected_frames: int) -> Iterator[Any]:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= expected_frames:
                break
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _reconstruct_keypoints(
    generation: Path,
    fourdanyone_root: Path,
    model_dir: Path,
    cache_path: Path,
    device: str,
):
    import numpy as np
    import torch

    if cache_path.is_file():
        keypoints = np.load(cache_path)
        if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
            return keypoints.astype(np.float32)

    sys.path.insert(0, str(fourdanyone_root))
    from fdanyone.assets import resolve_regressor
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Rerun geometry requested {resolved_device}, but CUDA is unavailable")

    motion = MotionResult.load(generation / "gvhmr")
    geometry = _body_geometry(
        motion=motion,
        regressor_path=resolve_regressor(model_dir=model_dir),
        gvhmr_root=fourdanyone_root / "third_party" / "GVHMR",
        device=resolved_device,
        include_mesh=False,
    )
    keypoints = geometry.keypoints_world.astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, keypoints)
    del geometry
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return keypoints


class RerunExporter:
    """Write the verified Colab camera/skeleton layout to an RRD file."""

    def __init__(
        self,
        *,
        generation: Path,
        output: Path,
        experiment: str,
        fourdanyone_root: Path,
        model_dir: Path,
        view_count: int = 4,
        device: str = "auto",
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.generation = Path(generation)
        self.output = Path(output)
        self.experiment = experiment
        self.fourdanyone_root = Path(fourdanyone_root)
        self.model_dir = Path(model_dir)
        self.view_count = view_count
        self.device = device
        self.on_progress = on_progress or (lambda _current, _total, _message: None)

    def export(self) -> Path:
        import numpy as np
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError as error:
            raise RuntimeError(
                "Rerun export requires: pip install 'fourda-3dgs-pipeline[rerun]'"
            ) from error

        metadata_path = self.generation / "metadata.json"
        cameras_path = self.generation / "cameras.json"
        if not metadata_path.is_file() or not cameras_path.is_file():
            raise FileNotFoundError(
                f"Rerun export requires metadata.json and cameras.json in {self.generation}"
            )

        metadata = json.loads(metadata_path.read_text())
        camera_payload = json.loads(cameras_path.read_text())
        output_meta = metadata["output"]
        num_frames = int(output_meta["frames_per_video"])
        fps = float(Fraction(output_meta["fps"]))
        all_cameras = camera_payload["cameras"]
        selected = select_camera_views(all_cameras, self.view_count)

        self.output.parent.mkdir(parents=True, exist_ok=True)
        keypoints = _reconstruct_keypoints(
            self.generation,
            self.fourdanyone_root,
            self.model_dir,
            self.output.parent / "keypoints_3d.npy",
            self.device,
        )

        sys.path.insert(0, str(self.fourdanyone_root))
        from fdanyone.skeleton.keypoints import KEYPOINT_NAMES, LINKS, keypoint_color

        expected_shape = (num_frames, len(KEYPOINT_NAMES), 3)
        if keypoints.shape != expected_shape:
            raise ValueError(
                f"Unexpected keypoints shape {keypoints.shape}; expected {expected_shape}"
            )

        rr.init(application_id="4DAnyone", recording_id=self.experiment, spawn=False)
        rr.save(str(self.output))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        for camera in all_cameras:
            camera_id = int(camera["camera_id"])
            camera_path = f"world/cameras/camera_{camera_id:02d}"
            camera_to_world = np.asarray(camera["camera_to_world"], dtype=np.float64)
            intrinsics = np.asarray(camera["K"], dtype=np.float64)
            rr.log(
                camera_path,
                rr.Transform3D(
                    translation=camera_to_world[:3, 3],
                    mat3x3=camera_to_world[:3, :3],
                ),
                static=True,
            )
            rr.log(
                f"{camera_path}/image",
                rr.Pinhole(
                    image_from_camera=intrinsics,
                    resolution=[int(camera["image_width"]), int(camera["image_height"])],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                static=True,
            )

        streams: dict[int, Iterator[Any]] = {}
        for camera in selected:
            camera_id = int(camera["camera_id"])
            video_path = self.generation / camera["video"]
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            streams[camera_id] = _read_video_frames(video_path, num_frames)

        point_colors = [keypoint_color(index) for index in range(len(KEYPOINT_NAMES))]
        link_pairs = [(int(first), int(second)) for _, first, second, _, _ in LINKS]
        link_colors = [color for _, _, _, color, _ in LINKS]

        for frame_index in range(num_frames):
            rr.set_time("frame", sequence=frame_index)
            rr.set_time("time", duration=frame_index / fps)
            frame_keypoints = keypoints[frame_index]
            rr.log(
                "world/skeleton/keypoints",
                rr.Points3D(frame_keypoints, colors=point_colors, radii=0.018),
            )
            rr.log(
                "world/skeleton/bones",
                rr.LineStrips3D(
                    [frame_keypoints[[first, second]] for first, second in link_pairs],
                    colors=link_colors,
                    radii=0.009,
                ),
            )
            for camera in selected:
                camera_id = int(camera["camera_id"])
                try:
                    image = next(streams[camera_id])
                except StopIteration as error:
                    raise RuntimeError(
                        f"Camera {camera_id} ended before frame {frame_index}"
                    ) from error
                rr.log(
                    f"views/camera_{camera_id:02d}",
                    rr.Image(image).compress(jpeg_quality=88),
                )
            if frame_index % 10 == 0 or frame_index + 1 == num_frames:
                self.on_progress(
                    frame_index + 1,
                    num_frames,
                    f"Writing Rerun frame {frame_index + 1}/{num_frames}",
                )

        camera_views = [
            rrb.Spatial2DView(
                name=(
                    f"Camera {int(camera['camera_id']):02d} — "
                    f"yaw {float(camera['yaw']):.0f}°"
                ),
                origin=f"views/camera_{int(camera['camera_id']):02d}",
            )
            for camera in selected
        ]
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="Animated skeleton and all cameras",
                        origin="world",
                    ),
                    rrb.Horizontal(*camera_views, column_shares=[1] * len(camera_views)),
                    row_shares=[3, 1],
                ),
                rrb.TimePanel(state="expanded"),
            ),
            make_active=True,
            make_default=True,
        )
        # Finalize the recording before an AWS worker starts uploading its directory.
        rr.disconnect()
        return self.output
