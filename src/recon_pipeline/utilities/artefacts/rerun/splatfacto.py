"""Record a trained static Gaussian scene, aligned cameras and metrics in Rerun.

Adapted from the user's working Nerfstudio_Splatfacto_Colab_Rerun notebook.
Run in the separately installed Rerun 0.36 environment.
"""

import argparse
from pathlib import Path
from recon_pipeline.utilities._output import report_progress, run_operation


def export_recording(args):
    import json
    import re
    from pathlib import Path

    import numpy as np
    import rerun as rr
    from PIL import Image
    from plyfile import PlyData, PlyElement
    from rerun import blueprint as rrb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    splat_path = args.splat.resolve()
    dataset_path = args.dataset.resolve()
    run_root = args.run_root.resolve()
    rerun_path = args.output.resolve()
    experiment_name = args.experiment_name
    run_timestamp = args.run_timestamp
    max_splats = args.max_splats
    view_count = args.view_count
    if max_splats < 1 or view_count < 1:
        raise ValueError("max-splats and view-count must be positive")
    if rerun_path.exists() and not args.replace_existing:
        raise FileExistsError(rerun_path)
    rerun_path.parent.mkdir(parents=True, exist_ok=True)
    report_progress(0.0, "Loading Gaussian PLY and aligned cameras")
    rr.init(
        "splatfacto_reconstruction",
        recording_id=f"{experiment_name}_{run_timestamp}",
        spawn=False,
    )
    rr.save(str(rerun_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Полный PLY остаётся на диске; для Rerun создаём ограниченную по размеру копию.
    # Сам PLY загружает официальный importer Rerun 0.36 — это надёжнее ручной сборки
    # GaussianSplats3D, ошибки которой archetype может только залогировать и вернуть пустой объект.
    vertex = PlyData.read(str(splat_path))["vertex"].data
    property_names = set(vertex.dtype.names or [])
    required = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
    }
    missing = sorted(required - property_names)
    if missing:
        raise ValueError(f"В standard Gaussian PLY отсутствуют поля: {missing}")

    count = len(vertex)
    if count == 0:
        raise ValueError("Экспортированный splat.ply не содержит гауссиан.")

    rng = np.random.default_rng(42)
    indices = (
        np.arange(count)
        if count <= max_splats
        else np.sort(rng.choice(count, max_splats, replace=False))
    )
    sampled_vertex = vertex[indices]

    centers = np.column_stack(
        [sampled_vertex[name] for name in ("x", "y", "z")]
    ).astype(np.float32)
    log_scales = np.column_stack(
        [sampled_vertex[f"scale_{axis}"] for axis in range(3)]
    ).astype(np.float32)
    scales = np.exp(np.clip(log_scales, -12, 3)).astype(np.float32)
    quats_wxyz = np.column_stack(
        [sampled_vertex[f"rot_{axis}"] for axis in range(4)]
    ).astype(np.float32)

    SH_C0 = 0.28209479177387814
    sh_dc = np.column_stack(
        [sampled_vertex[f"f_dc_{channel}"] for channel in range(3)]
    ).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    opacity_raw = np.asarray(sampled_vertex["opacity"], dtype=np.float32)
    alpha = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -30, 30)))
    rgba = np.clip(np.column_stack([rgb, alpha]) * 255.0, 0, 255).astype(np.uint8)

    arrays = {
        "centers": centers,
        "log_scales": log_scales,
        "quaternions": quats_wxyz,
        "sh_dc": sh_dc,
        "opacity": opacity_raw,
    }
    non_finite = {
        name: int((~np.isfinite(values)).sum()) for name, values in arrays.items()
    }
    if any(non_finite.values()):
        raise ValueError(f"Перед записью Rerun найдены NaN/Inf: {non_finite}")
    if float(alpha.max()) <= 0.0:
        raise ValueError("Все экспортированные гауссианы имеют нулевую opacity.")

    model_bounds_min = centers.min(axis=0)
    model_bounds_max = centers.max(axis=0)
    scale_quantiles = np.quantile(scales, [0.01, 0.50, 0.99])
    alpha_quantiles = np.quantile(alpha, [0.01, 0.50, 0.99])

    rerun_ply_path = rerun_path.with_suffix(".sample.ply")
    PlyData([PlyElement.describe(sampled_vertex, "vertex")], text=False).write(
        str(rerun_ply_path)
    )
    rr.log_file_from_path(
        rerun_ply_path,
        entity_path_prefix="world/scene",
        static=True,
    )

    # Отдельный диагностический point cloud использует те же центры. Если он виден,
    # а сплаты нет, проблема точно в Gaussian visualizer/WebGPU, а не в координатах.
    debug_count = min(len(centers), 10_000)
    debug_indices = np.linspace(0, len(centers) - 1, debug_count, dtype=np.int64)
    rr.log(
        "world/debug/gaussian_centers",
        rr.Points3D(
            centers[debug_indices],
            colors=rgba[debug_indices, :3],
            radii=0.003,
        ),
        static=True,
    )

    transforms = json.loads((dataset_path / "transforms.json").read_text())
    all_frames = transforms["frames"]
    dataparser = json.loads((run_root / "dataparser_transforms.json").read_text())
    world_transform = np.eye(4, dtype=np.float32)
    world_transform[:3, :4] = np.asarray(dataparser["transform"], dtype=np.float32)
    world_scale = float(dataparser["scale"])

    if len(all_frames) < view_count:
        raise ValueError(
            f"Для раскладки нужно {view_count} камер; найдено {len(all_frames)}"
        )

    def transformed_camera(frame):
        source_c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        c2w = world_transform @ source_c2w
        c2w[:3, 3] *= world_scale
        return c2w

    # Находим среднее кольцо по высоте камер после dataparser-преобразования.
    all_c2w = [transformed_camera(frame) for frame in all_frames]
    camera_centers = np.stack([c2w[:3, 3] for c2w in all_c2w])
    rounded_heights = np.round(camera_centers[:, 2], 3)
    levels, level_counts = np.unique(rounded_heights, return_counts=True)
    eligible_levels = levels[level_counts >= view_count]
    if len(eligible_levels) >= 2:
        middle_level = eligible_levels[
            np.argmin(np.abs(eligible_levels - np.median(camera_centers[:, 2])))
        ]
        ring_indices = np.flatnonzero(rounded_heights == middle_level)
    else:
        ring_indices = np.arange(len(all_frames))

    ring_center = np.median(camera_centers[ring_indices, :2], axis=0)
    ring_angles = np.arctan2(
        camera_centers[ring_indices, 1] - ring_center[1],
        camera_centers[ring_indices, 0] - ring_center[0],
    )
    ring_order = ring_indices[np.argsort(ring_angles)]
    selected_indices = ring_order[
        np.floor(np.arange(view_count) * len(ring_order) / view_count).astype(int)
    ]

    camera_entities = []
    camera_views = []

    def field(frame, name):
        value = frame.get(name, transforms.get(name))
        if value is None:
            raise KeyError(f"Не найден параметр камеры {name}")
        return value

    for slot, frame_index in enumerate(selected_indices):
        frame = all_frames[int(frame_index)]
        entity = f"world/cameras/camera_{slot:02d}"
        camera_entities.append(entity)
        c2w = all_c2w[int(frame_index)]

        width, height = int(field(frame, "w")), int(field(frame, "h"))
        intrinsics = np.array(
            [
                [field(frame, "fl_x"), 0, field(frame, "cx")],
                [0, field(frame, "fl_y"), field(frame, "cy")],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        rr.log(
            entity,
            rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
            static=True,
        )
        rr.log(
            entity,
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=[width, height],
                camera_xyz=rr.ViewCoordinates.RUB,
                image_plane_distance=0.25,
            ),
            static=True,
        )
        image_path = dataset_path / str(frame["file_path"]).removeprefix("./")
        rr.log(
            f"{entity}/image",
            rr.Image(np.asarray(Image.open(image_path).convert("RGBA"))),
            static=True,
        )
        camera_views.append(
            rrb.Spatial2DView(
                name=f"Input camera {int(frame_index):02d}",
                origin=entity,
                contents=[f"{entity}/image"],
            )
        )

    # TensorBoard scalars → временная шкала Rerun.
    event_files = sorted(run_root.glob("**/events.out.tfevents.*"))
    logged_metrics = 0
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
            for event in accumulator.Scalars(tag):
                rr.set_time("iteration", sequence=int(event.step))
                rr.log(f"metrics/{safe_tag}", rr.Scalars(float(event.value)))
                logged_metrics += 1

    scene_view = rrb.Vertical(
        rrb.Spatial3DView(
            name="Gaussian splats and cameras",
            origin="world",
            contents=["world/scene/**", *camera_entities],
        ),
        rrb.Horizontal(*camera_views, column_shares=[1] * view_count),
        row_shares=[3, 1],
    )
    debug_view = rrb.Spatial3DView(
        name="Debug Gaussian centers",
        origin="world",
        contents=["world/debug/**", *camera_entities],
    )
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            scene_view,
            debug_view,
            rrb.TimeSeriesView(name="Training metrics", origin="metrics"),
            active_tab=0,
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )
    rr.send_blueprint(blueprint)
    rr.get_global_data_recording().flush()
    rr.disconnect()
    if not rerun_path.is_file() or rerun_path.stat().st_size == 0:
        raise RuntimeError(f"Recording export produced no data: {rerun_path}")

    report_progress(1.0, "Reconstruction recording ready")
    return {
        "rerun_path": str(rerun_path),
        "visualized_splats": int(len(indices)),
        "full_splat_count": int(count),
        "logged_metrics": int(logged_metrics),
        "selected_camera_indices": selected_indices.tolist(),
        "selected_ring_size": int(len(ring_indices)),
        "debug_point_count": int(debug_count),
        "model_bounds_min": model_bounds_min.tolist(),
        "model_bounds_max": model_bounds_max.tolist(),
        "scale_quantiles_01_50_99": scale_quantiles.tolist(),
        "alpha_quantiles_01_50_99": alpha_quantiles.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("splat", "dataset", "run-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment-name", default="splatfacto")
    parser.add_argument("--run-timestamp", default="recording")
    parser.add_argument("--max-splats", type=int, default=100_000)
    parser.add_argument("--view-count", type=int, default=4)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args(argv)
    run_operation(lambda: export_recording(args))


if __name__ == "__main__":
    main()
