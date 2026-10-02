"""Operation parameters from the working Colab official-RGBA profile."""

import argparse
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    max_num_iterations: int = 60_000
    stop_split_at: int = 32_000
    cull_alpha_thresh: float = 0.02
    densify_grad_thresh: float = 0.0003
    densify_size_thresh: float = 0.0075
    split_screen_size: float = 0.03
    num_downscales: int = 1
    resolution_schedule: int = 2_000
    cull_scale_thresh: float = 0.10
    stop_screen_size_at: int = 32_000
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 10.0
    sh_degree: int = 2
    mask_threshold: int = 127
    mask_erosion_pixels: int = 1

    def __post_init__(self):
        defaults = {
            name: field.default
            for name, field in TrainingProfile.__dataclass_fields__.items()
        }
        for name, default in defaults.items():
            value = getattr(self, name)
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"{name} must be a boolean")
            elif isinstance(default, int) and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(f"{name} must be an integer")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_num_iterations < 1 or self.resolution_schedule < 1:
            raise ValueError("iterations and resolution schedule must be positive")
        if not 0 <= self.sh_degree <= 3 or not 0 <= self.mask_threshold <= 255:
            raise ValueError("SH degree must be 0..3 and mask threshold 0..255")


def add_profile_arguments(parser, prefix=""):
    for name, default in asdict(TrainingProfile()).items():
        options = {"default": default}
        if isinstance(default, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(default)
        parser.add_argument("--" + prefix + name.replace("_", "-"), **options)


def profile_arguments(profile):
    arguments = []
    for name in TrainingProfile.__dataclass_fields__:
        value = getattr(profile, name)
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            arguments.append("--" + ("" if value else "no-") + flag)
        else:
            arguments.extend(["--" + flag, str(value)])
    return arguments
