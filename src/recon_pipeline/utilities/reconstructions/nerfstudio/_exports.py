"""Supported standalone Gaussian export formats and filenames."""

EXPORT_FILENAMES = {
    "ply": "splat.ply",
    "compressed_ply": "splat.compressed.ply",
    "spz": "splat.spz",
    "sog": "splat.sog",
}


def validate_export_formats(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("export_formats must be a non-empty list")
    if any(not isinstance(item, str) or item not in EXPORT_FILENAMES for item in value):
        raise ValueError(f"export_formats supports only: {', '.join(EXPORT_FILENAMES)}")
    if len(set(value)) != len(value):
        raise ValueError("export_formats must not contain duplicates")
    return tuple(value)
