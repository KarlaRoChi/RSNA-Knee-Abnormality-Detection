"""Dataset and feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pydicom

from .config import DEFAULT_MAX_SLICES, ID_COLUMN


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata and validate that the study identifier is present."""
    frame = pd.read_csv(path)
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{path} must contain the {ID_COLUMN!r} column")
    return frame


def _dicom_files(study_root: Path) -> list[Path]:
    return sorted(
        path
        for path in study_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dcm", ""}
    )


def _read_pixels(path: Path) -> np.ndarray | None:
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        pixels = dataset.pixel_array.astype(np.float32)
    except (OSError, ValueError, AttributeError, pydicom.errors.InvalidDicomError):
        return None
    if pixels.size == 0 or not np.isfinite(pixels).any():
        return None
    pixels = np.nan_to_num(pixels, copy=False)
    low, high = np.percentile(pixels, [1, 99])
    if high > low:
        pixels = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return pixels


def image_features(
    image_root: str | Path,
    study_id: str,
    max_slices: int = DEFAULT_MAX_SLICES,
) -> np.ndarray:
    """Extract stable low-cost statistics from a study's DICOM slices."""
    files = _dicom_files(Path(image_root) / study_id)[:max_slices]
    slices = [pixels for path in files if (pixels := _read_pixels(path)) is not None]
    if not slices:
        return np.zeros(10, dtype=np.float32)
    values = np.concatenate([pixels.ravel() for pixels in slices])
    return np.asarray(
        [
            values.mean(),
            values.std(),
            np.percentile(values, 1),
            np.percentile(values, 10),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            np.percentile(values, 90),
            np.percentile(values, 99),
            float(len(slices)) / max_slices,
        ],
        dtype=np.float32,
    )


def report_text(row: pd.Series) -> str:
    """Find a report-like field without assuming one Kaggle column name."""
    candidates = ("report", "Report", "text", "Text", "radiology_report")
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return ""


def build_inputs(
    metadata: pd.DataFrame,
    image_root: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Return report texts and one image-statistics row per study."""
    texts: list[str] = []
    features: list[np.ndarray] = []
    for _, row in metadata.iterrows():
        study_id = str(row[ID_COLUMN])
        texts.append(report_text(row))
        features.append(image_features(image_root, study_id))
    return texts, np.vstack(features)
