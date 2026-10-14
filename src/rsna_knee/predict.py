"""Generate a Kaggle-compatible submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import ID_COLUMN, TARGET_COLUMNS
from .data import build_inputs, load_metadata
from .model import MultimodalBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("submission.csv"), type=Path)
    args = parser.parse_args()

    metadata = load_metadata(args.test_csv)
    texts, images = build_inputs(metadata, args.image_root)
    predictions = np.clip(
        MultimodalBaseline.load(str(args.model)).predict_proba(texts, images), 0.0, 1.0
    )
    submission = metadata[[ID_COLUMN]].copy()
    submission[TARGET_COLUMNS] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(submission)} studies")


if __name__ == "__main__":
    main()
