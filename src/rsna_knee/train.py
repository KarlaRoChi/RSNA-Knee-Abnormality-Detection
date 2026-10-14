"""Train the baseline model from Kaggle metadata and study folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .config import TARGET_COLUMNS
from .data import build_inputs, load_metadata
from .metrics import macro_auc, validate_targets
from .model import MultimodalBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metadata = load_metadata(args.train_csv)
    validate_targets(list(metadata.columns))
    texts, images = build_inputs(metadata, args.image_root)
    targets = metadata[TARGET_COLUMNS].to_numpy(dtype=np.int8)

    model = MultimodalBaseline(
        text=TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=1),
        scaler=StandardScaler(),
        classifiers=[],
    )
    model.fit(texts, images, targets)
    train_score = macro_auc(targets, model.predict_proba(texts, images))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"Saved {args.output}; training macro-AUC={train_score:.5f}")


if __name__ == "__main__":
    main()
