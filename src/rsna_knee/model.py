"""A CPU-friendly multimodal baseline."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class MultimodalBaseline:
    """TF-IDF report features combined with aggregate MRI features."""

    text: TfidfVectorizer
    scaler: StandardScaler
    classifiers: list[LogisticRegression]

    def _features(self, texts: list[str], images: np.ndarray, fit: bool) -> object:
        text_features = (
            self.text.fit_transform(texts)
            if fit
            else self.text.transform(texts)
        )
        image_features = (
            self.scaler.fit_transform(images)
            if fit
            else self.scaler.transform(images)
        )
        return hstack([text_features, image_features]).tocsr()

    def fit(self, texts: list[str], images: np.ndarray, targets: np.ndarray) -> None:
        features = self._features(texts, images, fit=True)
        self.classifiers = []
        for index in range(targets.shape[1]):
            classifier = LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=42,
            )
            classifier.fit(features, targets[:, index])
            self.classifiers.append(classifier)

    def predict_proba(self, texts: list[str], images: np.ndarray) -> np.ndarray:
        if not self.classifiers:
            raise RuntimeError("The model must be fitted before prediction")
        features = self._features(texts, images, fit=False)
        return np.column_stack(
            [classifier.predict_proba(features)[:, 1] for classifier in self.classifiers]
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "MultimodalBaseline":
        model = joblib.load(path)
        if not isinstance(model, MultimodalBaseline):
            raise TypeError(f"{path} does not contain a MultimodalBaseline")
        return model
