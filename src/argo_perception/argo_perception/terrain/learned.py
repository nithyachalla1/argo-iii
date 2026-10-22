"""Trained terrain model: gradient-boosted decision trees over per-cell measurements."""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FEATURE_NAMES
from .labels import IGNORE, NAMES


def cell_truth(mask: np.ndarray, rows: int, cols: int, min_share: float = 0.7) -> np.ndarray:
    """Majority label per cell; IGNORE where no label covers at least min_share of the cell."""
    shares = np.dstack(
        [cv2.resize((mask == label).astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA) for label in NAMES]
    )
    labels = shares.argmax(axis=2).astype(np.uint8)
    labels[shares.max(axis=2) < min_share] = IGNORE
    return labels


class TerrainModel:
    def __init__(self, trees: HistGradientBoostingClassifier):
        self.trees = trees

    @classmethod
    def train(cls, samples: np.ndarray, labels: np.ndarray, rounds: int = 300) -> "TerrainModel":
        trees = HistGradientBoostingClassifier(max_iter=rounds, random_state=0)
        trees.fit(samples, labels)
        return cls(trees)

    def predict_cells(self, features: np.ndarray) -> np.ndarray:
        rows, cols, count = features.shape
        return self.trees.predict(features.reshape(-1, count)).astype(np.uint8).reshape(rows, cols)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"trees": self.trees, "features": list(FEATURE_NAMES)}, path, compress=3)

    @classmethod
    def load(cls, path) -> "TerrainModel":
        stored = joblib.load(path)
        if stored["features"] != list(FEATURE_NAMES):
            raise ValueError(f"{path} was trained with different measurements; retrain it with scripts/train_terrain.py")
        return cls(stored["trees"])
