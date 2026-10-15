import numpy as np

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY

N_CLASSES = 3


def confusion(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Pixel counts indexed [true label, predicted label]; IGNORE pixels are skipped."""
    if truth.shape != predicted.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape}, predicted {predicted.shape}")
    keep = truth != IGNORE
    pairs = truth[keep].astype(np.int64) * N_CLASSES + predicted[keep]
    return np.bincount(pairs, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def class_iou(matrix: np.ndarray) -> np.ndarray:
    """Overlap score per label; NaN for a label that never appears."""
    hits = np.diag(matrix).astype(float)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - hits
    return np.divide(hits, union, out=np.full(len(hits), np.nan), where=union > 0)


def ground_iou(matrix: np.ndarray) -> tuple[float, float]:
    """Overlap for (ground, non-ground), with flat and rocky merged into ground."""
    ground = [FLAT, ROCKY]
    merged = np.array(
        [
            [matrix[np.ix_(ground, ground)].sum(), matrix[ground, NON_GROUND].sum()],
            [matrix[NON_GROUND, ground].sum(), matrix[NON_GROUND, NON_GROUND]],
        ]
    )
    scores = class_iou(merged)
    return float(scores[0]), float(scores[1])


def flat_rocky_accuracy(matrix: np.ndarray) -> float:
    """Of the pixels that truly are ground, the share given the right flat/rocky label."""
    total = matrix[[FLAT, ROCKY]].sum()
    if total == 0:
        return float("nan")
    return float((matrix[FLAT, FLAT] + matrix[ROCKY, ROCKY]) / total)
