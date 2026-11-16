import numpy as np

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY, UNKNOWN

N_CLASSES = 3
N_PREDICTED = 4  # the classes plus UNKNOWN


def confusion(truth: np.ndarray, predicted: np.ndarray, n_predicted: int = N_CLASSES) -> np.ndarray:
    """Pixel counts indexed [true label, predicted label]; IGNORE pixels are skipped.

    Pass n_predicted=N_PREDICTED when the prediction may contain UNKNOWN: the matrix then has a fourth column.
    """
    if truth.shape != predicted.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape}, predicted {predicted.shape}")
    keep = truth != IGNORE
    pairs = truth[keep].astype(np.int64) * n_predicted + predicted[keep]
    return np.bincount(pairs, minlength=N_CLASSES * n_predicted).reshape(N_CLASSES, n_predicted)


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


def selective_scores(matrix: np.ndarray) -> dict[str, float]:
    """Summarise a [true, predicted] matrix that has an UNKNOWN column (see N_PREDICTED).

    coverage:          share of scored pixels that got a real label rather than UNKNOWN.
    accuracy:          of those labelled pixels, the share labelled correctly (3 classes).
    missed_object:     of true non-ground pixels, the share labelled FLAT or ROCKY. The dangerous mistake.
    object_unknown:    of true non-ground pixels, the share sent to UNKNOWN (flagged, so not silently driven over).
    """
    if matrix.shape != (N_CLASSES, N_PREDICTED):
        raise ValueError(f"expected a {N_CLASSES}x{N_PREDICTED} matrix, got {matrix.shape}")
    total = matrix.sum()
    labelled = matrix[:, :N_CLASSES]
    objects = matrix[NON_GROUND].sum()
    nan = float("nan")
    return {
        "coverage": float(labelled.sum() / total) if total else nan,
        "accuracy": float(np.trace(labelled) / labelled.sum()) if labelled.sum() else nan,
        "missed_object": float(matrix[NON_GROUND, [FLAT, ROCKY]].sum() / objects) if objects else nan,
        "object_unknown": float(matrix[NON_GROUND, UNKNOWN] / objects) if objects else nan,
    }
