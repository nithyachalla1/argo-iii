from dataclasses import dataclass

import cv2
import numpy as np

from .config import TerrainConfig
from .features import cell_features, learned_features, prepare
from .ground_model import ground_distance, object_mask
from .labels import FLAT, NON_GROUND, ROCKY

_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class TerrainResult:
    labels: np.ndarray
    cell_labels: np.ndarray
    roughness: np.ndarray | None = None
    ground_distance: np.ndarray | None = None


def _ground_roughness(roughness: np.ndarray, objects: np.ndarray) -> np.ndarray:
    # Average over neighbours that are clear of any object, so an object's outline
    # does not make the ground next to it look rough.
    clear = cv2.erode((~objects).astype(np.uint8), _KERNEL).astype(np.float32)
    total = cv2.blur(roughness * clear, (3, 3))
    weight = cv2.blur(clear, (3, 3))
    return np.where(weight > 0, total / np.maximum(weight, 1e-6), roughness)


def _rule_labels(work: np.ndarray, cfg: TerrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    colour, roughness = cell_features(work, cfg)
    distance = ground_distance(colour, cfg)
    objects = object_mask(distance, cfg)

    rocky = (_ground_roughness(roughness, objects) > cfg.rocky_threshold).astype(np.uint8)
    rocky = cv2.morphologyEx(rocky, cv2.MORPH_OPEN, _KERNEL)
    rocky = cv2.morphologyEx(rocky, cv2.MORPH_CLOSE, _KERNEL)

    cell_labels = np.where(objects, NON_GROUND, np.where(rocky == 1, ROCKY, FLAT)).astype(np.uint8)
    return cell_labels, roughness, distance


def classify_terrain(
    image: np.ndarray, cfg: TerrainConfig = TerrainConfig(), model=None, combine: bool = False
) -> TerrainResult:
    """Label every pixel of a BGR image as FLAT, ROCKY or NON_GROUND.

    With a TerrainModel the trained model decides; without one the hand-set rules do.
    With combine, the model decides and anything the rules call an object is also NON_GROUND.
    """
    if combine and model is None:
        raise ValueError("combine needs a trained model")
    work = prepare(image, cfg)
    if model is None:
        cell_labels, roughness, distance = _rule_labels(work, cfg)
    else:
        cell_labels, roughness, distance = model.predict_cells(learned_features(work, cfg)), None, None
        if combine:
            rule_labels, roughness, distance = _rule_labels(work, cfg)
            cell_labels = np.where(rule_labels == NON_GROUND, NON_GROUND, cell_labels).astype(np.uint8)

    height, width = image.shape[:2]
    labels = cv2.resize(cell_labels, (width, height), interpolation=cv2.INTER_NEAREST)
    return TerrainResult(labels, cell_labels, roughness, distance)
