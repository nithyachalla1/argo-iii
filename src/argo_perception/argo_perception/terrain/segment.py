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
    # Trained model only: the model's chance for each label it gave (0..1), per pixel and per cell.
    # None for the hand-set rules, which have no such number.
    confidence: np.ndarray | None = None
    cell_confidence: np.ndarray | None = None


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
    """Label every pixel of a BGR image as FLAT, ROCKY or NON_GROUND (or UNKNOWN).

    With a TerrainModel the trained model decides; without one the hand-set rules do.
    With combine, the model decides and anything the rules call an object is also NON_GROUND.
    With a model, cells where its confidence is under cfg.min_confidence are labelled UNKNOWN
    instead of being forced into a class. The hand-set rules give no confidence, so never say UNKNOWN.
    """
    if combine and model is None:
        raise ValueError("combine needs a trained model")
    work = prepare(image, cfg)
    cell_confidence = None
    if model is None:
        cell_labels, roughness, distance = _rule_labels(work, cfg)
    else:
        cell_labels, cell_confidence = model.predict_cells(
            learned_features(work, cfg), cfg.non_ground_threshold, cfg.min_confidence
        )
        roughness, distance = None, None
        if combine:
            # A cell the rules call an object is blocked outright, so it is never UNKNOWN.
            rule_labels, roughness, distance = _rule_labels(work, cfg)
            is_object = rule_labels == NON_GROUND
            cell_labels = np.where(is_object, NON_GROUND, cell_labels).astype(np.uint8)
            cell_confidence = np.where(is_object, 1.0, cell_confidence).astype(np.float32)

    height, width = image.shape[:2]
    labels = cv2.resize(cell_labels, (width, height), interpolation=cv2.INTER_NEAREST)
    confidence = None
    if cell_confidence is not None:
        confidence = cv2.resize(cell_confidence, (width, height), interpolation=cv2.INTER_NEAREST)
    return TerrainResult(labels, cell_labels, roughness, distance, confidence, cell_confidence)
