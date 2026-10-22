import cv2
import numpy as np

from .config import TerrainConfig


def _merge_close(centres: np.ndarray, limit: float) -> np.ndarray:
    """Group index per cluster, joining clusters whose centres are within `limit`."""
    group = np.arange(len(centres))
    for i in range(len(centres)):
        for j in range(i):
            if np.linalg.norm(centres[i] - centres[j]) < limit:
                group[group == group[i]] = group[j]
    return group


def ground_distance(colour: np.ndarray, cfg: TerrainConfig) -> np.ndarray:
    """Colour distance of every cell from the nearest ground colour."""
    rows, cols = colour.shape[:2]
    samples = np.ascontiguousarray(colour.reshape(-1, 3), dtype=np.float32)
    k = min(cfg.n_clusters, len(samples))

    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, assignment, centres = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    assignment = assignment.ravel()

    if cfg.ground_prior == "bottom":
        strip_rows = max(1, round(rows * cfg.bottom_strip_frac))
        counted = assignment.reshape(rows, cols)[-strip_rows:].ravel()
        min_share = cfg.bottom_cluster_share
    else:
        counted = assignment
        min_share = cfg.ground_min_frac

    # k-means splits one evenly lit surface into several clusters, so shares are
    # counted per group of near-identical clusters rather than per cluster.
    group = _merge_close(centres, cfg.non_ground_distance / 2)
    share = np.bincount(group[counted], minlength=k) / counted.size
    ground_groups = share >= min_share
    ground_groups[np.argmax(share)] = True

    ground_centres = centres[ground_groups[group]]
    distances = np.linalg.norm(samples[:, None, :] - ground_centres[None, :, :], axis=2)
    return distances.min(axis=1).reshape(rows, cols).astype(np.float32)


def object_mask(distance: np.ndarray, cfg: TerrainConfig) -> np.ndarray:
    """Cells that are off-colour from the ground and form a large, solid blob."""
    candidates = (distance > cfg.non_ground_distance).astype(np.uint8)
    if cfg.object_gap_cells > 0:
        # Joins parts of one object split by a band that happens to match the ground colour.
        size = 2 * cfg.object_gap_cells + 1
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    min_cells = max(1, round(cfg.min_object_frac * candidates.size))

    count, components, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    objects = np.zeros(candidates.shape, dtype=bool)
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        if area < min_cells:
            continue
        blob = components == index
        hull = cv2.convexHull(cv2.findNonZero(blob.astype(np.uint8)))
        hull_cells = np.zeros(candidates.shape, dtype=np.uint8)
        cv2.fillConvexPoly(hull_cells, hull, 1)
        if area / hull_cells.sum() >= cfg.min_object_solidity:
            objects |= blob
    return objects
