import cv2
import numpy as np

from .config import TerrainConfig


def prepare(image: np.ndarray, cfg: TerrainConfig) -> np.ndarray:
    """Resize to the working width, with both sides a whole number of cells."""
    h, w = image.shape[:2]
    cols = cfg.work_width // cfg.cell_size
    rows = max(1, round(h * cols / w))
    size = (cols * cfg.cell_size, rows * cfg.cell_size)
    interpolation = cv2.INTER_AREA if size[0] < w else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def _pool(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    # INTER_AREA with a whole-number shrink factor is an exact per-cell mean.
    return cv2.resize(values, (cols, rows), interpolation=cv2.INTER_AREA)


def _measure(bgr: np.ndarray, cfg: TerrainConfig) -> dict[str, np.ndarray]:
    """Per-cell measurements shared by the hand-set rules and the trained model."""
    rows = bgr.shape[0] // cfg.cell_size
    cols = bgr.shape[1] // cfg.cell_size

    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lab_mean = _pool(lab, rows, cols)
    lab_std = np.sqrt(np.maximum(_pool(lab * lab, rows, cols) - lab_mean * lab_mean, 0.0))

    lightness = lab[..., 0] / 100.0

    # Subtracting a wide blur removes slow shading, so a shadow gradient is not read as texture.
    detail = lightness - cv2.GaussianBlur(lightness, (0, 0), cfg.cell_size)
    mean = _pool(detail, rows, cols)
    mean_sq = _pool(detail * detail, rows, cols)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    gx = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)

    grey = cv2.GaussianBlur((lightness * 255).astype(np.uint8), (3, 3), 0)
    edges = cv2.Canny(grey, cfg.canny_low, cfg.canny_high)

    return {
        "lab_mean": lab_mean,
        "lab_std": lab_std,
        "texture": texture,
        "gradient": _pool(cv2.magnitude(gx, gy), rows, cols),
        "gradient_x": _pool(np.abs(gx), rows, cols),
        "gradient_y": _pool(np.abs(gy), rows, cols),
        "edge_density": _pool(edges.astype(np.float32) / 255.0, rows, cols),
    }


def cell_features(bgr: np.ndarray, cfg: TerrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return per-cell colour (rows, cols, 3) and roughness (rows, cols) in 0..1."""
    m = _measure(bgr, cfg)
    colour = m["lab_mean"].copy()
    colour[..., 0] *= cfg.lightness_weight
    roughness = (
        np.clip(m["texture"] / cfg.texture_scale, 0, 1)
        + np.clip(m["gradient"] / cfg.gradient_scale, 0, 1)
        + np.clip(m["edge_density"] / cfg.edge_scale, 0, 1)
    ) / 3.0
    return colour, roughness.astype(np.float32)


_LOCAL = [
    "lightness", "green_red", "blue_yellow",
    "lightness_spread", "green_red_spread", "blue_yellow_spread",
    "texture", "edge_strength", "edge_strength_x", "edge_strength_y", "edge_density",
]
_COLOUR = ["lightness", "green_red", "blue_yellow"]
_CONTEXT_SIZES = (9, 21)
FEATURE_NAMES = (
    _LOCAL
    + [f"{name}_nearby" for name in _LOCAL]
    + ["lightness_vs_bottom", "green_red_vs_bottom", "blue_yellow_vs_bottom", "height_in_photo"]
    + [f"{name}_vs_area{size}" for size in _CONTEXT_SIZES for name in _COLOUR + ["texture"]]
    + [f"{name}_vs_below" for name in _COLOUR]
    + [f"{name}_vs_column_bottom" for name in _COLOUR]
)


def _context(lab_mean: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """How much each cell stands out from what is around and beneath it.

    An object differs from the ground it stands on even when it is unlike anything
    in the training photos, so these help the model find objects it has not seen.
    """
    parts = []
    for size in _CONTEXT_SIZES:
        parts.append(lab_mean - cv2.blur(lab_mean, (size, size)))
        parts.append((texture - cv2.blur(texture, (size, size)))[..., None])
    rows = lab_mean.shape[0]
    below = np.concatenate([lab_mean[2:], np.repeat(lab_mean[-1:], min(2, rows), axis=0)])
    parts.append(lab_mean - below)
    parts.append(lab_mean - lab_mean[-4:].mean(axis=0, keepdims=True))
    return np.dstack(parts)


def learned_features(bgr: np.ndarray, cfg: TerrainConfig) -> np.ndarray:
    """Per-cell measurements for the trained model, shape (rows, cols, len(FEATURE_NAMES))."""
    m = _measure(bgr, cfg)
    local = np.dstack(
        [m["lab_mean"], m["lab_std"], m["texture"], m["gradient"], m["gradient_x"], m["gradient_y"], m["edge_density"]]
    )
    rows, cols = local.shape[:2]
    nearby = np.dstack([cv2.blur(local[..., i], (5, 5)) for i in range(local.shape[2])])

    strip_rows = max(1, round(rows * cfg.bottom_strip_frac))
    vs_bottom = m["lab_mean"] - m["lab_mean"][-strip_rows:].mean(axis=(0, 1))
    height = np.repeat(np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None], cols, axis=1)
    context = _context(m["lab_mean"], m["texture"])
    return np.dstack([local, nearby, vs_bottom, height, context]).astype(np.float32)
