"""Top-down costmap in metres from a rover's-eye label image, assuming the ground is flat."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY

# Cost values follow the ROS nav2 costmap convention.
FREE = 0
MAX_BUFFER = 252
NEAR_OBSTACLE = 253
BLOCKED = 254
UNKNOWN = 255

_SAMPLES = 4  # points per cell side looked up in the label image


@dataclass(frozen=True)
class CostmapConfig:
    # Camera placement. Placeholders until the mount is decided (docs/hardware_dependencies.md).
    camera_height_m: float = 0.30
    camera_pitch_deg: float = 20.0
    horizontal_fov_deg: float = 70.0

    resolution_m: float = 0.10
    forward_m: float = 4.0
    side_m: float = 2.0

    flat_cost: int = 0
    rocky_cost: int = 60
    blocked_ratio: float = 0.4

    rover_radius_m: float = 0.25
    inflation_radius_m: float = 0.6
    inflation_decay: float = 5.0

    def __post_init__(self):
        if self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be above 0")
        if not 0 <= self.camera_pitch_deg < 90:
            raise ValueError("camera_pitch_deg must be at least 0 (level) and below 90 (straight down)")
        if not 0 < self.horizontal_fov_deg < 180:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")
        if self.resolution_m <= 0 or self.forward_m < self.resolution_m or self.side_m < self.resolution_m:
            raise ValueError("resolution_m must be above 0, and forward_m and side_m at least one cell")
        for name in ("flat_cost", "rocky_cost"):
            if not FREE <= getattr(self, name) <= MAX_BUFFER:
                raise ValueError(f"{name} must be between {FREE} and {MAX_BUFFER}")
        if not 0 <= self.blocked_ratio < 1:
            raise ValueError("blocked_ratio must be at least 0 and below 1")
        if not 0 <= self.rover_radius_m <= self.inflation_radius_m:
            raise ValueError("rover_radius_m must be at least 0 and no more than inflation_radius_m")
        if self.inflation_decay <= 0:
            raise ValueError("inflation_decay must be above 0")


def load_costmap_config(path) -> CostmapConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return CostmapConfig(**data)


@dataclass(frozen=True)
class Costmap:
    """Costs indexed [row, col]: row 0 is the far edge, the last row is nearest the rover,
    and columns run from left to right as seen from the rover."""

    costs: np.ndarray
    resolution_m: float
    forward_m: float
    side_m: float

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        """(forward, left) in metres from the point on the ground under the camera."""
        return (
            self.forward_m - (row + 0.5) * self.resolution_m,
            self.side_m - (col + 0.5) * self.resolution_m,
        )


def ground_to_pixel(x, y, image_shape, cfg: CostmapConfig):
    """Pixel (u, v) where flat ground at (x forward, y left) appears, and whether it is in front of the camera."""
    height, width = image_shape[:2]
    focal = (width / 2) / np.tan(np.radians(cfg.horizontal_fov_deg) / 2)
    pitch = np.radians(cfg.camera_pitch_deg)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    z = -cfg.camera_height_m

    ahead = x * np.cos(pitch) - z * np.sin(pitch)
    right = -y
    down = -x * np.sin(pitch) - z * np.cos(pitch)
    valid = ahead > 1e-6
    safe = np.where(valid, ahead, 1.0)
    return width / 2 + focal * right / safe, height / 2 + focal * down / safe, valid


@lru_cache(maxsize=8)
def _lookup_maps(image_shape: tuple[int, int], cfg: CostmapConfig) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = _grid_shape(cfg)
    step = cfg.resolution_m / _SAMPLES
    x = cfg.forward_m - (np.arange(rows * _SAMPLES) + 0.5) * step
    y = cfg.side_m - (np.arange(cols * _SAMPLES) + 0.5) * step
    u, v, valid = np.broadcast_arrays(*ground_to_pixel(x[:, None], y[None, :], image_shape, cfg))
    # Points behind the camera are sent off the image, where the lookup returns IGNORE.
    u = np.where(valid, u, -1.0).astype(np.float32)
    v = np.where(valid, v, -1.0).astype(np.float32)
    return u, v


def _grid_shape(cfg: CostmapConfig) -> tuple[int, int]:
    return round(cfg.forward_m / cfg.resolution_m), round(2 * cfg.side_m / cfg.resolution_m)


def _add_buffer(costs: np.ndarray, cfg: CostmapConfig) -> np.ndarray:
    blocked = costs == BLOCKED
    if not blocked.any():
        return costs
    distance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5) * cfg.resolution_m
    buffer = np.where(
        distance <= cfg.inflation_radius_m,
        np.round(MAX_BUFFER * np.exp(-cfg.inflation_decay * np.maximum(distance - cfg.rover_radius_m, 0.0))),
        0,
    )
    buffer[distance <= cfg.rover_radius_m] = NEAR_OBSTACLE
    known = (costs != UNKNOWN) & ~blocked
    out = costs.copy()
    out[known] = np.maximum(costs[known], buffer[known]).astype(np.uint8)
    return out


def build_costmap(labels: np.ndarray, cfg: CostmapConfig = CostmapConfig()) -> Costmap:
    """Project a full-size label image (from classify_terrain) onto the ground in front of the rover."""
    u, v = _lookup_maps(tuple(labels.shape[:2]), cfg)
    seen_labels = cv2.remap(
        labels, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=int(IGNORE)
    )
    rows, cols = _grid_shape(cfg)
    cells = seen_labels.reshape(rows, _SAMPLES, cols, _SAMPLES).transpose(0, 2, 1, 3).reshape(rows, cols, -1)

    seen = (cells != IGNORE).sum(axis=2)
    blocked = (cells == NON_GROUND).sum(axis=2)
    ground_cost = cfg.flat_cost * (cells == FLAT).sum(axis=2) + cfg.rocky_cost * (cells == ROCKY).sum(axis=2)
    ground = np.maximum(seen - blocked, 1)

    costs = np.round(ground_cost / ground).astype(np.uint8)
    costs[blocked > cfg.blocked_ratio * seen] = BLOCKED
    costs[seen == 0] = UNKNOWN
    return Costmap(_add_buffer(costs, cfg), cfg.resolution_m, cfg.forward_m, cfg.side_m)
