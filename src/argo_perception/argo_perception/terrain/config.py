from dataclasses import dataclass
from pathlib import Path

import yaml

GROUND_PRIORS = ("none", "bottom")


@dataclass(frozen=True)
class TerrainConfig:
    work_width: int = 640
    cell_size: int = 16

    lightness_weight: float = 0.5
    texture_scale: float = 0.12
    gradient_scale: float = 0.5
    edge_scale: float = 0.25
    canny_low: int = 50
    canny_high: int = 150

    ground_prior: str = "none"
    n_clusters: int = 5
    ground_min_frac: float = 0.25
    bottom_strip_frac: float = 0.2
    bottom_cluster_share: float = 0.15

    non_ground_distance: float = 24.0
    min_object_frac: float = 0.008
    min_object_solidity: float = 0.5
    object_gap_cells: int = 2

    rocky_threshold: float = 0.5

    def __post_init__(self):
        if self.ground_prior not in GROUND_PRIORS:
            raise ValueError(f"ground_prior must be one of {GROUND_PRIORS}, got {self.ground_prior!r}")
        if self.work_width < self.cell_size or self.cell_size < 2:
            raise ValueError("work_width must be at least cell_size, and cell_size at least 2")


def load_config(path) -> TerrainConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return TerrainConfig(**data)
