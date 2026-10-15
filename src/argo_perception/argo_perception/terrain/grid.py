import numpy as np

from .labels import NON_GROUND, ROCKY


def to_occupancy_grid(
    labels: np.ndarray,
    cell_size: int = 25,
    rocky_blocked: bool = False,
    blocked_ratio: float = 0.4,
) -> np.ndarray:
    """Grid of 1 (blocked) and 0 (free), indexed [row, col], for a grid route planner."""
    blocked = labels == NON_GROUND
    if rocky_blocked:
        blocked |= labels == ROCKY

    rows = labels.shape[0] // cell_size
    cols = labels.shape[1] // cell_size
    cells = blocked[: rows * cell_size, : cols * cell_size].reshape(rows, cell_size, cols, cell_size)
    return (cells.mean(axis=(1, 3)) > blocked_ratio).astype(int)
