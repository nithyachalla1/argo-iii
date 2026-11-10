"""The pipeline steps. run() chains them together."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor

GROUND_COLOR = "#2a78d6"    # blue
# One color per obstacle cluster, always assigned in this order.
CLUSTER_COLORS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FREE_CELL = "#a7b3c2"       # slate gray
OCCUPIED_CELL = "#e06c6c"   # soft red
INK = "#52514e"


# ---- Step 1: load and visualize ----

def load_points(points, config):
    # Undo the camera's pitch so z points straight up (no-op when pitch is 0).
    pitch = np.radians(config["camera_pitch_deg"])
    c, s = np.cos(pitch), np.sin(pitch)
    rotate_about_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.asarray(points, float) @ rotate_about_y.T


def plot_raw_cloud(points, config, path):
    fig, ax = _new_3d_axes("Input point cloud")
    ax.scatter(*points.T, s=_dot_size(points), color=INK, label="points")
    _draw_camera(ax, config)
    ax.legend(loc="upper left")
    _save(fig, path)


# ---- Step 2: ground plane with RANSAC ----

def ransac_ground_plane(points, config):
    # Fit z = a*x + b*y + c with scikit-learn's RANSAC: it repeatedly fits
    # random 3-point samples, keeps the fit most points agree with, then refits
    # on those inliers. LinearRegression copes with a single scan line (no
    # spread in y) by giving it b = 0.
    def flat_enough(model, X, z):
        # Reject candidates too steep to be ground. Without this, sklearn can
        # prefer a plane tilted up through an obstacle, since it breaks ties
        # in inlier count by R², which favors sloped fits.
        slope_deg = np.degrees(np.arctan(np.hypot(*model.coef_)))
        return slope_deg <= config["max_ground_slope_deg"]

    ransac = RANSACRegressor(
        LinearRegression(),
        min_samples=3,
        is_model_valid=flat_enough,
        residual_threshold=config["ransac_threshold_m"],
        max_trials=config["ransac_iterations"],
        random_state=config["random_seed"],
    )
    ransac.fit(points[:, :2], points[:, 2])
    (a, b), c = ransac.estimator_.coef_, ransac.estimator_.intercept_
    return np.array([a, b, c]), ransac.inlier_mask_


def height_above_plane(points, plane):
    # Signed perpendicular distance to the plane; positive = above it.
    a, b, c = plane
    return (points[:, 2] - (a * points[:, 0] + b * points[:, 1] + c)) / np.sqrt(a**2 + b**2 + 1)


# ---- Step 3: classify obstacles ----

def classify(points, plane, config):
    heights = height_above_plane(points, plane)
    is_obstacle = heights > config["ground_threshold_m"]
    return heights, is_obstacle


# ---- Step 4: group obstacle points into separate objects ----

def cluster_obstacles(points, is_obstacle, config):
    # DBSCAN on the top-down (x, y) positions: points within cluster_eps_m of
    # each other join the same object. Returns an id per point: 0, 1, 2... for
    # each object, -1 for ground and for stray points with too few neighbors.
    clusters = np.full(len(points), -1)
    if is_obstacle.any():
        dbscan = DBSCAN(eps=config["cluster_eps_m"], min_samples=config["cluster_min_points"])
        clusters[is_obstacle] = dbscan.fit_predict(points[is_obstacle, :2])
    return clusters


def plot_segmented_cloud(points, plane, clusters, is_obstacle, config, path):
    fig, ax = _new_3d_axes("Segmented point cloud")
    size = _dot_size(points)

    # Draw the fitted ground plane as a translucent surface.
    a, b, c = plane
    X, Y = np.meshgrid([points[:, 0].min(), points[:, 0].max()],
                       [min(points[:, 1].min(), -0.5), max(points[:, 1].max(), 0.5)])
    ax.plot_surface(X, Y, a * X + b * Y + c, color=GROUND_COLOR, alpha=0.15)

    ax.scatter(*points[~is_obstacle].T, s=size, color=GROUND_COLOR, label="ground")
    for i in range(clusters.max() + 1):
        color = CLUSTER_COLORS[i] if i < len(CLUSTER_COLORS) else INK
        ax.scatter(*points[clusters == i].T, s=size, color=color, label=f"obstacle {i + 1}")
    strays = is_obstacle & (clusters == -1)
    if strays.any():
        ax.scatter(*points[strays].T, s=size * 2, color=INK, marker="x", label="stray (ignored)")
    _draw_camera(ax, config)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    _save(fig, path)


# ---- Step 5: project obstacles onto a 2D grid ----

def build_occupancy_grid(points, config):
    # grid[row, col]: row = forward distance (x), col = sideways (y). 1 = obstacle.
    res = config["map_resolution_m"]
    n_rows = int(round(config["map_forward_m"] / res))
    n_cols = int(round(config["map_width_m"] / res))
    grid = np.zeros((n_rows, n_cols), np.uint8)

    rows = np.floor(points[:, 0] / res).astype(int)
    cols = np.floor((points[:, 1] + config["map_width_m"] / 2) / res).astype(int)
    on_map = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    grid[rows[on_map], cols[on_map]] = 1
    return grid


def plot_grid(grid, config, path):
    # Each cell is its own rounded tile with a white gap around it.
    res = config["map_resolution_m"]
    half_width = config["map_width_m"] / 2
    gap = res * 0.08

    tiles, colors = [], []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            x, y = -half_width + col * res + gap / 2, row * res + gap / 2
            tiles.append(FancyBboxPatch((x, y), res - gap, res - gap,
                                        boxstyle=f"round,pad=0,rounding_size={res * 0.12}"))
            colors.append(OCCUPIED_CELL if grid[row, col] else FREE_CELL)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.add_collection(PatchCollection(tiles, facecolors=colors, edgecolors="none"))
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(0, config["map_forward_m"])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot(0, 0, marker="^", markersize=10, color=INK, linestyle="none", label="camera", clip_on=False)
    ax.plot([], [], marker="s", markersize=8, color=FREE_CELL, linestyle="none", label="free")
    ax.plot([], [], marker="s", markersize=8, color=OCCUPIED_CELL, linestyle="none", label="obstacle")
    ax.set_xlabel("y, sideways (m)")
    ax.set_ylabel("x, forward (m)")
    ax.set_title(f"Occupancy grid ({res} m cells)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _save(fig, path)


# ---- Step 6: score against the known answer ----

def evaluate(is_obstacle, truth):
    # Compare predicted labels with the true ones (1 = obstacle, 0 = ground).
    if truth is None:
        return None
    truth = np.asarray(truth, bool)
    return {
        "accuracy": float((is_obstacle == truth).mean()),
        "ground_called_obstacle": int((is_obstacle & ~truth).sum()),  # false alarms
        "obstacle_called_ground": int((~is_obstacle & truth).sum()),  # misses
    }


# ---- Step 7: export ----

def export(out_dir, points, heights, clusters, plane, grid, scores, config):
    with open(out_dir / "points_classified.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "height_above_ground_m", "label", "cluster"])
        for (x, y, z), h, cluster in zip(points, heights, clusters):
            label = "ground" if h <= config["ground_threshold_m"] else "obstacle" if cluster >= 0 else "stray"
            writer.writerow([round(x, 4), round(y, 4), round(z, 4), round(h, 4), label, cluster])

    np.savetxt(out_dir / "occupancy_grid.csv", grid, fmt="%d", delimiter=",")

    summary = {
        "ground_plane": {"a": plane[0], "b": plane[1], "c": plane[2],
                         "equation": "z = a*x + b*y + c"},
        "num_points": len(points),
        "num_ground": int((heights <= config["ground_threshold_m"]).sum()),
        "num_obstacle_clusters": int(clusters.max() + 1),
        "num_stray": int(((heights > config["ground_threshold_m"]) & (clusters == -1)).sum()),
        "occupied_cells": int(grid.sum()),
        "grid_shape_rows_cols": list(grid.shape),
        "scores": scores,
        "config": config,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


# ---- Pipeline ----

def run(points, config, out_dir, truth=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(points, config)
    plot_raw_cloud(points, config, out_dir / "1_input_cloud.png")

    plane, _ = ransac_ground_plane(points, config)
    heights, is_obstacle = classify(points, plane, config)
    clusters = cluster_obstacles(points, is_obstacle, config)
    plot_segmented_cloud(points, plane, clusters, is_obstacle, config, out_dir / "2_segmented_cloud.png")

    # Only clustered points go on the map, so stray points don't block cells.
    grid = build_occupancy_grid(points[clusters >= 0], config)
    plot_grid(grid, config, out_dir / "3_occupancy_grid.png")

    scores = evaluate(is_obstacle, truth)
    return export(out_dir, points, heights, clusters, plane, grid, scores, config)


# ---- Plot helpers ----

def _new_3d_axes(title):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x, forward (m)")
    ax.set_ylabel("y, sideways (m)")
    ax.set_zlabel("z, up (m)")
    ax.set_title(title)
    return fig, ax


def _dot_size(points):
    # Big dots for a handful of points, small ones for a dense cloud.
    return 40 if len(points) < 100 else 3


def _draw_camera(ax, config):
    ax.scatter(0, 0, config["camera_height_m"], marker="^", s=80, color=INK, label="camera")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
