"""Settings for the ground-segmentation pipeline. Edit these, then run main.py."""

CONFIG = {
    # Camera mounting (used to level the cloud and draw the camera in plots)
    "camera_height_m": 0.40,
    "camera_pitch_deg": 0.0,

    # Classification
    "ground_threshold_m": 0.10,   # points higher than this above the ground plane are obstacles
    "ransac_threshold_m": 0.05,   # max distance from a candidate plane to count as an inlier
    "ransac_iterations": 100,
    "max_ground_slope_deg": 10.0, # steeper candidate planes can't be the ground
    "random_seed": 0,             # makes RANSAC repeatable

    # Clustering obstacle points into separate objects (DBSCAN, top-down)
    "cluster_eps_m": 0.25,        # points closer than this join the same object
    "cluster_min_points": 2,      # fewer neighbors than this = stray point, ignored

    # 2D occupancy grid (x = forward, y = left/right, camera at x=0, y=0)
    "map_resolution_m": 0.10,
    "map_width_m": 6.0,           # spans y from -3 m to +3 m
    "map_forward_m": 5.0,         # spans x from 0 m to 5 m
}
