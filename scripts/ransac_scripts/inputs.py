"""Input point clouds. Each row is one point: [x forward, y left, z up] in meters.

Every scene returns (points, truth). truth[i] is 1 if point i really is an
obstacle and 0 if it is ground, so the pipeline can score itself. It is None
when the right answer isn't known.
"""
import numpy as np

POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.01],
    [1.0, 0.0, -0.02],
    [1.5, 0.0, 0.00],
    [2.0, 0.0, 0.35],
    [2.1, 0.0, 0.40],
])


def sample():
    # The original 6 points: 4 on the floor, 2 on something 2 m ahead.
    return POINTS, np.array([0, 0, 0, 0, 1, 1])


def synthetic(seed=0):
    # A gently sloped, noisy floor with a box, a pole, a low bump (should
    # stay ground) and a few stray points floating in the air.
    rng = np.random.default_rng(seed)
    slope = 0.02  # floor rises 2 cm per meter forward

    def floor(x):
        return slope * x

    # Floor: 1500 points with 1 cm of sensor noise.
    x = rng.uniform(0.3, 5.0, 1500)
    y = rng.uniform(-3.0, 3.0, 1500)
    ground = np.c_[x, y, floor(x) + rng.normal(0, 0.01, 1500)]

    # Bump: 5 cm high, low enough to drive over, so it counts as ground.
    x = rng.uniform(1.0, 1.2, 60)
    y = rng.uniform(0.5, 2.0, 60)
    bump = np.c_[x, y, floor(x) + rng.uniform(0, 0.05, 60)]

    # Box: 0.4 x 0.5 m footprint, 0.5 m tall.
    x = rng.uniform(2.0, 2.4, 300)
    y = rng.uniform(-1.2, -0.7, 300)
    box = np.c_[x, y, floor(x) + rng.uniform(0, 0.5, 300)]

    # Pole: 10 cm radius, 1 m tall, centered at (3.5, 1.0).
    angle = rng.uniform(0, 2 * np.pi, 200)
    x = 3.5 + 0.1 * np.cos(angle)
    y = 1.0 + 0.1 * np.sin(angle)
    pole = np.c_[x, y, floor(x) + rng.uniform(0, 1.0, 200)]

    # Strays: isolated junk points (dust, reflections) clustering should drop.
    x = rng.uniform(0.3, 5.0, 15)
    strays = np.c_[x, rng.uniform(-3, 3, 15), floor(x) + rng.uniform(0.3, 1.5, 15)]

    points = np.vstack([ground, bump, box, pole, strays])
    truth = np.r_[np.zeros(len(ground) + len(bump)), np.ones(len(box) + len(pole) + len(strays))]
    return points, truth.astype(int)


SCENES = {"sample": sample, "synthetic": synthetic}
