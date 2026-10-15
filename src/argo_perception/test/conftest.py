import cv2
import numpy as np
import pytest

from argo_perception.terrain import FLAT, NON_GROUND, ROCKY

HEIGHT, WIDTH = 480, 640
SOIL = np.array([150, 170, 190], dtype=np.float32)
OBJECT_BGR = (200, 60, 30)
SKY_BGR = (235, 200, 150)


def _soil(rng, height=HEIGHT, width=WIDTH):
    image = SOIL + rng.normal(0, 2, (height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def _scatter_rocks(image, rng, x0, x1, y0, y1, count):
    for _ in range(count):
        centre = (int(rng.integers(x0, x1)), int(rng.integers(y0, y1)))
        axes = (int(rng.integers(3, 10)), int(rng.integers(3, 10)))
        shade = tuple(float(v) for v in SOIL * rng.uniform(0.35, 1.3))
        cv2.ellipse(image, centre, axes, float(rng.uniform(0, 180)), 0, 360, shade, -1)


@pytest.fixture
def mixed_scene():
    """Left half flat, right half rocky, with a solid object on the flat side."""
    rng = np.random.default_rng(0)
    image = _soil(rng)
    rocky = _soil(rng)
    _scatter_rocks(rocky, rng, 0, WIDTH, 0, HEIGHT, 2500)
    image[:, WIDTH // 2 :] = rocky[:, WIDTH // 2 :]
    cv2.rectangle(image, (80, 160), (207, 303), OBJECT_BGR, -1)

    truth = np.full((HEIGHT, WIDTH), FLAT, dtype=np.uint8)
    truth[:, WIDTH // 2 :] = ROCKY
    truth[160:304, 80:208] = NON_GROUND
    return image, truth


@pytest.fixture
def flat_scene():
    return _soil(np.random.default_rng(1))


@pytest.fixture
def rocky_scene():
    rng = np.random.default_rng(2)
    image = _soil(rng)
    _scatter_rocks(image, rng, 0, WIDTH, 0, HEIGHT, 2500)
    return image


@pytest.fixture
def horizon_scene():
    """Rover's-eye view: sky over the top 60%, flat ground below."""
    image = _soil(np.random.default_rng(3))
    image[: int(HEIGHT * 0.6)] = SKY_BGR
    return image
