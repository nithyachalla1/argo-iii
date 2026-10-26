import cv2
import numpy as np

from .costmap import BLOCKED, MAX_BUFFER, NEAR_OBSTACLE, UNKNOWN, Costmap
from .labels import COLOURS_BGR, NAMES


def _legend(out: np.ndarray, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
    scale = max(out.shape[1] / 640.0, 0.5)
    box, pad = int(14 * scale), int(8 * scale)
    for row, (name, colour) in enumerate(entries):
        top = pad + row * (box + pad)
        cv2.rectangle(out, (pad, top), (pad + box, top + box), colour, -1)
        cv2.rectangle(out, (pad, top), (pad + box, top + box), (255, 255, 255), 1)
        origin = (pad * 2 + box, top + box - 2)
        cv2.putText(out, name, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), max(3, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, name, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)


def overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the image by label and add a legend."""
    tint = np.zeros_like(image)
    for label, colour in COLOURS_BGR.items():
        tint[labels == label] = colour
    out = cv2.addWeighted(tint, alpha, image, 1 - alpha, 0)
    _legend(out, [(NAMES[label], colour) for label, colour in COLOURS_BGR.items()])
    return out


_GREEN, _YELLOW, _RED = np.array([0, 200, 0]), np.array([0, 220, 255]), np.array([0, 0, 255])
_DARK_RED, _GREY = (0, 0, 140), (128, 128, 128)


def _cost_colours() -> np.ndarray:
    """BGR colour for every cost value 0..255."""
    share = np.arange(MAX_BUFFER + 1)[:, None] / MAX_BUFFER
    ramp = np.where(share < 0.5, _GREEN + (_YELLOW - _GREEN) * share * 2, _YELLOW + (_RED - _YELLOW) * (share - 0.5) * 2)
    table = np.zeros((256, 3), dtype=np.uint8)
    table[: MAX_BUFFER + 1] = np.round(ramp)
    table[[NEAR_OBSTACLE, BLOCKED]] = _DARK_RED
    table[UNKNOWN] = _GREY
    return table


def costmap_image(costmap: Costmap, scale: int = 10) -> np.ndarray:
    """Colour picture of a costmap, far edge at the top, with a line every metre and the rover at the bottom."""
    big = cv2.resize(costmap.costs, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    out = _cost_colours()[big]
    height, width = out.shape[:2]

    metre = scale / costmap.resolution_m
    for i in range(1, int(costmap.forward_m) + 1):
        y = int(round(height - i * metre))
        cv2.line(out, (0, y), (width, y), (255, 255, 255), 1)
    for i in range(-int(costmap.side_m), int(costmap.side_m) + 1):
        x = int(round(width / 2 + i * metre))
        cv2.line(out, (x, 0), (x, height), (255, 255, 255), 1)

    tip = (width // 2, height - int(metre * 0.3))
    cv2.fillConvexPoly(out, np.array([tip, (tip[0] - scale, height - 1), (tip[0] + scale, height - 1)]), (255, 0, 0))

    _legend(out, [("free", tuple(int(c) for c in _GREEN)), ("costly", tuple(int(c) for c in _RED)),
                  ("blocked", _DARK_RED), ("unknown", _GREY)])
    return out
