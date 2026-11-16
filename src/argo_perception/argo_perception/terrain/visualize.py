import cv2
import numpy as np

from .costmap import BLOCKED, MAX_BUFFER, NEAR_OBSTACLE, UNKNOWN, Costmap
from .labels import COLOURS_BGR, PREDICTION_NAMES, UNKNOWN


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
    # unknown is listed only when the picture has some, so rules-only output keeps its three-entry legend
    shown = [label for label in COLOURS_BGR if label != UNKNOWN or (labels == UNKNOWN).any()]
    _legend(out, [(PREDICTION_NAMES[label], COLOURS_BGR[label]) for label in shown])
    return out


def confidence_image(image: np.ndarray, confidence: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Photo tinted by how sure the model is: dark purple is a guess, bright yellow is certain.

    Uses one fixed 0..1 scale (not stretched to the photo), so two photos can be compared by eye.
    A colour bar on the right edge marks 0, 0.5 and 1.
    """
    shade = np.clip(np.round(confidence * 255), 0, 255).astype(np.uint8)
    tint = cv2.applyColorMap(shade, cv2.COLORMAP_VIRIDIS)
    out = cv2.addWeighted(tint, alpha, image, 1 - alpha, 0)

    scale = max(out.shape[1] / 640.0, 0.5)
    bar_w, pad = int(16 * scale), int(8 * scale)
    bar_h = min(out.shape[0] - 2 * pad, int(160 * scale))
    left, top = out.shape[1] - pad - bar_w, pad
    ramp = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None].repeat(bar_w, axis=1)
    out[top : top + bar_h, left : left + bar_w] = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS)
    cv2.rectangle(out, (left, top), (left + bar_w, top + bar_h), (255, 255, 255), 1)
    for value in (0.0, 0.5, 1.0):
        y = int(round(top + (1 - value) * (bar_h - 1)))
        origin = (left - int(34 * scale), min(max(y + 4, top + 10), top + bar_h))
        text = f"{value:.1f}"
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), max(3, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
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
