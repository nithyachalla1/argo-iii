import cv2
import numpy as np

from .labels import COLOURS_BGR, NAMES


def overlay(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Tint the image by label and add a legend."""
    tint = np.zeros_like(image)
    for label, colour in COLOURS_BGR.items():
        tint[labels == label] = colour
    out = cv2.addWeighted(tint, alpha, image, 1 - alpha, 0)

    scale = max(image.shape[1] / 640.0, 0.5)
    box, pad = int(14 * scale), int(8 * scale)
    for row, (label, colour) in enumerate(COLOURS_BGR.items()):
        top = pad + row * (box + pad)
        cv2.rectangle(out, (pad, top), (pad + box, top + box), colour, -1)
        cv2.rectangle(out, (pad, top), (pad + box, top + box), (255, 255, 255), 1)
        origin = (pad * 2 + box, top + box - 2)
        cv2.putText(out, NAMES[label], origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), max(3, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(out, NAMES[label], origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
    return out
