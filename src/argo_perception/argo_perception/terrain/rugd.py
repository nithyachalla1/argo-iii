"""Convert RUGD's colour-coded label images to flat / rocky / non-ground."""

import numpy as np

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY

# name -> (RGB colour in RUGD's label images, our label). From RUGD_annotation-colormap.txt.
RUGD_CLASSES = {
    "void": ((0, 0, 0), IGNORE),
    "dirt": ((108, 64, 20), FLAT),
    "sand": ((255, 229, 204), FLAT),
    "grass": ((0, 102, 0), FLAT),
    "tree": ((0, 255, 0), NON_GROUND),
    "pole": ((0, 153, 153), NON_GROUND),
    "water": ((0, 128, 255), NON_GROUND),
    "sky": ((0, 0, 255), NON_GROUND),
    "vehicle": ((255, 255, 0), NON_GROUND),
    "container/generic-object": ((255, 0, 127), NON_GROUND),
    "asphalt": ((64, 64, 64), FLAT),
    "gravel": ((255, 128, 0), ROCKY),
    "building": ((255, 0, 0), NON_GROUND),
    "mulch": ((153, 76, 0), FLAT),
    "rock-bed": ((102, 102, 0), ROCKY),
    "log": ((102, 0, 0), NON_GROUND),
    "bicycle": ((0, 255, 128), NON_GROUND),
    "person": ((204, 153, 255), NON_GROUND),
    "fence": ((102, 0, 204), NON_GROUND),
    "bush": ((255, 153, 204), NON_GROUND),
    "sign": ((0, 102, 102), NON_GROUND),
    "rock": ((153, 204, 255), NON_GROUND),
    # A bridge is driven over in some frames and is a structure in others, so it is not scored.
    "bridge": ((102, 255, 255), IGNORE),
    "concrete": ((101, 101, 11), FLAT),
    "picnic-table": ((114, 85, 47), NON_GROUND),
}


def _colour_key(red, green, blue):
    return (red << 16) | (green << 8) | blue


_LOOKUP = np.full(1 << 24, IGNORE, dtype=np.uint8)
_KNOWN = np.zeros(1 << 24, dtype=bool)
for (_rgb, _label) in RUGD_CLASSES.values():
    _LOOKUP[_colour_key(*_rgb)] = _label
    _KNOWN[_colour_key(*_rgb)] = True


def rugd_to_labels(annotation_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return our label image and the number of pixels whose colour is not a RUGD class."""
    pixels = annotation_bgr.astype(np.uint32)
    keys = _colour_key(pixels[..., 2], pixels[..., 1], pixels[..., 0])
    return _LOOKUP[keys], int((~_KNOWN[keys]).sum())
