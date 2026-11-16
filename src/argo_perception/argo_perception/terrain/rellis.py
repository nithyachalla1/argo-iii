"""Convert RELLIS-3D's label ids to flat / rocky / non-ground."""

import numpy as np

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY

# id -> (name, our label). Ids from RELLIS-3D's ontology.yaml (the same colour scheme as RUGD).
RELLIS_CLASSES = {
    0: ("void", IGNORE),
    1: ("dirt", FLAT),
    3: ("grass", FLAT),
    4: ("tree", NON_GROUND),
    5: ("pole", NON_GROUND),
    6: ("water", NON_GROUND),
    7: ("sky", NON_GROUND),
    8: ("vehicle", NON_GROUND),
    9: ("object", NON_GROUND),
    10: ("asphalt", FLAT),
    12: ("building", NON_GROUND),
    15: ("log", NON_GROUND),
    17: ("person", NON_GROUND),
    18: ("fence", NON_GROUND),
    19: ("bush", NON_GROUND),
    23: ("concrete", FLAT),
    27: ("barrier", NON_GROUND),
    31: ("puddle", NON_GROUND),  # standing water, treated like RUGD's water
    # Not scored: whether mud is drivable depends on the rover and the mud, and RUGD has no equivalent.
    # Use scripts/evaluate_confidence.py output on photos with mud to see how sure the model is about it.
    33: ("mud", IGNORE),
    34: ("rubble", ROCKY),
}

_LOOKUP = np.full(256, IGNORE, dtype=np.uint8)
_KNOWN = np.zeros(256, dtype=bool)
for _id, (_name, _label) in RELLIS_CLASSES.items():
    _LOOKUP[_id] = _label
    _KNOWN[_id] = True


def rellis_to_labels(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return our label image and the number of pixels whose id is not a RELLIS class."""
    return _LOOKUP[ids], int((~_KNOWN[ids]).sum())
