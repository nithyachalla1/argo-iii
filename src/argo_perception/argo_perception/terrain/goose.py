"""Convert GOOSE label ids to flat / rocky / non-ground."""

import numpy as np

from .labels import FLAT, IGNORE, NON_GROUND, ROCKY

# label_key -> (name, our label). Keys from GOOSE's goose_label_mapping.csv.
# Classes that could be ground or not, or are not a surface to drive on, are IGNORE (not scored)
# rather than guessed: snow, leaves, moss, crops, rail_track, tree_root, bridge.
GOOSE_CLASSES = {
    0: ("undefined", IGNORE),
    1: ("traffic_cone", NON_GROUND),
    2: ("snow", IGNORE),
    3: ("cobble", ROCKY),
    4: ("obstacle", NON_GROUND),
    5: ("leaves", IGNORE),
    6: ("street_light", NON_GROUND),
    7: ("bikeway", FLAT),
    8: ("ego_vehicle", IGNORE),
    9: ("pedestrian_crossing", FLAT),
    10: ("road_block", NON_GROUND),
    11: ("road_marking", FLAT),
    12: ("car", NON_GROUND),
    13: ("bicycle", NON_GROUND),
    14: ("person", NON_GROUND),
    15: ("bus", NON_GROUND),
    16: ("forest", NON_GROUND),
    17: ("bush", NON_GROUND),
    18: ("moss", IGNORE),
    19: ("traffic_light", NON_GROUND),
    20: ("motorcycle", NON_GROUND),
    21: ("sidewalk", FLAT),
    22: ("curb", NON_GROUND),
    23: ("asphalt", FLAT),
    24: ("gravel", ROCKY),
    25: ("boom_barrier", NON_GROUND),
    26: ("rail_track", IGNORE),
    27: ("tree_crown", NON_GROUND),
    28: ("tree_trunk", NON_GROUND),
    29: ("debris", NON_GROUND),
    30: ("crops", IGNORE),
    31: ("soil", FLAT),
    32: ("rider", NON_GROUND),
    33: ("animal", NON_GROUND),
    34: ("truck", NON_GROUND),
    35: ("on_rails", NON_GROUND),
    36: ("caravan", NON_GROUND),
    37: ("trailer", NON_GROUND),
    38: ("building", NON_GROUND),
    39: ("wall", NON_GROUND),
    40: ("rock", NON_GROUND),  # as in RUGD, a rock is an object, not rocky ground
    41: ("fence", NON_GROUND),
    42: ("guard_rail", NON_GROUND),
    43: ("bridge", IGNORE),  # as in RUGD
    44: ("tunnel", NON_GROUND),
    45: ("pole", NON_GROUND),
    46: ("traffic_sign", NON_GROUND),
    47: ("misc_sign", NON_GROUND),
    48: ("barrier_tape", NON_GROUND),
    49: ("kick_scooter", NON_GROUND),
    50: ("low_grass", FLAT),
    51: ("high_grass", FLAT),  # as RUGD's grass
    52: ("scenery_vegetation", NON_GROUND),
    53: ("sky", NON_GROUND),
    54: ("water", NON_GROUND),
    55: ("wire", NON_GROUND),
    56: ("outlier", IGNORE),
    57: ("heavy_machinery", NON_GROUND),
    58: ("container", NON_GROUND),
    59: ("hedge", NON_GROUND),
    60: ("barrel", NON_GROUND),
    61: ("pipe", NON_GROUND),
    62: ("tree_root", IGNORE),
    63: ("military_vehicle", NON_GROUND),
}

_LOOKUP = np.full(256, IGNORE, dtype=np.uint8)
_KNOWN = np.zeros(256, dtype=bool)
for _key, (_name, _label) in GOOSE_CLASSES.items():
    _LOOKUP[_key] = _label
    _KNOWN[_key] = True


def goose_to_labels(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Return our label image and the number of pixels whose id is not a GOOSE class."""
    return _LOOKUP[ids], int((~_KNOWN[ids]).sum())
