FLAT = 0
ROCKY = 1
NON_GROUND = 2
UNKNOWN = 3  # prediction only: the model is not confident enough to say. Never appears in ground truth.
IGNORE = 255

# NAMES are the labels a photo can truly have; PREDICTION_NAMES adds the one the classifier may answer with.
NAMES = {FLAT: "flat", ROCKY: "rocky", NON_GROUND: "non_ground"}
PREDICTION_NAMES = {**NAMES, UNKNOWN: "unknown"}

COLOURS_BGR = {
    FLAT: (0, 200, 0),
    ROCKY: (0, 165, 255),
    NON_GROUND: (0, 0, 255),
    UNKNOWN: (200, 80, 160),
}
