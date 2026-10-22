"""Convert LabelMe outline files (.json) into label images (.png) for scoring.

Areas with no outline are written as 255 and are skipped when scoring.
Later outlines are drawn over earlier ones.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import IGNORE, NAMES  # noqa: E402

LABEL_IDS = {name: label for label, name in NAMES.items()}


def shapes_to_mask(annotation: dict) -> np.ndarray:
    mask = np.full((annotation["imageHeight"], annotation["imageWidth"]), IGNORE, dtype=np.uint8)
    for shape in annotation["shapes"]:
        name = shape["label"].strip().lower()
        if name not in LABEL_IDS:
            raise ValueError(f"unknown label {shape['label']!r}; use one of {sorted(LABEL_IDS)}")
        points = np.round(np.array(shape["points"], dtype=np.float64)).astype(np.int32)
        kind = shape.get("shape_type", "polygon")
        if kind == "polygon":
            cv2.fillPoly(mask, [points], LABEL_IDS[name])
        elif kind == "rectangle":
            cv2.rectangle(mask, tuple(map(int, points[0])), tuple(map(int, points[1])), LABEL_IDS[name], -1)
        else:
            raise ValueError(f"unsupported shape type {kind!r}; draw polygons or rectangles")
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="folder of LabelMe .json files")
    parser.add_argument("--out", type=Path, required=True, help="folder for the .png label images")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.json"))
    if not files:
        print(f"No .json files found in {args.input}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in files:
        try:
            mask = shapes_to_mask(json.loads(path.read_text()))
        except (ValueError, KeyError) as error:
            print(f"{path.name}: {error}", file=sys.stderr)
            failed += 1
            continue
        cv2.imwrite(str(args.out / f"{path.stem}.png"), mask)
        print(f"{path.name}: wrote {path.stem}.png")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
