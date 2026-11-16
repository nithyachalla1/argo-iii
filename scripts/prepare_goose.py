"""Convert GOOSE 2D frames and labels to flat / rocky / non-ground, ready for evaluate_confidence.py.

GOOSE: https://goose-dataset.de (CC BY-SA). Download a split, e.g.
  https://goose-dataset.de/storage/goose_2d_val.zip   (2.9 GB, photos and labels)
unzip it into its own empty folder, then:

  python scripts/prepare_goose.py --data <unzipped folder> --split val

Result, inside --root:
  images/   links to the visible-light photos, named <scene>_<frame>, e.g. 2022-12-07_aying_hills_0000
  masks/    label images (0 flat, 1 rocky, 2 non-ground, 255 not scored)
  splits/   <split>.txt, listing every photo of that split

Each scene (one drive, in one place and weather) becomes a group in evaluate_confidence.py.
The photos come from a camera behind a car windshield, higher than a rover's camera.
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain.goose import goose_to_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="the unzipped folder, holding images/ and labels/")
    parser.add_argument("--split", default="val", help="train or val (the test split has no labels)")
    parser.add_argument("--root", type=Path, default=REPO / "data" / "terrain" / "goose")
    args = parser.parse_args()

    scenes = sorted(p for p in (args.data / "labels" / args.split).iterdir() if p.is_dir())
    if not scenes:
        print(f"No scenes found in {args.data / 'labels' / args.split}", file=sys.stderr)
        return 1
    for sub in ("images", "masks", "splits"):
        (args.root / sub).mkdir(parents=True, exist_ok=True)

    names, unknown_pixels, missing = [], 0, 0
    for scene in scenes:
        for label_path in sorted(scene.glob("*_labelids.png")):
            stem = label_path.name[: -len("_labelids.png")]
            frame_id = stem.split("__")[1].split("_")[0]  # '0623'
            photo = args.data / "images" / args.split / scene.name / f"{stem}_windshield_vis.png"
            ids = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if not photo.exists() or ids is None or ids.ndim != 2:
                missing += 1
                continue
            name = f"{scene.name}_{frame_id}"
            mask, unknown = goose_to_labels(ids)
            unknown_pixels += unknown
            cv2.imwrite(str(args.root / "masks" / f"{name}.png"), mask)
            link = args.root / "images" / f"{name}.png"
            if not link.exists():
                link.symlink_to(photo.resolve())
            names.append(name)
    (args.root / "splits" / f"{args.split}.txt").write_text("\n".join(names) + "\n")

    print(f"Converted {len(names)} photos from {len(scenes)} scenes into {args.root}; {missing} skipped")
    if unknown_pixels:
        print(f"Warning: {unknown_pixels} pixels had an id that is not a GOOSE class; they are not scored", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
