"""Score the terrain classifier against hand-labelled photos.

Expects, inside the data folder:
  images/<name>.jpg|png    the photos
  masks/<name>.png         label images (0 flat, 1 rocky, 2 non-ground, 255 skip)
  splits/<split>.txt       one photo name per line, without the file extension
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import NAMES, classify_terrain, load_config  # noqa: E402
from argo_perception.terrain.config import GROUND_PRIORS  # noqa: E402
from argo_perception.terrain.learned import TerrainModel  # noqa: E402
from argo_perception.terrain.metrics import (  # noqa: E402
    N_CLASSES,
    class_iou,
    confusion,
    flat_rocky_accuracy,
    ground_iou,
)


def find_image(folder: Path, name: str) -> Path | None:
    matches = sorted(folder.glob(f"{name}.*"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "terrain")
    parser.add_argument("--split", default="tune", help="name of the list in splits/ (default: tune)")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "terrain_classifier.yaml")
    parser.add_argument("--ground-prior", choices=GROUND_PRIORS, help="override the setting in the config file")
    parser.add_argument("--model", type=Path, help="trained model file; without it the hand-set rules are used")
    parser.add_argument(
        "--combine", action="store_true", help="with --model: also mark anything the rules call an object as non-ground"
    )
    args = parser.parse_args()
    if args.combine and not args.model:
        parser.error("--combine needs --model")

    cfg = load_config(args.config)
    if args.ground_prior:
        cfg = dataclasses.replace(cfg, ground_prior=args.ground_prior)
    model = TerrainModel.load(args.model) if args.model else None

    split_file = args.data / "splits" / f"{args.split}.txt"
    if not split_file.exists():
        print(f"Missing {split_file}", file=sys.stderr)
        return 1
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    seconds = []
    skipped = 0
    for name in names:
        image_path = find_image(args.data / "images", name)
        mask_path = args.data / "masks" / f"{name}.png"
        image = cv2.imread(str(image_path)) if image_path else None
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if image is None or truth is None:
            print(f"{name}: photo or label image missing, skipped", file=sys.stderr)
            skipped += 1
            continue
        if truth.shape != image.shape[:2]:
            print(f"{name}: label image is {truth.shape}, photo is {image.shape[:2]}, skipped", file=sys.stderr)
            skipped += 1
            continue

        start = time.perf_counter()
        result = classify_terrain(image, cfg, model, args.combine)
        seconds.append(time.perf_counter() - start)

        matrix = confusion(truth, result.labels)
        total += matrix
        ground, non_ground = ground_iou(matrix)
        print(f"{name:24s} ground {ground:.2f}  non-ground {non_ground:.2f}  flat/rocky {flat_rocky_accuracy(matrix):.2f}")

    scored = len(names) - skipped
    if scored == 0:
        print("No photos were scored.", file=sys.stderr)
        return 1

    rules = f"hand-set rules, ground_prior={cfg.ground_prior}"
    method = f"trained model {args.model.name}" if model else rules
    if args.combine:
        method += f" combined with {rules}"
    print(f"\nSplit '{args.split}': {scored} photos scored, {skipped} skipped, {method}")
    print("\nOverlap per label (1.00 is perfect):")
    for label, score in enumerate(class_iou(total)):
        print(f"  {NAMES[label]:11s} {score:.3f}")
    ground, non_ground = ground_iou(total)
    print(f"\nGround vs non-ground overlap:  ground {ground:.3f}, non-ground {non_ground:.3f}")
    print(f"Flat vs rocky, share correct:  {flat_rocky_accuracy(total):.3f}")

    print("\nPixel counts (rows: true label, columns: predicted label):")
    print(" " * 13 + "".join(f"{NAMES[label]:>12s}" for label in range(N_CLASSES)))
    for label in range(N_CLASSES):
        print(f"  {NAMES[label]:11s}" + "".join(f"{count:12d}" for count in total[label]))

    print(f"\nTime per photo: median {np.median(seconds) * 1000:.0f} ms, slowest {max(seconds) * 1000:.0f} ms")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
