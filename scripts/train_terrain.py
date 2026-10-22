"""Train the terrain model on labelled photos and save it.

Uses the same data folder layout as scripts/evaluate_terrain.py. Score the result with:
  python scripts/evaluate_terrain.py --data <folder> --split val --model <saved model>
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import IGNORE, NAMES, load_config  # noqa: E402
from argo_perception.terrain.features import learned_features, prepare  # noqa: E402
from argo_perception.terrain.learned import TerrainModel, cell_truth  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "terrain" / "rugd")
    parser.add_argument("--split", default="train", help="name of the list in splits/ (default: train)")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "terrain_classifier.yaml")
    parser.add_argument("--out", type=Path, default=REPO / "models" / "terrain_rugd.joblib")
    parser.add_argument("--every", type=int, default=5, help="use every Nth photo; video frames repeat (default: 5)")
    parser.add_argument("--cells-per-photo", type=int, default=250, help="random cells sampled from each photo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_file = args.data / "splits" / f"{args.split}.txt"
    if not split_file.exists():
        print(f"Missing {split_file}", file=sys.stderr)
        return 1
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()][:: args.every]

    rng = np.random.default_rng(0)
    samples, labels = [], []
    skipped = 0
    for name in names:
        matches = sorted((args.data / "images").glob(f"{name}.*"))
        image = cv2.imread(str(matches[0])) if matches else None
        mask_path = args.data / "masks" / f"{name}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if image is None or mask is None or mask.shape != image.shape[:2]:
            print(f"{name}: photo or label image missing or mismatched, skipped", file=sys.stderr)
            skipped += 1
            continue

        features = learned_features(prepare(image, cfg), cfg)
        truth = cell_truth(mask, *features.shape[:2]).ravel()
        usable = np.flatnonzero(truth != IGNORE)
        chosen = rng.choice(usable, min(args.cells_per_photo, len(usable)), replace=False)
        samples.append(features.reshape(-1, features.shape[2])[chosen])
        labels.append(truth[chosen])

    if not samples:
        print("No photos could be used for training.", file=sys.stderr)
        return 1
    samples, labels = np.vstack(samples), np.concatenate(labels)

    print(f"Training on {len(labels):,} cells from {len(names) - skipped} photos (split '{args.split}', every {args.every})")
    for label, name in NAMES.items():
        print(f"  {name:11s} {np.mean(labels == label):6.1%} of cells")

    start = time.perf_counter()
    model = TerrainModel.train(samples, labels)
    model.save(args.out)
    print(f"Trained in {time.perf_counter() - start:.0f} s. Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
