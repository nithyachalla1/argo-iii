"""Label terrain photos as flat / rocky / non-ground and save an overlay and a label image.

With --model, also save a confidence picture (<name>_confidence.jpg) and the numbers (<name>_confidence.npy);
add --min-confidence to label cells the model is unsure of as unknown instead of guessing.

With --costmap, also save a top-down costmap of the ground in front of the rover
(<name>_costmap.png to look at, <name>_costmap.npy for planners).
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import (  # noqa: E402
    build_costmap,
    classify_terrain,
    confidence_image,
    costmap_image,
    load_config,
    load_costmap_config,
    overlay,
)
from argo_perception.terrain import UNKNOWN  # noqa: E402
from argo_perception.terrain.config import GROUND_PRIORS  # noqa: E402
from argo_perception.terrain.learned import TerrainModel  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="a photo, or a folder of photos")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: out/)")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "terrain_classifier.yaml")
    parser.add_argument("--ground-prior", choices=GROUND_PRIORS, help="override the setting in the config file")
    parser.add_argument("--model", type=Path, help="trained model file; without it the hand-set rules are used")
    parser.add_argument(
        "--combine", action="store_true", help="with --model: also mark anything the rules call an object as non-ground"
    )
    parser.add_argument(
        "--min-confidence", type=float, help="with --model: label cells the model is less sure of than this (0..1) as unknown"
    )
    parser.add_argument("--costmap", action="store_true", help="also save a top-down costmap (rover's-eye photos only)")
    parser.add_argument("--costmap-config", type=Path, default=REPO / "config" / "costmap.yaml")
    args = parser.parse_args()
    if args.combine and not args.model:
        parser.error("--combine needs --model")
    if args.min_confidence is not None and not args.model:
        parser.error("--min-confidence needs --model: the hand-set rules give no confidence")

    cfg = load_config(args.config)
    if args.ground_prior:
        cfg = dataclasses.replace(cfg, ground_prior=args.ground_prior)
    if args.min_confidence is not None:
        cfg = dataclasses.replace(cfg, min_confidence=args.min_confidence)
    model = TerrainModel.load(args.model) if args.model else None
    costmap_cfg = load_costmap_config(args.costmap_config) if args.costmap else None
    if args.costmap and model is None and cfg.ground_prior != "bottom":
        print("Warning: --costmap expects rover's-eye photos; use --model or --ground-prior bottom", file=sys.stderr)

    images = find_images(args.input)
    if not images:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not read {path}", file=sys.stderr)
            failed += 1
            continue
        result = classify_terrain(image, cfg, model, args.combine)
        cv2.imwrite(str(args.out / f"{path.stem}_overlay.jpg"), overlay(image, result.labels))
        cv2.imwrite(str(args.out / f"{path.stem}_labels.png"), result.labels)
        written = [f"{path.stem}_overlay.jpg", f"{path.stem}_labels.png"]
        note = ""
        if result.confidence is not None:
            cv2.imwrite(str(args.out / f"{path.stem}_confidence.jpg"), confidence_image(image, result.confidence))
            np.save(args.out / f"{path.stem}_confidence.npy", result.confidence)
            written += [f"{path.stem}_confidence.jpg", f"{path.stem}_confidence.npy"]
            note = f" ({(result.labels == UNKNOWN).mean():.1%} unknown, mean confidence {result.confidence.mean():.2f})"
        if costmap_cfg:
            costmap = build_costmap(result.labels, costmap_cfg)
            cv2.imwrite(str(args.out / f"{path.stem}_costmap.png"), costmap_image(costmap))
            np.save(args.out / f"{path.stem}_costmap.npy", costmap.costs)
            written += [f"{path.stem}_costmap.png", f"{path.stem}_costmap.npy"]
        print(f"{path.name}: wrote {', '.join(written)}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
