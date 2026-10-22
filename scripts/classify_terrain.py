"""Label terrain photos as flat / rocky / non-ground and save an overlay and a label image."""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import classify_terrain, load_config, overlay  # noqa: E402
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
    args = parser.parse_args()
    if args.combine and not args.model:
        parser.error("--combine needs --model")

    cfg = load_config(args.config)
    if args.ground_prior:
        cfg = dataclasses.replace(cfg, ground_prior=args.ground_prior)
    model = TerrainModel.load(args.model) if args.model else None

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
        print(f"{path.name}: wrote {path.stem}_overlay.jpg and {path.stem}_labels.png")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
