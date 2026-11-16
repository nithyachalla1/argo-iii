"""Measure what a confidence threshold buys: fewer wrong labels, at the price of more "unknown".

Runs the trained model once per photo, then re-labels every pixel whose confidence is under each
threshold as UNKNOWN and scores the result. Uses the same data folder layout as evaluate_terrain.py:
  images/<name>.jpg|png    the photos
  masks/<name>.png         label images (0 flat, 1 rocky, 2 non-ground, 255 skip)
  splits/<split>.txt       one photo name per line, without the file extension

Photos are grouped by the part of the name before the last underscore (RUGD: park-8_00001 is group
park-8), so a folder of photos from several places gives one row per place. Works on any dataset
converted to this layout, such as a new set of unfamiliar environments.
"""

import argparse
import csv
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import UNKNOWN, classify_terrain, load_config  # noqa: E402
from argo_perception.terrain.learned import TerrainModel  # noqa: E402
from argo_perception.terrain.metrics import N_PREDICTED, confusion, selective_scores  # noqa: E402

DEFAULT_THRESHOLDS = (0.0, 0.5, 0.7, 0.9)


def group_of(name: str) -> str:
    return name.rsplit("_", 1)[0] if "_" in name else name


def summarise(matrices: dict[float, np.ndarray]) -> list[dict]:
    """One row per threshold. matrices[0.0] (no UNKNOWN) is the forced-guess baseline."""
    baseline = matrices[min(matrices)]
    rows = []
    for threshold, matrix in sorted(matrices.items()):
        scores = selective_scores(matrix)
        # How often the removed pixels would have been wrong had the model been forced to guess.
        removed = matrix[:, UNKNOWN].sum()
        right_if_forced = np.trace(baseline[:, :3]) - np.trace(matrix[:, :3])
        scores["removed_were_wrong"] = float(1 - right_if_forced / removed) if removed else float("nan")
        scores["threshold"] = threshold
        rows.append(scores)
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'threshold':>9s} {'labelled':>9s} {'correct':>8s} {'removed were wrong':>19s} {'objects missed':>15s} {'objects unknown':>16s}")
    for r in rows:
        print(
            f"  {r['threshold']:9.2f} {r['coverage']:9.1%} {r['accuracy']:8.1%} {"-" if r["removed_were_wrong"] != r["removed_were_wrong"] else format(r["removed_were_wrong"], ".1%"):>19s}"
            f" {r['missed_object']:15.1%} {r['object_unknown']:16.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "terrain" / "rugd")
    parser.add_argument("--split", default="val", help="name of the list in splits/ (default: val)")
    parser.add_argument("--model", type=Path, default=REPO / "models" / "terrain_rugd.joblib")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "terrain_classifier.yaml")
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--every", type=int, default=1, help="score every Nth photo (default: 1, all)")
    parser.add_argument("--combine", action="store_true", help="also mark anything the rules call an object as non-ground")
    parser.add_argument("--csv", type=Path, help="also write the per-group table to this file")
    args = parser.parse_args()

    thresholds = sorted(set(args.thresholds) | {0.0})  # 0 is needed as the baseline
    cfg = dataclasses.replace(load_config(args.config), min_confidence=0.0)
    model = TerrainModel.load(args.model)

    split_file = args.data / "splits" / f"{args.split}.txt"
    if not split_file.exists():
        print(f"Missing {split_file}", file=sys.stderr)
        return 1
    names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()][:: args.every]

    zeros = lambda: {t: np.zeros((3, N_PREDICTED), dtype=np.int64) for t in thresholds}  # noqa: E731
    total, by_group = zeros(), defaultdict(zeros)
    scored = skipped = 0
    for name in names:
        matches = sorted((args.data / "images").glob(f"{name}.*"))
        image = cv2.imread(str(matches[0])) if matches else None
        mask_path = args.data / "masks" / f"{name}.png"
        truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if image is None or truth is None or truth.shape != image.shape[:2]:
            print(f"{name}: photo or label image missing or mismatched, skipped", file=sys.stderr)
            skipped += 1
            continue
        result = classify_terrain(image, cfg, model, args.combine)
        group = group_of(name)
        for t in thresholds:
            predicted = np.where(result.confidence < t, UNKNOWN, result.labels).astype(np.uint8)
            matrix = confusion(truth, predicted, N_PREDICTED)
            total[t] += matrix
            by_group[group][t] += matrix
        scored += 1

    if not scored:
        print("No photos were scored.", file=sys.stderr)
        return 1

    print(f"Split '{args.split}': {scored} photos scored, {skipped} skipped, model {args.model.name}"
          f"{' + rules' if args.combine else ''}, every {args.every}")
    print("labelled = share of pixels given a real label; correct = of those, share right (flat/rocky/non-ground);")
    print("removed were wrong = of the pixels sent to unknown, share a forced guess would have got wrong;")
    print("objects missed = true non-ground pixels labelled flat or rocky (the dangerous error);")
    print("objects unknown = true non-ground pixels flagged unknown instead.")
    print_table("All photos", summarise(total))
    rows_by_group = {g: summarise(m) for g, m in sorted(by_group.items())}
    for g, rows in rows_by_group.items():
        print_table(f"Group {g}", rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "threshold", "labelled", "correct", "removed_were_wrong", "objects_missed", "objects_unknown"])
            for g, rows in [("ALL", summarise(total)), *rows_by_group.items()]:
                for r in rows:
                    writer.writerow([g, r["threshold"], f"{r['coverage']:.4f}", f"{r['accuracy']:.4f}",
                                     f"{r['removed_were_wrong']:.4f}", f"{r['missed_object']:.4f}", f"{r['object_unknown']:.4f}"])
        print(f"\nWrote {args.csv}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
