"""Download the RUGD dataset and convert its labels to flat / rocky / non-ground.

Result, inside the --root folder, ready for scripts/evaluate_terrain.py:
  images/   links to the RUGD photos
  masks/    label images (0 flat, 1 rocky, 2 non-ground, 255 not scored)
  splits/   test.txt, and tune.txt divided into train.txt and val.txt. Split by whole
            video so that near-identical neighbouring frames never land in two lists

RUGD: http://rugd.vision  (Wigness et al., IROS 2019). Cite the paper if you publish results.
The photos are not ours to redistribute; keep the --root folder out of git.
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain import NAMES  # noqa: E402
from argo_perception.terrain.rugd import rugd_to_labels  # noqa: E402

BASE_URL = "http://rugd.vision/data/"
FRAMES = "RUGD_frames-with-annotations"
ANNOTATIONS = "RUGD_annotations"
TEST_SEQUENCES = {"creek", "park-1", "trail-7", "trail-13"}
VAL_SEQUENCES = {"park-8", "trail-5"}


def download(name: str, folder: Path) -> Path:
    """Fetch <name>.zip into folder, continuing a partial download if one exists."""
    target = folder / f"{name}.zip"
    request = urllib.request.Request(BASE_URL + target.name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers["Content-Length"])
    have = target.stat().st_size if target.exists() else 0
    if have == total:
        return target
    if have > total:
        have = 0

    request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request) as response:
        resumed = response.status == 206
        if not resumed:
            have = 0
        with open(target, "ab" if resumed else "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                have += len(chunk)
                print(f"\r{target.name}: {have / 1e6:,.0f} of {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    if have != total:
        raise OSError(f"{target.name} is incomplete ({have} of {total} bytes); run again to continue")
    return target


def unpack(archive: Path, folder: Path) -> None:
    if (folder / archive.stem).is_dir():
        return
    print(f"Unpacking {archive.name} ...")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(folder)


def sequence_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO / "data" / "terrain" / "rugd")
    parser.add_argument("--download", action="store_true", help="fetch and unpack RUGD first (about 5.4 GB)")
    parser.add_argument("--frames", type=Path, help="folder of RUGD photos (default: the downloaded copy)")
    parser.add_argument("--annotations", type=Path, help="folder of RUGD label images (default: the downloaded copy)")
    parser.add_argument("--every", type=int, default=1, help="keep every Nth frame of each video (default: all)")
    args = parser.parse_args()

    downloads = args.root / "_download"
    if args.download:
        downloads.mkdir(parents=True, exist_ok=True)
        for name in (ANNOTATIONS, FRAMES):
            unpack(download(name, downloads), downloads)
    frames_dir = args.frames or downloads / FRAMES
    annotations_dir = args.annotations or downloads / ANNOTATIONS
    for folder in (frames_dir, annotations_dir):
        if not folder.is_dir():
            print(f"Missing {folder}. Run with --download, or pass --frames and --annotations.", file=sys.stderr)
            return 1

    frames = {path.stem: path for path in frames_dir.rglob("*.png")}
    annotations = sorted(annotations_dir.rglob("*.png"))
    by_sequence: dict[str, list[Path]] = {}
    for path in annotations:
        by_sequence.setdefault(sequence_of(path.stem), []).append(path)

    for name in ("images", "masks", "splits"):
        (args.root / name).mkdir(parents=True, exist_ok=True)

    split_names = {"train": [], "val": [], "test": []}
    pixel_counts = np.zeros(256, dtype=np.int64)
    no_photo = unknown_pixels = 0
    for sequence, paths in sorted(by_sequence.items()):
        split = "test" if sequence in TEST_SEQUENCES else "val" if sequence in VAL_SEQUENCES else "train"
        for path in paths[:: args.every]:
            frame = frames.get(path.stem)
            if frame is None:
                no_photo += 1
                continue
            mask, unknown = rugd_to_labels(cv2.imread(str(path), cv2.IMREAD_COLOR))
            unknown_pixels += unknown
            pixel_counts += np.bincount(mask.ravel(), minlength=256)
            cv2.imwrite(str(args.root / "masks" / f"{path.stem}.png"), mask)

            link = args.root / "images" / frame.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(frame.resolve())
            split_names[split].append(path.stem)

    split_names["tune"] = sorted(split_names["train"] + split_names["val"])
    for split, names in split_names.items():
        (args.root / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")

    converted = len(split_names["tune"]) + len(split_names["test"])
    if converted == 0:
        print("No photos were converted.", file=sys.stderr)
        return 1

    print(f"Converted {converted} photos from {len(by_sequence)} videos into {args.root}")
    print("  " + "    ".join(f"{split}: {len(split_names[split])} photos" for split in ("train", "val", "test")))
    scored = pixel_counts[: len(NAMES)].sum()
    for label, name in NAMES.items():
        print(f"  {name:11s} {pixel_counts[label] / scored:6.1%} of scored pixels")
    print(f"  not scored  {pixel_counts[255] / pixel_counts.sum():6.1%} of all pixels")
    if no_photo:
        print(f"Warning: {no_photo} label images had no matching photo and were skipped.", file=sys.stderr)
    if unknown_pixels:
        print(f"Warning: {unknown_pixels} pixels had a colour that is not a RUGD class; they are not scored.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
