"""Convert RELLIS-3D camera frames and labels to flat / rocky / non-ground, ready for evaluate_confidence.py.

Download from https://github.com/unmannedlab/RELLIS-3D (Google Drive links; if it refuses with "too many users",
wait or download in a browser): "Full Images", "Full Image Annotations, ID Format" and "Image Split File".
Unpack each into its own empty folder, then:

  python scripts/prepare_rellis.py --frames <images folder> --annotations <annotations folder> --splits <split folder>

Result, inside --root:
  images/   links to the photos, named <sequence>_<frame>, e.g. 00000_frame000308
  masks/    label images (0 flat, 1 rocky, 2 non-ground, 255 not scored)
  splits/   train.txt, val.txt, test.txt (RELLIS's own split) and all.txt

RELLIS-3D: Jiang et al. 2020, arXiv 2011.12954. Licence CC BY-NC-SA 3.0: keep --root out of git.
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "argo_perception"))

from argo_perception.terrain.rellis import rellis_to_labels  # noqa: E402


def name_of(relative: str) -> str:
    """'00000/pylon_camera_node/frame000308-1581624683_550.jpg' -> '00000_frame000308'."""
    path = Path(relative)
    return f"{path.parts[0]}_{path.stem.split('-')[0]}"


def find_one(folder: Path, relative: str) -> Path | None:
    direct = folder / relative
    if direct.exists():
        return direct
    matches = list(folder.rglob(relative))  # allows an extra top-level folder such as Rellis-3D/
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO / "data" / "terrain" / "rellis")
    parser.add_argument("--frames", type=Path, required=True, help="folder holding the <sequence>/pylon_camera_node photos")
    parser.add_argument("--annotations", type=Path, required=True, help="folder holding <sequence>/pylon_camera_node_label_id")
    parser.add_argument("--splits", type=Path, required=True, help="folder holding train.lst, val.lst and test.lst")
    args = parser.parse_args()

    (args.root / "images").mkdir(parents=True, exist_ok=True)
    (args.root / "masks").mkdir(exist_ok=True)
    (args.root / "splits").mkdir(exist_ok=True)

    everything, unknown_pixels, missing = [], 0, 0
    for split in ("train", "val", "test"):
        names = []
        for line in (args.splits / f"{split}.lst").read_text().splitlines():
            if not line.strip():
                continue
            frame_rel, label_rel = line.split()[:2]
            name = name_of(frame_rel)
            frame, label = find_one(args.frames, frame_rel), find_one(args.annotations, label_rel)
            if frame is None or label is None:
                missing += 1
                continue
            ids = cv2.imread(str(label), cv2.IMREAD_UNCHANGED)
            if ids is None or ids.ndim != 2:
                missing += 1
                continue
            mask, unknown = rellis_to_labels(ids)
            unknown_pixels += unknown
            cv2.imwrite(str(args.root / "masks" / f"{name}.png"), mask)
            link = args.root / "images" / f"{name}{frame.suffix}"
            if not link.exists():
                link.symlink_to(frame.resolve())
            names.append(name)
        (args.root / "splits" / f"{split}.txt").write_text("\n".join(names) + "\n")
        everything += names
        print(f"{split}: {len(names)} photos")
    (args.root / "splits" / "all.txt").write_text("\n".join(everything) + "\n")

    print(f"Converted {len(everything)} photos into {args.root}; {missing} listed photos missing or unreadable")
    if unknown_pixels:
        print(f"Warning: {unknown_pixels} pixels had an id that is not a RELLIS class; they are not scored", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
