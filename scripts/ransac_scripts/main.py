"""Run the pipeline and save everything to outputs/{name}_{date}/.

    python main.py                                 -> outputs/2026-10-22_143005/
    python main.py --name test                     -> outputs/test_2026-10-22_143005/
    python main.py --scene synthetic --name synth  -> generated test scene
"""
import argparse
from datetime import datetime
from pathlib import Path

from config import CONFIG
from inputs import SCENES
from pipeline import run

OUTPUTS = Path(__file__).parent / "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="", help="optional label for this run's folder")
    parser.add_argument("--scene", default="sample", choices=SCENES, help="which input to run on")
    args = parser.parse_args()

    date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = OUTPUTS / (f"{args.name}_{date}" if args.name else date)

    points, truth = SCENES[args.scene]()
    summary = run(points, CONFIG, out_dir, truth)

    plane = summary["ground_plane"]
    print(f"Ground plane: z = {plane['a']:.3f}x + {plane['b']:.3f}y + {plane['c']:.3f}")
    print(f"{summary['num_ground']} ground points, {summary['num_obstacle_clusters']} obstacles, "
          f"{summary['num_stray']} stray points, {summary['occupied_cells']} occupied cells")
    if summary["scores"]:
        s = summary["scores"]
        print(f"Accuracy {s['accuracy']:.1%}: {s['ground_called_obstacle']} false alarms, "
              f"{s['obstacle_called_ground']} misses")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
