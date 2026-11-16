# argo_perception

Perception code for A.R.G.O. Currently one feature: the terrain classifier.

## Terrain classifier

Labels every part of a photo as one of:

| Value | Name | Meaning |
|---|---|---|
| 0 | `flat` | smooth ground |
| 1 | `rocky` | rough ground |
| 2 | `non_ground` | an object, or anything that is not ground |
| 3 | `unknown` | the trained model is not confident enough to say (trained model only, see below) |

### Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run it on photos

```bash
.venv/bin/python scripts/classify_terrain.py path/to/photo_or_folder --out out/
```

For each photo this writes `<name>_overlay.jpg` (the photo tinted green / orange / red) and `<name>_labels.png` (the label values).

For photos taken from the rover looking forward, add `--ground-prior bottom`. Without it, sky is mistaken for ground.

### Use the trained model

By default the classifier uses hand-set rules. A model trained on the public RUGD dataset scores much higher on trail and park photos taken from rover height:

```bash
.venv/bin/python scripts/classify_terrain.py path/to/photos --out out/ --model models/terrain_rugd.joblib
```

The model is for rover's-eye photos only, and it can miss objects unlike anything in RUGD. Read the limits in `docs/decisions.md` before relying on it.

Add `--combine --ground-prior bottom` to also mark anything the hand-set rules call an object as non-ground. This catches objects the model has never seen, such as a traffic cone, but blocks more real ground. The numbers are in `docs/decisions.md`.

To retrain it (needs the RUGD download, see `data/terrain/README.md`):

```bash
.venv/bin/python scripts/train_terrain.py
.venv/bin/python scripts/evaluate_terrain.py --data data/terrain/rugd --split val --model models/terrain_rugd.joblib
```

### Use it from Python

Run your script with `PYTHONPATH=src/argo_perception` so Python can find the package.

```python
import cv2
from argo_perception.terrain import classify_terrain, to_occupancy_grid

image = cv2.imread("photo.jpg")
result = classify_terrain(image)  # hand-set rules

# or, with the trained model:
# from argo_perception.terrain.learned import TerrainModel
# result = classify_terrain(image, model=TerrainModel.load("models/terrain_rugd.joblib"))
# add combine=True to also block anything the rules call an object
grid = to_occupancy_grid(result.labels, cell_size=25, rocky_blocked=True)
```

`grid` holds 1 for blocked cells and 0 for free cells, indexed `[row, col]`. It can be passed directly to a grid route planner such as `a_star_search` on the `candidate_map` branch.

### Confidence and unknown

With `--model`, every cell has a confidence from 0 to 1: the model's chance for the label it gave. With the trained model, cells whose confidence is under `min_confidence` are labelled `unknown` instead of being guessed. The default is 0.7; change it in the config file or with `--min-confidence`, and 0 turns it off. Each photo also gets `<name>_confidence.jpg` (dark purple is a guess, yellow is certain, one fixed scale for every photo) and `<name>_confidence.npy`.

```bash
.venv/bin/python scripts/classify_terrain.py photos/ --out out/ --model models/terrain_rugd.joblib --ground-prior bottom --min-confidence 0.7
```

In Python, `classify_terrain(image, cfg, model)` returns `result.confidence` (per pixel; `None` for the hand-set rules) and puts `UNKNOWN` in `result.labels` when a cell's confidence is under `cfg.min_confidence`. In the costmap an unknown area is 255 ("no information"); in `to_occupancy_grid` it counts as blocked.

To measure a threshold on labelled photos, in any folder laid out as in `data/terrain/README.md`:

```bash
.venv/bin/python scripts/evaluate_confidence.py --data path/to/data --split test --thresholds 0.5 0.7 0.9
```

It prints one table per group of photos (the name before the last underscore). Results for RUGD and for GOOSE, an unfamiliar dataset, are in `docs/decisions.md`. `scripts/prepare_goose.py` and `scripts/prepare_rellis.py` convert those datasets into the layout above.

### Top-down costmap

`build_costmap` projects the labels onto the ground in front of the rover and returns a grid in metres (10 cm cells, 4 m ahead and 2 m to each side by default). Costs follow the ROS nav2 convention:

| Cost | Meaning |
|---|---|
| 0 | flat |
| 60 | rocky |
| 1 to 252 | close to an obstacle; higher is closer |
| 253 | so close to an obstacle that the rover's centre must not go there |
| 254 | blocked (non-ground) |
| 255 | unknown: not visible in the photo, or mostly labelled `unknown` |

```python
from argo_perception.terrain import build_costmap, costmap_image, load_costmap_config

costmap = build_costmap(result.labels, load_costmap_config("config/costmap.yaml"))
costmap.costs                 # uint8 array: row 0 is the far edge, the last row is nearest the rover
costmap.cell_centre(39, 20)   # (metres ahead, metres to the left) of a cell
cv2.imwrite("costmap.png", costmap_image(costmap))
```

From the command line, add `--costmap` to `scripts/classify_terrain.py`. It also saves `<name>_costmap.png` and `<name>_costmap.npy`.

The camera height, tilt and field of view in `config/costmap.yaml` are placeholders. Distances are only right once they match the real camera. With wrong values the map is badly wrong, not slightly off: see `docs/decisions.md`.

### Settings

All thresholds are in `config/terrain_classifier.yaml`, with a comment on each.

### Tests

```bash
.venv/bin/python -m pytest
```

### Scoring against labelled photos

See `data/terrain/README.md`.

### Limits

See the terrain entry in `docs/decisions.md`.
