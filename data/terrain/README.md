# Terrain test photos

Photos and hand labels used to tune and score the terrain classifier.
No photos have been collected yet. This file says what is needed and how to label it.

## Folder layout

```text
data/terrain/
  images/   the photos (.jpg or .png)
  json/     LabelMe outline files, one per photo
  masks/    label images made from the outlines, one .png per photo
  splits/
    tune.txt   photo names used for adjusting settings (about 30)
    test.txt   photo names held back for the final score (about 20)
```

Names in `tune.txt` and `test.txt` are one per line, without the file extension.
Decide the split once, before tuning, and do not move photos between the lists afterwards.
A score on photos that were also used for tuning is not a fair score.

## What to collect

About 50 photos, each showing ground and at least one object:

- Half taken from rover height (camera 20 to 40 cm above the ground, looking forward).
- Half taken looking straight down.

Include these hard cases on purpose:

- hard shadows across flat ground
- grass
- gravel
- one large rock on flat ground
- a photo that is entirely flat ground
- a photo that is entirely rocky ground

Record where every photo came from in the table at the bottom of this file.
Check the licence before adding a photo that someone else took.

## The three labels

| Label name | Use it for |
|---|---|
| `flat` | Ground the rover could drive over at full speed: packed dirt, pavement, short grass. |
| `rocky` | Ground that is still ground but rough: gravel, loose stones, rubble, broken pavement. |
| `non_ground` | Anything that is not ground: objects, walls, plants taller than the rover's clearance, people, sky. |

Rules for unclear cases:

- A single rock too big to drive over is `non_ground`. Many small rocks are `rocky`.
- A shadow is labelled as whatever it falls on.
- Leave an area with no outline if you cannot decide. Unlabelled areas are not scored.
- Outline the object itself, not its shadow.

## How to label

1. Install LabelMe: `pip install labelme`, then run `labelme data/terrain/images --output data/terrain/json`.
2. Draw a polygon around each region and type one of the three label names.
3. Convert the outlines to label images:

   ```bash
   python scripts/labelme_to_mask.py data/terrain/json --out data/terrain/masks
   ```

4. Score the classifier:

   ```bash
   python scripts/evaluate_terrain.py --split tune
   ```

## Public dataset: RUGD

RUGD is a public set of about 7,400 labelled photos taken from a small ground robot on trails, parks and a village (http://rugd.vision, Wigness et al., IROS 2019). It gives labelled data without any hand labelling, but it is not our course, so our own photos remain the final test.

Download and convert it (about 5.4 GB, into `data/terrain/rugd/`, which git ignores):

```bash
python scripts/prepare_rugd.py --download
python scripts/evaluate_terrain.py --data data/terrain/rugd --split tune --ground-prior bottom
```

Train the model on it and score the result:

```bash
python scripts/train_terrain.py
python scripts/evaluate_terrain.py --data data/terrain/rugd --split val --model models/terrain_rugd.joblib
```

Use the validation list while making choices. Score the test list only once, at the end.

RUGD labels 24 kinds of thing. They are converted to our three labels like this:

| Our label | RUGD classes |
|---|---|
| `flat` | dirt, sand, grass, asphalt, concrete, mulch |
| `rocky` | gravel, rock-bed |
| `non_ground` | tree, bush, log, rock, water, sky, pole, fence, sign, building, vehicle, bicycle, person, picnic-table, container/generic-object |
| not scored | void, bridge |

Two things to keep in mind:

- RUGD says what the ground is made of, not how rough it is. Smooth packed gravel still counts as `rocky` here.
- The photos are split by whole video, because neighbouring frames of one video look almost the same. `creek`, `park-1`, `trail-7` and `trail-13` are the test list. `park-8` and `trail-5` are the validation list. The other 12 videos are the training list. `tune.txt` is training and validation together.

Do not copy RUGD photos into the repository. Cite the RUGD paper in anything published.

## Storage

Keep the photos in this folder if they total a few megabytes.
If they are larger, store them outside the repository and write the location here.

## Photo sources

| Photo | Taken by / source | Licence | View (rover / top-down) |
|---|---|---|---|
| | | | |
