# Design Decisions

## Terrain classifier uses classical OpenCV on a single photo

**Decision.** The first terrain classifier (`src/argo_perception/argo_perception/terrain/`) labels each part of one colour photo as `flat`, `rocky` or `non_ground`, using OpenCV and NumPy only. It does not use a neural network or depth.

**Why.**
- No labelled photos exist yet, so there is nothing to train a network on.
- OpenCV is already in the planned stack and is what the existing route-finder prototype uses.
- Hand-set rules can be read, explained and adjusted by anyone on the team.

**How it decides.**
- Colour decides ground versus not ground. The most common colours in the photo (or, for rover's-eye photos, the colours along the bottom edge) are taken to be ground. A large solid patch of a different colour is `non_ground`.
- Texture decides flat versus rocky. Ground with a lot of fine brightness variation and many edges is `rocky`.
- Colour and texture are kept separate on purpose. If texture were also used to find the ground, a gravel patch the same colour as the dirt around it would be mistaken for an object.

**Known limits.**
- A single photo carries no height information. A large boulder is labelled `non_ground`, not `rocky`.
- Grass and hard shadows can be read as rough.
- Part of an object that matches the ground colour can be missed. Small gaps are filled (`object_gap_cells`), large ones are not.
- Roughness changes with camera distance, so settings tuned at one camera height may not suit another.
- With `ground_prior: none`, sky in a rover's-eye photo is mistaken for ground. Use `ground_prior: bottom` for those photos.

**Status of the settings.** The values in `config/terrain_classifier.yaml` were set by eye on a small number of web photos and have not been tuned.

**First measured score (2026-10-12).** On the 5,512 tuning photos of the public RUGD dataset, with `ground_prior: bottom` and the untuned settings (overlap is intersection-over-union, 1.00 is perfect):

| Measure | Score | Target |
|---|---|---|
| Ground overlap | 0.53 | 0.80 |
| Non-ground overlap | 0.28 | 0.80 |
| Flat vs rocky, share of ground pixels correct | 38% | 75% |

The hand-set rules miss the targets by a wide margin on this data. Two causes account for most of it: trees and bushes are close to the ground in colour in woodland photos, so they are not found as non-ground; and grass and leaf litter are visually busy, so they are called rocky while RUGD counts them as flat. See `data/terrain/README.md` for how to reproduce this.

**What would replace this.** Depth from the stereo cameras listed in `docs/interfaces.md` would remove the first three limits.

## Terrain classifier: a trained model alongside the hand-set rules

**Decision.** A trained model (`models/terrain_rugd.joblib`) is available as an option (`--model`). The hand-set rules stay the default. The model is gradient-boosted decision trees from scikit-learn, trained on 26 measurements per cell from the RUGD training videos.

**Why a trained model.** The rules scored far below target on RUGD. Trees and bushes are the same colour as woodland ground, and grass is as visually busy as gravel, so no single threshold separates them.

**Why scikit-learn and not OpenCV's own trees.** OpenCV 5 no longer ships its machine-learning module, so `cv2.ml` is not available with the current `opencv-python`. scikit-learn is the one added dependency.

**Why gradient-boosted trees.** Compared with a random forest on the validation videos, it scored slightly higher and its saved file is 0.3 MB instead of 8.7 MB. It is slower, about 25 ms per photo instead of about 18 ms. The hand-set rules take about 13 ms.

**Scores on RUGD (2026-10-12).** Overlap is intersection-over-union; 1.00 is perfect. The model was trained on 12 videos. Validation is 2 other videos, test is 4 more. The test photos were scored once, after all choices were made.

| Measure | Target | Rules, test | Model, validation | Model, test |
|---|---|---|---|---|
| Ground overlap | 0.80 | 0.47 | 0.91 | 0.796 |
| Non-ground overlap | 0.80 | 0.31 | 0.92 | 0.85 |
| Flat vs rocky, share of ground pixels correct | 75% | 53% | 90% | 81% |
| Rocky overlap | none set | 0.17 | 0.62 | 0.44 |

**Known limits of the model.**
- It is for rover's-eye photos only. One of its measurements is how high in the photo a cell sits, which was by far the most used measurement in the random-forest comparison, so it will not work on top-down photos.
- It only knows what RUGD showed it: trails, parks, a creek and a village. On web photos of streets it labelled distant road as non-ground and missed a traffic cone that the hand-set rules found. An object unlike anything in RUGD can be labelled as ground.
- Rocky is its weakest label. On the test videos about half of the true rocky pixels were labelled flat.
- The test score is lower than the validation score, so expect a further drop on terrain that differs more from RUGD, including our own course.
- The model file is a Python pickle tied to the scikit-learn version that wrote it. Only load model files from a trusted source, and retrain with `scripts/train_terrain.py` after upgrading scikit-learn.

**Next candidates.** Add our own labelled course photos to the training data; train a second model without the height measurement for top-down photos.

## Terrain classifier: model and rules combined

**Decision.** `--combine` (with `--model`) lets the model decide, then also marks as `non_ground` anything the hand-set rules call an object. It is an option, not the default.

**Why.** The model can label an object unlike anything in RUGD as ground. The rules look for anything that differs in colour from the ground, so they do not depend on having seen the object before. On the web photo with traffic cones, the model alone labelled most of the nearest cone as ground and the combination labelled it `non_ground`.

**What it costs (RUGD validation videos, 2026-10-12, `ground_prior: bottom`).**

| Measure | Model alone | Combined |
|---|---|---|
| Ground overlap | 0.91 | 0.88 |
| Non-ground overlap | 0.92 | 0.90 |
| Flat vs rocky, share of ground pixels correct | 90% | 86% |
| True non-ground pixels labelled as ground | 4.0% | 3.5% |
| True ground pixels labelled as non-ground | 4.8% | 8.9% |
| Time per photo | 23 ms | 36 ms |

On RUGD the combination scores lower: it nearly doubles the ground wrongly blocked and removes only about an eighth of the missed non-ground. Most of the extra blocking is grass and other ground that differs in colour from the bottom of the photo. The benefit is on objects RUGD does not contain, and that has been seen on one photo, not measured. The combination has not been scored on the RUGD test videos.

**When to use it.** Where driving into an unfamiliar object is worse than taking a detour. Measure it on our own course photos before making it the default.
