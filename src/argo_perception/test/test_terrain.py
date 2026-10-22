import dataclasses
from pathlib import Path

import numpy as np
import pytest

from argo_perception.terrain import (
    FLAT,
    IGNORE,
    NON_GROUND,
    ROCKY,
    TerrainConfig,
    classify_terrain,
    load_config,
    overlay,
    to_occupancy_grid,
)
from argo_perception.terrain.features import cell_features, prepare
from argo_perception.terrain.metrics import class_iou, confusion, flat_rocky_accuracy, ground_iou

REPO = Path(__file__).resolve().parents[3]


def share(labels, label):
    return float((labels == label).mean())


# --- measurements ---


def test_rocky_cells_measure_rougher_than_flat_cells(mixed_scene):
    image, _ = mixed_scene
    cfg = TerrainConfig()
    _, roughness = cell_features(prepare(image, cfg), cfg)
    half = roughness.shape[1] // 2
    assert np.median(roughness[:, half + 2 :]) > 5 * np.median(roughness[:, : half - 2])


def test_prepare_gives_a_whole_number_of_cells():
    cfg = TerrainConfig()
    work = prepare(np.zeros((517, 933, 3), dtype=np.uint8), cfg)
    assert work.shape[1] == cfg.work_width
    assert work.shape[0] % cfg.cell_size == 0


# --- classification ---


def test_mixed_scene_matches_truth(mixed_scene):
    image, truth = mixed_scene
    result = classify_terrain(image)
    assert result.labels.shape == truth.shape
    scores = class_iou(confusion(truth, result.labels))
    assert scores[FLAT] > 0.9
    assert scores[ROCKY] > 0.9
    assert scores[NON_GROUND] > 0.9


def test_all_flat_photo_has_no_rocky_or_object(flat_scene):
    labels = classify_terrain(flat_scene).labels
    assert share(labels, FLAT) == 1.0


def test_all_rocky_photo_is_rocky_with_no_object(rocky_scene):
    labels = classify_terrain(rocky_scene).labels
    assert share(labels, ROCKY) > 0.95
    assert share(labels, NON_GROUND) == 0.0


def test_bottom_prior_marks_sky_as_non_ground(horizon_scene):
    labels = classify_terrain(horizon_scene, TerrainConfig(ground_prior="bottom")).labels
    horizon = int(horizon_scene.shape[0] * 0.6)
    assert share(labels[: horizon - 16], NON_GROUND) == 1.0
    assert share(labels[horizon + 16 :], FLAT) == 1.0


def test_small_off_colour_speck_is_not_an_object(flat_scene):
    image = flat_scene.copy()
    image[200:216, 300:316] = (200, 60, 30)
    assert share(classify_terrain(image).labels, NON_GROUND) == 0.0


def test_ground_coloured_band_across_an_object_is_filled(flat_scene):
    image = flat_scene.copy()
    image[100:340, 240:400] = (200, 60, 30)
    band = slice(208, 240)  # two whole rows of 16 px cells
    image[band, 240:400] = flat_scene[band, 240:400]

    filled = classify_terrain(image).labels
    assert share(filled[band, 256:384], NON_GROUND) == 1.0

    unfilled = classify_terrain(image, TerrainConfig(object_gap_cells=0)).labels
    assert share(unfilled[band, 256:384], NON_GROUND) == 0.0


def test_output_matches_input_size_for_odd_shapes(flat_scene):
    image = flat_scene[:301, :417]
    assert classify_terrain(image).labels.shape == (301, 417)


def test_same_photo_gives_same_labels(mixed_scene):
    image, _ = mixed_scene
    assert np.array_equal(classify_terrain(image).labels, classify_terrain(image).labels)


def test_overlay_keeps_image_shape(mixed_scene):
    image, truth = mixed_scene
    assert overlay(image, truth).shape == image.shape


# --- settings ---


def test_yaml_file_matches_built_in_defaults():
    assert load_config(REPO / "config" / "terrain_classifier.yaml") == TerrainConfig()


def test_unknown_ground_prior_is_rejected():
    with pytest.raises(ValueError):
        TerrainConfig(ground_prior="top")


def test_unknown_setting_in_yaml_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("rocky_treshold: 0.4\n")
    with pytest.raises(TypeError):
        load_config(path)


# --- route-finder grid ---


def test_grid_shape_and_blocked_cells():
    labels = np.full((100, 150), FLAT, dtype=np.uint8)
    labels[0:25, 0:25] = NON_GROUND
    labels[50:75, 50:75] = ROCKY
    labels[75:100, 100:109] = NON_GROUND  # 36% of one cell: below the 40% cut-off

    grid = to_occupancy_grid(labels, cell_size=25)
    assert grid.shape == (4, 6)
    assert grid[0, 0] == 1
    assert grid.sum() == 1

    with_rocky = to_occupancy_grid(labels, cell_size=25, rocky_blocked=True)
    assert with_rocky[2, 2] == 1
    assert with_rocky.sum() == 2


def test_grid_ignores_leftover_pixels_at_the_edges():
    labels = np.full((60, 60), NON_GROUND, dtype=np.uint8)
    assert to_occupancy_grid(labels, cell_size=25).shape == (2, 2)


# --- scoring ---


def test_perfect_prediction_scores_one():
    truth = np.array([[FLAT, ROCKY], [NON_GROUND, NON_GROUND]], dtype=np.uint8)
    matrix = confusion(truth, truth.copy())
    assert np.allclose(class_iou(matrix), 1.0)
    assert ground_iou(matrix) == (1.0, 1.0)
    assert flat_rocky_accuracy(matrix) == 1.0


def test_scores_for_a_known_mistake():
    truth = np.array([FLAT, FLAT, ROCKY, ROCKY, NON_GROUND, NON_GROUND], dtype=np.uint8)
    predicted = np.array([FLAT, ROCKY, ROCKY, ROCKY, NON_GROUND, FLAT], dtype=np.uint8)
    matrix = confusion(truth, predicted)
    assert np.allclose(class_iou(matrix), [1 / 3, 2 / 3, 1 / 2])
    assert ground_iou(matrix) == (4 / 5, 1 / 2)
    assert flat_rocky_accuracy(matrix) == 3 / 4


def test_ignored_pixels_are_not_scored():
    truth = np.array([FLAT, IGNORE, IGNORE], dtype=np.uint8)
    predicted = np.array([FLAT, ROCKY, NON_GROUND], dtype=np.uint8)
    assert confusion(truth, predicted).sum() == 1


def test_label_missing_from_photo_scores_nan():
    truth = np.array([FLAT, FLAT], dtype=np.uint8)
    scores = class_iou(confusion(truth, truth.copy()))
    assert scores[FLAT] == 1.0
    assert np.isnan(scores[ROCKY]) and np.isnan(scores[NON_GROUND])


def test_confusion_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        confusion(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 3), dtype=np.uint8))


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TerrainConfig().cell_size = 8


# --- RUGD label conversion ---


def test_rugd_colours_map_to_our_labels():
    from argo_perception.terrain.rugd import RUGD_CLASSES, rugd_to_labels

    names = ["dirt", "gravel", "rock-bed", "rock", "sky", "void", "bridge"]
    annotation = np.array([[RUGD_CLASSES[name][0][::-1] for name in names]], dtype=np.uint8)  # RGB -> BGR
    labels, unknown = rugd_to_labels(annotation)
    assert labels.tolist() == [[FLAT, ROCKY, ROCKY, NON_GROUND, NON_GROUND, IGNORE, IGNORE]]
    assert unknown == 0


def test_rugd_unknown_colour_is_counted_and_not_scored():
    from argo_perception.terrain.rugd import rugd_to_labels

    labels, unknown = rugd_to_labels(np.array([[[1, 2, 3]]], dtype=np.uint8))
    assert labels.tolist() == [[IGNORE]]
    assert unknown == 1


def test_every_rugd_class_has_a_distinct_colour():
    from argo_perception.terrain.rugd import RUGD_CLASSES

    colours = [colour for colour, _ in RUGD_CLASSES.values()]
    assert len(RUGD_CLASSES) == 25
    assert len(set(colours)) == 25


# --- trained model ---


def _train_on(image, truth):
    from argo_perception.terrain.features import learned_features
    from argo_perception.terrain.learned import TerrainModel, cell_truth

    cfg = TerrainConfig()
    features = learned_features(prepare(image, cfg), cfg)
    labels = cell_truth(truth, *features.shape[:2]).ravel()
    keep = labels != IGNORE
    return TerrainModel.train(features.reshape(-1, features.shape[2])[keep], labels[keep], rounds=30)


def test_cell_truth_takes_the_majority_and_skips_mixed_cells():
    from argo_perception.terrain.learned import cell_truth

    mask = np.full((32, 48), FLAT, dtype=np.uint8)
    mask[:, 16:32] = ROCKY
    mask[:4, 32:] = NON_GROUND  # a quarter of the top-right cell
    mask[16:, 32:40] = NON_GROUND  # half of the bottom-right cell
    truth = cell_truth(mask, rows=2, cols=3)
    assert truth.tolist() == [[FLAT, ROCKY, FLAT], [FLAT, ROCKY, IGNORE]]


def test_feature_names_match_feature_count(flat_scene):
    from argo_perception.terrain.features import FEATURE_NAMES, learned_features

    cfg = TerrainConfig()
    features = learned_features(prepare(flat_scene, cfg), cfg)
    assert features.shape[2] == len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_trained_model_learns_the_scene_it_was_shown(mixed_scene):
    image, truth = mixed_scene
    result = classify_terrain(image, model=_train_on(image, truth))
    assert result.labels.shape == truth.shape
    assert np.all(class_iou(confusion(truth, result.labels)) > 0.9)
    assert result.roughness is None


def test_saved_model_gives_the_same_labels_after_loading(mixed_scene, tmp_path):
    from argo_perception.terrain.learned import TerrainModel

    image, truth = mixed_scene
    model = _train_on(image, truth)
    path = tmp_path / "models" / "model.joblib"
    model.save(path)
    reloaded = TerrainModel.load(path)
    assert np.array_equal(classify_terrain(image, model=model).labels, classify_terrain(image, model=reloaded).labels)


def test_model_trained_with_other_measurements_is_rejected(tmp_path):
    import joblib

    from argo_perception.terrain.learned import TerrainModel

    path = tmp_path / "old.joblib"
    joblib.dump({"trees": None, "features": ["something_else"]}, path)
    with pytest.raises(ValueError):
        TerrainModel.load(path)


# --- model and rules combined ---


class _SeesOnlyFlat:
    def predict_cells(self, features):
        return np.full(features.shape[:2], FLAT, dtype=np.uint8)


def test_combine_adds_objects_the_model_missed(mixed_scene):
    image, _ = mixed_scene
    rules = classify_terrain(image).labels
    combined = classify_terrain(image, model=_SeesOnlyFlat(), combine=True).labels
    assert (rules == NON_GROUND).any()
    assert np.array_equal(combined == NON_GROUND, rules == NON_GROUND)
    assert np.all(combined[rules != NON_GROUND] == FLAT)


def test_combine_without_a_model_is_rejected(flat_scene):
    with pytest.raises(ValueError):
        classify_terrain(flat_scene, combine=True)
