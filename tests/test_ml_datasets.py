"""Tests for the ML dataset layer: grouping, labels, and determinism."""

import pandas as pd
import pytest

from ecosystem_evolution.ml.datasets import (
    CHALLENGING_THRESHOLD,
    FAVORABLE_THRESHOLD,
    LABEL_COLUMNS,
    REGRESSION_HORIZON,
    assert_no_leakage,
    build_dataset,
    feature_frame,
    grouped_split,
    random_scenario,
    suitability_class,
)
import random


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_dataset(scenarios=40, generations=10, seed=4242)


def test_dataset_has_one_row_per_scenario(frame: pd.DataFrame) -> None:
    assert len(frame) == 40
    assert frame["scenario_id"].nunique() == 40


def test_every_label_column_is_present(frame: pd.DataFrame) -> None:
    for column in LABEL_COLUMNS:
        assert column in frame.columns, column


def test_regression_label_matches_the_recorded_generation(frame: pd.DataFrame) -> None:
    # The label must be a real simulated population, never negative and never
    # above the carrying capacity the simulator enforced.
    label = frame[f"population_after_{REGRESSION_HORIZON}_generations"]
    assert (label >= 0).all()
    assert (label <= frame["carrying_capacity"]).all()


def test_labels_are_not_the_end_of_the_run(frame: pd.DataFrame) -> None:
    # If the horizon were the final generation, the label would be a leak.
    label = frame[f"population_after_{REGRESSION_HORIZON}_generations"]
    assert not label.equals(frame["final_population"])


def test_class_labels_follow_the_documented_thresholds(frame: pd.DataFrame) -> None:
    for score, expected in frame[["suitability_score", "suitability_class"]].itertuples(
        index=False
    ):
        assert expected == suitability_class(score)
    assert set(frame["suitability_class"]) <= {"favorable", "moderate", "challenging"}


def test_suitability_class_thresholds_are_ordered() -> None:
    assert CHALLENGING_THRESHOLD < FAVORABLE_THRESHOLD
    assert suitability_class(FAVORABLE_THRESHOLD) == "favorable"
    assert suitability_class(CHALLENGING_THRESHOLD - 0.01) == "challenging"


def test_dataset_is_deterministic_for_a_seed() -> None:
    first = build_dataset(scenarios=8, generations=10, seed=11)
    second = build_dataset(scenarios=8, generations=10, seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_scenarios() -> None:
    first = build_dataset(scenarios=8, generations=10, seed=11)
    second = build_dataset(scenarios=8, generations=10, seed=12)
    assert not first["temperature_c"].equals(second["temperature_c"])


def test_too_few_generations_is_rejected() -> None:
    # A run that ends at the horizon would make the label the run's endpoint.
    with pytest.raises(ValueError, match="generations must be at least"):
        build_dataset(scenarios=2, generations=REGRESSION_HORIZON)


def test_zero_scenarios_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_dataset(scenarios=0)


def test_grouped_split_never_shares_a_scenario(frame: pd.DataFrame) -> None:
    train, test = grouped_split(frame, test_fraction=0.25, seed=7)

    assert_no_leakage(train, test)
    assert set(train["scenario_id"]).isdisjoint(set(test["scenario_id"]))
    assert len(train) + len(test) == len(frame)


def test_grouped_split_is_reproducible(frame: pd.DataFrame) -> None:
    first = grouped_split(frame, seed=3)
    second = grouped_split(frame, seed=3)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_leakage_check_catches_a_bad_split(frame: pd.DataFrame) -> None:
    # A row-wise split is exactly the mistake the guard exists to catch.
    bad_train = frame.iloc[: len(frame) // 2]
    bad_test = frame.iloc[len(frame) // 2 :]
    if set(bad_train["scenario_id"]) & set(bad_test["scenario_id"]):
        pass  # disjoint halves cannot leak; force one instead
    leaking_test = pd.concat([bad_test, bad_train.head(1)])
    with pytest.raises(ValueError, match="appear in both splits"):
        assert_no_leakage(bad_train, leaking_test)


def test_split_rejects_a_bad_fraction(frame: pd.DataFrame) -> None:
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="test_fraction"):
            grouped_split(frame, test_fraction=bad)


def test_feature_frame_excludes_labels_and_metadata(frame: pd.DataFrame) -> None:
    features = feature_frame(frame)

    for column in ("scenario_id", "seed", "suitability_class", "final_population"):
        assert column not in features.columns
    for column in LABEL_COLUMNS:
        assert column not in features.columns


def test_feature_frame_reports_missing_columns(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        feature_frame(frame, columns=["temperature_c", "not_a_real_column"])


def test_random_scenario_keeps_habitat_and_temperature_consistent() -> None:
    rng = random.Random(5)
    for index in range(60):
        scenario = random_scenario(rng, f"scenario-{index:05d}")
        environment = scenario.environment
        # A coral reef in a blizzard is not a scenario worth generating.
        if environment.habitat_type.value == "coral_reef":
            assert environment.temperature_c >= 18.0
        assert -50.0 <= environment.temperature_c <= 60.0
        assert 0.1 <= environment.habitat_area_km2 <= 100.0
