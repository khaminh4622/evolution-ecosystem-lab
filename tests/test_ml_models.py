"""Tests for the ML models: that they learn, and that the scores are honest.

These tests use a shared, moderately sized dataset. A tiny dataset makes the
held-out score noisy enough to be meaningless: an early version of this work
scored R2 = -15 on 150 scenarios purely from an unlucky split, and the same code
scored +0.3 on 250. The fixtures below are sized to keep the assertions stable.
"""

from pathlib import Path

import pandas as pd
import pytest

from ecosystem_evolution.ml.datasets import (
    assert_no_leakage,
    build_dataset,
    feature_frame,
    grouped_split,
)
from ecosystem_evolution.ml.models import (
    CAPACITY_LABEL,
    CLASS_LABEL,
    POPULATION_LABEL,
    check_against_formula,
    load_model,
    save_models,
    train_all,
    train_population_regressor,
    train_suitability_classifier,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_dataset(scenarios=250, generations=10, seed=99)


def test_classifier_beats_guessing_the_majority_class(frame: pd.DataFrame) -> None:
    _, score = train_suitability_classifier(frame)

    train, _ = grouped_split(frame)
    majority = train[CLASS_LABEL].value_counts(normalize=True).max()
    assert score.score > majority, (
        f"accuracy {score.score:.3f} does not beat the majority class {majority:.3f}"
    )


def test_classifier_reports_its_own_held_out_scenarios(frame: pd.DataFrame) -> None:
    _, score = train_suitability_classifier(frame)

    assert score.train_scenarios + score.test_scenarios == frame["scenario_id"].nunique()
    assert score.test_rows > 0
    assert set(score.extra) == {"macro_f1"}


def test_regressor_beats_predicting_the_mean(frame: pd.DataFrame) -> None:
    _, score = train_population_regressor(frame)

    # R2 above zero means better than always guessing the training mean.
    assert score.score > 0.0, f"R2 was {score.score:.3f}, no better than the mean"
    assert score.extra["baseline_r2"] < 0.1


def test_regressor_error_is_smaller_than_the_label_spread(frame: pd.DataFrame) -> None:
    _, score = train_population_regressor(frame)

    spread = frame[POPULATION_LABEL].std()
    assert score.extra["rmse"] < spread, (
        f"rmse {score.extra['rmse']:.2f} is no better than the label spread {spread:.2f}"
    )


def test_capacity_model_approaches_the_exact_formula(frame: pd.DataFrame) -> None:
    _, score = check_against_formula(frame)

    # The simulator computes capacity in closed form, so a learned model should
    # track it closely; a poor score here would indicate a pipeline problem.
    assert score.score > 0.5, f"R2 was {score.score:.3f}"
    assert score.extra["exact_formula_mae"] == 0.0


def test_models_are_reproducible_for_a_fixed_seed(frame: pd.DataFrame) -> None:
    first, first_score = train_suitability_classifier(frame, random_state=0)
    second, second_score = train_suitability_classifier(frame, random_state=0)

    assert first_score.score == pytest.approx(second_score.score)
    train, test = grouped_split(frame)
    assert (first.predict(feature_frame(test)) == second.predict(feature_frame(test))).all()


def test_a_different_random_state_still_learns(frame: pd.DataFrame) -> None:
    _, score = train_suitability_classifier(frame, random_state=13)
    train, _ = grouped_split(frame)
    majority = train[CLASS_LABEL].value_counts(normalize=True).max()
    assert score.score > majority


def test_train_all_returns_every_task(frame: pd.DataFrame) -> None:
    models, scores = train_all(frame)

    assert set(models) == {
        "suitability_classifier",
        "population_regressor",
        "extinction_classifier",
        "capacity_check",
    }
    assert {score.task for score in scores} == set(models)
    for score in scores:
        assert 0.0 <= score.score <= 1.0


def test_models_can_be_saved_and_reloaded(frame: pd.DataFrame) -> None:
    import os

    models, _ = train_all(frame, n_jobs=1)
    _, test = grouped_split(frame)
    features = feature_frame(test)

    # Model artifacts are build output, so they are written outside the project
    # tree: some restricted environments mount it read-only.
    base = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "."
    directory = os.path.join(base, "evolution-ml-artifacts")
    try:
        written = save_models(models, directory)

        assert len(written) == 4
        before = models["suitability_classifier"].predict(features)
        reloaded = load_model(os.path.join(directory, "suitability_classifier.joblib"))
        after = reloaded.predict(features)

        # Preprocessing must travel with the model, so predictions match exactly.
        assert (before == after).all()
    finally:
        for name in (
            "suitability_classifier",
            "population_regressor",
            "extinction_classifier",
            "capacity_check",
        ):
            try:
                os.unlink(os.path.join(directory, f"{name}.joblib"))
            except OSError:
                pass


def test_predictions_are_never_impossible(frame: pd.DataFrame) -> None:
    model, _ = train_population_regressor(frame)
    _, test = grouped_split(frame)

    predicted = model.predict(feature_frame(test))

    assert (predicted >= 0).all(), "a population cannot be negative"


def test_grouped_split_is_what_makes_the_score_honest(frame: pd.DataFrame) -> None:
    """A row-wise split inflates the score, which is why grouping is required.

    The dataset has one row per scenario, so overlap needs several rows per
    scenario to demonstrate. This test builds such a frame on purpose: the
    multi-generation rows of one run share an environment, which is exactly the
    situation that makes a row-wise split leak.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline

    from ecosystem_evolution.ml.models import _preprocessor

    # Rows from several generations of the same scenarios.
    repeated = pd.concat([frame, frame, frame], ignore_index=True)
    repeated["row_id"] = range(len(repeated))

    # Honest: whole scenarios held out.
    honest_train, honest_test = grouped_split(repeated, test_fraction=0.25, seed=7)
    assert_no_leakage(honest_train, honest_test)
    honest_model = Pipeline(
        [
            ("prepare", _preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=0)),
        ]
    )
    honest_model.fit(feature_frame(honest_train), honest_train[CLASS_LABEL])
    honest_score = accuracy_score(
        honest_test[CLASS_LABEL], honest_model.predict(feature_frame(honest_test))
    )

    # Dishonest: rows shuffled, so the same scenario lands on both sides.
    shuffled = repeated.sample(frac=1.0, random_state=1).reset_index(drop=True)
    cut = int(len(shuffled) * 0.75)
    leaky_train, leaky_test = shuffled.iloc[:cut], shuffled.iloc[cut:]
    assert set(leaky_train["scenario_id"]) & set(leaky_test["scenario_id"])

    leaky_model = Pipeline(
        [
            ("prepare", _preprocessor()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=0)),
        ]
    )
    leaky_model.fit(feature_frame(leaky_train), leaky_train[CLASS_LABEL])
    leaky_score = accuracy_score(
        leaky_test[CLASS_LABEL], leaky_model.predict(feature_frame(leaky_test))
    )

    # The leaked split reports a better score on the same underlying task.
    assert leaky_score > honest_score, (
        f"expected the leaky split ({leaky_score:.3f}) to flatter the model "
        f"more than the grouped split ({honest_score:.3f})"
    )
