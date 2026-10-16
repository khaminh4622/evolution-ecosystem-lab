"""Fit and evaluate the teaching models, then check them against the simulator.

Three tasks are trained on simulated scenarios:

``suitability_classifier``
    Predicts the documented ``favorable`` / ``moderate`` / ``challenging`` class
    from the environment. A fast pre-screen for a classroom UI.

``population_regressor``
    Predicts the population five generations ahead. This replaces a
    sixty-seed simulation sweep with one instant estimate.

``capacity_check``
    Predicts carrying capacity and compares it with the closed-form equation the
    simulator already uses. This is the honest test of what the ML is worth: if
    a model merely re-derives a formula that is already known exactly, its value
    is in the pipeline, not in new knowledge about foxes.

Every model is fitted on a scenario-grouped split, so the reported scores come
from scenarios the model has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .datasets import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REGRESSION_HORIZON,
    assert_no_leakage,
    feature_frame,
    grouped_split,
)

CLASS_LABEL = "suitability_class"
POPULATION_LABEL = f"population_after_{REGRESSION_HORIZON}_generations"
CAPACITY_LABEL = "carrying_capacity"


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A model's held-out score, with the counts needed to judge it."""

    task: str
    metric: str
    score: float
    train_rows: int
    test_rows: int
    train_scenarios: int
    test_scenarios: int
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "score": round(self.score, 4),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_scenarios": self.train_scenarios,
            "test_scenarios": self.test_scenarios,
            **{name: round(value, 4) for name, value in self.extra.items()},
        }

    def describe(self) -> str:
        extra = ", ".join(f"{key}={value:.3f}" for key, value in self.extra.items())
        return (
            f"{self.task}: {self.metric}={self.score:.3f} "
            f"({self.test_scenarios} held-out scenarios, {self.test_rows} rows"
            + (f", {extra}" if extra else "")
            + ")"
        )


def _preprocessor() -> ColumnTransformer:
    """One-hot encode the two categorical controls, pass numbers through."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )


def train_suitability_classifier(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[Pipeline, Evaluation]:
    """Classify a scenario as favorable, moderate, or challenging.

    ``n_jobs`` defaults to 1. Random forests are fast enough single-threaded at
    these dataset sizes, and single-threaded fitting avoids spawning worker
    processes, which some restricted environments block.
    """
    train, test = grouped_split(frame, test_fraction=test_fraction, seed=seed)
    assert_no_leakage(train, test)

    model = Pipeline(
        steps=[
            ("prepare", _preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    model.fit(feature_frame(train), train[CLASS_LABEL])
    predicted = model.predict(feature_frame(test))

    evaluation = Evaluation(
        task="suitability_classifier",
        metric="accuracy",
        score=float(accuracy_score(test[CLASS_LABEL], predicted)),
        train_rows=len(train),
        test_rows=len(test),
        train_scenarios=int(train["scenario_id"].nunique()),
        test_scenarios=int(test["scenario_id"].nunique()),
        extra={"macro_f1": float(f1_score(test[CLASS_LABEL], predicted, average="macro"))},
    )
    return model, evaluation


def train_population_regressor(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[Pipeline, Evaluation]:
    """Predict the population five generations ahead.

    Read the R2 score with care. The label is severely skewed: most scenarios
    collapse toward zero while a few grow large, so a handful of rows hold most
    of the variance and R2 swings widely between splits. The mean absolute error
    is the more stable summary, and the extinction classifier below is the more
    trustworthy prediction task.
    """
    train, test = grouped_split(frame, test_fraction=test_fraction, seed=seed)
    assert_no_leakage(train, test)

    model = Pipeline(
        steps=[
            ("prepare", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=1,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    model.fit(feature_frame(train), train[POPULATION_LABEL])
    predicted = model.predict(feature_frame(test))
    actual = test[POPULATION_LABEL].to_numpy()

    # Baseline: always predict the training mean. R2 is measured against this.
    baseline = np.full_like(actual, fill_value=float(train[POPULATION_LABEL].mean()))
    variance_share = _top_row_variance_share(train[POPULATION_LABEL].to_numpy(), 5)
    evaluation = Evaluation(
        task="population_regressor",
        metric="r2",
        score=float(r2_score(actual, predicted)),
        train_rows=len(train),
        test_rows=len(test),
        train_scenarios=int(train["scenario_id"].nunique()),
        test_scenarios=int(test["scenario_id"].nunique()),
        extra={
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(root_mean_squared_error(actual, predicted)),
            "baseline_r2": float(r2_score(actual, baseline)),
            "top5_variance_share": variance_share,
        },
    )
    return model, evaluation


def _top_row_variance_share(values: np.ndarray, count: int) -> float:
    """Fraction of total variance held by the largest ``count`` values.

    A high share means the score is dominated by a few extreme scenarios, which
    is why a single R2 number can be misleading on this label.
    """
    if len(values) == 0:
        return 0.0
    mean = float(values.mean())
    total = float(((values - mean) ** 2).sum())
    if total == 0.0:
        return 0.0
    largest = np.sort(values)[-count:]
    return float((((largest - mean) ** 2).sum()) / total)


def train_extinction_classifier(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[Pipeline, Evaluation]:
    """Predict whether a scenario's population collapses to zero.

    This is the most useful and most trustworthy of the tasks: the outcome is
    balanced enough to score honestly, and it answers the question a student
    actually asks, which is whether their scenario survives at all.
    """
    labelled = frame.assign(
        extinct=(frame[POPULATION_LABEL] == 0).astype(int)
    )
    train, test = grouped_split(labelled, test_fraction=test_fraction, seed=seed)
    assert_no_leakage(train, test)

    model = Pipeline(
        steps=[
            ("prepare", _preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=n_jobs
                ),
            ),
        ]
    )
    model.fit(feature_frame(train), train["extinct"])
    predicted = model.predict(feature_frame(test))
    probability = model.predict_proba(feature_frame(test))[:, 1]

    evaluation = Evaluation(
        task="extinction_classifier",
        metric="roc_auc",
        score=float(roc_auc_score(test["extinct"], probability)),
        train_rows=len(train),
        test_rows=len(test),
        train_scenarios=int(train["scenario_id"].nunique()),
        test_scenarios=int(test["scenario_id"].nunique()),
        extra={
            "accuracy": float(accuracy_score(test["extinct"], predicted)),
            "base_rate": float(test["extinct"].mean()),
        },
    )
    return model, evaluation


def check_against_formula(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
    random_state: int = 0,
    n_jobs: int | None = None,
) -> tuple[Pipeline, Evaluation]:
    """Learn carrying capacity and compare it with the known exact equation.

    The simulator computes capacity in closed form, so a perfect model is
    possible. The gap between the model and exact arithmetic is a direct
    measurement of what the machine learning adds here: nothing about foxes,
    only a pipeline that could be reused if the equation were unknown.
    """
    train, test = grouped_split(frame, test_fraction=test_fraction, seed=seed)
    assert_no_leakage(train, test)

    model = Pipeline(
        steps=[
            ("prepare", _preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=1,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    model.fit(feature_frame(train), train[CAPACITY_LABEL])
    predicted = model.predict(feature_frame(test))
    actual = test[CAPACITY_LABEL].to_numpy()

    # Exact-arithmetic reference: the simulator's own value, no learning.
    exact_mae = 0.0
    evaluation = Evaluation(
        task="capacity_check",
        metric="r2",
        score=float(r2_score(actual, predicted)),
        train_rows=len(train),
        test_rows=len(test),
        train_scenarios=int(train["scenario_id"].nunique()),
        test_scenarios=int(test["scenario_id"].nunique()),
        extra={
            "mae": float(mean_absolute_error(actual, predicted)),
            "exact_formula_mae": exact_mae,
        },
    )
    return model, evaluation


def train_all(
    frame: pd.DataFrame, *, seed: int = 7, n_jobs: int | None = None
) -> tuple[dict[str, Pipeline], list[Evaluation]]:
    """Fit every task on one dataset and return the models with their scores."""
    classifier, classifier_score = train_suitability_classifier(
        frame, seed=seed, n_jobs=n_jobs
    )
    regressor, regressor_score = train_population_regressor(
        frame, seed=seed, n_jobs=n_jobs
    )
    extinction, extinction_score = train_extinction_classifier(
        frame, seed=seed, n_jobs=n_jobs
    )
    capacity, capacity_score = check_against_formula(frame, seed=seed, n_jobs=n_jobs)
    models = {
        "suitability_classifier": classifier,
        "population_regressor": regressor,
        "extinction_classifier": extinction,
        "capacity_check": capacity,
    }
    return models, [classifier_score, regressor_score, extinction_score, capacity_score]


def save_models(models: dict[str, Pipeline], directory: str | Path) -> list[Path]:
    """Persist each fitted pipeline, preprocessing included."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in models.items():
        path = destination / f"{name}.joblib"
        joblib.dump(model, path)
        written.append(path)
    return written


def load_model(path: str | Path) -> Pipeline:
    """Load a persisted pipeline; preprocessing travels with the model."""
    return joblib.load(Path(path))
