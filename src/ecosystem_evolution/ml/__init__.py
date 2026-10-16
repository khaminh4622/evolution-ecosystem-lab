"""Machine learning on simulated experiments.

What this package is for: turning the simulator's experiments into a dataset,
then training small models that answer a classroom question quickly.

What it is *not*: evidence about real foxes. Every label comes from this
simulator, so a model here can only reproduce the simulator's own behaviour. The
simulator's rules are hand-written formulas, which means a model that predicts
them well has demonstrated a working pipeline, not discovered biology. That
distinction is stated in the project README and must be kept wherever these
models are presented.

The one safeguard worth remembering: all rows from one scenario share an
environment, so splits are made on ``scenario_id``. Splitting rows at random
would leak and report a score the model has not earned.
"""

from .datasets import (
    FAVORABLE_THRESHOLD,
    CHALLENGING_THRESHOLD,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    REGRESSION_HORIZON,
    ScenarioParameters,
    assert_no_leakage,
    build_dataset,
    feature_frame,
    grouped_split,
    random_scenario,
    suitability_class,
)
from .models import (
    Evaluation,
    check_against_formula,
    load_model,
    save_models,
    train_all,
    train_extinction_classifier,
    train_population_regressor,
    train_suitability_classifier,
)

__all__ = [
    "CHALLENGING_THRESHOLD",
    "FAVORABLE_THRESHOLD",
    "FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "REGRESSION_HORIZON",
    "Evaluation",
    "ScenarioParameters",
    "assert_no_leakage",
    "build_dataset",
    "check_against_formula",
    "feature_frame",
    "grouped_split",
    "load_model",
    "random_scenario",
    "save_models",
    "suitability_class",
    "train_all",
    "train_extinction_classifier",
    "train_population_regressor",
    "train_suitability_classifier",
]
