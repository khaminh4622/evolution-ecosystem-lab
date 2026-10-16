"""Walk through the machine-learning pipeline step by step, printing real output.

Requires the optional machine-learning dependencies::

    pip install -e ".[ml]"

Run::

    python -m ecosystem_evolution.ml_walkthrough_example

This is the teaching example for the ML phase: it shows a scenario becoming a
row of data, how the split keeps scenarios apart, how the models score, how one
prediction looks on a scenario the model never saw, and that a saved pipeline
reloads identically. It also states what the demonstration does and does not
prove.
"""

from __future__ import annotations

import os

from .ml.datasets import (
    REGRESSION_HORIZON,
    build_dataset,
    feature_frame,
    grouped_split,
)
from .ml.models import POPULATION_LABEL, load_model, save_models, train_all

SCENARIOS = 250


def _rule(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    _rule("STEP 1: generate scenarios and simulate each one")
    frame = build_dataset(scenarios=SCENARIOS, generations=10, seed=20260101)
    print(f"{len(frame)} scenarios simulated -> DataFrame with {frame.shape[1]} columns")

    first = frame.iloc[0]
    print("\nthe first scenario, as the model sees it:")
    print(f"  scenario_id        {first['scenario_id']}")
    print(f"  habitat_type       {first['habitat_type']}")
    print(f"  temperature_c      {first['temperature_c']}")
    print(f"  food_availability  {first['food_availability']}")
    print(f"  initial_size       {first['initial_size']}")
    print("  --- labels, produced by simulating the scenario ---")
    print(f"  suitability_class              {first['suitability_class']}")
    print(
        f"  population_after_{REGRESSION_HORIZON}_generations "
        f"{first[f'population_after_{REGRESSION_HORIZON}_generations']}"
    )

    print()
    _rule("STEP 2: split by scenario, never by row")
    train, test = grouped_split(frame, test_fraction=0.25, seed=7)
    print(f"train: {len(train):>3} rows / {train['scenario_id'].nunique():>3} scenarios")
    print(f"test:  {len(test):>3} rows / {test['scenario_id'].nunique():>3} scenarios")
    overlap = set(train["scenario_id"]) & set(test["scenario_id"])
    print(f"scenarios appearing on both sides: {len(overlap)}  <- must be 0")

    print()
    _rule("STEP 3: train the four models")
    models, scores = train_all(frame)
    for score in scores:
        print(f"  {score.describe()}")

    print()
    _rule("STEP 4: use a model on a scenario it has never seen")
    new_scenario = test.iloc[[0]]
    features = feature_frame(new_scenario)
    classifier = models["suitability_classifier"]
    extinction = models["extinction_classifier"]
    regressor = models["population_regressor"]

    predicted_label = classifier.predict(features)[0]
    classes = list(classifier.classes_)
    probability = classifier.predict_proba(features)[0]
    extinct_probability = extinction.predict_proba(features)[0][
        list(extinction.classes_).index(1)
    ]

    print(f"  scenario {new_scenario['scenario_id'].iloc[0]}")
    print(
        f"    habitat {new_scenario['habitat_type'].iloc[0]}, "
        f"{new_scenario['temperature_c'].iloc[0]} C"
    )
    print(
        f"    suitability class: actual={new_scenario['suitability_class'].iloc[0]:<12} "
        f"predicted={predicted_label}"
    )
    print(
        "    class probabilities: "
        + ", ".join(f"{name}={value:.2f}" for name, value in zip(classes, probability))
    )
    print(f"    extinction risk:     {extinct_probability:.0%}")
    print(
        f"    population in {REGRESSION_HORIZON} generations: "
        f"actual={new_scenario[POPULATION_LABEL].iloc[0]}, "
        f"predicted={regressor.predict(features)[0]:.1f}"
    )

    print()
    _rule("STEP 5: save the pipelines and load one back")
    directory = os.path.join(os.environ.get("TEMP") or ".", "evolution-models-demo")
    written = save_models(models, directory)
    for path in written:
        print(f"  wrote {os.path.basename(str(path)):<32} {os.path.getsize(path):>9,} bytes")
    reloaded = load_model(os.path.join(directory, "suitability_classifier.joblib"))
    print(
        "  reloaded model predicts the same: "
        f"{(reloaded.predict(features) == classifier.predict(features)).all()}"
    )

    print()
    _rule("What this proves, and what it does not")
    print("  Proves: the pipeline works on unseen scenarios and persists correctly.")
    print("  Does NOT prove: anything about real foxes. Every label came from the")
    print("  simulator, whose rules are hand-written formulas.")
    print(
        "  A single correct prediction is not accuracy: the 62 held-out scenarios"
    )
    print("  in step 3 are the measurement. Trust the extinction probability more")
    print("  than the population point estimate.")


if __name__ == "__main__":
    main()
