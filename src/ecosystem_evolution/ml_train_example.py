"""Train the teaching models and report honest, held-out scores.

Requires the optional machine-learning dependencies::

    pip install -e ".[ml]"

Run::

    python -m ecosystem_evolution.ml_train_example

Every score comes from scenarios the model never saw while fitting. The output
also states plainly what these models are worth: a model trained on this
simulator reproduces the simulator, and says nothing about real foxes.
"""

from __future__ import annotations

import os

from .ml.datasets import build_dataset
from .ml.models import save_models, train_all

SCENARIOS = 250


def main() -> None:
    frame = build_dataset(scenarios=SCENARIOS, generations=10, seed=20260101)
    print(f"Training on {len(frame)} simulated scenarios\n")

    models, scores = train_all(frame)
    for score in scores:
        print(score.describe())

    regressor = next(s for s in scores if s.task == "population_regressor")
    extinction = next(s for s in scores if s.task == "extinction_classifier")
    capacity = next(s for s in scores if s.task == "capacity_check")

    print(
        f"\nRead the regressor with care: {regressor.extra['top5_variance_share']:.0%} "
        "of the label's variance sits in 5 of"
        f" {SCENARIOS} scenarios."
    )
    print(
        "Most scenarios collapse toward zero while a few grow large, so R2 swings"
        " widely between splits."
    )
    print(f"Mean absolute error ({regressor.extra['mae']:.1f} foxes) is the steadier summary.")

    print(
        f"\nThe extinction question is the one to trust: ROC-AUC "
        f"{extinction.score:.3f} against a base rate of {extinction.extra['base_rate']:.2f}."
    )
    print("That answers what a student actually asks: does this scenario survive?")

    print(
        f"\nThe capacity check is a pipeline test, not new knowledge. The simulator"
        f" already computes capacity exactly, so a model can only approximate it"
        f" (mean absolute error {capacity.extra['mae']:.0f} foxes)."
    )

    print("\nHonest reading of all of this:")
    print("  * Every label comes from this simulator, so the models reproduce it.")
    print("  * The simulator's rules are hand-written formulas, so 'learning' them")
    print("    demonstrates a working pipeline, not biology.")
    print("  * These models must never be presented as evidence about real foxes.")

    # Model artifacts are build output, so they are written outside the source
    # tree; the repository ignores models/*.joblib either way.
    base = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "."
    destination = os.path.join(base, "evolution-models")
    try:
        written = save_models(models, destination)
        print(f"\nSaved {len(written)} pipelines (preprocessing included) to {destination}")
    except OSError as error:
        print(f"\nCould not save models: {error}")


if __name__ == "__main__":
    main()
