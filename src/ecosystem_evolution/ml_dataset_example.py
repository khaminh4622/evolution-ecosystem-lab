"""Build an ML dataset from simulated scenarios and show what is in it.

Requires the optional machine-learning dependencies::

    pip install -e ".[ml]"

Run::

    python -m ecosystem_evolution.ml_dataset_example
"""

from __future__ import annotations

from .ml.datasets import REGRESSION_HORIZON, build_dataset, grouped_split

SCENARIOS = 250


def main() -> None:
    frame = build_dataset(scenarios=SCENARIOS, generations=10, seed=20260101)
    print(f"Simulated {len(frame)} scenarios -> {frame.shape[0]} rows, {frame.shape[1]} columns")

    print("\nLabel balance (suitability_class):")
    for name, count in frame["suitability_class"].value_counts().items():
        print(f"  {name:<12} {count:>4}  ({count / len(frame):.0%})")

    label = f"population_after_{REGRESSION_HORIZON}_generations"
    print(f"\nRegression label ({label}):")
    print(f"  extinct (0 foxes): {(frame[label] == 0).sum()} of {len(frame)} scenarios")
    print(f"  mean {frame[label].mean():.1f}, median {frame[label].median():.0f}, max {frame[label].max()}")

    train, test = grouped_split(frame, test_fraction=0.25, seed=7)
    print("\nScenario-grouped split (never splits a scenario across both sides):")
    print(f"  train: {len(train):>3} rows from {train['scenario_id'].nunique():>3} scenarios")
    print(f"  test:  {len(test):>3} rows from {test['scenario_id'].nunique():>3} scenarios")

    print("\nWhy grouping matters: every row of a scenario shares one environment,")
    print("so a row-wise split would let a model memorise the scenario instead of")
    print("learning the relationship.")


if __name__ == "__main__":
    main()
