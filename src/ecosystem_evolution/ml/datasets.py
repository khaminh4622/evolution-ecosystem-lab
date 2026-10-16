"""Build scenario-grouped machine-learning datasets from simulated experiments.

Every row this module produces comes from a real run of the simulator, never
from a hand-written rule, with one documented exception: the classification
label ``suitability_class`` is derived from ``SpeciesProfile.suitability``,
which is itself a documented teaching formula rather than a measurement.

The important guarantee is grouping. All rows from one simulated scenario share
the same ``scenario_id``, so train/test splits must be made on ``scenario_id``.
Splitting rows at random would put near-duplicate generations of the same run on
both sides and report an accuracy the model has not earned.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Sequence

import pandas as pd

from ..environment import EnvironmentConfig, HabitatType, Season
from ..species import FOX, SpeciesProfile
from ..simulation import (
    EvolutionSimulator,
    InitialPopulationSpec,
    SimulationConfig,
)

# Suffix of the regression label: population this many generations ahead.
REGRESSION_HORIZON = 5

# Thresholds turning a 0-1 suitability score into a class label. Chosen so the
# three classes are meaningful in this model: below 0.45 a population almost
# always shrinks, above 0.75 it usually grows back to carrying capacity.
FAVORABLE_THRESHOLD = 0.75
CHALLENGING_THRESHOLD = 0.45

CATEGORICAL_FEATURES = ("habitat_type", "season")

NUMERIC_FEATURES = (
    "temperature_c",
    "rainfall",
    "humidity",
    "food_availability",
    "water_availability",
    "vegetation_cover",
    "predator_pressure",
    "habitat_area_km2",
    "pollution_pressure",
    "initial_size",
    "initial_trait_mean",
    "initial_trait_variation",
)

LABEL_COLUMNS = (
    "suitability_class",
    "suitability_score",
    "population_after_5_generations",
    "carrying_capacity",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def suitability_class(score: float) -> str:
    """Turn a 0-1 suitability score into one of three documented classes."""
    if score >= FAVORABLE_THRESHOLD:
        return "favorable"
    if score >= CHALLENGING_THRESHOLD:
        return "moderate"
    return "challenging"


@dataclass(frozen=True, slots=True)
class ScenarioParameters:
    """One randomly drawn experiment: an environment plus a starting population."""

    scenario_id: str
    environment: EnvironmentConfig
    population_size: int
    trait_mean: float
    trait_variation: float
    generations: int
    seed: int


def random_scenario(rng: random.Random, scenario_id: str, generations: int = 10) -> ScenarioParameters:
    """Draw one plausible scenario.

    Temperature and habitat are drawn together, because a coral reef at -20 C is
    not a scenario a student would ask about; drawing them independently would
    waste most of the dataset on impossible combinations.
    """
    habitat = rng.choice(list(HabitatType))
    temperature_ranges = {
        HabitatType.DESERT: (-5.0, 50.0),
        HabitatType.GRASSLAND: (-15.0, 40.0),
        HabitatType.FOREST: (-25.0, 35.0),
        HabitatType.TUNDRA: (-40.0, 15.0),
        HabitatType.WETLAND: (-10.0, 38.0),
        HabitatType.COASTAL: (-5.0, 35.0),
        HabitatType.CORAL_REEF: (18.0, 34.0),
    }
    low, high = temperature_ranges[habitat]
    environment = EnvironmentConfig(
        temperature_c=round(rng.uniform(low, high), 2),
        rainfall=round(rng.random(), 3),
        humidity=round(rng.random(), 3),
        food_availability=round(rng.random(), 3),
        water_availability=round(rng.random(), 3),
        vegetation_cover=round(rng.random(), 3),
        predator_pressure=round(rng.random(), 3),
        pollution_pressure=round(rng.random() * 0.6, 3),
        habitat_type=habitat,
        habitat_area_km2=round(rng.uniform(0.5, 20.0), 2),
        season=rng.choice(list(Season)),
    )
    return ScenarioParameters(
        scenario_id=scenario_id,
        environment=environment,
        population_size=rng.randint(10, 60),
        trait_mean=round(rng.uniform(0.3, 0.7), 3),
        trait_variation=round(rng.uniform(0.05, 0.30), 3),
        generations=generations,
        seed=rng.randrange(1_000_000),
    )


def _scenario_row(
    scenario: ScenarioParameters,
    species: SpeciesProfile,
    result,
) -> dict[str, Any]:
    """One training row: the scenario's inputs and its outcome."""
    environment = scenario.environment
    score = species.suitability(environment)
    horizon = min(REGRESSION_HORIZON, result.generations[-1].generation)
    record = result.generations[horizon]
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "species_id": species.id,
        "seed": scenario.seed,
        "temperature_c": environment.temperature_c,
        "rainfall": environment.rainfall,
        "humidity": environment.humidity,
        "food_availability": environment.food_availability,
        "water_availability": environment.water_availability,
        "vegetation_cover": environment.vegetation_cover,
        "predator_pressure": environment.predator_pressure,
        "pollution_pressure": environment.pollution_pressure,
        "habitat_area_km2": environment.habitat_area_km2,
        "habitat_type": environment.habitat_type.value,
        "season": environment.season.value,
        "initial_size": scenario.population_size,
        "initial_trait_mean": scenario.trait_mean,
        "initial_trait_variation": scenario.trait_variation,
        "suitability_score": score,
        "suitability_class": suitability_class(score),
        "carrying_capacity": result.generations[0].carrying_capacity,
        "final_population": result.final_population,
        f"population_after_{REGRESSION_HORIZON}_generations": record.population_size,
    }
    return row


def build_dataset(
    scenarios: int = 200,
    *,
    generations: int = 10,
    seed: int = 20260101,
    species: SpeciesProfile = FOX,
    mutation_probability: float = 0.10,
) -> pd.DataFrame:
    """Simulate ``scenarios`` random experiments and return one row each.

    Deterministic for a given ``seed``: the same call always returns the same
    frame, which is what makes a reported score reproducible.

    ``generations`` must exceed the regression horizon by a clear margin. If the
    run ends soon after generation 5, the label is nearly the run's final
    population, and a model can appear accurate by predicting the endpoint
    rather than by learning anything. The default of 10 generations keeps the
    label 5 steps away from the end of the run.
    """
    if scenarios < 1:
        raise ValueError("scenarios must be at least 1")
    if generations < REGRESSION_HORIZON * 2:
        raise ValueError(
            f"generations must be at least {REGRESSION_HORIZON * 2}, so the "
            f"{REGRESSION_HORIZON}-generation label is not simply the end of the run"
        )

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(scenarios):
        scenario = random_scenario(rng, f"scenario-{index:05d}", generations=generations)
        result = EvolutionSimulator(
            scenario.environment,
            SimulationConfig(
                generations=scenario.generations,
                seed=scenario.seed,
                mutation_probability=mutation_probability,
            ),
            species=species,
        ).run(
            InitialPopulationSpec(
                size=scenario.population_size,
                trait_mean=scenario.trait_mean,
                trait_variation=scenario.trait_variation,
            )
        )
        rows.append(_scenario_row(scenario, species, result))
    return pd.DataFrame(rows)


def grouped_split(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole scenario, so no scenario appears on both sides.

    This is the safeguard the schema document requires. A row-wise split would
    leak: every row of a scenario shares one environment, so a model could
    memorise the scenario instead of learning the relationship.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    scenario_ids = sorted(frame["scenario_id"].unique())
    if len(scenario_ids) < 2:
        raise ValueError("need at least two scenarios to split")
    rng = random.Random(seed)
    rng.shuffle(scenario_ids)
    cut = max(1, int(round(len(scenario_ids) * test_fraction)))
    test_ids = set(scenario_ids[:cut])
    is_test = frame["scenario_id"].isin(test_ids)
    return frame.loc[~is_test].reset_index(drop=True), frame.loc[is_test].reset_index(drop=True)


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any scenario id appears in both splits."""
    overlap = set(train["scenario_id"]) & set(test["scenario_id"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} scenario(s) appear in both splits, e.g. {sorted(overlap)[:3]}"
        )


def feature_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return just the feature columns, dropping unused metadata and labels."""
    chosen = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    missing = [name for name in chosen if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame[chosen].copy()
