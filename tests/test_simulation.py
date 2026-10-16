from ecosystem_evolution.environment import EnvironmentConfig, HabitatType, Season
from ecosystem_evolution.genetics import Genome
from ecosystem_evolution.logging import ExperimentRecord
from ecosystem_evolution.simulation import (
    EvolutionSimulator,
    InitialPopulationSpec,
    SimulationConfig,
)


def _environment(**overrides: object) -> EnvironmentConfig:
    defaults: dict[str, object] = {
        "temperature_c": 10.0,
        "rainfall": 0.6,
        "humidity": 0.6,
        "food_availability": 0.65,
        "water_availability": 0.65,
        "vegetation_cover": 0.65,
        "predator_pressure": 0.2,
        "habitat_type": HabitatType.FOREST,
        "habitat_area_km2": 2.0,
        "pollution_pressure": 0.05,
        "season": Season.SPRING,
    }
    defaults.update(overrides)
    return EnvironmentConfig(**defaults)  # type: ignore[arg-type]


def test_same_seed_yields_exact_same_replay() -> None:
    environment = _environment()
    config = SimulationConfig(generations=12, seed=917, mutation_probability=0.15)
    spec = InitialPopulationSpec(size=35, trait_mean=0.5, trait_variation=0.22)

    first = EvolutionSimulator(environment, config).run(spec)
    second = EvolutionSimulator(environment, config).run(spec)

    assert first.as_dict() == second.as_dict()
    assert len(first.generations) == 13  # baseline plus 12 experimental generations


def test_environment_selects_different_thermal_genomes() -> None:
    cold = _environment(temperature_c=-20.0)
    hot = _environment(temperature_c=42.0)
    cold_adapted = Genome(0.95, 0.05, 0.35, 0.70, 0.45)
    hot_adapted = Genome(0.05, 0.95, 0.35, 0.70, 0.45)

    cold_sim = EvolutionSimulator(cold, SimulationConfig(seed=1))
    hot_sim = EvolutionSimulator(hot, SimulationConfig(seed=1))

    assert cold_sim.survival_probability(cold_adapted, 20) > cold_sim.survival_probability(hot_adapted, 20)
    assert hot_sim.survival_probability(hot_adapted, 20) > hot_sim.survival_probability(cold_adapted, 20)


def test_capacity_prevents_uncontrolled_growth() -> None:
    config = SimulationConfig(
        generations=10,
        seed=81,
        base_carrying_capacity_per_km2=12,
        minimum_carrying_capacity=5,
    )
    result = EvolutionSimulator(_environment(habitat_area_km2=1.0), config).run(
        InitialPopulationSpec(size=250, trait_mean=0.5, trait_variation=0.2)
    )

    assert all(
        record.population_size <= record.carrying_capacity
        for record in result.generations[1:]
    )


def test_favorable_conditions_support_population_growth() -> None:
    favorable = _environment(
        temperature_c=15.0,
        food_availability=0.85,
        water_availability=0.85,
        vegetation_cover=0.70,
        predator_pressure=0.08,
        pollution_pressure=0.01,
        habitat_area_km2=2.0,
    )
    result = EvolutionSimulator(favorable, SimulationConfig(generations=10, seed=90210)).run(
        InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18)
    )

    assert result.final_population >= 30


def test_founders_have_unique_ids_and_export_rows_keep_seed() -> None:
    result = EvolutionSimulator(_environment(), SimulationConfig(generations=2, seed=55)).run(
        InitialPopulationSpec(size=20)
    )
    ids = [animal.id for animal in result.replay[0].animals]
    rows = ExperimentRecord("identity-check", "Traits are inherited.", result).generation_rows()

    assert len(ids) == len(set(ids))
    assert {row["seed"] for row in rows} == {55}
    assert all("dist_fur_insulation" in row for row in rows)
