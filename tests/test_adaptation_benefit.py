import pytest

from ecosystem_evolution.genetics import Genome
from ecosystem_evolution.simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from tests.test_simulation import _environment


def _hot():
    return _environment(temperature_c=30.0, food_availability=0.7, water_availability=0.7,
                        vegetation_cover=0.6, predator_pressure=0.15, pollution_pressure=0.03,
                        habitat_area_km2=3.0)


def _sim(benefit: float, env=None) -> EvolutionSimulator:
    return EvolutionSimulator(env or _hot(), SimulationConfig(adaptation_benefit=benefit))


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_adaptation_benefit_must_be_between_zero_and_one(bad: float) -> None:
    with pytest.raises(ValueError, match="adaptation_benefit"):
        SimulationConfig(adaptation_benefit=bad)


def test_zero_benefit_is_plain_species_suitability() -> None:
    sim = _sim(0.0)
    base = sim.species.suitability(sim.environment)
    for fitness in (0.0, 0.5, 1.0):
        assert sim.effective_suitability(fitness) == pytest.approx(base)


def test_benefit_only_raises_suitability_and_never_lowers_it() -> None:
    sim = _sim(0.5)
    base = sim.species.suitability(sim.environment)
    assert sim.effective_suitability(base - 0.2) == pytest.approx(base)
    assert sim.effective_suitability(base + 0.2) == pytest.approx(base + 0.5 * 0.2)


def test_better_adapted_fox_survives_more_often_only_with_benefit() -> None:
    env = _hot()
    targets = env.trait_targets()
    adapted = Genome(**targets)
    unadapted = Genome(**{k: 0.5 for k in targets})
    with_benefit, without = _sim(0.5, env), _sim(0.0, env)

    assert with_benefit.survival_probability(adapted, 30) > without.survival_probability(adapted, 30)
    assert with_benefit.survival_probability(adapted, 30) > with_benefit.survival_probability(unadapted, 30)


def test_adaptation_benefit_reduces_hot_weather_extinctions() -> None:
    env = _hot()
    spec = InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18)

    def extinctions(benefit: float) -> int:
        return sum(
            EvolutionSimulator(env, SimulationConfig(generations=30, seed=seed,
                mutation_probability=0.10, adaptation_benefit=benefit)).run(spec).final_population == 0
            for seed in range(30)
        )

    assert extinctions(0.5) < extinctions(0.0)
