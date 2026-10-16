from ecosystem_evolution.genetics import Genome
from ecosystem_evolution.simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from tests.test_simulation import _environment


def test_special_founder_genomes_are_present_in_the_baseline_population() -> None:
    adapted = Genome(0.05, 0.95, 0.90, 0.80, 0.40)
    result = EvolutionSimulator(_environment(), SimulationConfig(generations=1, seed=3)).run(
        InitialPopulationSpec(size=5, founder_genomes=(adapted, adapted))
    )

    assert [animal.genome for animal in result.replay[0].animals[:2]] == [adapted, adapted]


def test_special_founders_cannot_exceed_starting_population() -> None:
    adapted = Genome(0.05, 0.95, 0.90, 0.80, 0.40)

    try:
        InitialPopulationSpec(size=1, founder_genomes=(adapted, adapted))
    except ValueError as error:
        assert "cannot contain more foxes" in str(error)
    else:
        raise AssertionError("Expected special-founder validation to fail")
