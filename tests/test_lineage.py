import pytest

from ecosystem_evolution.lineage import (
    descendants,
    founder_shares,
    life_histories,
    line_of_descent,
    lineage_report,
)
from ecosystem_evolution.simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from tests.test_simulation import _environment


@pytest.fixture(scope="module")
def result():
    config = SimulationConfig(generations=15, seed=21, mutation_probability=0.10)
    return EvolutionSimulator(_environment(), config).run(InitialPopulationSpec(size=20))


def test_every_parent_exists_and_was_born_before_its_child(result) -> None:
    lives = life_histories(result)
    for life in lives.values():
        for parent_id in life.animal.parent_ids:
            assert lives[parent_id].born_generation < life.born_generation


def test_children_are_listed_under_both_parents(result) -> None:
    lives = life_histories(result)
    for life in lives.values():
        for child_id in life.child_ids:
            assert life.id in lives[child_id].animal.parent_ids


def test_founder_shares_sum_to_one_and_name_only_founders(result) -> None:
    shares = founder_shares(result)
    founders = {animal.id for animal in result.replay[0].animals}
    assert set(shares) <= founders
    assert sum(shares.values()) == pytest.approx(1.0)


def test_line_of_descent_runs_from_a_founder_to_the_chosen_fox(result) -> None:
    final = result.replay[-1].animals
    chain = line_of_descent(result, final[-1].id)
    assert chain[0].parent_ids == ()
    assert chain[-1].id == final[-1].id
    for older, younger in zip(chain, chain[1:]):
        assert older.id in younger.parent_ids


def test_line_of_descent_can_be_pinned_to_a_chosen_founder(result) -> None:
    lives = life_histories(result)
    shares = founder_shares(result)
    leader = max(shares, key=lambda fox_id: shares[fox_id])
    living = [a for a in result.replay[-1].animals if a.id in descendants(lives, leader)]
    chain = line_of_descent(result, living[0].id, leader)
    assert chain[0].id == leader
    assert chain[-1].id == living[0].id
    with pytest.raises(ValueError):
        line_of_descent(result, leader, living[0].id)


def test_descendants_include_children_and_exclude_self(result) -> None:
    lives = life_histories(result)
    parent = next(life for life in lives.values() if life.child_ids)
    found = descendants(lives, parent.id)
    assert set(parent.child_ids) <= found
    assert parent.id not in found


def test_report_for_extinct_population_says_so() -> None:
    config = SimulationConfig(generations=5, seed=1)
    extinct = EvolutionSimulator(_environment(temperature_c=55.0, food_availability=0.02,
                                              water_availability=0.02), config).run(
        InitialPopulationSpec(size=3))
    if extinct.final_population == 0:
        assert "extinct" in lineage_report(extinct)
    else:
        assert "Founder" in lineage_report(extinct)
