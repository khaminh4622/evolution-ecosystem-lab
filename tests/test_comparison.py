from ecosystem_evolution.comparison import compare_environment_change
from ecosystem_evolution.locations import environment_for_location
from ecosystem_evolution.simulation import InitialPopulationSpec, SimulationConfig


def test_comparison_keeps_the_original_environment_unchanged() -> None:
    original = environment_for_location("sonoran_desert")
    comparison = compare_environment_change(
        original,
        {"food_availability": 0.80},
        config=SimulationConfig(generations=2, seed=77),
        initial_population=InitialPopulationSpec(size=10),
    )

    assert comparison.original_environment.food_availability == 0.30
    assert comparison.changed_environment.food_availability == 0.80
    assert comparison.original_result.config.seed == comparison.changed_result.config.seed == 77


def test_same_inputs_produce_matching_before_after_timeline() -> None:
    original = environment_for_location("pacific_northwest_forest")
    comparison = compare_environment_change(
        original,
        {},
        config=SimulationConfig(generations=2, seed=77),
        initial_population=InitialPopulationSpec(size=10),
    )

    assert comparison.original_result.as_dict() == comparison.changed_result.as_dict()
