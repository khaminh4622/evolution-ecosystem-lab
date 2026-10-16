import pytest

from ecosystem_evolution.environment import HabitatType
from ecosystem_evolution.locations import environment_for_location, get_location


def test_desert_location_supplies_desert_resource_defaults() -> None:
    environment = environment_for_location("sonoran_desert")

    assert environment.habitat_type is HabitatType.DESERT
    assert environment.temperature_c > 30
    assert environment.water_availability < 0.2
    assert environment.food_availability < 0.4


def test_player_can_adjust_a_location_assumption_without_changing_preset() -> None:
    adjusted = environment_for_location("sonoran_desert", food_availability=0.65)

    assert adjusted.food_availability == 0.65
    assert get_location("sonoran_desert").environment.food_availability == 0.30


def test_unknown_location_explains_available_choices() -> None:
    with pytest.raises(ValueError, match="sonoran_desert"):
        get_location("not-a-place")
