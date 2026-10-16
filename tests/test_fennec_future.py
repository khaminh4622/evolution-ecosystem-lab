from ecosystem_evolution.environment import HabitatType
from ecosystem_evolution.locations import environment_for_location
from ecosystem_evolution.species import FENNEC_FOX, FOX


def test_fennec_profile_is_more_suitable_than_general_fox_in_hot_desert() -> None:
    desert = environment_for_location("sonoran_desert", temperature_c=40.0)

    assert desert.habitat_type is HabitatType.DESERT
    assert FENNEC_FOX.suitability(desert) > FOX.suitability(desert)
