from ecosystem_evolution import desert_example
from ecosystem_evolution.environment import Season
from ecosystem_evolution.locations import environment_for_location


def test_desert_example_shows_both_open_desert_and_oasis(capsys) -> None:
    desert_example.main()
    output = capsys.readouterr().out

    assert "Open Sonoran desert" in output
    assert "0 of 60 runs survived" in output
    assert "A red fox needs regular water" in output
    assert "Oasis at 30 C:" in output
    assert "One line of descent" in output


def test_open_desert_keeps_the_preset_climate_unchanged() -> None:
    """The honest scenario must not inflate resources to force survival."""
    environment = environment_for_location("sonoran_desert", **desert_example.OPEN_DESERT)

    assert environment.rainfall == 0.12
    assert environment.humidity == 0.18
    assert environment.water_availability == 0.12
    assert environment.food_availability == 0.30


def test_oasis_water_agrees_with_its_rainfall_and_humidity() -> None:
    """The bug this replaced: desert rainfall with rainforest water.

    A place cannot be both. The oasis raises all three together, so the
    scenario describes one coherent habitat.
    """
    oasis = environment_for_location("sonoran_desert", **desert_example.OASIS)

    assert oasis.rainfall >= 0.40, "an oasis must rain more than a desert"
    assert oasis.humidity >= 0.50
    assert oasis.water_availability >= 0.60
    assert oasis.season is Season.SUMMER
