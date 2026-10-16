from ecosystem_evolution import world_location_example
from ecosystem_evolution.environment import HabitatType
from ecosystem_evolution.location_data import (
    LocationEnvironment,
    MapLocation,
    WeatherObservation,
    environment_from_weather,
)
from ecosystem_evolution.simulation import (
    EvolutionSimulator,
    InitialPopulationSpec,
    SimulationConfig,
)


def _stub_place(name: str, weather: WeatherObservation):
    """Return a fake environment_for_place that honours the caller's overrides."""
    location = MapLocation(name, 33.4484, -112.074, "United States")

    def fake_environment_for_place(place, habitat_type, **overrides):
        return LocationEnvironment(
            location=location,
            weather=weather,
            environment=environment_from_weather(
                location, weather, HabitatType.DESERT, **overrides
            ),
        )

    return fake_environment_for_place


def test_world_location_example_reports_the_honest_outcome(capsys, monkeypatch) -> None:
    """The shipped scenario is deliberately unsurvivable, and says why.

    Phoenix in July is dry enough that the derived water keeps no foxes alive,
    so the example must report the extinction and explain it rather than
    inflating the resources to force a survivors' table.
    """
    weather = WeatherObservation(31.1, 0.12, 0.0, 0.04, "2026-07-15T12:00")
    monkeypatch.setattr(
        world_location_example,
        "environment_for_place",
        _stub_place("Phoenix", weather),
    )

    world_location_example.main(include_lineage=False)
    output = capsys.readouterr().out

    assert "Map selection: Phoenix, United States" in output
    assert "Weather at selection: 31.1 C" in output
    # Resources are derived from the real weather, not overridden.
    assert "Derived from that weather: water 0.11" in output
    assert "0 of 60 runs survived" in output
    assert "Why no foxes were left" in output


def test_derived_resources_are_not_inflated() -> None:
    """No resource override, so a dry place stays dry."""
    import inspect

    source = inspect.getsource(world_location_example.build_environment)
    assert "water_availability" not in source
    assert "food_availability" not in source


def test_capacity_check_reports_the_habitat_limit(monkeypatch) -> None:
    weather = WeatherObservation(40.0, 0.05, 0.0, 0.02, "2026-07-15T12:00")
    stub = _stub_place("Dry place", weather)
    monkeypatch.setattr(
        world_location_example,
        "build_environment",
        lambda: stub(
            "Dry place",
            HabitatType.DESERT,
            food_availability=0.10,
            water_availability=0.05,
            vegetation_cover=0.05,
            habitat_area_km2=0.5,
        ),
    )

    report = world_location_example.check_capacity()

    assert "carrying capacity" in report
    assert "derived water" in report


def test_capacity_check_is_quiet_when_the_habitat_is_large_enough(monkeypatch) -> None:
    weather = WeatherObservation(31.1, 0.12, 0.0, 0.04, "2026-07-15T12:00")
    monkeypatch.setattr(
        world_location_example, "environment_for_place", _stub_place("Phoenix", weather)
    )

    report = world_location_example.check_capacity()

    assert "carrying capacity" in report
    assert "over capacity" not in report
