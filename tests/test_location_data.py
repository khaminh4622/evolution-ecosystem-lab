from ecosystem_evolution.environment import HabitatType, Season
from ecosystem_evolution.location_data import (
    MapLocation,
    WeatherObservation,
    environment_for_coordinates,
    environment_for_place,
    environment_from_weather,
    search_place,
)


def _fake_fetch(url: str) -> dict[str, object]:
    if "geocoding-api" in url:
        return {
            "results": [
                {
                    "name": "Phoenix",
                    "latitude": 33.4484,
                    "longitude": -112.074,
                    "country": "United States",
                    "timezone": "America/Phoenix",
                }
            ]
        }
    return {
        "current": {
            "temperature_2m": 40.0,
            "relative_humidity_2m": 18,
            "precipitation": 0.0,
            "soil_moisture_0_to_1cm": 0.08,
            "time": "2026-07-15T12:00",
        }
    }


def test_place_search_returns_coordinates_from_map_provider() -> None:
    location = search_place("Phoenix, Arizona", fetch_json=_fake_fetch)[0]

    assert location.name == "Phoenix"
    assert location.latitude == 33.4484
    assert location.country == "United States"


def test_weather_bridge_creates_editable_desert_environment() -> None:
    selected = environment_for_place(
        "Phoenix, Arizona",
        HabitatType.DESERT,
        fetch_json=_fake_fetch,
        food_availability=0.40,
    )

    assert selected.environment.habitat_type is HabitatType.DESERT
    assert selected.environment.temperature_c == 40.0
    assert selected.environment.humidity == 0.18
    assert selected.environment.food_availability == 0.40
    assert selected.environment.water_availability < 0.3
    assert selected.environment.season is Season.SUMMER


def test_weather_resource_estimates_are_bounded() -> None:
    location = MapLocation("Test", -20.0, 30.0)
    weather = WeatherObservation(55.0, 1.0, 100.0, 1.0, "2026-01-15T12:00")
    environment = environment_from_weather(location, weather, HabitatType.WETLAND)

    assert environment.season is Season.SUMMER
    assert all(
        0.0 <= value <= 1.0
        for value in (
            environment.rainfall,
            environment.food_availability,
            environment.water_availability,
            environment.vegetation_cover,
        )
    )


def test_map_coordinates_can_start_an_experiment_without_place_search() -> None:
    selected = environment_for_coordinates(
        "Clicked desert", 33.4484, -112.074, HabitatType.DESERT, fetch_json=_fake_fetch
    )

    assert selected.location.name == "Clicked desert"
    assert selected.environment.temperature_c == 40.0
