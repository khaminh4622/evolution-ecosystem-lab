"""Bridge a map selection and weather observations to simulation inputs.

The module uses Open-Meteo's public geocoding and forecast endpoints. Weather
values are observed/forecast data; food, vegetation, predator pressure, and
pollution remain transparent classroom estimates that a player must be able to
review and change before an experiment begins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from typing import Any, Callable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .environment import EnvironmentConfig, HabitatType, Season

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

JsonFetcher = Callable[[str], Mapping[str, Any]]


class LocationDataError(RuntimeError):
    """Raised when a map or weather provider cannot supply usable data."""


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "evolution-ecosystem-lab/0.1"})
    try:
        with urlopen(request, timeout=10) as response:  # nosec B310: fixed HTTPS endpoints
            payload = json.load(response)
    except OSError as error:
        raise LocationDataError("Could not load location or weather data.") from error
    if not isinstance(payload, dict):
        raise LocationDataError("The location service returned an invalid response.")
    return payload


@dataclass(frozen=True, slots=True)
class MapLocation:
    """A place selected in a map search or clicked on a map."""

    name: str
    latitude: float
    longitude: float
    country: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WeatherObservation:
    """Current weather values returned for one map location."""

    temperature_c: float
    humidity: float
    precipitation_mm: float
    soil_moisture: float | None
    observed_at: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocationEnvironment:
    """The source data and editable environment created for a map location."""

    location: MapLocation
    weather: WeatherObservation
    environment: EnvironmentConfig

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload for a game UI to display before simulation."""
        return {
            "location": self.location.as_dict(),
            "weather": self.weather.as_dict(),
            "environment": self.environment.as_dict(),
        }


def search_place(query: str, fetch_json: JsonFetcher = _fetch_json) -> tuple[MapLocation, ...]:
    """Resolve a player-entered place name into coordinates for a map picker."""
    cleaned = query.strip()
    if len(cleaned) < 2:
        raise ValueError("place search must contain at least two characters")
    payload = fetch_json(f"{GEOCODING_URL}?{urlencode({'name': cleaned, 'count': 5, 'language': 'en', 'format': 'json'})}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise LocationDataError(f"No map locations found for {cleaned!r}.")
    locations: list[MapLocation] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            locations.append(
                MapLocation(
                    name=str(result["name"]),
                    latitude=float(result["latitude"]),
                    longitude=float(result["longitude"]),
                    country=str(result["country"]) if result.get("country") else None,
                    timezone=str(result["timezone"]) if result.get("timezone") else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not locations:
        raise LocationDataError("The map service returned no usable coordinates.")
    return tuple(locations)


def current_weather(
    location: MapLocation, fetch_json: JsonFetcher = _fetch_json
) -> WeatherObservation:
    """Load weather at a map location; this does not claim to measure food."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,soil_moisture_0_to_1cm",
        "timezone": "auto",
    }
    payload = fetch_json(f"{FORECAST_URL}?{urlencode(params)}")
    current = payload.get("current")
    if not isinstance(current, dict):
        raise LocationDataError("The weather service returned no current conditions.")
    try:
        soil_value = current.get("soil_moisture_0_to_1cm")
        return WeatherObservation(
            temperature_c=float(current["temperature_2m"]),
            humidity=_clamp_unit(float(current["relative_humidity_2m"]) / 100.0),
            precipitation_mm=max(0.0, float(current["precipitation"])),
            soil_moisture=(
                _clamp_unit(float(soil_value)) if soil_value is not None else None
            ),
            observed_at=str(current["time"]) if current.get("time") else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LocationDataError("The weather service returned incomplete conditions.") from error


_RESOURCE_BASELINES: dict[HabitatType, tuple[float, float, float]] = {
    # food, water, vegetation: broad teaching assumptions, not ecological survey data
    HabitatType.FOREST: (0.65, 0.65, 0.80),
    HabitatType.GRASSLAND: (0.55, 0.50, 0.55),
    HabitatType.DESERT: (0.25, 0.15, 0.15),
    HabitatType.TUNDRA: (0.30, 0.45, 0.20),
    HabitatType.WETLAND: (0.70, 0.90, 0.85),
    HabitatType.COASTAL: (0.50, 0.80, 0.35),
    HabitatType.CORAL_REEF: (0.55, 0.95, 0.65),
}


def _season_for(location: MapLocation, observed_at: str | None) -> Season:
    if not observed_at:
        return Season.SPRING
    try:
        month = datetime.fromisoformat(observed_at).month
    except ValueError:
        return Season.SPRING
    northern = (Season.WINTER, Season.SPRING, Season.SUMMER, Season.AUTUMN)
    southern = (Season.SUMMER, Season.AUTUMN, Season.WINTER, Season.SPRING)
    seasons = northern if location.latitude >= 0 else southern
    return seasons[(month % 12) // 3]


def environment_from_weather(
    location: MapLocation,
    weather: WeatherObservation,
    habitat_type: HabitatType,
    **overrides: object,
) -> EnvironmentConfig:
    """Make editable resource estimates from weather plus a player-selected habitat.

    A weather snapshot alone cannot determine food, predators, or pollution.
    The formulas below intentionally expose starting estimates for those model
    inputs, and callers may override every value before simulation.
    """
    base_food, base_water, base_vegetation = _RESOURCE_BASELINES[habitat_type]
    rain_signal = _clamp_unit(weather.precipitation_mm / 5.0)
    soil = weather.soil_moisture if weather.soil_moisture is not None else base_water
    water = _clamp_unit(0.45 * base_water + 0.25 * weather.humidity + 0.20 * soil + 0.10 * rain_signal)
    vegetation = _clamp_unit(0.55 * base_vegetation + 0.25 * water + 0.20 * weather.humidity)
    temperature_quality = _clamp_unit(1.0 - max(abs(weather.temperature_c - 18.0) - 20.0, 0.0) / 60.0)
    food = _clamp_unit((0.55 * base_food + 0.25 * water + 0.20 * vegetation) * (0.70 + 0.30 * temperature_quality))
    defaults: dict[str, object] = {
        "temperature_c": weather.temperature_c,
        "rainfall": rain_signal,
        "humidity": weather.humidity,
        "food_availability": food,
        "water_availability": water,
        "vegetation_cover": vegetation,
        "predator_pressure": 0.25,
        "habitat_type": habitat_type,
        "habitat_area_km2": 1.0,
        "pollution_pressure": 0.05,
        "season": _season_for(location, weather.observed_at),
    }
    defaults.update(overrides)
    return EnvironmentConfig(**defaults)  # type: ignore[arg-type]


def environment_for_place(
    place_name: str,
    habitat_type: HabitatType,
    *,
    fetch_json: JsonFetcher = _fetch_json,
    **overrides: object,
) -> LocationEnvironment:
    """Resolve a place, load current weather, and create editable game inputs."""
    location = search_place(place_name, fetch_json=fetch_json)[0]
    weather = current_weather(location, fetch_json=fetch_json)
    return LocationEnvironment(
        location=location,
        weather=weather,
        environment=environment_from_weather(location, weather, habitat_type, **overrides),
    )


def environment_for_coordinates(
    name: str,
    latitude: float,
    longitude: float,
    habitat_type: HabitatType,
    *,
    fetch_json: JsonFetcher = _fetch_json,
    **overrides: object,
) -> LocationEnvironment:
    """Create game inputs from coordinates clicked on a world map."""
    location = MapLocation(name=name, latitude=latitude, longitude=longitude)
    weather = current_weather(location, fetch_json=fetch_json)
    return LocationEnvironment(
        location=location,
        weather=weather,
        environment=environment_from_weather(location, weather, habitat_type, **overrides),
    )
