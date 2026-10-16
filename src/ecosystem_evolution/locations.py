"""Curated location presets for classroom ecosystem experiments.

These are transparent starting assumptions, not live measurements or wildlife
management advice. A player can always override a value before running an
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .environment import EnvironmentConfig, HabitatType, Season


@dataclass(frozen=True, slots=True)
class LocationProfile:
    """A named place with the model inputs needed to start an experiment."""

    id: str
    display_name: str
    description: str
    environment: EnvironmentConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "environment": self.environment.as_dict(),
        }


SONORAN_DESERT = LocationProfile(
    id="sonoran_desert",
    display_name="Sonoran Desert — hot, dry scenario",
    description="A classroom desert starting point with scarce water and vegetation.",
    environment=EnvironmentConfig(
        temperature_c=38.0,
        rainfall=0.12,
        humidity=0.18,
        food_availability=0.30,
        water_availability=0.12,
        vegetation_cover=0.18,
        predator_pressure=0.30,
        habitat_type=HabitatType.DESERT,
        habitat_area_km2=1.5,
        pollution_pressure=0.08,
        season=Season.SUMMER,
    ),
)

PACIFIC_NORTHWEST_FOREST = LocationProfile(
    id="pacific_northwest_forest",
    display_name="Pacific Northwest forest — cool, wet scenario",
    description="A classroom forest starting point with plentiful vegetation and water.",
    environment=EnvironmentConfig(
        temperature_c=12.0,
        rainfall=0.75,
        humidity=0.78,
        food_availability=0.68,
        water_availability=0.82,
        vegetation_cover=0.88,
        predator_pressure=0.22,
        habitat_type=HabitatType.FOREST,
        habitat_area_km2=1.5,
        pollution_pressure=0.04,
        season=Season.AUTUMN,
    ),
)

ARCTIC_TUNDRA = LocationProfile(
    id="arctic_tundra",
    display_name="Arctic tundra — cold, sparse scenario",
    description="A classroom tundra starting point with cold temperatures and limited food.",
    environment=EnvironmentConfig(
        temperature_c=-18.0,
        rainfall=0.18,
        humidity=0.55,
        food_availability=0.28,
        water_availability=0.48,
        vegetation_cover=0.22,
        predator_pressure=0.35,
        habitat_type=HabitatType.TUNDRA,
        habitat_area_km2=1.5,
        pollution_pressure=0.03,
        season=Season.WINTER,
    ),
)

LOCATION_CATALOG: dict[str, LocationProfile] = {
    profile.id: profile
    for profile in (SONORAN_DESERT, PACIFIC_NORTHWEST_FOREST, ARCTIC_TUNDRA)
}


def get_location(location_id: str) -> LocationProfile:
    """Look up a classroom location preset by its stable identifier."""
    try:
        return LOCATION_CATALOG[location_id]
    except KeyError as error:
        available = ", ".join(sorted(LOCATION_CATALOG))
        raise ValueError(f"unknown location {location_id!r}; choose one of {available}") from error


def environment_for_location(
    location_id: str, **overrides: object
) -> EnvironmentConfig:
    """Return a location's assumptions, optionally adjusted by the player."""
    return replace(get_location(location_id).environment, **overrides)
