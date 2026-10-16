"""Validated environment controls and transparent trait-target equations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping


class HabitatType(str, Enum):
    FOREST = "forest"
    GRASSLAND = "grassland"
    DESERT = "desert"
    TUNDRA = "tundra"
    WETLAND = "wetland"
    COASTAL = "coastal"
    CORAL_REEF = "coral_reef"


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {value!r}")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Controls exposed to students, with bounded values for safe simulation."""

    temperature_c: float = 15.0
    rainfall: float = 0.55
    humidity: float = 0.55
    food_availability: float = 0.60
    water_availability: float = 0.60
    vegetation_cover: float = 0.60
    predator_pressure: float = 0.25
    habitat_type: HabitatType = HabitatType.FOREST
    habitat_area_km2: float = 1.0
    pollution_pressure: float = 0.05
    season: Season = Season.SPRING

    def __post_init__(self) -> None:
        if not -50.0 <= self.temperature_c <= 60.0:
            raise ValueError("temperature_c must be between -50 and 60 Celsius")
        if not 0.1 <= self.habitat_area_km2 <= 100.0:
            raise ValueError("habitat_area_km2 must be between 0.1 and 100")
        for name in (
            "rainfall",
            "humidity",
            "food_availability",
            "water_availability",
            "vegetation_cover",
            "predator_pressure",
            "pollution_pressure",
        ):
            _unit(name, getattr(self, name))
        if not isinstance(self.habitat_type, HabitatType):
            raise ValueError("habitat_type must be a HabitatType")
        if not isinstance(self.season, Season):
            raise ValueError("season must be a Season")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["habitat_type"] = self.habitat_type.value
        result["season"] = self.season.value
        return result

    @property
    def resource_quality(self) -> float:
        """Food and water opportunity, intentionally independent of individual genes."""
        return (
            0.45 * self.food_availability
            + 0.30 * self.water_availability
            + 0.15 * self.vegetation_cover
            + 0.10 * self.rainfall
        )

    @property
    def temperature_comfort(self) -> float:
        return 1.0 - min(abs(self.temperature_c - 15.0) / 55.0, 1.0)

    @property
    def heat_target(self) -> float:
        """The heat tolerance this place favors, from its temperature alone."""
        return _sigmoid((self.temperature_c - 16.0) / 10.0)

    def trait_targets(self) -> Mapping[str, float]:
        """Return the documented target values used only to calculate fitness."""
        heat_target = self.heat_target
        habitat_camouflage = {
            HabitatType.FOREST: 0.70,
            HabitatType.GRASSLAND: 0.45,
            HabitatType.DESERT: 0.78,
            HabitatType.TUNDRA: 0.22,
            HabitatType.WETLAND: 0.55,
            HabitatType.COASTAL: 0.45,
            HabitatType.CORAL_REEF: 0.68,
        }[self.habitat_type]
        season_shift = {
            Season.SPRING: 0.00,
            Season.SUMMER: 0.04,
            Season.AUTUMN: -0.02,
            Season.WINTER: -0.06,
        }[self.season]
        return {
            "fur_insulation": _sigmoid((12.0 - self.temperature_c) / 10.0),
            "heat_tolerance": heat_target,
            "water_efficiency": _clamp(
                1.0
                - (
                    0.65 * self.water_availability
                    + 0.20 * self.rainfall
                    + 0.15 * self.humidity
                )
            ),
            "camouflage": _clamp(habitat_camouflage + season_shift),
            "body_size": _clamp(
                0.35 + 0.35 * self.food_availability - 0.20 * heat_target
            ),
        }


