"""Teaching profiles for threatened wildlife scenarios.

Conservation categories are catalog metadata, not outputs from the simulator. The
environment preferences below are deliberately broad educational parameters; they
must not be used as species-management advice or population forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .environment import EnvironmentConfig, HabitatType


class ConservationStatus(str, Enum):
    CRITICALLY_ENDANGERED = "Critically Endangered"
    ENDANGERED = "Endangered"
    NOT_ASSESSED = "Teaching baseline (not a conservation assessment)"


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class SpeciesProfile:
    """A transparent scenario profile used to vary weather sensitivity by species."""

    id: str
    common_name: str
    scientific_name: str
    conservation_status: ConservationStatus
    primary_habitat: HabitatType
    preferred_temperature_c: float
    temperature_tolerance_c: float
    preferred_rainfall: float
    preferred_humidity: float
    preferred_food: float
    preferred_water: float
    preferred_vegetation: float
    capacity_modifier: float
    lesson_note: str

    def suitability(self, environment: EnvironmentConfig) -> float:
        """Return a 0–1 instructional suitability score for this profile.

        Weather is intentionally influential: temperature is 45% of the score;
        rainfall and humidity together are 35%. Resources are the other 20%.
        This transparent simplification makes the conditions teachable, rather
        than claiming a validated species-distribution model.
        """
        temperature_match = _clamp_unit(
            1.0
            - abs(environment.temperature_c - self.preferred_temperature_c)
            / self.temperature_tolerance_c
        )
        rainfall_match = 1.0 - abs(environment.rainfall - self.preferred_rainfall)
        humidity_match = 1.0 - abs(environment.humidity - self.preferred_humidity)
        food_match = 1.0 - abs(environment.food_availability - self.preferred_food)
        water_match = 1.0 - abs(environment.water_availability - self.preferred_water)
        vegetation_match = 1.0 - abs(
            environment.vegetation_cover - self.preferred_vegetation
        )
        habitat_match = 1.0 if environment.habitat_type == self.primary_habitat else 0.70
        return _clamp_unit(
            0.45 * temperature_match
            + 0.20 * rainfall_match
            + 0.15 * humidity_match
            + 0.07 * food_match
            + 0.05 * water_match
            + 0.05 * vegetation_match
            + 0.03 * habitat_match
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "conservation_status": self.conservation_status.value,
            "primary_habitat": self.primary_habitat.value,
            "preferred_temperature_c": self.preferred_temperature_c,
            "temperature_tolerance_c": self.temperature_tolerance_c,
            "preferred_rainfall": self.preferred_rainfall,
            "preferred_humidity": self.preferred_humidity,
            "preferred_food": self.preferred_food,
            "preferred_water": self.preferred_water,
            "preferred_vegetation": self.preferred_vegetation,
            "capacity_modifier": self.capacity_modifier,
            "lesson_note": self.lesson_note,
        }


FOX = SpeciesProfile(
    "fox", "Fox (teaching baseline)", "Vulpes vulpes", ConservationStatus.NOT_ASSESSED,
    HabitatType.FOREST, 15, 22, .55, .55, .60, .60, .60, 1.0,
    "A generalist baseline used to introduce inherited variation and selection.",
)
FENNEC_FOX = SpeciesProfile(
    "fennec_fox", "Fennec Fox (teaching profile)", "Vulpes zerda", ConservationStatus.NOT_ASSESSED,
    HabitatType.DESERT, 35, 18, .15, .25, .55, .45, .30, .85,
    "A desert-adapted fox profile for comparing survival and inherited variation in hot, dry scenarios.",
)
AMUR_LEOPARD = SpeciesProfile(
    "amur_leopard", "Amur Leopard", "Panthera pardus orientalis", ConservationStatus.CRITICALLY_ENDANGERED,
    HabitatType.FOREST, 5, 18, .55, .55, .70, .60, .80, .28,
    "Warm, dry conditions reduce suitability in this cool forest scenario.",
)
SUMATRAN_ORANGUTAN = SpeciesProfile(
    "sumatran_orangutan", "Sumatran Orangutan", "Pongo abelii", ConservationStatus.CRITICALLY_ENDANGERED,
    HabitatType.FOREST, 26, 9, .85, .90, .80, .85, .95, .20,
    "Heat alone is not enough: high humidity, rainfall, forest cover, and food support the scenario.",
)
AFRICAN_FOREST_ELEPHANT = SpeciesProfile(
    "african_forest_elephant", "African Forest Elephant", "Loxodonta cyclotis", ConservationStatus.CRITICALLY_ENDANGERED,
    HabitatType.FOREST, 25, 11, .78, .78, .82, .82, .88, .12,
    "Large animals receive a lower capacity modifier because the same habitat supports fewer individuals.",
)
HAWKSBILL_TURTLE = SpeciesProfile(
    "hawksbill_turtle", "Hawksbill Turtle", "Eretmochelys imbricata", ConservationStatus.CRITICALLY_ENDANGERED,
    HabitatType.CORAL_REEF, 27, 6, .65, .75, .65, 1.00, .70, .35,
    "Warm coral-reef conditions and water quality are simplified through the habitat, temperature, and pollution controls.",
)
VAQUITA = SpeciesProfile(
    "vaquita", "Vaquita", "Phocoena sinus", ConservationStatus.CRITICALLY_ENDANGERED,
    HabitatType.COASTAL, 20, 8, .20, .65, .55, 1.00, .10, .16,
    "A coastal scenario where pollution and predator-pressure controls stand in for several human-caused pressures.",
)
RED_PANDA = SpeciesProfile(
    "red_panda", "Red Panda", "Ailurus fulgens", ConservationStatus.ENDANGERED,
    HabitatType.FOREST, 14, 9, .72, .75, .70, .70, .90, .42,
    "Cool, humid forests with dense vegetation produce the highest scenario suitability.",
)
BORNEAN_ELEPHANT = SpeciesProfile(
    "bornean_elephant", "Bornean Elephant", "Elephas maximus borneensis", ConservationStatus.ENDANGERED,
    HabitatType.FOREST, 26, 10, .80, .85, .82, .85, .90, .14,
    "A tropical forest scenario where drought or forest-cover loss lowers suitability.",
)

SPECIES_CATALOG: dict[str, SpeciesProfile] = {
    profile.id: profile
    for profile in (
        FOX,
        FENNEC_FOX,
        AMUR_LEOPARD,
        SUMATRAN_ORANGUTAN,
        AFRICAN_FOREST_ELEPHANT,
        HAWKSBILL_TURTLE,
        VAQUITA,
        RED_PANDA,
        BORNEAN_ELEPHANT,
    )
}


def get_species(species_id: str) -> SpeciesProfile:
    try:
        return SPECIES_CATALOG[species_id]
    except KeyError as error:
        available = ", ".join(sorted(SPECIES_CATALOG))
        raise ValueError(f"unknown species {species_id!r}; choose one of {available}") from error
