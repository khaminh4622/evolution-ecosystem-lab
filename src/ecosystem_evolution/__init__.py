"""Headless, reproducible evolution simulation for the ecosystem lab."""

from .comparison import ScenarioComparison, compare_environment_change
from .environment import EnvironmentConfig, HabitatType, Season
from .genetics import Animal, Genome, Trait
from .location_data import (
    LocationDataError,
    LocationEnvironment,
    MapLocation,
    WeatherObservation,
    environment_for_coordinates,
    environment_for_place,
    environment_from_weather,
    search_place,
)
from .locations import LOCATION_CATALOG, LocationProfile, environment_for_location, get_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from .species import FENNEC_FOX, SPECIES_CATALOG, SpeciesProfile, get_species

__all__ = [
    "Animal",
    "EnvironmentConfig",
    "EvolutionSimulator",
    "FENNEC_FOX",
    "Genome",
    "HabitatType",
    "InitialPopulationSpec",
    "LOCATION_CATALOG",
    "LocationProfile",
    "LocationDataError",
    "LocationEnvironment",
    "MapLocation",
    "Season",
    "ScenarioComparison",
    "SimulationConfig",
    "SPECIES_CATALOG",
    "SpeciesProfile",
    "Trait",
    "WeatherObservation",
    "environment_for_coordinates",
    "environment_for_place",
    "environment_for_location",
    "environment_from_weather",
    "compare_environment_change",
    "get_location",
    "get_species",
    "search_place",
]
