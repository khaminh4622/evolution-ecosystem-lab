"""Start a fox experiment from a real map-search and weather snapshot.

The map and weather give the temperature, humidity, rainfall, and soil moisture.
Habitat and the resource estimates stay player choices, and the experiment runs
the same population table as the winter and summer examples.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import HabitatType
from .location_data import environment_for_place
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig

PLACE = "Phoenix, Arizona"
HABITAT = HabitatType.DESERT


def build_environment():
    """Resolve the place, load its current weather, and make editable inputs.

    Every resource below is a reviewed estimate. Water, vegetation, and food
    start from the desert baseline unless they are set here, and the default
    habitat area is small: it decides the carrying capacity, so a small value
    caps the population however well the foxes adapt.

    Phoenix in July is 31 C and about 12% humidity, which is harsh for the
    teaching baseline fox. A rich, well-watered corridor is needed for the
    population to hold: at this weather, a poor habitat drives the crowding term
    of the survival equation through its floor and every fox dies in the first
    generation. Open desert at the weather defaults supports only about four
    foxes and never survives.
    """
    return environment_for_place(
        PLACE,
        HABITAT,
        food_availability=0.85,
        water_availability=0.85,
        vegetation_cover=0.70,
        predator_pressure=0.05,
        pollution_pressure=0.01,
        habitat_area_km2=5.0,
    )


def main(include_lineage: bool = True) -> None:
    selected = build_environment()
    print(f"Map selection: {selected.location.name}, {selected.location.country}")
    print(
        f"Weather at selection: {selected.weather.temperature_c:.1f} C, "
        f"humidity {selected.weather.humidity:.0%}, "
        f"rainfall {selected.weather.precipitation_mm:.1f} mm"
    )
    print()

    show_adaptation_story(
        f"{selected.location.name} at {selected.weather.temperature_c:.0f} C",
        selected.environment,
        generations=30,
        include_lineage=include_lineage,
    )


def check_capacity() -> str:
    """Report a start size the habitat can hold, without running 60 seeds.

    Dropping a large group into a tiny habitat drives the crowding term of the
    survival equation through the 0.01 floor and every fox dies in one
    generation, which the population table alone does not explain.
    """
    selected = build_environment()
    environment = selected.environment
    carrying_capacity = EvolutionSimulator(
        environment, SimulationConfig(generations=1)
    ).carrying_capacity()
    start_size = InitialPopulationSpec().size
    report = [
        f"{selected.location.name}: current weather "
        f"{selected.weather.temperature_c:.1f} C",
        f"  water {environment.water_availability:.2f}, "
        f"vegetation {environment.vegetation_cover:.2f}, "
        f"food {environment.food_availability:.2f}, "
        f"area {environment.habitat_area_km2:.1f} km2",
        f"  carrying capacity {carrying_capacity} for {start_size} starting foxes"
        + ("  (over capacity: expect extinction)" if start_size > carrying_capacity else ""),
    ]
    return "\n".join(report)


if __name__ == "__main__":
    # --no-lineage prints the population table only; --check prints the capacity
    # report and runs no simulation.
    if "--check" in sys.argv[1:]:
        print(check_capacity())
    else:
        main(include_lineage="--no-lineage" not in sys.argv[1:])
