"""Start a fox experiment from a real map-search and weather snapshot.

The map and weather give the temperature, humidity, rainfall, and soil moisture.
Those four values feed the resource formulas in ``location_data``, so water,
vegetation, and food are *estimated from the real weather* rather than invented.
Habitat type and habitat area are the two things a weather snapshot cannot tell
us, so they remain your choices, and every value stays editable.

The result for Phoenix in July is extinction, and that is the honest answer
rather than a bug. The city is 30 C with about 13% humidity and no rain, so the
derived water availability is near 0.11. A red fox is a temperate generalist;
at that water level no starting group persists, at any size or habitat area
tested. Changing the species to the desert profile does not save it either,
because this model's suitability and survival equations still weigh the weather
heavily. A place that cannot support a species is a valid lesson, so the example
reports it plainly instead of inflating the water to force a survivors' table.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import HabitatType
from .location_data import environment_for_place
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig

PLACE = "Phoenix, Arizona"
HABITAT = HabitatType.DESERT

# The desert baseline supports very few foxes here, so the starting group is
# small enough for the habitat to hold. Starting more foxes than the carrying
# capacity drives the crowding term of the survival equation through its floor
# and kills every fox in the first generation; the population table cannot
# explain that, so this example avoids the trap instead of hiding it.
START_SIZE = 4


def build_environment():
    """Resolve the place, load its current weather, and derive editable inputs.

    No resource is overridden. Water, vegetation, and food come from the
    documented formulas applied to the real weather, which for Phoenix in July
    means a genuinely dry habitat (water near 0.1). Only habitat type and area
    are chosen, because a weather snapshot cannot measure either.
    """
    return environment_for_place(
        PLACE,
        HABITAT,
        habitat_area_km2=1.0,
    )


def main(include_lineage: bool = True) -> None:
    selected = build_environment()
    environment = selected.environment
    print(f"Map selection: {selected.location.name}, {selected.location.country}")
    print(
        f"Weather at selection: {selected.weather.temperature_c:.1f} C, "
        f"humidity {selected.weather.humidity:.0%}, "
        f"rainfall {selected.weather.precipitation_mm:.1f} mm"
    )
    print(
        f"Derived from that weather: water {environment.water_availability:.2f}, "
        f"vegetation {environment.vegetation_cover:.2f}, "
        f"food {environment.food_availability:.2f}"
    )
    print(
        "  These follow from the real measurements. Habitat type and area are"
        " choices,\n  and predator pressure and pollution are estimates that"
        " should be reviewed."
    )
    print()

    show_adaptation_story(
        f"{selected.location.name} at {selected.weather.temperature_c:.0f} C",
        selected.environment,
        generations=30,
        start_size=START_SIZE,
        include_lineage=include_lineage,
    )

    carrying_capacity = EvolutionSimulator(
        environment, SimulationConfig(generations=1)
    ).carrying_capacity()
    print()
    print("Why no foxes were left:")
    print(
        f"  Water availability of {environment.water_availability:.2f} is the"
        " constraint. A red fox\n  is a temperate generalist, and this habitat"
        f" holds only about {carrying_capacity} foxes even\n  before selection."
        " Adjust the resources, the habitat area, or start from a"
    )
    print(
        "  wetter place with the same map bridge; the derived values above are"
        " estimates\n  you are free to review and change."
    )


def check_capacity() -> str:
    """Report how many foxes this habitat can hold, and flag an over-large start.

    Dropping a large group into a tiny habitat drives the crowding term of the
    survival equation through the 0.01 floor and every fox dies in one
    generation, which the population table alone does not explain.
    """
    selected = build_environment()
    environment = selected.environment
    carrying_capacity = EvolutionSimulator(
        environment, SimulationConfig(generations=1)
    ).carrying_capacity()
    start_size = START_SIZE
    report = [
        f"{selected.location.name}: current weather "
        f"{selected.weather.temperature_c:.1f} C, "
        f"humidity {selected.weather.humidity:.0%}, "
        f"rainfall {selected.weather.precipitation_mm:.1f} mm",
        f"  derived water {environment.water_availability:.2f}, "
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
