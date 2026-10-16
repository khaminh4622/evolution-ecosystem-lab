"""Red foxes in a 30 C desert: adapt, breed, and grow back toward capacity.

The built-in desert preset is very scarce (it supports about 6 foxes), so every
run dies out there. This example gives the player the same kind of improvements
as the fennec example: more food and water, and fewer predators.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .locations import environment_for_location


def main(include_lineage: bool = True) -> None:
    desert = environment_for_location(
        "sonoran_desert",
        temperature_c=30.0,
        food_availability=0.85,
        water_availability=0.75,
        vegetation_cover=0.60,
        predator_pressure=0.05,
        pollution_pressure=0.01,
        habitat_area_km2=3.0,
    )
    show_adaptation_story("Desert at 30 C", desert, include_lineage=include_lineage)


if __name__ == "__main__":
    # --no-lineage prints the population table only.
    main(include_lineage="--no-lineage" not in sys.argv[1:])
