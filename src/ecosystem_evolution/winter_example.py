"""Red foxes in a 0 C winter forest: adapt, breed, and grow back toward capacity.

Winter only nudges the camouflage target slightly in this model; the cold
itself (0 C) is what pushes fur insulation up. Food and water are set to
generous values, so the cold is the main challenge.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import Season
from .locations import environment_for_location


def main(include_lineage: bool = True) -> None:
    winter_forest = environment_for_location(
        "pacific_northwest_forest",
        temperature_c=0.0,
        season=Season.WINTER,
        food_availability=0.70,
        water_availability=0.70,
        vegetation_cover=0.60,
        predator_pressure=0.15,
        pollution_pressure=0.03,
        habitat_area_km2=3.0,
    )
    show_adaptation_story(
        "Winter forest at 0 C", winter_forest, include_lineage=include_lineage
    )


if __name__ == "__main__":
    # --no-lineage prints the population table only.
    main(include_lineage="--no-lineage" not in sys.argv[1:])
