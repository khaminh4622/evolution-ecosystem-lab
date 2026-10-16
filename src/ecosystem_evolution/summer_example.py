"""Red foxes in a 30 C summer forest: the warm-weather counterpart to winter.

Same Pacific Northwest preset and the same generous food and water as the 0 C
winter example, with the temperature raised to 30 C in summer. Summer shifts the
camouflage target slightly up. The cold of the winter run selects for thick fur;
here the heat does the opposite, pushing fur insulation down and heat tolerance
up, and the population still grows back to capacity.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import Season
from .locations import environment_for_location


def main(include_lineage: bool = True) -> None:
    summer_forest = environment_for_location(
        "pacific_northwest_forest",
        temperature_c=30.0,
        season=Season.SUMMER,
        food_availability=0.70,
        water_availability=0.70,
        vegetation_cover=0.60,
        predator_pressure=0.15,
        pollution_pressure=0.03,
        habitat_area_km2=3.0,
    )
    show_adaptation_story(
        "Summer forest at 30 C", summer_forest, include_lineage=include_lineage
    )


if __name__ == "__main__":
    # --no-lineage prints the population table only.
    main(include_lineage="--no-lineage" not in sys.argv[1:])
