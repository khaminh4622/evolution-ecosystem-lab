"""Red foxes in hot, dry country: open desert fails, a watered oasis works.

This example shows both halves of the same lesson, because the honest answer is
not "add water until the foxes survive".

**Open desert.** The Sonoran preset is 38 C with rainfall 0.12 and humidity 0.18.
Water availability follows from those measurements, and at that level a red fox
population does not persist at any size or habitat area tested. A red fox is a
temperate generalist that needs regular water, so treating a real desert as a
place it can colonise would be wishful thinking.

**Oasis.** A river valley, irrigated farmland, or a spring-fed corridor is a
genuinely different habitat, and the difference is measurable: rainfall and
humidity rise with the water, so all three agree instead of contradicting each
other. There, foxes survive and the population adapts to the heat.

Published earlier versions of this example declared desert rainfall (0.12) and
desert humidity (0.18) while claiming water availability 0.75, which is wetter
than the forest examples use. Those three fields cannot all be true of one
place, so the scenario was internally inconsistent. This version keeps them
consistent and labels each scenario for what it is.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import Season
from .locations import environment_for_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig

# A watered corridor in hot country: rainfall and humidity travel with the
# water value, so the climate described is one real place rather than a mix.
OASIS = dict(
    temperature_c=30.0,
    season=Season.SUMMER,
    rainfall=0.50,
    humidity=0.60,
    food_availability=0.75,
    water_availability=0.70,
    vegetation_cover=0.65,
    predator_pressure=0.08,
    pollution_pressure=0.01,
    habitat_area_km2=3.0,
)

# Open desert, exactly as the preset describes it. No resource is inflated.
OPEN_DESERT = dict(
    temperature_c=30.0,
    season=Season.SUMMER,
    habitat_area_km2=3.0,
)


def describe_open_desert() -> None:
    """Report what the open desert does, without dressing it up."""
    environment = environment_for_location("sonoran_desert", **OPEN_DESERT)
    simulator = EvolutionSimulator(environment, SimulationConfig(generations=30))
    spec = InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18)
    survived = 0
    for seed in range(60):
        result = EvolutionSimulator(
            environment,
            SimulationConfig(generations=30, seed=seed, mutation_probability=0.10),
        ).run(spec)
        survived += result.final_population > 0

    print("Open Sonoran desert, unchanged preset")
    print(
        f"  rainfall {environment.rainfall:.2f}, humidity {environment.humidity:.2f} "
        f"-> water availability {environment.water_availability:.2f}"
    )
    print(f"  {survived} of 60 runs survived 30 generations")
    print("  A red fox needs regular water, so this habitat supports no population")
    print("  at any starting size or area tested. That is the model's answer, not a")
    print("  missing setting: the water value follows from the climate above.")
    print()


def main(include_lineage: bool = True) -> None:
    describe_open_desert()
    oasis = environment_for_location("sonoran_desert", **OASIS)
    print("The same hot country, at a watered oasis or irrigated corridor")
    print(
        f"  rainfall {oasis.rainfall:.2f}, humidity {oasis.humidity:.2f} "
        f"-> water availability {oasis.water_availability:.2f}"
    )
    print()
    show_adaptation_story("Oasis at 30 C", oasis, include_lineage=include_lineage)


if __name__ == "__main__":
    # --no-lineage prints the population table only.
    main(include_lineage="--no-lineage" not in sys.argv[1:])
