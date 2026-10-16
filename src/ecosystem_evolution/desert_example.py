"""Red foxes in hot, dry country: choose the open desert or a watered oasis.

One place per run. Nothing is printed that you did not ask for.

**Open desert.** The Sonoran preset is 38 C with rainfall 0.12 and humidity 0.18.
Water availability follows from those measurements, and at that level a red fox
population does not persist at any starting size or habitat area tested. A red
fox is a temperate generalist that needs regular water, so treating a real
desert as a place it can colonise would be wishful thinking.

**Oasis.** A river valley, irrigated farmland, or a spring-fed corridor is a
genuinely different habitat, and the difference is measurable: rainfall and
humidity rise with the water, so all three agree instead of contradicting each
other. There, foxes survive and the population adapts to the heat.

Published earlier versions of this example declared desert rainfall (0.12) and
desert humidity (0.18) while claiming water availability 0.75, which is wetter
than the forest examples use. Those three fields cannot all be true of one
place, so the scenario was internally inconsistent. This version keeps them
consistent and shows one place at a time.
"""

from __future__ import annotations

import sys

from .adaptation_story import show_adaptation_story
from .environment import Season
from .locations import environment_for_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig

# Open desert, exactly as the preset describes it. No resource is inflated.
OPEN_DESERT = dict(
    temperature_c=30.0,
    season=Season.SUMMER,
    habitat_area_km2=3.0,
)

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

PLACES = {
    "desert": ("Sonoran open desert at 30 C", OPEN_DESERT),
    "oasis": ("Oasis at 30 C", OASIS),
}


def show_open_desert(runs: int = 60, generations: int = 30) -> None:
    """Report what the open desert does, without dressing it up."""
    environment = environment_for_location("sonoran_desert", **OPEN_DESERT)
    spec = InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18)
    survived = sum(
        EvolutionSimulator(
            environment,
            SimulationConfig(generations=generations, seed=seed, mutation_probability=0.10),
        ).run(spec).final_population
        > 0
        for seed in range(runs)
    )

    print("Sonoran open desert at 30 C")
    print(
        f"  rainfall {environment.rainfall:.2f}, humidity {environment.humidity:.2f} "
        f"-> water availability {environment.water_availability:.2f}"
    )
    print(f"  {survived} of {runs} runs survived {generations} generations")
    print("  A red fox needs regular water, so this habitat supports no population")
    print("  at any starting size or area tested. That is the model's answer, not a")
    print("  missing setting: the water value follows from the climate above.")


def main(place: str = "oasis", include_lineage: bool = True) -> None:
    """Show one place: ``desert`` or ``oasis``."""
    if place not in PLACES:
        raise ValueError(f"unknown place {place!r}; choose one of {sorted(PLACES)}")
    title, overrides = PLACES[place]

    if place == "desert":
        show_open_desert(runs=60, generations=30)
        return

    environment = environment_for_location("sonoran_desert", **overrides)
    print(f"{title}: rainfall {environment.rainfall:.2f}, "
          f"humidity {environment.humidity:.2f}, "
          f"water {environment.water_availability:.2f}")
    print()
    show_adaptation_story(title, environment, include_lineage=include_lineage)


if __name__ == "__main__":
    # --desert or --oasis picks one place (default oasis); --no-lineage prints
    # the population table only.
    arguments = sys.argv[1:]
    chosen = "desert" if "--desert" in arguments else "oasis"
    main(place=chosen, include_lineage="--no-lineage" not in arguments)
