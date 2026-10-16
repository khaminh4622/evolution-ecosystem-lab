"""Follow a fox family in hot weather: adapt, breed, and pass traits on."""

from .lineage import lineage_report
from .locations import environment_for_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig


def main() -> None:
    hot_forest = environment_for_location(
        "pacific_northwest_forest",
        temperature_c=30.0,
        food_availability=0.70,
        water_availability=0.70,
        vegetation_cover=0.60,
        predator_pressure=0.15,
        pollution_pressure=0.03,
        habitat_area_km2=3.0,
    )
    spec = InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18)

    # Most hot-weather runs go extinct, so look at the surviving run with the
    # largest final population and say so rather than hiding the others.
    runs = [
        EvolutionSimulator(
            hot_forest, SimulationConfig(generations=40, seed=seed, mutation_probability=0.10)
        ).run(spec)
        for seed in range(60)
    ]
    survivors = [(seed, run) for seed, run in enumerate(runs) if run.final_population > 0]
    print(f"Hot weather (30 C): {len(survivors)} of {len(runs)} runs survived 40 generations.")
    if not survivors:
        return
    seed, best = max(survivors, key=lambda pair: pair[1].final_population)
    print(f"Showing seed {seed}, the largest surviving population.\n")
    print(lineage_report(best))


if __name__ == "__main__":
    main()
