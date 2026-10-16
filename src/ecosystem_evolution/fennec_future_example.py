"""Show a desert-adapted fox population over 30 simulated generations."""

from .locations import environment_for_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from .species import FENNEC_FOX


def main() -> None:
    # Temperature stays at 40 C. The player improves other survivable conditions.
    desert = environment_for_location(
        "sonoran_desert",
        temperature_c=40.0,
        food_availability=0.85,
        water_availability=0.75,
        vegetation_cover=0.60,
        predator_pressure=0.05,
        pollution_pressure=0.01,
        habitat_area_km2=3.0,
    )
    result = EvolutionSimulator(
        desert,
        SimulationConfig(generations=30, seed=20261016, mutation_probability=0.10),
        species=FENNEC_FOX,
    ).run(InitialPopulationSpec(size=30, trait_mean=0.50, trait_variation=0.18))

    print("Fennec fox future: 40 C desert scenario")
    for generation in (0, 1, 5, 10, 20, 30):
        record = result.generations[generation]
        heat = None if record.average_traits is None else record.average_traits["heat_tolerance"]
        print(
            f"Generation {generation}: population={record.population_size}; "
            f"average heat tolerance={heat if heat is None else round(heat, 3)}"
        )


if __name__ == "__main__":
    main()
