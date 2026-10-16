"""Follow two desert-adapted fox founders through a 40 C desert future."""
from .genetics import Genome
from .locations import environment_for_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from .species import FENNEC_FOX


def main() -> None:
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
    # Two founders fit this model's hot-desert trait targets particularly well.
    desert_adapted = Genome(
        fur_insulation=0.08,
        heat_tolerance=0.97,
        water_efficiency=0.46,
        camouflage=0.82,
        body_size=0.46,
    )
    result = EvolutionSimulator(
        desert,
        SimulationConfig(generations=30, seed=20261016, mutation_probability=0.10),
        species=FENNEC_FOX,
    ).run(
        InitialPopulationSpec(
            size=30,
            trait_mean=0.50,
            trait_variation=0.18,
            founder_genomes=(desert_adapted, desert_adapted),
        )
    )

    print("Two desert-adapted Fennec founders in a 40 C desert")
    for generation in (0, 1, 5, 10, 20, 30):
        record = result.generations[generation]
        print(
            f"Generation {generation}: population={record.population_size}; "
            f"average heat tolerance="
            f"{None if record.average_traits is None else round(record.average_traits['heat_tolerance'], 3)}"
        )


if __name__ == "__main__":
    main()
