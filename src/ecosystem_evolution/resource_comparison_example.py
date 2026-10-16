"""Compare original desert resources with player-improved resources."""

from .comparison import compare_environment_change
from .locations import environment_for_location
from .simulation import InitialPopulationSpec, SimulationConfig
from .species import FENNEC_FOX


def main() -> None:
    # This is the map-selected starting environment: 40 C remains unchanged.
    original = environment_for_location("sonoran_desert", temperature_c=40.0)
    comparison = compare_environment_change(
        original,
        {
            "food_availability": 0.85,
            "water_availability": 0.75,
            "vegetation_cover": 0.60,
            "predator_pressure": 0.05,
            "pollution_pressure": 0.01,
            "habitat_area_km2": 3.0,
        },
        config=SimulationConfig(generations=30, seed=20261016, mutation_probability=0.10),
        initial_population=InitialPopulationSpec(size=30, trait_mean=0.50, trait_variation=0.18),
        species=FENNEC_FOX,
    )
    print("40 C desert: original resources versus player-edited resources")
    for row in comparison.timeline_rows():
        if row["generation"] in (0, 1, 5, 10, 20, 30):
            print(
                f"Generation {row['generation']}: original={row['original_population']}; "
                f"changed={row['changed_population']}"
            )


if __name__ == "__main__":
    main()
