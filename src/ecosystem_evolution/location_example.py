"""Run a named-location experiment with editable environmental assumptions."""

from .locations import environment_for_location, get_location
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig


def main() -> None:
    location = get_location("sonoran_desert")
    # A player can change a preset before starting the experiment.
    environment = environment_for_location("sonoran_desert", food_availability=0.40)
    result = EvolutionSimulator(
        environment,
        SimulationConfig(generations=30, seed=20261016),
    ).run(InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18))

    print(location.display_name)
    print(f"Final population after 30 generations: {result.final_population}")


if __name__ == "__main__":
    main()
