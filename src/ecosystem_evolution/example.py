"""Create a reproducible sample experiment for a demo or dashboard prototype."""

from pathlib import Path

from .environment import EnvironmentConfig, HabitatType, Season
from .logging import ExperimentRecord
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig


def main() -> None:
    result = EvolutionSimulator(
        EnvironmentConfig(
            temperature_c=2.0,
            rainfall=0.65,
            humidity=0.70,
            food_availability=0.55,
            water_availability=0.70,
            vegetation_cover=0.75,
            predator_pressure=0.25,
            habitat_type=HabitatType.FOREST,
            habitat_area_km2=1.5,
            pollution_pressure=0.05,
            season=Season.WINTER,
        ),
        SimulationConfig(generations=10, seed=20261016),
    ).run(InitialPopulationSpec(size=30, trait_mean=0.5, trait_variation=0.18))
    experiment = ExperimentRecord(
        experiment_id="winter-forest-demo",
        hypothesis="Cold conditions will favor foxes with lower heat tolerance.",
        result=result,
    )
    output = Path("data/generated")
    experiment.write_json(output / "winter-forest-demo.json")
    experiment.write_csv(output / "winter-forest-demo.csv")
    print(f"Final population: {result.final_population}")


if __name__ == "__main__":
    main()
