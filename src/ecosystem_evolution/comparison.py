"""Run an original environment and player-edited environment side by side."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .environment import EnvironmentConfig
from .simulation import (
    EvolutionSimulator,
    InitialPopulationSpec,
    SimulationConfig,
    SimulationResult,
)
from .species import FOX, SpeciesProfile


@dataclass(frozen=True, slots=True)
class ScenarioComparison:
    """Two reproducible runs that begin with the same population and seed."""

    original_environment: EnvironmentConfig
    changed_environment: EnvironmentConfig
    original_result: SimulationResult
    changed_result: SimulationResult

    def timeline_rows(self) -> list[dict[str, object]]:
        """Small, UI-ready rows for a before/after population chart."""
        return [
            {
                "generation": original.generation,
                "original_population": original.population_size,
                "changed_population": changed.population_size,
                "original_heat_tolerance": (
                    None if original.average_traits is None else original.average_traits["heat_tolerance"]
                ),
                "changed_heat_tolerance": (
                    None if changed.average_traits is None else changed.average_traits["heat_tolerance"]
                ),
            }
            for original, changed in zip(
                self.original_result.generations, self.changed_result.generations, strict=True
            )
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "original_environment": self.original_environment.as_dict(),
            "changed_environment": self.changed_environment.as_dict(),
            "original_result": self.original_result.as_dict(),
            "changed_result": self.changed_result.as_dict(),
        }


def compare_environment_change(
    original_environment: EnvironmentConfig,
    changes: Mapping[str, object],
    *,
    config: SimulationConfig,
    initial_population: InitialPopulationSpec = InitialPopulationSpec(),
    species: SpeciesProfile = FOX,
) -> ScenarioComparison:
    """Compare a map's original resources with the player's edited resources.

    Both runs use the same configuration, starting population, and random seed.
    Differences therefore come from the changed environment inputs rather than a
    different random experiment.
    """
    changed_environment = replace(original_environment, **dict(changes))
    original_result = EvolutionSimulator(original_environment, config, species).run(
        initial_population
    )
    changed_result = EvolutionSimulator(changed_environment, config, species).run(
        initial_population
    )
    return ScenarioComparison(
        original_environment=original_environment,
        changed_environment=changed_environment,
        original_result=original_result,
        changed_result=changed_result,
    )
