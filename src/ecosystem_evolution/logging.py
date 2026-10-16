"""Portable experiment records for dashboard, CSV export, and later ML datasets."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .simulation import SimulationResult

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    result: SimulationResult
    schema_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            **self.result.as_dict(),
        }

    def generation_rows(self) -> list[dict[str, object]]:
        """Flatten each generation without losing its scenario-level provenance."""
        environment = self.result.environment.as_dict()
        config = self.result.config.as_dict()
        initial = {
            f"initial_{key}": value
            for key, value in asdict(self.result.initial_population_spec).items()
        }
        rows: list[dict[str, object]] = []
        for record in self.result.generations:
            row: dict[str, object] = {
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "hypothesis": self.hypothesis,
                "species_id": self.result.species.id,
                "species_name": self.result.species.common_name,
                "conservation_status": self.result.species.conservation_status.value,
                "generation": record.generation,
                "population_size": record.population_size,
                "carrying_capacity": record.carrying_capacity,
                "species_suitability": record.species_suitability,
                "survivors": record.survivors,
                "births": record.births,
                "survival_rate": record.survival_rate,
                "reproduction_rate": record.reproduction_rate,
                **environment,
                **config,
                **initial,
            }
            for trait, value in (record.average_traits or {}).items():
                row[f"avg_{trait}"] = value
            for trait, distribution in record.trait_distributions.items():
                row[f"dist_{trait}"] = json.dumps(distribution, separators=(",", ":"))
            rows.append(row)
        return rows

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        rows = self.generation_rows()
        if not rows:
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
