"""Genome value objects and two-parent particulate inheritance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Iterable


class Trait(str, Enum):
    FUR_INSULATION = "fur_insulation"
    HEAT_TOLERANCE = "heat_tolerance"
    WATER_EFFICIENCY = "water_efficiency"
    CAMOUFLAGE = "camouflage"
    BODY_SIZE = "body_size"


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class Genome:
    fur_insulation: float
    heat_tolerance: float
    water_efficiency: float
    camouflage: float
    body_size: float

    def __post_init__(self) -> None:
        for trait in Trait:
            value = getattr(self, trait.value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{trait.value} must be between 0 and 1; got {value!r}")

    def value(self, trait: Trait) -> float:
        return getattr(self, trait.value)

    def as_dict(self) -> dict[str, float]:
        return {trait.value: self.value(trait) for trait in Trait}

    @classmethod
    def from_values(cls, values: dict[str, float]) -> "Genome":
        return cls(**{trait.value: float(values[trait.value]) for trait in Trait})

    @classmethod
    def random(cls, rng: random.Random, mean: float, variation: float) -> "Genome":
        return cls(
            **{
                trait.value: clamp_unit(rng.gauss(mean, variation))
                for trait in Trait
            }
        )


@dataclass(frozen=True, slots=True)
class Animal:
    id: str
    genome: Genome
    parent_ids: tuple[str, str] | tuple[()] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "parent_ids": list(self.parent_ids),
            "genome": self.genome.as_dict(),
        }


def inherit_genome(
    parent_a: Genome,
    parent_b: Genome,
    rng: random.Random,
    mutation_probability: float,
    mutation_magnitude: float,
) -> Genome:
    """Copy each trait from one randomly chosen parent, then mutate and clamp.

    Traits segregate independently, like separate loci. Unlike averaging
    ("blending") the parents, this keeps variation in the population from
    halving every generation, so selection always has something to act on.
    """
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("mutation_probability must be between 0 and 1")
    if not 0.0 <= mutation_magnitude <= 1.0:
        raise ValueError("mutation_magnitude must be between 0 and 1")

    child: dict[str, float] = {}
    for trait in Trait:
        donor = parent_a if rng.random() < 0.5 else parent_b
        value = donor.value(trait)
        if rng.random() < mutation_probability:
            value += rng.uniform(-mutation_magnitude, mutation_magnitude)
        child[trait.value] = clamp_unit(value)
    return Genome.from_values(child)


def mean_genome(animals: Iterable[Animal]) -> dict[str, float] | None:
    population = list(animals)
    if not population:
        return None
    return {
        trait.value: sum(animal.genome.value(trait) for animal in population)
        / len(population)
        for trait in Trait
    }
