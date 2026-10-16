"""Deterministic generational selection model, independent from any renderer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable

from .environment import EnvironmentConfig
from .genetics import Animal, Genome, Trait, inherit_genome, mean_genome
from .species import FOX, SpeciesProfile


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class InitialPopulationSpec:
    size: int = 30
    trait_mean: float = 0.50
    trait_variation: float = 0.18
    founder_genomes: tuple[Genome, ...] = ()

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("initial population size must be at least 1")
        if not 0.0 <= self.trait_mean <= 1.0:
            raise ValueError("trait_mean must be between 0 and 1")
        if not 0.0 <= self.trait_variation <= 1.0:
            raise ValueError("trait_variation must be between 0 and 1")
        if len(self.founder_genomes) > self.size:
            raise ValueError("founder_genomes cannot contain more foxes than size")
        if not all(isinstance(genome, Genome) for genome in self.founder_genomes):
            raise ValueError("founder_genomes must contain Genome values")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    generations: int = 10
    seed: int | None = None
    mutation_probability: float = 0.05
    mutation_magnitude: float = 0.08
    base_carrying_capacity_per_km2: float = 80.0
    minimum_carrying_capacity: int = 4
    adaptation_benefit: float = 0.5

    def __post_init__(self) -> None:
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 0.0 <= self.mutation_magnitude <= 1.0:
            raise ValueError("mutation_magnitude must be between 0 and 1")
        if self.base_carrying_capacity_per_km2 <= 0:
            raise ValueError("base_carrying_capacity_per_km2 must be positive")
        if self.minimum_carrying_capacity < 1:
            raise ValueError("minimum_carrying_capacity must be at least 1")
        if not 0.0 <= self.adaptation_benefit <= 1.0:
            raise ValueError("adaptation_benefit must be between 0 and 1")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    population_size: int
    carrying_capacity: int
    species_suitability: float
    survivors: int | None
    births: int | None
    survival_rate: float | None
    reproduction_rate: float | None
    average_traits: dict[str, float] | None
    trait_distributions: dict[str, list[int]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PopulationSnapshot:
    generation: int
    animals: tuple[Animal, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "animals": [animal.as_dict() for animal in self.animals],
        }


@dataclass(frozen=True, slots=True)
class SimulationResult:
    environment: EnvironmentConfig
    config: SimulationConfig
    species: SpeciesProfile
    initial_population_spec: InitialPopulationSpec
    generations: tuple[GenerationRecord, ...]
    replay: tuple[PopulationSnapshot, ...]

    @property
    def final_population(self) -> int:
        return self.generations[-1].population_size

    def as_dict(self) -> dict[str, object]:
        return {
            "species": self.species.as_dict(),
            "environment": self.environment.as_dict(),
            "simulation_config": self.config.as_dict(),
            "initial_population_spec": asdict(self.initial_population_spec),
            "results": [record.as_dict() for record in self.generations],
            "replay": [snapshot.as_dict() for snapshot in self.replay],
        }


class EvolutionSimulator:
    """A seeded, single-species simulation that can run without Unity or an API."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        config: SimulationConfig,
        species: SpeciesProfile = FOX,
    ) -> None:
        self.environment = environment
        self.config = config
        self.species = species
        self._rng = random.Random(config.seed)
        self._animal_sequence = 0

    def run(self, initial: InitialPopulationSpec = InitialPopulationSpec()) -> SimulationResult:
        """Run generation zero plus the requested number of selection steps."""
        # Reset lets one simulator object reproduce the same complete experiment on repeat.
        self._rng = random.Random(self.config.seed)
        self._animal_sequence = 0
        population = self.create_initial_population(initial)
        capacity = self.carrying_capacity()
        records = [self._record_baseline(population, capacity)]
        replay = [PopulationSnapshot(0, tuple(population))]

        for generation in range(1, self.config.generations + 1):
            population, record = self._advance_generation(population, generation, capacity)
            records.append(record)
            replay.append(PopulationSnapshot(generation, tuple(population)))

        return SimulationResult(
            environment=self.environment,
            config=self.config,
            species=self.species,
            initial_population_spec=initial,
            generations=tuple(records),
            replay=tuple(replay),
        )

    def create_initial_population(self, spec: InitialPopulationSpec) -> list[Animal]:
        special_founders = [
            Animal(id=self._next_id(), genome=genome)
            for genome in spec.founder_genomes
        ]
        random_founders = [
            Animal(
                id=self._next_id(),
                genome=Genome.random(self._rng, spec.trait_mean, spec.trait_variation),
            )
            for _ in range(spec.size - len(special_founders))
        ]
        return special_founders + random_founders

    def carrying_capacity(self) -> int:
        env = self.environment
        scaled_capacity = (
            self.config.base_carrying_capacity_per_km2
            * env.habitat_area_km2
            * (0.20 + 0.80 * env.food_availability)
            * (0.20 + 0.80 * env.water_availability)
            * (0.30 + 0.70 * env.vegetation_cover)
            * (1.0 - 0.45 * env.pollution_pressure)
            * self.species.capacity_modifier
        )
        return max(self.config.minimum_carrying_capacity, round(scaled_capacity))

    def trait_fitness(self, genome: Genome) -> float:
        """Return environmental trait match; this does not mutate an animal."""
        targets = self.environment.trait_targets()
        scores = {
            trait.value: 1.0 - abs(genome.value(trait) - targets[trait.value])
            for trait in Trait
        }
        predator_shift = 0.10 * self.environment.predator_pressure
        weights = {
            Trait.FUR_INSULATION.value: 0.25,
            Trait.HEAT_TOLERANCE.value: 0.25,
            Trait.WATER_EFFICIENCY.value: 0.20,
            Trait.CAMOUFLAGE.value: 0.20 + predator_shift,
            Trait.BODY_SIZE.value: 0.10 - predator_shift,
        }
        return sum(scores[name] * weight for name, weight in weights.items())

    def effective_suitability(self, trait_fitness: float) -> float:
        """Species-level suitability, raised when traits fit the local weather.

        ``species.suitability`` is a coarse, species-wide fit to the climate. An
        animal whose inherited traits match local conditions is held back less
        by a poor species-level fit, so adaptation can widen the climate a
        population tolerates. The boost only closes ``adaptation_benefit`` of
        the gap up to the animal's trait fitness and never lowers suitability.
        With ``adaptation_benefit = 0`` this is the plain species suitability.
        """
        base = self.species.suitability(self.environment)
        return base + self.config.adaptation_benefit * max(0.0, trait_fitness - base)

    def survival_probability(self, genome: Genome, population_size: int) -> float:
        """Documented survival equation before stochastic sampling."""
        crowding = max(0.0, population_size / self.carrying_capacity() - 1.0)
        env = self.environment
        fitness = self.trait_fitness(genome)
        probability = (
            0.02
            + 0.61 * fitness
            + 0.20 * env.resource_quality
            + 0.07 * env.temperature_comfort
            - 0.20 * env.predator_pressure
            - 0.15 * env.pollution_pressure
            - 0.24 * crowding
            + 0.28 * self.effective_suitability(fitness)
            - 0.14
        )
        return _clamp(probability, 0.01, 0.98)

    def _advance_generation(
        self, population: list[Animal], generation: int, capacity: int
    ) -> tuple[list[Animal], GenerationRecord]:
        previous_size = len(population)
        selected = [
            animal
            for animal in population
            if self._rng.random() < self.survival_probability(animal.genome, previous_size)
        ]
        survivors = self._cap_population(selected, capacity)
        births = self._reproduce(survivors, capacity)
        next_population = survivors + births
        record = GenerationRecord(
            generation=generation,
            population_size=len(next_population),
            carrying_capacity=capacity,
            species_suitability=self.species.suitability(self.environment),
            survivors=len(survivors),
            births=len(births),
            survival_rate=(len(survivors) / previous_size if previous_size else 0.0),
            reproduction_rate=(len(births) / len(survivors) if survivors else 0.0),
            average_traits=mean_genome(next_population),
            trait_distributions=self._trait_distributions(next_population),
        )
        return next_population, record

    def _cap_population(self, survivors: list[Animal], capacity: int) -> list[Animal]:
        """Apply the hard capacity limit after survival; fit animals get a slight edge."""
        if len(survivors) <= capacity:
            return survivors
        ranked = sorted(
            survivors,
            key=lambda animal: self._rng.random()
            ** (1.0 / max(self.trait_fitness(animal.genome), 0.001)),
            reverse=True,
        )
        return ranked[:capacity]

    def _reproduce(self, survivors: list[Animal], capacity: int) -> list[Animal]:
        if len(survivors) < 2 or len(survivors) >= capacity:
            return []
        parents = survivors.copy()
        self._rng.shuffle(parents)
        offspring: list[Animal] = []
        env = self.environment
        for first_index in range(0, len(parents) - 1, 2):
            if len(survivors) + len(offspring) >= capacity:
                break
            parent_a, parent_b = parents[first_index], parents[first_index + 1]
            pair_fitness = (self.trait_fitness(parent_a.genome) + self.trait_fitness(parent_b.genome)) / 2.0
            birth_probability = _clamp(
                0.06
                + 1.50 * pair_fitness * (env.resource_quality**1.6)
                - 0.20 * env.predator_pressure
                - 0.15 * env.pollution_pressure,
                0.0,
                0.90,
            )
            birth_probability *= 0.25 + 0.75 * self.effective_suitability(pair_fitness)
            if self._rng.random() >= birth_probability:
                continue
            offspring.append(self._child(parent_a, parent_b))
            if (
                len(survivors) + len(offspring) < capacity
                and self._rng.random() < 0.50
            ):
                offspring.append(self._child(parent_a, parent_b))
        return offspring

    def _child(self, parent_a: Animal, parent_b: Animal) -> Animal:
        return Animal(
            id=self._next_id(),
            parent_ids=(parent_a.id, parent_b.id),
            genome=inherit_genome(
                parent_a.genome,
                parent_b.genome,
                self._rng,
                self.config.mutation_probability,
                self.config.mutation_magnitude,
            ),
        )

    def _next_id(self) -> str:
        self._animal_sequence += 1
        return f"fox-{self._animal_sequence:06d}"

    @staticmethod
    def _trait_distributions(population: Iterable[Animal]) -> dict[str, list[int]]:
        """Count the inherited trait in five equal-width bins for a histogram."""
        counts = {trait.value: [0, 0, 0, 0, 0] for trait in Trait}
        for animal in population:
            for trait in Trait:
                bin_index = min(4, int(animal.genome.value(trait) * 5))
                counts[trait.value][bin_index] += 1
        return counts

    def _record_baseline(self, population: list[Animal], capacity: int) -> GenerationRecord:
        return GenerationRecord(
            generation=0,
            population_size=len(population),
            carrying_capacity=capacity,
            species_suitability=self.species.suitability(self.environment),
            survivors=None,
            births=None,
            survival_rate=None,
            reproduction_rate=None,
            average_traits=mean_genome(population),
            trait_distributions=self._trait_distributions(population),
        )
