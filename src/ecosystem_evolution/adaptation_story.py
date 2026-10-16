"""Shared printout for the "adapt, breed, grow back" examples."""

from __future__ import annotations

from .environment import EnvironmentConfig
from .lineage import lineage_report
from .simulation import EvolutionSimulator, InitialPopulationSpec, SimulationConfig
from .species import FOX, SpeciesProfile

TRAITS = ("fur_insulation", "heat_tolerance", "water_efficiency", "camouflage", "body_size")


def show_adaptation_story(
    title: str,
    environment: EnvironmentConfig,
    species: SpeciesProfile = FOX,
    runs: int = 60,
    generations: int = 40,
    include_lineage: bool = True,
    start_size: int = 30,
) -> None:
    """Run many seeds, then walk through one typical surviving run in detail.

    The printout has two sections: a population table (population size, mean
    trait match, mean traits per generation) and the lineage report for one
    founder. Set ``include_lineage`` to False to print the table alone.

    ``start_size`` exists because a habitat can only hold so many foxes:
    starting far more than the carrying capacity kills every fox in the first
    generation, and the table alone cannot explain why.
    """
    spec = InitialPopulationSpec(size=start_size, trait_mean=0.5, trait_variation=0.18)

    def simulator(seed: int) -> EvolutionSimulator:
        return EvolutionSimulator(
            environment,
            SimulationConfig(generations=generations, seed=seed, mutation_probability=0.10),
            species=species,
        )

    results = [(seed, simulator(seed).run(spec)) for seed in range(runs)]
    survivors = sorted(
        (pair for pair in results if pair[1].final_population > 0),
        key=lambda pair: pair[1].final_population,
    )
    print(f"{title}: {len(survivors)} of {runs} runs survived {generations} generations.")
    if not survivors:
        return
    # A middle-of-the-pack survivor, not the best or the worst.
    seed, result = survivors[len(survivors) // 2]
    sim = simulator(seed)
    targets = environment.trait_targets()
    print(
        f"Showing seed {seed}, a typical survivor "
        f"(carrying capacity {result.generations[0].carrying_capacity})."
    )
    print("\nTraits this place favors: " + ", ".join(f"{t[:4]} {targets[t]:.2f}" for t in TRAITS))

    print("\ngen   foxes  trait match  " + "  ".join(t[:4] for t in TRAITS))
    shown = sorted({g for g in (0, 5, 10, 15, 20, 30, generations) if g <= generations})
    for generation in shown:
        record = result.generations[generation]
        animals = result.replay[generation].animals
        if not animals:
            print(f"{generation:>3}   {record.population_size:>5}  (extinct)")
            break
        match = sum(sim.trait_fitness(a.genome) for a in animals) / len(animals)
        traits = record.average_traits
        print(
            f"{generation:>3}   {record.population_size:>5}  {match:>11.2f}  "
            + "  ".join(f"{traits[t]:.2f}" for t in TRAITS)
        )
    if include_lineage:
        print("\n" + lineage_report(result))
