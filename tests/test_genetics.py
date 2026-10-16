import random

from ecosystem_evolution.genetics import Genome, Trait, inherit_genome


def test_each_trait_comes_from_exactly_one_parent_without_mutation() -> None:
    first = Genome(0.2, 0.4, 0.6, 0.8, 0.1)
    second = Genome(0.8, 0.6, 0.4, 0.2, 0.9)

    for seed in range(50):
        child = inherit_genome(first, second, random.Random(seed), 0.0, 0.0)
        for trait in Trait:
            assert child.value(trait) in (first.value(trait), second.value(trait))


def test_traits_segregate_independently_and_both_parents_contribute() -> None:
    first = Genome(0.0, 0.0, 0.0, 0.0, 0.0)
    second = Genome(1.0, 1.0, 1.0, 1.0, 1.0)
    rng = random.Random(11)

    children = [inherit_genome(first, second, rng, 0.0, 0.0) for _ in range(200)]

    # Mixed children (some traits from each parent) must occur.
    assert any(len({child.value(trait) for trait in Trait}) == 2 for child in children)
    # Across many children each parent supplies roughly half of each trait.
    for trait in Trait:
        from_second = sum(child.value(trait) == 1.0 for child in children)
        assert 70 <= from_second <= 130


def test_inheritance_preserves_variation_unlike_blending() -> None:
    rng = random.Random(2)
    population = [Genome.random(rng, 0.5, 0.2) for _ in range(200)]

    def variance(genomes: list[Genome]) -> float:
        values = [g.value(Trait.CAMOUFLAGE) for g in genomes]
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    start = variance(population)
    for _ in range(8):
        population = [
            inherit_genome(rng.choice(population), rng.choice(population), rng, 0.0, 0.0)
            for _ in range(200)
        ]

    # Blending would shrink variance by about 2**8; segregation keeps most of it.
    assert variance(population) > 0.4 * start


def test_mutation_is_bounded_even_at_trait_edges() -> None:
    low = Genome(0.0, 0.0, 0.0, 0.0, 0.0)
    high = Genome(1.0, 1.0, 1.0, 1.0, 1.0)

    for seed in range(20):
        child = inherit_genome(low, high, random.Random(seed), 1.0, 1.0)
        assert all(0.0 <= child.value(trait) <= 1.0 for trait in Trait)


def test_all_five_traits_exist() -> None:
    assert [trait.value for trait in Trait] == [
        "fur_insulation",
        "heat_tolerance",
        "water_efficiency",
        "camouflage",
        "body_size",
    ]
