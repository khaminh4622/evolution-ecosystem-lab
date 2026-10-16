"""Follow individual foxes and their descendants through a finished run.

Everything here is read from the replay the simulator already records, so it
adds no new simulation rules. The model has no juvenile stage: a newborn joins
the population immediately, and it can become a parent only after it survives
one generation's selection step.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .genetics import Animal, Trait
from .simulation import SimulationResult


@dataclass(frozen=True, slots=True)
class FoxLife:
    """One fox's recorded life: when it appeared, how long it lived, its young."""

    animal: Animal
    born_generation: int
    last_generation_alive: int
    child_ids: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.animal.id

    @property
    def generations_lived(self) -> int:
        return self.last_generation_alive - self.born_generation


def life_histories(result: SimulationResult) -> dict[str, FoxLife]:
    """Return a life record for every fox that ever appeared in the replay."""
    born: dict[str, int] = {}
    last: dict[str, int] = {}
    animals: dict[str, Animal] = {}
    for snapshot in result.replay:
        for animal in snapshot.animals:
            born.setdefault(animal.id, snapshot.generation)
            last[animal.id] = snapshot.generation
            animals[animal.id] = animal
    children: dict[str, list[str]] = defaultdict(list)
    for animal in animals.values():  # replay order keeps children oldest-first
        for parent_id in animal.parent_ids:
            children[parent_id].append(animal.id)
    return {
        fox_id: FoxLife(animals[fox_id], born[fox_id], last[fox_id], tuple(children[fox_id]))
        for fox_id in animals
    }


def descendants(lives: dict[str, FoxLife], fox_id: str) -> set[str]:
    """All children, grandchildren and so on of one fox (not the fox itself)."""
    found: set[str] = set()
    stack = list(lives[fox_id].child_ids)
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(lives[current].child_ids)
    return found


def founder_shares(result: SimulationResult) -> dict[str, float]:
    """Share of the final population's ancestry that comes from each founder.

    Each parent contributes half, so the shares always sum to 1. An empty dict
    means the population went extinct.
    """
    lives = life_histories(result)
    memo: dict[str, dict[str, float]] = {}

    def ancestry(fox_id: str) -> dict[str, float]:
        if fox_id in memo:
            return memo[fox_id]
        parents = lives[fox_id].animal.parent_ids
        if not parents:
            memo[fox_id] = {fox_id: 1.0}
        else:
            mixed: dict[str, float] = defaultdict(float)
            for parent_id in parents:
                for founder, share in ancestry(parent_id).items():
                    mixed[founder] += share / len(parents)
            memo[fox_id] = dict(mixed)
        return memo[fox_id]

    final = result.replay[-1].animals
    if not final:
        return {}
    totals: dict[str, float] = defaultdict(float)
    for animal in final:
        for founder, share in ancestry(animal.id).items():
            totals[founder] += share / len(final)
    return dict(totals)


def line_of_descent(
    result: SimulationResult, fox_id: str, founder_id: str | None = None
) -> list[Animal]:
    """Walk back from a fox to a founder, giving one line through the family tree.

    The result runs founder first. Without ``founder_id`` the walk follows each
    fox's first-listed parent. With it, the walk stays inside that founder's
    descendants, so the line ends at the founder you asked about. The whole
    tree is not shown because it doubles in size with each generation back.
    """
    lives = life_histories(result)
    allowed: set[str] | None = None
    if founder_id is not None:
        allowed = descendants(lives, founder_id) | {founder_id}
        if fox_id not in allowed:
            raise ValueError(f"{fox_id} is not a descendant of {founder_id}")
    chain = [lives[fox_id].animal]
    while chain[-1].parent_ids:
        parents = [
            pid for pid in chain[-1].parent_ids if allowed is None or pid in allowed
        ]
        chain.append(lives[parents[0]].animal)
    chain.reverse()
    return chain


def _traits(animal: Animal) -> str:
    return " ".join(f"{trait.value[:4]} {animal.genome.value(trait):.2f}" for trait in Trait)


def lineage_report(result: SimulationResult) -> str:
    """A plain-text story: the leading founder, its young, and a line of descent."""
    final = result.replay[-1].animals
    if not final:
        return "The population went extinct, so there is no living lineage to follow."

    lives = life_histories(result)
    shares = founder_shares(result)
    leader_id = max(shares, key=lambda fox_id: shares[fox_id])
    leader = lives[leader_id]
    kids = descendants(lives, leader_id)
    living_kids = [fox for fox in final if fox.id in kids]

    lines = [
        f"Founder {leader_id} supplies {shares[leader_id]:.0%} of the ancestry of the "
        f"{len(final)} foxes alive in generation {result.replay[-1].generation}.",
        f"  Alive generations {leader.born_generation}-{leader.last_generation_alive}, "
        f"{len(leader.child_ids)} direct children, {len(kids)} descendants in all, "
        f"{len(living_kids)} still alive.",
        f"  Founder traits: {_traits(leader.animal)}",
        "",
        "One line of descent, founder to a living fox (each fox is a parent of the next):",
    ]
    target = living_kids[0]  # the leading founder always has living descendants
    for animal in line_of_descent(result, target.id, leader_id):
        life = lives[animal.id]
        lines.append(
            f"  gen {life.born_generation:>2}  {animal.id}  {_traits(animal)}  "
            f"({len(life.child_ids)} children)"
        )
    return "\n".join(lines)
