# Simulation assumptions and equations

## Scope

The model represents one simplified fox population. A fox has a persistent identifier and a genome with normalized traits in `[0, 1]`:

`fur_insulation`, `heat_tolerance`, `water_efficiency`, `camouflage`, and `body_size`.

A fox's genome is fixed for life. Choosing a different location never rewrites it: the place changes the *targets* those traits are graded against, so the same starting population adapts toward thicker fur in a cold place and toward more heat tolerance in a hot one.

Each simulation step is one generation. Generations overlap: the survivors of generation *g* are the parents of the newborns, and they also remain in the population of generation *g+1* alongside those newborns. There is no age structure, so a surviving fox is simply as likely to die next step as any other fox with the same traits. This keeps the classroom model inspectable, but it is not a full age-structured population model.

## Environment controls

All availability/pressure values are normalized to `[0, 1]`. Temperature is in °C and habitat area is km². Inputs are validated at the configuration boundary to prevent physically nonsensical simulation values.

## Location presets

The location selector is a set of curated classroom presets, not a live data
service. Choosing a place fills the environment controls with transparent,
editable assumptions. For example, `sonoran_desert` starts hot and dry, with
low food, water, and vegetation. The saved experiment still records the exact
values used, including any player edits.

The optional world-location bridge can additionally start from a map search and
current weather snapshot. It imports temperature, humidity, precipitation, and
soil moisture. Habitat must be selected by the player, and food, vegetation,
predator pressure, and pollution remain editable estimates; they cannot be
claimed as direct measurements from weather alone.

| Control | Valid range | Effect |
| --- | --- | --- |
| Temperature | -50 to 60 °C | Sets cold/heat trait targets; extreme values lower background quality |
| Rainfall, humidity | 0–1 | Contribute to water stress and water-efficiency target |
| Food, water, vegetation | 0–1 | Affect resource quality and carrying capacity |
| Predator, pollution pressure | 0–1 | Directly reduce survival/reproduction; predation weights camouflage |
| Habitat type | forest, grassland, desert, tundra, wetland | Sets camouflage target and small body-size shift |
| Habitat area | 0.1–100 km² | Scales carrying capacity |

## Trait targets

The target functions are intentionally explicit rather than learned:

\[
T_{insulation} = sigmoid((12 - temperature)/10)
\]
\[
T_{heat} = sigmoid((temperature - 16)/10)
\]
\[
T_{water} = clamp(1 - (0.65 water + 0.20 rainfall + 0.15 humidity))
\]

Camouflage comes from a documented habitat table. Body size is `clamp(0.35 + 0.35 * food - 0.20 * heat_target)`. A trait's match is `1 - abs(value - target)`. The weighted trait fitness is a mean of insulation (0.25), heat tolerance (0.25), water efficiency (0.20), camouflage (0.20 + 0.10 × predator pressure), and body size (0.10). Predator pressure shifts up to 0.10 weight from size to camouflage, in proportion to the pressure (full shift at pressure 1).

Because the targets are derived from the environment, the same genome scores differently in each location: that is what makes one fixed fox population adapt in a different direction depending on where it is placed, and it is the only way location enters a fox's fitness.

## Survival and selection

First compute resource quality:

\[
R = 0.45 food + 0.30 water + 0.15 vegetation + 0.10 rainfall
\]

Temperature comfort is `1 - min(abs(temp - 15) / 55, 1)`. Individual survival is sampled from:

\[
p_{survival} = clamp(0.02 + 0.61F + 0.20R + 0.07C - 0.20P - 0.15L - 0.24 max(0, N/K - 1) + 0.28S_{eff} - 0.14, 0.01, 0.98)
\]

where *F* is trait fitness, *C* is temperature comfort, *P* predator pressure, *L* pollution pressure, *N* population, *K* carrying capacity, and *S_eff* the animal's effective suitability, described under *Adaptation benefit* below. The constant `-0.14` is a calibration offset that offsets the positive terms so typical survival rates stay realistic; it is a tuning value, not a biological quantity. Fitness changes each fox's chance to survive; it never directly edits an inherited trait.

## Capacity and reproduction

\[
K = max(4, round(base\_capacity * area * (0.20 + 0.80food)(0.20 + 0.80water)(0.30 + 0.70vegetation)(1 - 0.45pollution)))
\]

Survivors are shuffled uniformly at random and paired off, so no fox is used twice per generation (with an odd number of survivors, one stays unpaired). Mating is therefore random with respect to fitness; fitter pairs have an advantage only through survival and through their higher chance of producing offspring, below. A pair's probability of producing offspring is:

\[
p_{birth} = S' \cdot clamp(0.06 + 1.50F_{pair}R^{1.6} - 0.20P - 0.15L, 0, 0.90), \quad S' = 0.25 + 0.75S_{eff}
\]

Successful pairs have one offspring and a 50% chance of a second. The exponent on resource quality makes scarce resources suppress reproduction more strongly than favorable conditions. *F_pair* is the mean trait fitness of the two parents, and *S_eff* is computed from *F_pair*. Because a fox inherits only heat tolerance, the pair's own contribution to breeding comes from *F_pair*, while *E* carries the place's contribution; both are scaled by resource quality. Birth stops at `K`; `K` is therefore a hard ceiling on the next population. Extinction is possible and remains extinct unless the player starts a new experiment.

## Adaptation benefit

Species suitability *S* is a coarse, species-wide fit to the weather. On its own it would penalize survival and breeding in the same way for every fox, however well that fox's inherited traits match local conditions, so adapting would stop the decline but never let a population grow back. To let adaptation widen the climate a population can tolerate, each animal (or breeding pair) uses an effective suitability:

\[
S_{eff} = S + a \cdot max(0, F - S)
\]

where *F* is the animal's trait fitness (for a pair, the pair's mean) and *a* is `SimulationConfig.adaptation_benefit` (0 to 1, default 0.5). The boost closes a fraction *a* of the gap between suitability and trait fitness. It never lowers *S*, and with `adaptation_benefit = 0` the model is the plain species suitability. At `1.0` a perfectly adapted fox would have no remaining penalty from poor species-level fit, so a hot place would be as easy as a warm one; the default 0.5 keeps hot weather harder than warm even for well-adapted foxes.

This is a modeling assumption chosen for teaching, not a measured quantity. Because a hot place's survival and birth rates start low, a population still shrinks while it adapts, so a small starting group can go extinct before the benefit builds up. A larger starting group, or more food and water, helps it get there.

## Inheritance and mutation

Inheritance is particulate, not blending. For each trait independently, the offspring copies the value of one parent, chosen with equal probability, so traits segregate like separate genes and an offspring can combine, for example, one parent's fur with the other's body size. Independently for each trait, mutation then occurs with the configured probability; its signed magnitude is drawn uniformly from `[-mutation_magnitude, +mutation_magnitude]`. The final value is clamped to `[0, 1]`. With mutation probability zero, every offspring trait exactly equals one of its parents' values.

This was chosen over averaging the parents because averaging halves the population's variation every generation (the classic objection to blending inheritance), leaving mutation as the only source of new variation. Copying keeps variation in the population, apart from random drift in small populations, so selection can keep acting on it.

## Reproducibility

All stochastic choices use an instance-local `random.Random(seed)`. A complete experiment preserves the seed, environment, simulation configuration, initial population specification, and schema version. Same code + same configuration + same seed produces the same record.

## Educational interpretation

Results demonstrate a model-based hypothesis test: students choose conditions, state a prediction, compare an observed run with its baseline or a second scenario, and justify a conclusion from population and trait data. Results are simulated evidence, not measurements from wild fox populations.
