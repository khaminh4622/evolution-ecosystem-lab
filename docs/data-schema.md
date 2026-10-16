# Experiment data schema

The simulation writes an `ExperimentRecord` as JSON and one `generation_rows()` table as CSV-compatible dictionaries. Schema version: `1.0`.

## Experiment JSON

```json
{
  "schema_version": "1.0",
  "experiment_id": "cool-forest-001",
  "hypothesis": "Foxes with more insulation will be selected in colder conditions.",
  "environment": { "temperature_c": 2.0, "...": "validated EnvironmentConfig fields" },
  "simulation_config": { "seed": 7, "generations": 10, "...": "SimulationConfig fields" },
  "initial_population_spec": { "size": 30, "trait_mean": 0.5, "trait_variation": 0.18 },
  "results": ["one GenerationRecord for baseline and each generation"],
  "replay": ["one PopulationSnapshot for baseline and each generation"]
}
```

Required reproducibility fields are `environment`, `simulation_config.seed`, all remaining simulation configuration, and the initial population specification. Results must never be stored without their configuration.

## Generation row (CSV)

One row is emitted per generation with these stable columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `experiment_id` | string | User/application experiment key |
| `generation` | integer | 0 is the baseline population |
| `population_size` | integer | Number of living animals after the step |
| `survivors` | integer | Animals surviving selection in this step |
| `births` | integer | New offspring produced in this step |
| `survival_rate` | number or null | survivors / previous population; baseline is null |
| `reproduction_rate` | number or null | births / survivors; baseline is null |
| `carrying_capacity` | integer | Calculated for the current environment |
| `avg_*` | number or null | Mean of every trait; null if extinct |
| `dist_*` | JSON string | Five-bin trait-frequency array for dashboard charts |

The environment controls, supplied seed, and initial-population fields are also flattened into every row for data set generation. This avoids accidental row-level train/test splits without scenario metadata.

## Replay snapshot

Each snapshot stores `generation` and a list of animals. Each animal has `id`, `parent_ids` (empty for founders), and the five genome values. Unity can render animals from this data but must not treat positions/meshes as simulation data.

## ML dataset conventions

Classification labels are produced by a documented suitability function and named `suitability_class` with values `favorable`, `moderate`, or `challenging`. Regression labels are named `population_after_5_generations` and only come from independent simulation experiments.

Every ML row includes `scenario_id`, `seed`, the full environment vector, starting population, initial trait means, and initial trait variation. Train/test partitions must group by `scenario_id` so generations or replicates from the same scenario cannot occur in both sets. Missing numeric values are imputed in a persisted pipeline; categorical values use a persisted one-hot encoder with unknown-category handling.

## What the ML predicts, and what that is worth

`ecosystem_evolution.ml` trains four models on simulated scenarios. Every score
is measured on scenarios held out by `scenario_id`.

| Task | Metric | Typical held-out score |
| --- | --- | --- |
| `extinction_classifier` | ROC-AUC | ~0.78 (base rate ~0.31) |
| `suitability_classifier` | accuracy | ~0.61 (majority class ~0.58) |
| `capacity_check` | R2 | ~0.80 |
| `population_regressor` | R2 | unstable, see below |

Three honest caveats belong with those numbers:

1. **The regressor's R2 is not a meaningful headline.** The label is severely skewed — roughly 30% of scenarios collapse to zero while a few grow large — so five rows of a 250-scenario dataset can hold about 87% of the label's variance. R2 then swings widely between splits. Mean absolute error is the steadier summary.
2. **The capacity model demonstrates a pipeline, not knowledge.** The simulator already computes carrying capacity in closed form, so the model can only approximate a value that is known exactly.
3. **Nothing here is evidence about real foxes.** The labels come from this simulator, whose rules are hand-written formulas chosen for teaching. A model that predicts them reproduces the simulator; it does not discover biology. Any use must keep that separation, as the project README requires.
