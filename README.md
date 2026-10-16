# Evolution Ecosystem Lab

An offline, deterministic simulation core for a Unity educational game about evolution. The core deliberately has no Unity dependency: the game, a FastAPI service, and model-training scripts can all consume the same experiment records.

## Implemented first milestone

- Individual fox genomes with unique identifiers and five normalized inherited traits. A genome is fixed for life: choosing a different location changes the values those traits are graded against, never the fox.
- Stochastic survival and reproduction driven by explicit environment–trait fitness rules.
- Sexual-style two-parent inheritance with configurable mutation and strict `[0, 1]` trait bounds.
- Resource-dependent carrying capacity, environmental pressure, complete generation metrics, and replay snapshots.
- Named desert, forest, and tundra presets that turn a chosen place into editable resource assumptions.
- An adjustable adaptation benefit (`SimulationConfig.adaptation_benefit`, default 0.5) so foxes that adapt to a harsh place can grow back toward capacity. Set it to 0 for the original rules.
- Reproducible runs when `SimulationConfig.seed` is set.
- JSON/CSV-ready experiment logging plus a documented schema.
- Unit tests for determinism, inheritance, mutation bounds, population limits, and environmental selection.

## Quick start

Python is required (3.10+). From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```

Run a sample experiment and write its artifacts:

```powershell
python -m ecosystem_evolution.example
```

Try a location-driven experiment:

```powershell
python -m ecosystem_evolution.location_example
```

`sonoran_desert`, `pacific_northwest_forest`, and `arctic_tundra` are built-in
classroom presets. Each supplies a habitat, temperature, rainfall, food, water,
vegetation, predator pressure, pollution, and season. They are editable
assumptions—not live measurements—so players can test questions such as “What
happens if this desert gets more food?”

To start from a real map search and current weather snapshot, run:

```powershell
python -m ecosystem_evolution.world_location_example
```

The map-data bridge searches a place—or accepts coordinates clicked on a
world map—then retrieves temperature, humidity, precipitation, and soil
moisture and makes editable simulation inputs. A player must choose the habitat
and review food, vegetation, predator pressure, and pollution: a weather
service does not directly measure those ecological resources. Network access is
required for this optional feature.

The bridge prints the same population table as the winter and summer examples.
Add `--no-lineage` for the table alone, or `--check` to see the habitat's
carrying capacity and whether the starting group is larger than it can hold
(a start far above capacity kills every fox in the first generation):

```powershell
python -m ecosystem_evolution.world_location_example --check
```

To see a hot-desert, long-term evolution example with a desert-adapted fox
profile, run:

```powershell
python -m ecosystem_evolution.fennec_future_example
```

Compare the original map-selected resources with a player's changes while
keeping the same temperature, starting foxes, and random seed:

```powershell
python -m ecosystem_evolution.resource_comparison_example
```

Follow two specially adapted founders among thirty starting foxes:

```powershell
python -m ecosystem_evolution.two_founders_example
```

`InitialPopulationSpec.founder_genomes` lets the game give one or more named
starting animals specific inherited traits. Those animals still need mates and
enough resources to produce descendants.

Follow one fox family through hot weather (a founder, its young, and a line of
descent to a living fox):

```powershell
python -m ecosystem_evolution.lineage_example
```

Watch red foxes in hot, dry country. Choose one place per run: a watered oasis
(the default), where foxes survive and adapt to the heat, or the open Sonoran
desert, where no population persists because the climate leaves almost no water:

```powershell
python -m ecosystem_evolution.desert_example              # oasis
python -m ecosystem_evolution.desert_example --desert     # open desert
```

Do the same in a 0 C winter forest (cold pushes fur insulation up):

```powershell
python -m ecosystem_evolution.winter_example
```

Or in a 30 C summer forest, where heat pushes fur insulation down and heat
tolerance up instead:

```powershell
python -m ecosystem_evolution.summer_example
```

The desert, winter, summer, location, and world-location examples print two
sections: a population table (generation, population size, mean trait match, mean
traits), then a lineage report for one founder. Add `--no-lineage` to print the
table alone, and see the world-location section below for `--check`:

```powershell
python -m ecosystem_evolution.summer_example --no-lineage
```

`ecosystem_evolution.lineage` reads the replay a run already records: when each fox
appeared, how many generations it lived, its children, each founder's share of the
final population's ancestry, and one line of descent from a founder to a living fox.
The model has no juvenile stage. A newborn joins the population immediately and can
become a parent only after surviving one generation's selection step.

## Machine learning (experimental)

`ecosystem_evolution.ml` turns simulated experiments into a dataset and trains
four teaching models on it. Install the extra first:

```powershell
pip install -e ".[ml]"
```

See what a dataset contains:

```powershell
python -m ecosystem_evolution.ml_dataset_example
```

Train the models and print held-out scores:

```powershell
python -m ecosystem_evolution.ml_train_example
```

Walk through the whole pipeline step by step, including one prediction on a
scenario the model never saw:

```powershell
python -m ecosystem_evolution.ml_walkthrough_example
```

**What these models are worth.** Three things must be said plainly:

1. **The labels come from this simulator, not from nature.** The simulator's rules
   are hand-written formulas chosen for teaching, so a model that predicts them
   well has demonstrated a working pipeline, not discovered biology.
2. **The population regressor's R2 is unstable and should not be quoted alone.**
   About 30% of scenarios collapse to zero while a few grow large, so five rows of
   a 250-scenario dataset can hold ~87% of the label's variance. Report mean
   absolute error instead.
3. **The most trustworthy task is the extinction classifier** (ROC-AUC ~0.78
   against a base rate of ~0.31), because its outcome is balanced enough to score
   honestly and it answers the question a student actually asks.

Splits are always made on `scenario_id`. Every row of a scenario shares one
environment, so a row-wise split would leak and report a score the model has not
earned; `tests/test_ml_models.py` demonstrates that gap directly.

Any model trained here predicts **this simulator's outcomes** and must be
presented separately from real biological evidence.

## Architecture

| Area | Location | Responsibility |
| --- | --- | --- |
| Environment configuration | `src/ecosystem_evolution/environment.py` | Validated controls and trait optima |
| Location presets | `src/ecosystem_evolution/locations.py` | Named places translated into editable environment inputs |
| World-location data | `src/ecosystem_evolution/location_data.py` | Map search and weather snapshot translated into editable inputs |
| Before/after comparison | `src/ecosystem_evolution/comparison.py` | Same-start comparison of original and player-edited resources |
| Genetics & inheritance | `src/ecosystem_evolution/genetics.py` | Genomes, mutation, parent mixing |
| Evolution simulation | `src/ecosystem_evolution/simulation.py` | Selection, reproduction, capacity, replay |
| Lineage & life histories | `src/ecosystem_evolution/lineage.py` | Parents, children, founder shares and lines of descent from a finished run |
| Data logging | `src/ecosystem_evolution/logging.py` | Experiment records, JSON and tabular rows |
| ML (experimental) | `src/ecosystem_evolution/ml/` | Dataset generation from simulated scenarios, grouped splits, and four teaching models |
| API (next phase) | `src/ecosystem_evolution/api/` | Unity/Python request boundary |
| Unity presentation | `unity/Assets/Scripts/` | Rendering, UI, replay controls; never fitness logic |

The equations, constraints, and planned Unity/API contract are in [docs/simulation-assumptions.md](docs/simulation-assumptions.md). The exact artifact formats are in [docs/data-schema.md](docs/data-schema.md).

## Scientific scope

The simulator is a transparent teaching model, not a validated model of real fox ecology. Any ML model trained from its experiments predicts **this simulator's outcomes**. It must be presented separately from real biological evidence.
