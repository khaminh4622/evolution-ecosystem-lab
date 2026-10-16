# Unity integration boundary

Unity should request an experiment from a local FastAPI service or load a generated `ExperimentRecord` JSON file. It should render `replay` snapshots and plot `results`; it must not copy survival, inheritance, mutation, or capacity equations into C#.

Phase 5 contract:

1. Send an environment, simulation config, initial-population spec, hypothesis, and experiment ID.
2. Receive the `docs/data-schema.md` JSON record.
3. Animate snapshots by animal `id`, updating meshes/animations separately from genomes.
4. Display prediction outputs as model estimates and simulation outputs as actual results, never as the same field.
