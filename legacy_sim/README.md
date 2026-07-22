# Archived monolith

`trash.py` is a historical, single-file snapshot retained only for comparison
with early simulation runs. It is not used by the supported `fieldbird-sim` CLI
and is intentionally excluded from linting, packaging, and release guarantees.

The maintained implementation lives under `src/fieldbird_sim/`. New work,
tests, and bug fixes must target that package. The compatibility runner in
`scripts/run.py` may execute this snapshot for historical investigations, but
its output is not covered by the public API or reproducibility contract.
