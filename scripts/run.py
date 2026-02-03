from pathlib import Path
from fieldbird_sim.run import run_legacy
from fieldbird_sim.paths import legacy_entry, project_root

def main() -> None:
    res = run_legacy(entry=legacy_entry(), cwd=project_root(), patch_pandas_parent_dir=True)
    if not res.ok:
        raise SystemExit(res.error)

if __name__ == "__main__":
    main()
