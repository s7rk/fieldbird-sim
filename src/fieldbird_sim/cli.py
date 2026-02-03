from __future__ import annotations

import argparse

from fieldbird_sim.legacy.kpis import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fieldbird-sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run the ported legacy simulation (no runpy, no legacy_sim/trash.py).")
    r.add_argument(
        "--use-synthetic/--use-csv-inputs",
        dest="use_synthetic",
        default=True,
        help="Use synthetic fields (default) or read CSV inputs.",
    )

    a = p.parse_args(argv)

    if a.cmd == "run":
        run(use_synthetic=bool(a.use_synthetic))
        return 0

    return 2
