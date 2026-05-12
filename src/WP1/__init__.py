"""
WP1 — YAML/CLI front-end for winged_drone_train.

This package is a thin orchestration layer around
``winged_drone_train.train`` and ``winged_drone_train.eval``:

  - ``WP1.config_loader``  — YAML + ``--cfg.section.key`` CLI overrides,
                              deep-merged onto the legacy dicts produced by
                              ``winged_drone_train.train.get_cfgs`` /
                              ``get_train_cfg``.
  - ``WP1.run_manager``    — timestamped ``logs/runs/<ts>_<exp>/`` layout
                              with TensorBoard / eval / plot subfolders.
  - ``WP1.train``          — CLI entry point that wraps
                              ``winged_drone_train.train`` and writes results
                              into the run folder above.
  - ``WP1.eval``           — CLI entry point that wraps
                              ``winged_drone_train.eval.evaluation`` and
                              points it at a WP1 run folder.

Quick start
-----------
.. code-block:: bash

    python -m WP1.train --cfg src/WP1/configs/default.yaml
    python -m WP1.eval  --run logs/runs/<timestamp>_<exp_name>
"""

from WP1.config_loader import (
    build_cfgs,
    deep_merge,
    dump_yaml,
    load_yaml,
    parse_cli_overrides,
)
from WP1.csv_logger import CSVLogger
from WP1.plotting import plot_run
from WP1.run_manager import RunManager

__all__ = [
    "CSVLogger",
    "RunManager",
    "build_cfgs",
    "deep_merge",
    "dump_yaml",
    "load_yaml",
    "parse_cli_overrides",
    "plot_run",
]
