"""
Run-folder management — standardised, timestamped directory layout.
====================================================================

Every training run is stored in a self-contained folder under
``logs/runs/`` with a deterministic structure that makes any run fully
reproducible from its artefacts alone.

Directory layout
----------------
::

    logs/runs/<YYYY-MM-DD_HH-MM-SS>_<exp_name>/
    ├── config.yaml          # frozen RunConfig snapshot
    ├── catalog.txt          # URDF filenames used (if multi-morph)
    ├── tb/                  # TensorBoard event files (includes model_*.pt checkpoints)
    ├── eval/                # per-iteration CSV logs
    └── plots/               # auto-generated diagnostic figures

The ``RunManager`` class creates this layout at the start of a run,
saves the config snapshot, and exposes the sub-directory paths as
attributes for other modules (``CSVLogger``, ``RLTrainingLogger``,
``plot_run``) to use.

Resume behaviour
----------------
When ``resume=True`` is passed, the manager looks for the most recent
existing folder whose name contains the experiment name and creates a
new folder with the suffix ``_resumed``.  This avoids overwriting the
original run while making the lineage clear.

Usage
-----
.. code-block:: python

    from WP1.config import RunConfig
    from WP1.run_manager import RunManager

    cfg = RunConfig(exp_name="my-experiment")
    run = RunManager(cfg)

    # Use run.log_dir for TensorBoard/checkpoints, run.eval_dir for CSVs, etc.
    print(run.run_dir)      # logs/runs/2025-06-01_12-00-00_my-experiment
    print(run.tb_dir)       # .../tb
    print(run.eval_dir)     # .../eval
    print(run.plots_dir)    # .../plots
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from WP1.config import RunConfig


class RunManager:
    """Create and manage a timestamped run directory.

    On instantiation the manager:

    1. Generates a folder name ``<timestamp>_<exp_name>`` (or appends
       ``_resumed`` if resuming).
    2. Creates the folder and all required sub-directories.
    3. Writes a ``config.yaml`` snapshot into the run folder.

    Attributes
    ----------
    cfg : RunConfig
        The configuration object for this run.
    root : Path
        Parent directory for all runs (default ``logs/runs``).
    run_dir : Path
        Full path to this run's top-level directory.
    tb_dir : Path
        Directory for TensorBoard event files and model checkpoints.
    eval_dir : Path
        Directory for per-iteration CSV evaluation logs.
    plots_dir : Path
        Directory for auto-generated diagnostic figures.
    """

    def __init__(
        self,
        cfg: RunConfig,
        root: str | Path = "logs/runs",
        resume: bool = False,
    ) -> None:
        """Initialise the run manager and create the directory layout.

        Parameters
        ----------
        cfg : RunConfig
            Full training configuration.  A YAML snapshot is written to
            ``<run_dir>/config.yaml``.
        root : str or Path
            Parent directory under which timestamped run folders are
            created.  Defaults to ``logs/runs``.
        resume : bool
            If ``True``, find the latest existing run folder matching
            ``cfg.exp_name`` and create a new folder with the suffix
            ``_resumed`` appended to its name.
        """
        self.cfg = cfg
        self.root = Path(root)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{stamp}_{cfg.exp_name}"

        if resume:
            # Find the latest existing folder for this experiment
            existing = self._find_latest(cfg.exp_name)
            if existing is not None:
                folder_name = existing.name + "_resumed"

        self.run_dir = self.root / folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.tb_dir = self.run_dir / "tb"
        self.eval_dir = self.run_dir / "eval"
        self.plots_dir = self.run_dir / "plots"

        for d in (self.tb_dir, self.eval_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        self.cfg.to_yaml(self.run_dir / "config.yaml")
        print(f"[RunManager] Run directory: {self.run_dir}")

    @property
    def log_dir(self) -> Path:
        """Path to the TensorBoard log directory.

        This is the directory passed to RSL-RL's ``OnPolicyRunner`` and to
        ``RLTrainingLogger`` as their ``log_dir``.
        """
        return self.tb_dir

    def save_catalog(self, catalog_path: Optional[str | Path]) -> None:
        """Copy ``catalog.txt`` from *catalog_path* into the run folder.

        If *catalog_path* is a directory, looks for ``catalog.txt`` inside
        it.  If *catalog_path* is a file, copies it directly.  Does nothing
        if *catalog_path* is ``None`` or the file does not exist.

        Parameters
        ----------
        catalog_path : str, Path, or None
            Path to the catalog directory or file.
        """
        if catalog_path is None:
            return
        src = Path(catalog_path)
        catalog_file = src / "catalog.txt" if src.is_dir() else src
        if catalog_file.is_file():
            shutil.copy2(catalog_file, self.run_dir / "catalog.txt")
            print(f"[RunManager] Saved catalog.txt from {catalog_file}")

    def _find_latest(self, exp_name: str) -> Optional[Path]:
        """Find the most recent run folder matching *exp_name*.

        Scans ``self.root`` for directories whose names contain
        *exp_name*, sorts them lexicographically (which is also
        chronological since names are timestamp-prefixed), and returns
        the last one.

        Parameters
        ----------
        exp_name : str
            Experiment name to search for.

        Returns
        -------
        Path or None
            Path to the latest matching run folder, or ``None`` if no
            match is found.
        """
        if not self.root.exists():
            return None
        candidates = sorted(
            [d for d in self.root.iterdir() if d.is_dir() and exp_name in d.name],
            key=lambda p: p.name,
        )
        return candidates[-1] if candidates else None
