"""Location of the result folders read by the figure and table scripts.

Each study folder (``Results_Cliff``, ``Results_lSweep_LO_GcCorr0.33``, ...) is taken from
``results/`` in the repository root when it exists there (written by the scripts in
``simulations/``), and otherwise from ``data/``, which holds the time histories of the
paper runs. The environment variable ``PAPER_RESULTS_DIR`` replaces ``results/`` by
another folder, for example a copy of the full result folders of the paper.
"""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("PAPER_RESULTS_DIR", REPOSITORY / "results")).resolve()
DATA = REPOSITORY / "data"


def study_folder(name: str) -> Path:
    """Folder of the study ``name``: in the results folder if it exists there, else in data/."""
    for root in (RESULTS, DATA):
        if (root / name).is_dir():
            return root / name
    return RESULTS / name


def require_field_outputs(case_dir: Path, figures: str, data_only: str) -> None:
    """Stop with a clear message if ``case_dir`` has no field outputs (Data/Outputs_*.hdf5).

    ``data/`` holds the time histories only; ``figures`` names what needs the fields and
    ``data_only`` the option that draws the figures available from ``data/``.
    """
    if any((Path(case_dir) / "Data").glob("Outputs_*.hdf5")):
        return
    raise SystemExit(
        f"{figures} need the full field outputs (Data/Outputs_*.hdf5) of\n    {case_dir}\n"
        "which are not included in data/ (time histories only). Re-run the simulation, which writes "
        "results/, or set PAPER_RESULTS_DIR to a folder that holds a copy of the full result folders "
        f"of the paper (the Results_* folders). {data_only}"
    )
