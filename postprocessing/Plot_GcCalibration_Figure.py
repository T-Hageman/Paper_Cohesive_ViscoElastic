"""Manuscript figure for the Gc-correction calibration (Appendix B).

Writes, in figures/ (repository root):
    GcCorrection_Geometry.pdf     (a) notched-bar geometry and extraction line
    GcCorrection_LoadDisp.pdf     (b) bar load-displacement curves, uncorrected/corrected
    GcCorrection_Calibration.pdf  (c) effective Gc versus element size
Run from postprocessing/ after simulations/Calibrate_Gc_Correction.py --analyse.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulations"))   # Calibrate_Gc_Correction
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from plotting.PlotLines_Appendix import FONTSIZE, FormatAndSave  # noqa: E402
import Calibrate_Gc_Correction as C  # noqa: E402
from plotting.paths import study_folder  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"   # figures/ in the repository root
FIGURES.mkdir(exist_ok=True)
RESULTS = study_folder("Results_GcCorrection")   # results/ or data/ (plotting/paths.py)
EXTRACT_UM = (120.0, 200.0)       # end displacements at which the local law is read
STATIONS = (0.07, 0.13)           # y-range of the extraction stations
CORRECTED = "c0.33"


def geometry_panel():
    """Panel (a): drawn by Plot_Geometry_Figures in the common sketch style."""

    from Plot_Geometry_Figures import notched_bar

    notched_bar(C.LX, C.LY, C.NOTCH_WIDTH, C.NOTCH_DEPTH, C.LX_CENTRE, STATIONS)


def load_curve(case_dir):
    with h5py.File(case_dir / "Data/time_data.hdf5") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["data_names"][()]]
        data = np.asarray(h["data"])
    u = np.abs(data[:, names.index("BC_Right_u")]) * 1e6             # um
    F = np.abs(data[:, names.index("BC_Right_F")]) / C.LIGAMENT / 1e6  # MPa, nominal ligament stress
    return u, F


def load_displacement_panel(ratios):
    fig, ax = plt.subplots()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for tag, style in (("uncorrected", "--"), (CORRECTED, "-")):
        for i, dx in enumerate(ratios):
            case_dir = RESULTS / tag / f"dx_{dx}"
            if not (case_dir / C.COMPLETION_FILE).is_file():
                continue
            u, F = load_curve(case_dir)
            ax.plot(u, F, color=colours[i % len(colours)], linestyle=style,
                    label=rf"$\ell/h^{{\mathrm{{el}}}}={dx}$" if tag == CORRECTED else "_nolegend_")
    for u_ext in EXTRACT_UM:
        ax.axvline(u_ext, color="C3", linestyle=":", linewidth=0.8)
    ax.plot([], [], "k--", label="uncorrected")
    ax.plot([], [], "k-", label=r"$G_\text{c}/(1+\frac{1}{3}h^{\mathrm{el}}/\ell)$")
    ax.set_xlabel(r"$u\;[\mu\text{m}]$")
    ax.set_ylabel(r"$\overline{\sigma}\;[\text{MPa}]$")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 0.42)
    ax.legend(fontsize=FONTSIZE, ncol=2, columnspacing=0.8, handlelength=1.5, loc="upper right")
    FormatAndSave(fig, ax, str(FIGURES / "GcCorrection_LoadDisp"))
    plt.close(fig)


def calibration_panel(rows):
    """Panel (c): Gc_num/Gc against h/ell, uncorrected and corrected with c = 1/4, 1/3, 1/2."""

    fig, ax = plt.subplots()
    # points only: the first-order lines 1 + c h/ell cluttered the panel
    coefficients = (("c0.25", 4, "^", ":"), ("c0.33", 3, "s", "-"), ("c0.5", 2, "v", "--"))
    series = [("uncorrected", "o", "uncorrected")] + [
        (tag, marker, rf"$G_\text{{c}}/(1+\frac{{1}}{{{denominator}}}h^{{\mathrm{{el}}}}/\ell)$")
        for tag, denominator, marker, _ in coefficients]
    for tag, marker, label in series:
        sub = sorted((r for r in rows if r["tag"] == tag), key=lambda r: r["h_mean"])
        if not sub:
            continue
        s = [r["h_mean"] / C.ELL for r in sub]
        g = [0.5 * (r["Gc_local_120um"] + r["Gc_local_200um"]) for r in sub]
        ax.plot(s, g, marker=marker, linestyle="none", markersize=3.5, markeredgewidth=0.5, label=label)
    ax.axhline(1.0, color="0.6", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"$h^{\mathrm{el}}/\ell$")
    ax.set_ylabel(r"$G_\text{c}^\text{num}/G_\text{c}$")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0.9, 1.3)
    ax.legend(fontsize=FONTSIZE, loc="upper left")
    FormatAndSave(fig, ax, str(FIGURES / "GcCorrection_Calibration"))
    plt.close(fig)


def main():
    rows = json.load(open(RESULTS / "analysis.json"))
    ratios = sorted({r["dxratio"] for r in rows})
    geometry_panel()
    load_displacement_panel(ratios)
    calibration_panel(rows)
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
