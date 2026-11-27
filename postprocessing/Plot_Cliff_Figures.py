"""Manuscript figures of the ice-cliff case (Section 4.2).

Reads the cliff runs (results/Results_Cliff/<case>, or data/ for the time histories of the
paper runs) and writes, at final printed size (see the figure conventions of Section 3):

- Cliff_Criterion_<Original|Opening>_<Elastic|Viscoelastic> (Fig. 16):  phase field next
                     to the cliff face for the strength surface of Section 2 and the
                     opening-only one, early in the lowering: the runs with the original
                     criterion at the step at which they were stopped, the opening-only
                     runs at the output closest to the same time (both from the runs with
                     the 0.25 m crack-advance limit).
- Cliff_Final_<Elastic|Viscoelastic>_<phasefield|ux> (Fig. 17):  phase field and horizontal
                     displacement of the whole cliff at the last output of the
                     opening-only runs with the 5 m crack-advance limit.
- Cliff_LengthScale (Fig. 18):  depth of the deepest crevasse against time and water level
                     for the phase-field length scales 1, 2 and 4 m, elastic (solid) and
                     viscoelastic (dashed) (opening-only criterion, 5 m crack-advance
                     limit; 8 x 6 cm).

The stills are 6.8 cm wide with a height that follows from the window; u_x is in mm.
The stills need the field outputs of the runs; Fig. 18 needs the time histories only.

The crack depth is the Crack_Depth measure (lowest point with phi > 0.5 below the surface),
and the depth of the notches before any crevasse has grown. Usage:

    python Plot_Cliff_Figures.py                       # Figs. 16, 17 and 18
    python Plot_Cliff_Figures.py --length-scale-only   # Fig. 18 only (from data/)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# The runs may still be writing their output: a reader that locks the file makes the solver
# fail at its next output.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.tri as mtri
import numpy as np

from plotting.paths import require_field_outputs, study_folder

ROOT = study_folder("Results_Cliff")   # results/ or data/ (plotting/paths.py)
FIGURES = Path(__file__).resolve().parents[1] / "figures"   # figures/ in the repository root
FIGURES.mkdir(exist_ok=True)
FONTSIZE = 7
LINEWIDTH = 0.5
LINE_SIZE_CM = (8.0, 6.0)
STILL_WIDTH_CM = 6.8
STILL_IMAGE_FRACTION = 0.77        # part of the width of a still taken by the field
RASTER_DPI = 600
CLIFF_HEIGHT = 200.0               # m (Cliff_Params.CLIFF_HEIGHT)
NOTCH_DEPTH = 10.0                 # m (Cliff_Params.NOTCH_DEPTH)
START_LEVEL = 182.0                # m, water level during the spin-up and at t = 0
LOWERING_TIME = 3600.0             # s over which the water level falls from START_LEVEL to zero
CASE = "new_wl_{rheology}_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_{limit}"
ORIGINAL_CASE = "new_wl_{rheology}_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_r2_fs1.5_nostag_csf1"
CLIFF_LENGTH = 500.0
CRITERION_WINDOW = (300.0, 500.0, 100.0, 200.0)     # x0, x1, y0, y1 [m]: upper half next to the face
ADVANCE_LIMITS = {"5": "csf20", "0.25": "csf1"}     # crack advance per step [m]: folder suffix
RHEOLOGIES = ("elastic", "viscoelastic")
LEVEL_TICKS = (180.0, 150.0, 120.0, 90.0, 60.0, 30.0, 0.0)
# Length-scale study (opening-only criterion): ell [m]: (case folder, colour, marker). All with a crack advance of 5 m per step.
LENGTH_SCALE_CASE = "new_wl_{rheology}_FreeSlip_l{ell}_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_adv5m"
LENGTH_SCALES = {1: (CASE.replace("{limit}", "csf20"), "#0072B2", "o"), 2: (LENGTH_SCALE_CASE.replace("{ell}", "2"), "#D55E00", "s"),
                 4: (LENGTH_SCALE_CASE.replace("{ell}", "4"), "#009E73", "^")}
LENGTH_SCALE_LINES = {"elastic": "-", "viscoelastic": "--"}
LENGTH_SCALE_MARK_EVERY = 450.0    # s between the markers of one curve

plt.rcParams.update({"font.size": FONTSIZE, "axes.linewidth": LINEWIDTH,
                     "xtick.major.width": LINEWIDTH, "ytick.major.width": LINEWIDTH,
                     "lines.linewidth": LINEWIDTH, "legend.frameon": False})


def load_history(case: str) -> dict | None:
    path = ROOT / case / "Data" / "time_data.hdf5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        names = [n.decode() for n in f["data_names"][:]]
        data = f["data"][:]
    return {name: data[:, i] for i, name in enumerate(names)}


def water_level(time):
    return START_LEVEL * np.clip(1.0 - np.asarray(time, dtype=float) / LOWERING_TIME, 0.0, 1.0)


def _save(fig, name, dpi=RASTER_DPI):
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi, facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi, facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Saved {FIGURES / name}.pdf")


def output_steps(case: str) -> list[int]:
    return sorted(int(path.stem.split("_")[-1]) for path in (ROOT / case / "Data").glob("Outputs_*.hdf5"))


def output_time(case: str, step: int) -> float:
    """Time of the output Outputs_<step>.hdf5 (row ``step`` of the time history)."""
    return float(load_history(case)["time"][step])


def output_near(case: str, time: float) -> int:
    """Output closest in time to ``time``; the last output if the run has not reached it."""
    times = load_history(case)["time"]
    steps = [step for step in output_steps(case) if step < len(times)]
    return min(steps, key=lambda step: abs(times[step] - time))


def load_field(case: str, step: int, field: str = "phasefield"):
    """Triangulation and DG1 nodal values of one output."""
    data = ROOT / case / "Data"
    with h5py.File(data / f"Outputs_{step}.hdf5", "r") as f:
        values = f[f"Fields/{field}"][:]
        epoch = int(f["epoch_tag"][()])
    with h5py.File(data / f"mesh_epoch_{epoch:03d}.hdf5", "r") as f:
        x = f["Mesh_DG1/X"][:]
        y = f["Mesh_DG1/Y"][:]
    n = x.shape[0]
    tri = mtri.Triangulation(x.ravel(), y.ravel(), np.arange(3 * n).reshape(n, 3))
    return tri, values[:, :3].ravel()


def field_still(case: str, step: int, name: str, field: str = "phasefield", window=None, scale: float | None = None):
    """Phase field or u_x [mm] at one output in the window (x0, x1, y0, y1), the whole
    cliff by default, with a scale bar of ``scale`` m in the lower left corner if given."""
    x0, x1, y0, y1 = window or (0.0, CLIFF_LENGTH, 0.0, CLIFF_HEIGHT)
    tri, values = load_field(case, step, field)
    if field == "phasefield":
        cmap, clim, title, colour = "inferno", (0.0, 1.0), r"$\phi$ [-]", "w"
        ticks, fmt = [0, 0.25, 0.5, 0.75, 1.0], "%.2f"
    elif field == "u_x":
        inside = (tri.x >= x0) & (tri.x <= x1) & (tri.y >= y0) & (tri.y <= y1)
        values = values * 1e3
        peak = float(values[inside].max())
        # Tick spacing that keeps the labels of the colour bar apart (at most eight ticks).
        spacing = next(step for step in (10.0, 20.0, 50.0, 100.0, 200.0, 500.0) if np.ceil(peak / 10.0) * 10.0 / step <= 7.0)
        limit = float(spacing * np.ceil(peak / spacing))
        cmap, clim, title, colour = "viridis", (0.0, limit), r"$u_x$ [mm]", "w"
        ticks, fmt = np.arange(0.0, limit + 1.0, spacing), "%.0f"
    else:
        raise ValueError(f"unsupported field {field}")
    width = STILL_WIDTH_CM
    image_height = STILL_IMAGE_FRACTION * width * (y1 - y0) / (x1 - x0)
    height = image_height + 0.3
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))
    ax = fig.add_axes((0.01, 0.15 / height, STILL_IMAGE_FRACTION, image_height / height))
    image = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, vmin=clim[0], vmax=clim[1], rasterized=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if scale is not None:
        xs, ys = x0 + 0.04 * (x1 - x0), y0 + 0.10 * (y1 - y0)
        ax.plot([xs, xs + scale], [ys, ys], color=colour, linewidth=1.0, solid_capstyle="butt")
        ax.text(xs + 0.5 * scale, ys + 0.03 * (y1 - y0), f"{scale:g} m", color=colour, ha="center", va="bottom",
                fontsize=FONTSIZE)
    cax = fig.add_axes((0.82, 0.15 / height + 0.02, 0.018, 0.68 * image_height / height))
    bar = fig.colorbar(image, cax=cax, ticks=ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONTSIZE, width=LINEWIDTH, length=2)
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    fig.text(0.80, (0.15 + 0.80 * image_height) / height, title, fontsize=FONTSIZE, ha="left", va="bottom")
    _save(fig, name)


def criterion_stills():
    """Original against opening-only strength surface, next to the cliff face. Both from
    runs with the 0.25 m crack-advance limit, so that the criterion is the only difference."""
    for rheology in RHEOLOGIES:
        original = ORIGINAL_CASE.format(rheology=rheology)
        opening = CASE.format(rheology=rheology, limit=ADVANCE_LIMITS["0.25"])
        for needed in (original, opening):
            require_field_outputs(ROOT / needed, "The criterion stills (Fig. 16)",
                                  'Without them, "python postprocessing/Plot_Cliff_Figures.py --length-scale-only" draws Fig. 18 from data/.')
        step = output_steps(original)[-1]
        time = output_time(original, step)
        other = output_near(opening, time)
        for case, number, criterion in ((original, step, "Original"), (opening, other, "Opening")):
            instant = output_time(case, number)
            print(f"criterion still: {criterion}, {rheology}: output {number}, t = {instant:.1f} s, "
                  f"water {water_level(instant):.1f} m, crack depth {load_history(case)['Crack_Depth'][number]:.1f} m")
            field_still(case, number, f"Cliff_Criterion_{criterion}_{rheology.capitalize()}", window=CRITERION_WINDOW,
                        scale=50.0)


def final_stills(limit: str):
    """Phase field and horizontal displacement of the whole cliff at the last output."""
    for rheology in RHEOLOGIES:
        case = CASE.format(rheology=rheology, limit=ADVANCE_LIMITS[limit])
        require_field_outputs(ROOT / case, "The final stills (Fig. 17)",
                              'Without them, "python postprocessing/Plot_Cliff_Figures.py --length-scale-only" draws Fig. 18 from data/.')
        step = output_steps(case)[-1]
        instant = output_time(case, step)
        print(f"final still: {rheology}: output {step}, t = {instant:.1f} s, water {water_level(instant):.1f} m, "
              f"crack depth {load_history(case)['Crack_Depth'][step]:.1f} m")
        for field, label in (("phasefield", "phasefield"), ("u_x", "ux")):
            field_still(case, step, f"Cliff_Final_{rheology.capitalize()}_{label}", field=field)


def length_scale_figure(name: str = "Cliff_LengthScale"):
    """Depth of the deepest crevasse against time and water level for the phase-field length
    scales of LENGTH_SCALES: colour and marker per length scale, elastic solid and
    viscoelastic dashed. The markers are placed at fixed times, shifted between the
    length scales so that they stay apart where the curves coincide."""
    fig = plt.figure(figsize=(LINE_SIZE_CM[0] / 2.54, LINE_SIZE_CM[1] / 2.54))
    ax = fig.add_axes((0.17, 0.17, 0.80, 0.66))
    end_time = LOWERING_TIME
    marker_style = dict(markersize=3.0, markerfacecolor="none", markeredgewidth=LINEWIDTH)
    for rheology, line_style in LENGTH_SCALE_LINES.items():
        for number, (ell, (case, colour, marker)) in enumerate(LENGTH_SCALES.items()):
            history = load_history(case.format(rheology=rheology, ell=ell))
            if history is None:
                print(f"length-scale figure: no output of {case.format(rheology=rheology, ell=ell)}, skipped")
                continue
            keep = history["time"] >= 0.0
            time = history["time"][keep]
            depth = np.maximum(history["Crack_Depth"][keep], NOTCH_DEPTH)
            end_time = max(end_time, time[-1])
            ax.plot(time, depth, color=colour, linestyle=line_style)
            marks = np.arange(LENGTH_SCALE_MARK_EVERY * (number + 1) / len(LENGTH_SCALES), time[-1], LENGTH_SCALE_MARK_EVERY)
            ax.plot(marks, np.interp(marks, time, depth), color=colour, linestyle="none", marker=marker, **marker_style)
            print(f"length-scale figure: {rheology}, ell = {ell} m: last step t = {time[-1]:.0f} s "
                  f"(water {water_level(time[-1]):.1f} m), depth {depth[-1]:.1f} m")
    for ell, (_, colour, marker) in LENGTH_SCALES.items():
        ax.plot([], [], color=colour, marker=marker, label=rf"$\ell={ell}\;\mathrm{{m}}$", **marker_style)
    for rheology, line_style in LENGTH_SCALE_LINES.items():
        ax.plot([], [], color="0.4", linestyle=line_style, label=rheology)
    ax.set_xlabel(r"$t\;[\mathrm{s}]$")
    ax.set_ylabel(r"$d_\mathrm{c}\;[\mathrm{m}]$")
    ax.set_xlim(0.0, end_time)
    ax.set_ylim(0.0, CLIFF_HEIGHT)
    ax.legend(loc="lower right", fontsize=FONTSIZE, handlelength=3.0)
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks([LOWERING_TIME * (1.0 - level / START_LEVEL) for level in LEVEL_TICKS])
    top.set_xticklabels([f"{level:.0f}" for level in LEVEL_TICKS])
    top.set_xlabel(r"$h_\mathrm{w}\;[\mathrm{m}]$")
    _save(fig, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--length-scale-only", action="store_true",
                        help="draw the length-scale figure (Fig. 18) only")
    args = parser.parse_args()
    length_scale_figure()
    if args.length_scale_only:
        return
    criterion_stills()
    final_stills("5")


if __name__ == "__main__":
    main()
