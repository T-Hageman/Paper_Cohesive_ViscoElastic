"""Manuscript figures of the sea-ice floe case (Section 4.1).

Reads the output of simulations/Run_Floe_Cases.py (results/Results_Floe/<case>, or data/ for
the time histories of the paper runs) and writes to figures/, at final printed size (see
the figure conventions of Section 3):

- SeaIce_Mesh (Fig. 12b):  adaptive mesh of the fragmentation case at its last pattern
                          stage, coloured by element size (6.8 x 3.0 cm)
- SeaIce_Pattern_<k>_phasefield / _ux (Fig. 14): phase field and u_x of the fragmentation
                          case at the output steps STAGES, one row per step (6.8 x 5.4 cm)
- SeaIce_Stress (Fig. 13): nominal stress against far-field strain of the three rates and
                          the elastic reference, up to the nucleation of the crack (8 x 6 cm)

All cases are simulated on the upper half of the floe (symmetry about the horizontal
centreline through the defect). The fragmentation case is the fast rate (PATTERN_CASE);
the other cases of the rate comparison (RATE_CASES, ELASTIC_CASE) that fracture were
stopped by hand just after nucleation. Forces and crack lengths of an upper-half model
are converted to the full floe (floe_fraction). The mesh and pattern figures need the
field outputs of the fragmentation case; Fig. 13 needs the time histories only. Usage:

    python Plot_Floe_Figures.py                 # Figs. 12b, 13 and 14
    python Plot_Floe_Figures.py --lines-only    # Fig. 13 only (from data/)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from plotting.paths import require_field_outputs, study_folder

ROOT = study_folder("Results_Floe")   # results/ or data/ (plotting/paths.py)
FIGURES = Path(__file__).resolve().parents[1] / "figures"   # figures/ in the repository root
FIGURES.mkdir(exist_ok=True)
FLOE_SIDE = 100.0
CENTRE = np.array([50.0, 50.0])
H_FLOE = 1.0                       # m, thickness used for the nominal stress (force per unit width / side)
FONTSIZE = 7
LINEWIDTH = 0.5
HALF_MESH_HEIGHT_CM = 3.0          # mesh figure of an upper-half model, at the width of a still
STILL_SIZE_CM = (6.8, 5.4)
LINE_SIZE_CM = (8.0, 6.0)
RASTER_DPI = 600
MESH_DPI = 1200                    # the mesh still needs individual elements to be visible
STAGES = (410, 1210, 2060)         # output steps of the fragmentation case: rows of the pattern figure
PATTERN_HALF_WIDTH = 20.0          # m, half width of the pattern window, which starts at the symmetry line
PATTERN_CASE = "fast_l0.25_sym"    # upper half of the floe; force and crack length are half-floe values
RATE_CASES = {                     # folder: legend label
    PATTERN_CASE: r"$\dot{\varepsilon}_\mathrm{ext}=10^{-8}\;\mathrm{s^{-1}}$",
    "intermediate_l0.25_sym": r"$\dot{\varepsilon}_\mathrm{ext}=2.5\cdot10^{-9}\;\mathrm{s^{-1}}$",
    "slow_l0.25_sym": r"$\dot{\varepsilon}_\mathrm{ext}=10^{-9}\;\mathrm{s^{-1}}$",
}
ELASTIC_CASE = "fast_l0.25_sym_elastic"
NUCLEATION_LENGTH = 0.1            # m of crack-length measure (per full floe) that marks nucleation
# Largest cohesive strain that counts as the strength surface being reached. From a local
# stress of about 0.7 f_t the return map classifies the opening mode as active and carries
# multipliers of 2e-12 to 1e-10, without the surface being reached; when it is, Eta_Max
# jumps to above 5e-8 within one step.
ONSET_ETA = 1.0e-9
DYNAMIC_DT = 1.0e-3                # s, time increments below this belong to the dynamic fracture

plt.rcParams.update({"font.size": FONTSIZE, "axes.linewidth": LINEWIDTH,
                     "xtick.major.width": LINEWIDTH, "ytick.major.width": LINEWIDTH,
                     "lines.linewidth": LINEWIDTH, "legend.frameon": False})


# ----------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------
def output_steps(case: str) -> list[int]:
    files = glob.glob(str(ROOT / case / "Data" / "Outputs_*.hdf5"))
    return sorted(int(Path(f).stem.split("_")[-1]) for f in files)


def load_field(case: str, step: int, field: str = "phasefield"):
    """Triangulation (centred on the defect) and DG1 nodal values of one output."""
    data = ROOT / case / "Data"
    with h5py.File(data / f"Outputs_{step}.hdf5", "r") as f:
        values = f[f"Fields/{field}"][:]
        epoch = int(f["epoch_tag"][()])
    with h5py.File(data / f"mesh_epoch_{epoch:03d}.hdf5", "r") as f:
        x = f["Mesh_DG1/X"][:] - CENTRE[0]
        y = f["Mesh_DG1/Y"][:] - CENTRE[1]
    n = x.shape[0]
    tri = mtri.Triangulation(x.ravel(), y.ravel(), np.arange(3 * n).reshape(n, 3))
    return tri, values[:, :3].ravel()


def floe_fraction(case: str) -> float:
    """Part of the floe that the case simulates: 0.5 for an upper-half model, else 1."""
    settings = json.loads((ROOT / case / "case_settings.json").read_text())
    return 0.5 if settings["parameters"].get("symmetric") else 1.0


def load_history(case: str) -> dict | None:
    """Time history of a case; the reaction force and the crack-length measure of an
    upper-half model are doubled, so that they hold for the full floe."""
    path = ROOT / case / "Data" / "time_data.hdf5"
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        names = [n.decode() for n in f["data_names"][:]]
        data = f["data"][:]
    history = {name: data[:, i] for i, name in enumerate(names)}
    for name in ("BC_Right_F", "Crack_Length"):
        history[name] = history[name] / floe_fraction(case)
    return history


def nucleation_step(history: dict) -> int | None:
    """First step at which the crack-length measure (per full floe) exceeds NUCLEATION_LENGTH."""
    started = np.nonzero(history["Crack_Length"] > NUCLEATION_LENGTH)[0]
    return int(started[0]) if started.size else None


def dynamic_onset(history: dict) -> int | None:
    """Last step before the time increment drops below DYNAMIC_DT: the start of the
    dynamic fracture, which the times "since nucleation" are counted from."""
    dynamic = np.nonzero(np.diff(history["time"]) < DYNAMIC_DT)[0]
    return int(dynamic[0]) if dynamic.size else None


def _figure(size_cm):
    return plt.figure(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))


def _save(fig, name, dpi=RASTER_DPI):
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi, facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi, facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"Saved {FIGURES / name}.pdf")


def _scale_bar(ax, half_width, colour="w", centre=(0.0, 0.0)):
    """Scale bar in the lower left corner of a window of +-half_width around centre."""
    length = 0.5
    for candidate in (1, 2, 5, 10, 20):
        if candidate <= 0.45 * half_width:
            length = candidate
    x0, y0 = centre[0] - 0.9 * half_width, centre[1] - 0.88 * half_width
    ax.plot([x0, x0 + length], [y0, y0], color=colour, linewidth=1.0, solid_capstyle="butt")
    ax.text(x0 + 0.5 * length, y0 + 0.04 * half_width, f"{length:g} m", color=colour,
            ha="center", va="bottom", fontsize=FONTSIZE)


def field_still(case: str, step: int, half_width: float, name: str, field: str = "phasefield",
                limit: float | None = None, centre=(0.0, 0.0)):
    """Phase field or u_x at one output, square window of +-half_width around ``centre``
    (in coordinates centred on the defect).

    u_x is drawn in mm with symmetric limits (+-limit, default: the largest |u_x| in
    the window, rounded up to 0.1 mm), as the Section 3 displacement stills.
    """
    tri, values = load_field(case, step, field)
    if field == "phasefield":
        cmap, clim, title = "inferno", (0.0, 1.0), r"$\phi$ [-]"
        ticks, fmt = [0, 0.25, 0.5, 0.75, 1.0], "%.2f"
    elif field == "u_x":
        values = values * 1e3
        if limit is None:
            limit = ux_limit(case, step, half_width, centre)
        cmap, clim, title = "coolwarm", (-limit, limit), r"$u_x$" + "\n" + r"[mm]"
        ticks, fmt = np.linspace(-limit, limit, 5), "%.2f"
    else:
        raise ValueError(f"unsupported field {field}")
    fig = _figure(STILL_SIZE_CM)
    aspect = STILL_SIZE_CM[1] / STILL_SIZE_CM[0]
    ax = fig.add_axes((0.02, 0.02, 0.96 * aspect, 0.96))
    image = ax.tripcolor(tri, values, shading="gouraud", cmap=cmap, vmin=clim[0], vmax=clim[1],
                         rasterized=True)
    ax.set_xlim(centre[0] - half_width, centre[0] + half_width)
    ax.set_ylim(centre[1] - half_width, centre[1] + half_width)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _scale_bar(ax, half_width, colour="w" if field == "phasefield" else "k", centre=centre)
    cax = fig.add_axes((0.80, 0.18, 0.018, 0.56))
    bar = fig.colorbar(image, cax=cax, ticks=ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONTSIZE, width=LINEWIDTH, length=2)
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(fmt))
    fig.text(0.80, 0.78, title, fontsize=FONTSIZE, ha="left", va="bottom")
    _save(fig, name)


def ux_limit(case: str, step: int, half_width: float, centre=(0.0, 0.0)) -> float:
    """Largest |u_x| [mm] in the window, rounded up to 0.1 mm."""
    tri, values = load_field(case, step, "u_x")
    inside = (np.abs(tri.x - centre[0]) <= half_width) & (np.abs(tri.y - centre[1]) <= half_width)
    return float(0.1 * np.ceil(np.abs(values[inside]).max() * 1e3 / 0.1))


def _mesh_edges(tri):
    """Edge segments of every triangle and the characteristic size h (longest edge) per edge.

    Same definition of h and the same per-edge colouring as the Section 3 mesh stills
    (plotting/PlotAnimations._mesh_grid).
    """
    corners = np.column_stack((tri.x, tri.y))[tri.triangles]            # (n, 3, 2)
    segments = np.stack((corners, np.roll(corners, -1, axis=1)), axis=2)  # (n, 3, 2, 2)
    lengths = np.linalg.norm(segments[:, :, 1] - segments[:, :, 0], axis=2)
    h = lengths.max(axis=1)
    return segments.reshape(-1, 2, 2), np.repeat(h, 3), h


def mesh_figure(case: str, step: int, name: str = "SeaIce_Mesh"):
    """Adaptive mesh of the simulated upper half of the floe, edges coloured by element
    size (as the Section 3 mesh stills)."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    tri, phi = load_field(case, step)
    print(f"mesh figure: step {step}, overall mesh only")
    segments, edge_h, h = _mesh_edges(tri)
    print(f"mesh figure: {len(h)} elements, h from {h.min():.3g} to {h.max():.3g} m")
    norm = LogNorm(vmin=h.min(), vmax=h.max())
    # the upper-half model is drawn as simulated, at the width of a still and about half its height
    fig = _figure((STILL_SIZE_CM[0], HALF_MESH_HEIGHT_CM))
    ax = fig.add_axes((0.02, 0.02, 0.96 * STILL_SIZE_CM[1] / STILL_SIZE_CM[0], 0.96))
    half = FLOE_SIDE / 2
    lines = LineCollection(segments, array=edge_h, cmap="viridis", norm=norm, linewidths=0.1,
                           rasterized=True)
    ax.add_collection(lines)
    ax.set_xlim(-half, half); ax.set_ylim(0.0, half)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes((0.80, 0.18, 0.018, 0.56))
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax,
                       ticks=np.geomspace(h.min(), h.max(), 5))
    bar.minorticks_off()
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=FONTSIZE, width=LINEWIDTH, length=2)
    bar.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: rf"${v:.2g}$"))
    fig.text(0.80, 0.78, r"$h^\mathrm{el}$ [m]", fontsize=FONTSIZE, ha="left", va="bottom")
    _save(fig, name, dpi=MESH_DPI)


def _line_axes():
    fig = _figure(LINE_SIZE_CM)
    ax = fig.add_axes((0.17, 0.17, 0.80, 0.78))
    return fig, ax


def stress_figure(name: str = "SeaIce_Stress"):
    """Nominal stress against far-field strain of the rate cases, each up to the last
    step before its crack nucleates (cross); the fracture that follows takes milliseconds
    at constant far-field strain. A circle marks the step at which the strength surface
    is first reached (Eta_Max > ONSET_ETA)."""
    fig, ax = _line_axes()
    cases = [(ELASTIC_CASE, "elastic", {"color": "k", "linestyle": "--"})]
    cases += [(case, label, {}) for case, label in RATE_CASES.items()]
    for case, label, style in cases:
        history = load_history(case)
        if history is None:
            continue
        strain = 2.0 * history["BC_Right_u"] / FLOE_SIDE
        stress = history["BC_Right_F"] / (FLOE_SIDE * H_FLOE) / 1e6
        nucleation = nucleation_step(history)
        last = len(strain) if nucleation is None else nucleation       # exclusive
        line, = ax.plot(strain[:last] * 1e5, stress[:last], label=label, **style)
        onset = np.argmax(history["Eta_Max"] > ONSET_ETA)
        if history["Eta_Max"][onset] > ONSET_ETA and onset < last:
            print(f"stress figure: {case}: strength surface reached at step {onset + 1}, strain {strain[onset]:.3e}, "
                  f"nominal stress {stress[onset]:.4f} MPa")
            ax.plot(strain[onset] * 1e5, stress[onset], "o", markersize=2.5, color=line.get_color(),
                    markerfacecolor="none", markeredgewidth=LINEWIDTH)
        if nucleation is not None:
            ax.plot(strain[last - 1] * 1e5, stress[last - 1], "x", markersize=3.5,
                    color=line.get_color(), markeredgewidth=LINEWIDTH)
        print(f"stress figure: {case}: " + (
            f"nucleation after step {last - 1}, t = {history['time'][last - 1]:.1f} s, "
            f"strain {strain[last - 1]:.3e}, nominal stress {stress[last - 1]:.4f} MPa"
            if nucleation is not None else
            f"no nucleation up to strain {strain[-1]:.3e}; stress {stress[-1]:.4f} MPa "
            f"(largest {stress.max():.4f} MPa)"))
    ax.set_xlabel(r"$\varepsilon_\mathrm{ext}\;[10^{-5}]$")
    ax.set_ylabel(r"$F/(Lh)\;[\mathrm{MPa}]$")
    ax.set_xlim(left=0.0); ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower right", fontsize=FONTSIZE)
    _save(fig, name)


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lines-only", action="store_true",
                        help="draw the stress figure (Fig. 13) only")
    args = parser.parse_args()

    stress_figure()
    if args.lines_only:
        return
    require_field_outputs(ROOT / PATTERN_CASE, "The mesh and pattern figures (Figs. 12b, 14)",
                          'Without them, "python postprocessing/Plot_Floe_Figures.py --lines-only" draws Fig. 13 from data/.')
    steps = output_steps(PATTERN_CASE)
    stages = [min(steps, key=lambda s: abs(s - target)) for target in STAGES]
    half = PATTERN_HALF_WIDTH
    centre = (0.0, half)
    history = load_history(PATTERN_CASE)
    t0 = history["time"][dynamic_onset(history)]
    limit = ux_limit(PATTERN_CASE, stages[-1], half, centre)
    print(f"pattern stages {stages}, window {2 * half:g} x {2 * half:g} m, u_x limit {limit:g} mm")
    for k, step in enumerate(stages, start=1):
        tri, phi = load_field(PATTERN_CASE, step)
        print(f"  stage {k}: step {step}, {1e3 * (history['time'][step - 1] - t0):.2f} ms after nucleation, "
              f"cracks to {tri.y[phi > 0.5].max():.1f} m from the defect, {len(tri.triangles)} elements")
        field_still(PATTERN_CASE, step, half, f"SeaIce_Pattern_{k}_phasefield", centre=centre)
        field_still(PATTERN_CASE, step, half, f"SeaIce_Pattern_{k}_ux", field="u_x", limit=limit,
                    centre=centre)
    mesh_figure(PATTERN_CASE, stages[-1])


if __name__ == "__main__":
    main()
