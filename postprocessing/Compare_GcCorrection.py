"""Effect of the element-size correction of the fracture energy (Fig. 10).

    python Compare_GcCorrection.py                                            # Fig. 10a (Gc = 10)
    python Compare_GcCorrection.py --study Results_dxSweep_LO --xmax 0.035    # Fig. 10b (Gc = 1)

Overlays the load-displacement curves of the element-size study ``--study`` without
(dashed) and with (solid) the correction Gc/(1 + 0.33 h/ell), the study folder
``<study>_GcCorr0.33``, and writes ``figures/gc_correction/<study>_GcCorr0.33_overlay.pdf``.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from plotting.paths import study_folder

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR.parent / "figures" / "gc_correction"   # Fig. 10: <study>_GcCorr<c>_overlay.pdf
COEFFICIENT = "0.33"


def load_case(case_dir):
    with h5py.File(case_dir / "Data/time_data.hdf5") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["data_names"][()]]
        data = np.asarray(h["data"])
    u = np.abs(data[:, names.index("BC_Top_u")]) * 1e3      # mm
    F = np.abs(data[:, names.index("BC_Top_F")]) / 1e6      # MPa (force per unit width / Lx = 1 m)
    return u, F


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="Results_dxSweep_LO_Gc10")
    parser.add_argument("--xmax", type=float, default=0.25, help="x-axis limit [mm]")
    args = parser.parse_args(argv)
    studies = {"uncorrected": study_folder(args.study),
               f"Gc/(1+{COEFFICIENT} h/ell)": study_folder(f"{args.study}_GcCorr{COEFFICIENT}")}

    results = {}
    for label, folder in studies.items():
        for case_dir in sorted(folder.glob("dx_*"), key=lambda p: int(p.name.split("_")[1])):
            if (case_dir / "Data/time_data.hdf5").is_file():
                results[(label, int(case_dir.name.split("_")[1]))] = load_case(case_dir)
    _manuscript_figure(args, results, studies)


def _manuscript_figure(args, results, studies):
    """8 x 6 cm overlay in the manuscript style: uncorrected dashed, corrected solid, legend below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from plotting.PlotLines import (
        FONTSIZE, HISTORY_MARKER_STYLE, HISTORY_MARKERS, FormatAndSave, Mark_Curve,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    labels = list(studies)
    styles = {labels[0]: "--", labels[1]: "-"}
    meshes = sorted({dx for _, dx in results})
    fig, ax = plt.subplots()
    for label in labels:
        for (lab, dx), (u, F) in sorted(results.items(), key=lambda kv: kv[0][1]):
            if lab != label:
                continue
            colour = colours[(dx - 1) % len(colours)]
            ax.plot(u, F, color=colour, linestyle=styles[label])
            Mark_Curve(ax, u, F, meshes.index(dx), len(meshes), args.xmax, colour,
                       shift=labels.index(label) / len(labels))
    handles = [
        Line2D([], [], color=colours[(dx - 1) % len(colours)], label=rf"$\ell/h^{{\mathrm{{el}}}}={dx}$",
               marker=HISTORY_MARKERS[number % len(HISTORY_MARKERS)], **HISTORY_MARKER_STYLE)
        for number, dx in enumerate(meshes)
    ]
    style_handles = [
        Line2D([], [], color="0.25", linestyle="--", label="uncorrected"),
        Line2D([], [], color="0.25", linestyle="-", label=r"$G_\text{c}/(1+\frac{1}{3}h^{\mathrm{el}}/\ell)$"),
        Line2D([], [], linestyle="none", label=""),
    ]
    ax.set_xlabel(r"$t\,\dot{\varepsilon}_{\text{ext}}\,L_y\;[\mathrm{mm}]$")
    ax.set_ylabel(r"$\overline{\sigma_{yy}}\;[\text{MPa}]$")
    ax.set_xlim([0.0, args.xmax])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    # Layout of the history panels of Plot_Lines: line styles in the first legend column.
    ax.set_ylim([0.0, 0.25])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    fig.legend(
        handles=style_handles + handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3,
        frameon=True, fancybox=False, framealpha=1.0, edgecolor="black",
        fontsize=FONTSIZE, borderpad=0.25, columnspacing=0.5, handletextpad=0.3,
    )
    save_name = str(OUTPUT / f"{args.study}_GcCorr{COEFFICIENT}_overlay")
    FormatAndSave(fig, ax, save_name, top=0.95, bottom=0.43)
    plt.close(fig)
    print(f"Figure written to {save_name}.pdf")


if __name__ == "__main__":
    main()
