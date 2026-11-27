"""Figures of the plate with a hole (Section 3), from the sweeps of simulations/Run_Plate_Sweeps.py.

    python Process_All_Data.py            # Figs. 8, 9 and 11 (line plots; from data/)
    python Process_All_Data.py --detail   # Figs. 5-7 (detail case Gc = 5 J/m^2, ell = 0.02 m,
                                          # medium rate; needs the field outputs)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from plotting.PlotAnimations import Plot_Animations_Case
from plotting.PlotLines import Plot_Lines
from plotting.paths import require_field_outputs, study_folder

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "simulations"))   # Small_Reference_Params
# Studies written by simulations/Run_Plate_Sweeps.py, read from results/ or data/ (plotting/paths.py)
LINE_PLOT_FOLDER = REPOSITORY / "figures"     # Figs. 8, 9, 11
ANIMATION_FOLDER = REPOSITORY / "figures" / "animations"   # Figs. 5-7: stills and movies per case
SPECIMEN_HEIGHT_M = 1.0  # Ly in Small_Reference_Params.py; height in metres.

# Result-folder suffix selecting the sweep variant: "" for the uncorrected runs,
# "_GcCorr0.33" for the runs with the element-size correction of the fracture energy.
STUDY_SUFFIX = "_GcCorr0.33"


def _study(name):
    """Return the result folder of a study, preferring the STUDY_SUFFIX variant.

    Studies that were not rerun with the current variant (e.g. the Gc=1 mesh
    study at the low and high rates) fall back to the base folder.
    """
    candidate = study_folder(f"{name}{STUDY_SUFFIX}")
    if STUDY_SUFFIX and not candidate.is_dir():
        print(f"{candidate} not found; using {name}")
        return study_folder(name)
    return candidate

# Detail case of Figs. 5-7
DETAIL_CASE_GC = 5.0
DETAIL_CASE_LENGTH_SCALE = 0.02
DETAIL_CASE_RATE = "medium"  # "low", "medium", or "high"

# Rate encodings are shared by mesh and length-scale comparisons.
RATE_CASE_SPECS = [
    {
        "case": "LO_LowRate",
        "rate": 1.0e-9,
        "label": r"$\dot{\varepsilon}_{\text{ext}}=1.0\cdot10^{-9}\;\text{s}^{-1}$",
        "line_style": ":",
        "marker": "^",
    },
    {
        "case": "LO",
        "rate": 1.5e-9,
        "label": r"$\dot{\varepsilon}_{\text{ext}}=1.5\cdot10^{-9}\;\text{s}^{-1}$",
        "line_style": "-",
        "marker": "o",
    },
    {
        "case": "LO_HighRate",
        "rate": 2.5e-9,
        "label": r"$\dot{\varepsilon}_{\text{ext}}=2.5\cdot10^{-9}\;\text{s}^{-1}$",
        "line_style": "--",
        "marker": "s",
    },
]
FRACTURE_ENERGY_SPECS = [
    (1.0, "", "#0072B2"),
    (2.5, "_Gc2.5", "#D55E00"),
    (5.0, "_Gc5.0", "#009E73"),
    (10.0, "_Gc10", "#CC79A7"),
]


def _ensure_failure_surface():
    """Generate the required reference surface from this sweep's material."""

    failure_surface_path = Path(__file__).resolve().parent / "FailureSurf.hdf5"
    if failure_surface_path.is_file():
        return

    from femcubes.solids.failure_surface_plot import plot_FailureSurface

    from Small_Reference_Params import _build_small_params

    params = _build_small_params(
        l=0.02,
        dxratio=5,
        rate=1.0e-9,
        include_plots=False,
        OutputFolder=str(failure_surface_path.parent),
        MeshRef=False,
    )
    print(f"Generating failure surface at {failure_surface_path}")
    plot_FailureSurface(
        params.Materials["Ice"],
        np.linspace(-1.0e6, 0.5e6, 200),
        save_path=failure_surface_path.with_suffix(".png"),
        export_hdf5=True,
    )
    if not failure_surface_path.is_file():
        raise RuntimeError(
            f"Failure-surface generator did not create {failure_surface_path}"
        )


def _postprocess_selected_l_sweep_case():
    """Run detailed post-processing for the case selected at the top of the file."""

    rate_cases = {
        "low": ("LO_LowRate", 1.0e-9),
        "medium": ("LO", 1.5e-9),
        "high": ("LO_HighRate", 2.5e-9),
    }
    if DETAIL_CASE_RATE not in rate_cases:
        raise ValueError(
            "DETAIL_CASE_RATE must be one of: low, medium, or high"
        )

    rate_case_name, history_strain_rate = rate_cases[DETAIL_CASE_RATE]
    root = Path(__file__).resolve().parent
    gc_suffix = (
        "" if np.isclose(DETAIL_CASE_GC, 1.0)
        else f"_Gc{DETAIL_CASE_GC:.1f}"
    )
    case_path = (
        root
        / _study(f"Results_lSweep_{rate_case_name}{gc_suffix}")
        / f"l_{DETAIL_CASE_LENGTH_SCALE:g}"
    )
    data_path = case_path / "Data"
    time_data_path = data_path / "time_data.hdf5"
    if not time_data_path.is_file():
        raise FileNotFoundError(f"Selected lSweep result is missing: {time_data_path}")
    require_field_outputs(case_path, "The detail figures of the selected case (Figs. 5-7)",
                          'Without them, "python postprocessing/Process_All_Data.py" (without --detail) draws '
                          "Figs. 8, 9 and 11 from data/.")

    _ensure_failure_surface()
    output_path = ANIMATION_FOLDER / case_path.parent.name / case_path.name
    print(
        f"Detailed post-processing for Gc={DETAIL_CASE_GC:g} J/m^2, "
        f"ell={DETAIL_CASE_LENGTH_SCALE:g} m, "
        f"loading rate={DETAIL_CASE_RATE}: {data_path}"
    )
    Plot_Animations_Case(
        data_path,
        output_path,
        history_strain_rate=history_strain_rate,
        history_height_m=SPECIMEN_HEIGHT_M,
    )


def _plot_local_dx_sweep():
    """Plot the standalone Gc=10 local mesh-refinement study."""

    dx_values = [1, 2, 3, 4, 5]
    file_sets = []
    legend_labels = []
    for rate_spec in RATE_CASE_SPECS:
        file_sets.append([])
        legend_labels.append([])
        for dx in dx_values:
            case_path = (
                _study(f"Results_dxSweep_{rate_spec['case']}_Gc10") / f"dx_{dx}"
            )
            file_sets[-1].append(str(case_path / "Data" / "time_data.hdf5"))
            legend_labels[-1].append(rf"$\ell/h^{{\mathrm{{el}}}}={dx}$")

    rate_labels = [spec["label"] for spec in RATE_CASE_SPECS]
    Plot_Lines(
        file_sets,
        [spec["line_style"] for spec in RATE_CASE_SPECS],
        np.asarray(dx_values, dtype=float),
        legend_labels,
        str(LINE_PLOT_FOLDER / "MeshRefinement_Study_Gc10"),
        r"\ell/h^{\mathrm{el}}\;[-]",
        rate_labels,
        case_colors=["#0072B2", "#D55E00", "#009E73"],
        case_markers=[spec["marker"] for spec in RATE_CASE_SPECS],
        peak_legend_columns=1,
        peak_legend_below=True,
        peak_x_scale="linear",
        peak_xticks=dx_values,
        history_style_labels=rate_labels,
        history_legend_rates_first=True,
        history_strain_rates=[spec["rate"] for spec in RATE_CASE_SPECS],
        history_height_m=SPECIMEN_HEIGHT_M,
        history_time_scale="linear",
        history_displacement_limit_mm=0.25,
        x_units="",
    )


def _plot_dx_sweep():
    """Plot the Gc=1 mesh-refinement study (Fig. 8)."""

    # Mesh refinement: all three loading rates, with low-order phase field.
    dx_values = [1, 2, 3, 4, 5]
    file_sets = []
    legend_labels = []
    for rate_spec in RATE_CASE_SPECS:
        file_sets.append([])
        legend_labels.append([])
        for dx in dx_values:
            # The Gc=1 mesh study (fig:elementSizeSensitivity) keeps the nominal-Gc runs for all rates: only the medium rate
            # was rerun with the correction (fig:GcCorrectionEffect b), which changes Gc=1 by <0.3%.
            case_path = study_folder(f"Results_dxSweep_{rate_spec['case']}") / f"dx_{dx}"
            file_sets[-1].append(str(case_path / "Data" / "time_data.hdf5"))
            legend_labels[-1].append(rf"$\ell/h^{{\mathrm{{el}}}}={dx}$")

    rate_labels = [spec["label"] for spec in RATE_CASE_SPECS]
    Plot_Lines(
        file_sets,
        [spec["line_style"] for spec in RATE_CASE_SPECS],
        np.asarray(dx_values, dtype=float),
        legend_labels,
        str(LINE_PLOT_FOLDER / "MeshRefinement_Study"),
        r"\ell/h^{\mathrm{el}}\;[-]",
        rate_labels,
        case_colors=["#0072B2", "#D55E00", "#009E73"],
        case_markers=[spec["marker"] for spec in RATE_CASE_SPECS],
        peak_legend_columns=1,
        peak_legend_below=True,
        peak_x_scale="linear",
        peak_xticks=dx_values,
        history_style_labels=rate_labels,
        history_legend_rates_first=True,
        history_strain_rates=[spec["rate"] for spec in RATE_CASE_SPECS],
        history_height_m=SPECIMEN_HEIGHT_M,
        history_time_scale="linear",
        x_units="",
    )


def main():
    """Line plots of Figs. 8, 9 and 11."""
    _plot_dx_sweep()

    # Keep the local Gc=10 study in its own history and peak-load figure pair.
    _plot_local_dx_sweep()

    # Length-scale refinement: colour identifies G_c; style and marker identify rate.
    length_scales = [0.1, 0.05, 0.02, 0.01, 0.005]
    file_sets = []
    legend_labels = []
    line_styles = []
    case_labels = []
    case_colors = []
    case_markers = []
    history_style_labels = []
    strain_rates = []
    history_groups = []
    for fracture_energy, folder_suffix, color in FRACTURE_ENERGY_SPECS:
        for rate_spec in RATE_CASE_SPECS:
            file_sets.append([])
            legend_labels.append([])
            line_styles.append(rate_spec["line_style"])
            case_labels.append(rf"$G_c={fracture_energy:g}\;\mathrm{{J\,m^{{-2}}}}$, " + rate_spec["label"])
            case_colors.append(color)
            case_markers.append(rate_spec["marker"])
            history_style_labels.append(rate_spec["label"])
            strain_rates.append(rate_spec["rate"])
            history_groups.append(rf"$G_c={fracture_energy:g}\;\mathrm{{J\,m^{{-2}}}}$")
            for length_scale in length_scales:
                case_path = _study(f"Results_lSweep_{rate_spec['case']}{folder_suffix}") / f"l_{length_scale}"
                file_sets[-1].append(str(case_path / "Data" / "time_data.hdf5"))
                legend_labels[-1].append(rf"$\ell={length_scale:g}\;\mathrm{{m}}$")

    # Gc = 5 J/m^2 alone, all three rates (Fig. 11b).
    rate_count = len(RATE_CASE_SPECS)
    for group_index, (fracture_energy, _, _) in enumerate(FRACTURE_ENERGY_SPECS):
        if fracture_energy != 5.0:
            continue
        case_slice = slice(group_index * rate_count, (group_index + 1) * rate_count)
        save_name = f"lSweep_Study_Gc{fracture_energy}"
        Plot_Lines(
            file_sets[case_slice],
            line_styles[case_slice],
            length_scales,
            legend_labels[case_slice],
            str(LINE_PLOT_FOLDER / save_name),
            r"\ell",
            case_labels[case_slice],
            case_colors=case_colors[case_slice],
            case_markers=case_markers[case_slice],
            peak_legend_columns=1,
            peak_x_scale="log",
            peak_xticks=length_scales,
            peak_figure_size_cm=(8, 6),
            peak_legend_below=True,
            history_style_labels=history_style_labels[case_slice],
            history_legend_rates_first=True,
            history_strain_rates=strain_rates[case_slice],
            history_height_m=SPECIMEN_HEIGHT_M,
            history_time_scale="linear",
        )

    peak_legend_handles = [
        Line2D([], [], color=color, linestyle="-", label=rf"$G_c={fracture_energy:g}\;\mathrm{{J\,m^{{-2}}}}$")
        for fracture_energy, _, color in FRACTURE_ENERGY_SPECS
    ] + [
        Line2D(
            [], [], color="0.25", linestyle=rate_spec["line_style"],
            marker=rate_spec["marker"], markersize=4.0, markeredgewidth=0.5,
            label=rate_spec["label"],
        )
        for rate_spec in RATE_CASE_SPECS
    ]

    # Separate history panels keep G_c identifiable in the combined figure.
    Plot_Lines(
        file_sets,
        line_styles,
        length_scales,
        legend_labels,
        str(LINE_PLOT_FOLDER / "lSweep_Study_AllCases"),
        r"\ell",
        case_labels,
        case_colors=case_colors,
        case_markers=case_markers,
        peak_legend_columns=2,
        peak_legend_below=True,
        peak_legend_handles=peak_legend_handles,
        peak_x_scale="log",
        peak_xticks=length_scales,
        peak_figure_size_cm=(8, 6),
        history_style_labels=history_style_labels,
        history_strain_rates=strain_rates,
        history_height_m=SPECIMEN_HEIGHT_M,
        history_groups=history_groups,
        history_time_scale="linear",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detail", action="store_true",
                        help="Figs. 5-7 of the detail case instead of the line plots (needs the field outputs)")
    if parser.parse_args().detail:
        _postprocess_selected_l_sweep_case()
    else:
        main()
