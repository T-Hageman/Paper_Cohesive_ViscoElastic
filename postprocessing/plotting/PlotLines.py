"""Plot time histories and peak loads for parameter and loading-rate sweeps."""
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np

FONTSIZE = 7
LINEWIDTH = 0.5
HISTORY_DISPLACEMENT_LIMIT_MM = 0.05
# A few markers per history curve keep coinciding curves identifiable: one shape
# per value of the swept parameter, each shifted along the curve relative to the others.
HISTORY_MARKERS = ("o", "s", "^", "D", "v")
HISTORY_MARKER_STYLE = dict(markersize=3.0, markerfacecolor="none", markeredgewidth=LINEWIDTH)
HISTORY_MARKS_PER_CURVE = 2.5  # across the displayed interval
HISTORY_STRESS_LIMIT_MPA = 0.25  # displayed stress range
# Sudden failure: the load falls more steeply than DROP_SLOPE (in axis fractions) over
# at least DROP_MINIMUM of the stress range.
DROP_SLOPE, DROP_MINIMUM = 20.0, 0.12


def Ensure_Folder(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


def Mark_Curve(axis, x, y, number, count, x_limit, colour, shift=0.0):
    """Mark curve `number` of a family of `count` curves a few times up to x_limit.

    The markers of the curves of one family alternate along x, so that each stays
    visible where the curves coincide; shift (between 0 and 1) moves them within
    their slot to separate families that share the axis. Near-vertical drops of the
    load (sudden failure) get a marker of their own, at a height that alternates
    between the curves in the same way.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    spacing = x_limit / HISTORY_MARKS_PER_CURVE
    slot = (number + 0.5 + shift) / count
    marks_x = np.arange(spacing * slot, min(float(x[-1]), x_limit), spacing)
    marks_y = np.interp(marks_x, x, y)

    steep = np.diff(y) / HISTORY_STRESS_LIMIT_MPA < -DROP_SLOPE * np.diff(x) / x_limit
    edges = np.flatnonzero(np.diff(np.concatenate(([False], steep, [False]))))
    for start, stop in zip(edges[::2], edges[1::2]):  # the drop spans points start..stop
        height = y[start] - y[stop]
        if height < DROP_MINIMUM * HISTORY_STRESS_LIMIT_MPA or x[start] > x_limit:
            continue
        drop_y = y[start] - height * (0.1 + 0.8 * (slot % 1.0))  # clear of both ends
        drop_x = np.interp(drop_y, y[start:stop + 1][::-1], x[start:stop + 1][::-1])
        # A marker spaced along x that falls on this drop would crowd its own marker.
        keep = (marks_x < x[start]) | (marks_x > x[stop])
        marks_x = np.append(marks_x[keep], drop_x)
        marks_y = np.append(marks_y[keep], drop_y)
    axis.plot(marks_x, marks_y, color=colour, linestyle="none", zorder=3,
              marker=HISTORY_MARKERS[number % len(HISTORY_MARKERS)], **HISTORY_MARKER_STYLE)


def FormatAndSave(fig, ax, SaveName, fontsize=7, top=0.97,
                  figure_size_cm=(8, 6), left=0.2, right=0.97, bottom=0.20):
    fig.set_size_inches(*(np.asarray(figure_size_cm) / 2.54))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.3)

    axes = np.atleast_1d(ax).ravel()
    legends = list(fig.legends)
    for axis in axes:
        axis.title.set_fontsize(fontsize)
        axis.xaxis.label.set_fontsize(fontsize)
        axis.yaxis.label.set_fontsize(fontsize)
        axis.tick_params(axis="both", which="both", labelsize=fontsize)
        for line in axis.get_lines():
            line.set_linewidth(LINEWIDTH)
        if axis.get_legend() is not None:
            legends.append(axis.get_legend())
    for legend in legends:
        for line in legend.get_lines():
            line.set_linewidth(LINEWIDTH)
        for label in legend.get_texts():
            label.set_fontsize(fontsize)

    save_options = {
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    fig.savefig(f"{SaveName}.png", dpi=1200, **save_options)
    fig.savefig(f"{SaveName}.pdf", format="pdf", dpi=1200, metadata={"CreationDate": None}, **save_options)


def Plot_Lines(
    fileSets,
    line_styles,
    xticks,
    legend_labels,
    SaveName,
    xName,
    caseLabels,
    case_colors=None,
    case_markers=None,
    peak_legend_columns=1,
    peak_legend_above=False,
    peak_legend_below=False,
    peak_legend_handles=None,
    peak_figure_size_cm=(8, 6),
    peak_x_scale="log",
    peak_xticks=None,
    history_style_labels=None,
    history_legend_rates_first=False,
    history_groups=None,
    history_time_scale="linear",
    x_units="m",
    history_strain_rates=None,
    history_height_m=1.0,
    history_displacement_limit_mm=HISTORY_DISPLACEMENT_LIMIT_MM,
):
    """Use curve colour for the swept parameter and line style for loading rate.

    Optional history groups get separate panels with shared axes, so additional
    material variations remain identifiable in the combined history figure.
    Supplying one strain rate per case plots t * strain_rate * Ly in millimetres;
    history_height_m supplies Ly in metres. The history displacement limit sets
    both the displayed interval and the corresponding peak-load calculation.
    Peak-load colours and markers are supplied independently.
    """
    if history_strain_rates is not None:
        history_strain_rates = np.asarray(history_strain_rates, dtype=float)
        if (history_strain_rates.shape != (len(fileSets),)
                or not np.all(np.isfinite(history_strain_rates))
                or np.any(history_strain_rates <= 0.0)):
            raise ValueError("Supply one finite, positive strain rate per case")
        if not np.isfinite(history_height_m) or history_height_m <= 0.0:
            raise ValueError("The specimen height must be finite and positive, in metres")
        if (
            not np.isfinite(history_displacement_limit_mm)
            or history_displacement_limit_mm <= 0.0
        ):
            raise ValueError("The history displacement limit must be finite and positive")
    Ensure_Folder(Path(SaveName).parent)
    groups = history_groups if history_groups is not None else [""] * len(fileSets)
    group_names = list(dict.fromkeys(groups))
    fig, history_axes = plt.subplots(1, len(group_names), sharex=True, sharey=True,
                                    squeeze=False)
    history_axes = history_axes.ravel()
    axes_by_group = dict(zip(group_names, history_axes))
    fig2, ax2 = plt.subplots()
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    history_x_min, history_x_max = np.inf, -np.inf
    cases_in_group = {name: [case for case, group in enumerate(groups) if group == name]
                      for name in group_names}

    for case, files in enumerate(fileSets):
        axis = axes_by_group[groups[case]]
        group_cases = cases_in_group[groups[case]]
        peak_load = np.full(len(xticks), np.nan)
        for i, filename in enumerate(files):
            print(f"Reading data from {filename}")
            if not Path(filename).is_file():
                print(f"File {filename} does not exist. Skipping.")
                continue
            # The driver writes simulation_time.txt only when a case has run to
            # completion; skip the partial history of a case that is still running.
            # stopped_after_peak.txt marks a case stopped by hand on its descending
            # branch because only its peak load is used: its history is truncated.
            case_folder = Path(filename).parent.parent
            if not any((case_folder / marker).is_file()
                       for marker in ("simulation_time.txt", "stopped_after_peak.txt")):
                print(f"Case {Path(filename).parent.parent} is incomplete. Skipping.")
                continue
            with h5py.File(filename, "r") as time_data:
                t = time_data["data"][:, 0]
                stress = time_data["data"][:, 4] * 1e-6

            history_x = (t if history_strain_rates is None else
                         t * history_strain_rates[case] * history_height_m * 1.0e3)
            valid = np.isfinite(history_x) & np.isfinite(stress)
            if history_time_scale == "log":
                valid &= history_x > 0.0
            if np.any(valid):
                axis.plot(history_x[valid], stress[valid], color=colour_cycle[i % len(colour_cycle)],
                          linestyle=line_styles[case])
                Mark_Curve(
                    axis, history_x[valid], stress[valid], i, len(files),
                    history_displacement_limit_mm if history_strain_rates is not None
                    else float(np.max(history_x[valid])),
                    colour_cycle[i % len(colour_cycle)],
                    shift=group_cases.index(case) / len(group_cases),
                )
                history_x_min = min(history_x_min, float(np.min(history_x[valid])))
                history_x_max = max(history_x_max, float(np.max(history_x[valid])))
            peak_in_range = valid.copy()
            if history_strain_rates is not None:
                peak_in_range &= (
                    (history_x >= 0.0)
                    & (history_x <= history_displacement_limit_mm)
                )
            if np.any(peak_in_range):
                peak_load[i] = np.max(stress[peak_in_range])

        case_color = (case_colors[case] if case_colors is not None
                      else colour_cycle[case % len(colour_cycle)])
        case_marker = case_markers[case] if case_markers is not None else "x"
        ax2.plot(
            xticks, peak_load, color=case_color, linestyle=line_styles[case],
            marker=case_marker, markersize=3.5 if case_markers is not None else 6.0,
            markeredgewidth=0.5, label=caseLabels[case],
        )

    for group_name, axis in axes_by_group.items():
        axis.set_title(group_name)
        axis.set_xlabel(
            r"$t\,\dot{\varepsilon}_{\text{ext}}\,L_y\;[\mathrm{mm}]$" if history_strain_rates is not None
            else r"$time\;[\mathrm{s}]$"
        )
        axis.set_xscale(history_time_scale)
        if history_strain_rates is not None:
            history_ticks = np.linspace(0.0, history_displacement_limit_mm, 6)
            axis.xaxis.set_major_locator(FixedLocator(history_ticks))
            axis.set_xlim(history_ticks[0], history_ticks[-1])  # mm.
        elif np.isfinite(history_x_min) and history_x_max > history_x_min:
            if history_time_scale == "log":
                axis.set_xlim(history_x_min / 1.05, history_x_max * 1.05)
            else:
                margin = 0.02 * (history_x_max - history_x_min)
                axis.set_xlim(history_x_min - margin, history_x_max + margin)
        history_y_locator = MaxNLocator(nbins=5)
        history_y_ticks = history_y_locator.tick_values(1e-3, 0.225)
        axis.yaxis.set_major_locator(FixedLocator(history_y_ticks))
        axis.set_ylim(history_y_ticks[0], history_y_ticks[-1])
    history_axes[0].set_ylabel(r"$\overline{\sigma_{yy}}\;[\mathrm{MPa}]$")

    parameter_handles = [
        Line2D([], [], color=colour_cycle[i % len(colour_cycle)], label=label,
               marker=HISTORY_MARKERS[i % len(HISTORY_MARKERS)], **HISTORY_MARKER_STYLE)
        for i, label in enumerate(legend_labels[0])
    ]
    history_handles = parameter_handles
    if history_style_labels is not None:
        style_handles = {}
        for label, style in zip(history_style_labels, line_styles):
            style_handles.setdefault(label, Line2D([], [], color="0.25",
                                                   linestyle=style, label=label))
        rate_handles = list(style_handles.values())
        history_handles = (
            rate_handles + parameter_handles if history_legend_rates_first
            else parameter_handles + rate_handles
        )
    fig.legend(
        handles=history_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3 if len(group_names) == 1 else 4,
        frameon=True, fancybox=False, framealpha=1.0, edgecolor="black",
        fontsize=7, borderpad=0.25, columnspacing=0.5, handletextpad=0.3,
    )
    FormatAndSave(
        fig, history_axes, SaveName, top=0.95,
        bottom=0.43 if len(group_names) == 1 else 0.30,
        figure_size_cm=(8, 6) if len(group_names) == 1 else (19, 7.5),
        left=0.2 if len(group_names) == 1 else 0.09,
    )
    plt.close(fig)

    unit_label = rf"\;[\mathrm{{{x_units}}}]" if x_units else ""
    ax2.set_xlabel(rf"${xName}{unit_label}$")
    ax2.set_ylabel(r"$\overline{\sigma}_\mathrm{peak}\;[\mathrm{MPa}]$")
    ax2.set_xscale(peak_x_scale)
    if peak_xticks is not None:
        peak_xticks = np.asarray(peak_xticks, dtype=float)
        x_min, x_max = float(np.min(peak_xticks)), float(np.max(peak_xticks))
        ax2.set_xticks(peak_xticks)
        ax2.set_xlim(x_min, x_max)
        y_locator = MaxNLocator(nbins=4)
        y_ticks = y_locator.tick_values(*ax2.get_ylim())
        ax2.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax2.set_ylim(y_ticks[0], y_ticks[-1])
    peak_axes_bottom = 0.20
    if peak_legend_below:
        if peak_legend_handles is None:
            legend_handles, legend_labels_for_figure = ax2.get_legend_handles_labels()
        else:
            legend_handles = peak_legend_handles
            legend_labels_for_figure = [handle.get_label() for handle in legend_handles]
        legend_rows = int(np.ceil(len(legend_handles) / peak_legend_columns))
        if peak_figure_size_cm[0] <= 8:
            # The rows of a marker legend are taller than those of a plain line legend.
            row_cm = 0.30 if peak_legend_handles is None else 0.35
            peak_footer_cm = 1.65 + row_cm * legend_rows
            peak_axes_bottom = peak_footer_cm / peak_figure_size_cm[1]
        else:
            peak_axes_bottom = 0.28 + 0.06 * (legend_rows - 1)
        fig2.legend(
            legend_handles, legend_labels_for_figure, loc="lower center",
            bbox_to_anchor=(0.5, 0.015), ncol=peak_legend_columns,
            frameon=True, fancybox=False, framealpha=1.0, edgecolor="black",
            fontsize=7,
            borderpad=0.25 if peak_legend_handles is not None else 0.35,
            columnspacing=0.4 if peak_legend_handles is not None else 0.8,
            handletextpad=0.25 if peak_legend_handles is not None else 0.4,
            handlelength=1.15 if peak_legend_handles is not None else 2.0,
        )
        peak_axes_top = 0.97
    elif peak_legend_above:
        legend_rows = int(np.ceil(len(fileSets) / peak_legend_columns))
        peak_axes_top = max(0.45, 0.97 - 0.085 * legend_rows)
        fig2.legend(
            *ax2.get_legend_handles_labels(), loc="lower center",
            bbox_to_anchor=(0.5, peak_axes_top + 0.015),
            ncol=peak_legend_columns, frameon=False, fontsize=7,
            columnspacing=0.8, handletextpad=0.4,
        )
    else:
        ax2.legend(fontsize=FONTSIZE, ncol=peak_legend_columns)
        peak_axes_top = 0.97
    peak_right = 0.92 if peak_x_scale == "log" and peak_xticks is not None else 0.97
    FormatAndSave(fig2, ax2, SaveName + "_PeakLoad", top=peak_axes_top,
                  bottom=peak_axes_bottom, figure_size_cm=peak_figure_size_cm,
                  right=peak_right)
    plt.close(fig2)
