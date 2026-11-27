"""Render a synchronized load--displacement history for one simulation case."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence
import warnings

import h5py
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np


DISPLACEMENT_TO_MM = 1.0e3
FORCE_TO_KN = 1.0e-3
FONT_SIZE = 7
TITLE_FONT_SIZE = 8
CASE_LABEL_FONT_SIZE = 5
FIGURE_SIZE_CM = (8.0, 8.0)
# The selected-states response curve is a manuscript line plot: same size
# and margins as the PlotLines figures, no title or case label.
SELECTED_STATES_SIZE_CM = (8.0, 6.0)
FIGURE_DPI = 300
VIDEO_SIZE_PX = (800, 800)
VIDEO_FPS = 10


class _FrameLike(Protocol):
    step: int
    time: float


class _SelectedStateLike(Protocol):
    name: str
    display_name: str
    frame_index: int


def _decode_names(raw_names: np.ndarray) -> list[str]:
    return [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]


def _read_history(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read time, top displacement, and reaction force from the full history."""

    time_path = data_dir / "time_data.hdf5"
    if not time_path.is_file():
        raise FileNotFoundError(f"Missing time-history file: {time_path}")

    with h5py.File(time_path, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(f"{time_path} does not contain the 'data' dataset")
        data = np.asarray(h5_file["data"], dtype=float)
        names = (
            _decode_names(h5_file["data_names"][...])
            if "data_names" in h5_file
            else []
        )

    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(f"{time_path}: 'data' must be a non-empty 2-D array")

    if "BC_Top_u" in names:
        displacement_column = names.index("BC_Top_u")
    elif data.shape[1] > 6:
        warnings.warn(
            f"{time_path} has no BC_Top_u name; using legacy column 6",
            stacklevel=2,
        )
        displacement_column = 6
    else:
        raise KeyError(f"{time_path} does not contain a BC_Top_u column")

    if "BC_Top_F" in names:
        force_column = names.index("BC_Top_F")
    elif data.shape[1] > 4:
        warnings.warn(
            f"{time_path} has no BC_Top_F name; using legacy column 4",
            stacklevel=2,
        )
        force_column = 4
    else:
        raise KeyError(f"{time_path} does not contain a BC_Top_F column")

    time_column = names.index("time") if "time" in names else 0
    time = data[:, time_column]
    displacement = data[:, displacement_column] * DISPLACEMENT_TO_MM
    force = data[:, force_column] * FORCE_TO_KN
    finite = np.isfinite(time) & np.isfinite(displacement) & np.isfinite(force)
    if not np.all(finite):
        invalid_rows = np.flatnonzero(~finite)
        raise ValueError(
            f"{time_path} contains non-finite time-history data at "
            f"row(s) {invalid_rows[:8].tolist()}"
        )
    return time, displacement, force


def _history_rows(frames: Sequence[_FrameLike], history_length: int) -> np.ndarray:
    """Map each saved output step to its zero-based time-history row."""

    steps = np.asarray([frame.step for frame in frames], dtype=int)
    rows = steps - 1
    if np.all((rows >= 0) & (rows < history_length)):
        return rows
    if history_length == len(frames):
        return np.arange(len(frames), dtype=int)
    raise ValueError(
        "Could not map saved output files to load--displacement history rows: "
        f"steps span {steps[0]}..{steps[-1]}, but the history has "
        f"{history_length} rows"
    )


def _limits_with_margin(values: np.ndarray) -> tuple[float, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span <= np.finfo(float).eps * max(1.0, abs(lower), abs(upper)):
        span = max(abs(lower), abs(upper), 1.0)
    margin = 0.06 * span
    return lower - margin, upper + margin


def _time_label(time: float) -> str:
    magnitude = abs(time)
    if magnitude != 0.0 and (magnitude >= 1.0e3 or magnitude < 1.0e-2):
        exponent = int(np.floor(np.log10(magnitude)))
        mantissa = time / 10.0**exponent
        return rf"{mantissa:.3f} \times 10^{{{exponent}}}"
    return f"{time:.4g}"


def _annotation(
    frame: _FrameLike,
    displacement: float,
    force: float,
    state_name: str | None = None,
) -> str:
    prefix = f"{state_name}; " if state_name else ""
    return (
        rf"{prefix}$t = {_time_label(frame.time)}\,\mathrm{{s}}$"
        "\n"
        rf"$u_{{\mathrm{{top}}}} = {displacement:.4g}\,\mathrm{{mm}}$; "
        rf"$F_{{\mathrm{{top}}}} = {force:.4g}\,\mathrm{{kN}}$"
    )


def _open_video_writer(path: Path, fps: int):
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    )


def _create_figure(
    displacement: np.ndarray,
    force: np.ndarray,
    video_size_px: tuple[int, int],
    case_label: str,
) -> tuple[Figure, FigureCanvasAgg, Line2D, Line2D, Line2D, object]:
    figure_width = FIGURE_SIZE_CM[0] / 2.54
    figure_height = FIGURE_SIZE_CM[1] / 2.54
    video_dpi = video_size_px[0] / figure_width
    figure = Figure(
        figsize=(figure_width, figure_height),
        dpi=video_dpi,
        facecolor="white",
        frameon=True,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.19, 0.15, 0.75, 0.65))

    future_line, = axis.plot(
        displacement,
        force,
        color="0.78",
        linewidth=0.8,
        solid_capstyle="round",
        zorder=1,
    )
    prior_line, = axis.plot(
        displacement[:1],
        force[:1],
        color="black",
        linewidth=0.9,
        solid_capstyle="round",
        zorder=2,
    )
    current_point, = axis.plot(
        displacement[0],
        force[0],
        marker="o",
        linestyle="none",
        markersize=4.0,
        markerfacecolor="#d62728",
        markeredgecolor="white",
        markeredgewidth=0.35,
        zorder=3,
    )

    axis.set_xlim(*_limits_with_margin(displacement))
    axis.set_ylim(*_limits_with_margin(force))
    axis.set_xlabel(
        r"$u_{\mathrm{top}}\;[\mathrm{mm}]$",
        fontsize=FONT_SIZE,
        labelpad=2,
    )
    axis.set_ylabel(
        r"$F_{\mathrm{top}}\;[\mathrm{kN}]$",
        fontsize=FONT_SIZE,
        labelpad=2,
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.tick_params(
        axis="both",
        which="both",
        labelsize=FONT_SIZE - 1,
        length=2.0,
        width=0.5,
        pad=2.0,
    )
    axis.grid(True, color="0.88", linestyle=":", linewidth=0.45)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    figure.text(
        0.025,
        0.975,
        "Load--displacement evolution",
        ha="left",
        va="top",
        fontsize=TITLE_FONT_SIZE,
    )
    figure.text(
        0.975,
        0.985,
        case_label,
        ha="right",
        va="top",
        fontsize=CASE_LABEL_FONT_SIZE,
        color="0.35",
    )
    annotation = figure.text(
        0.025,
        0.925,
        "",
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        linespacing=1.15,
    )

    prior_handle = Line2D([], [], color="black", linewidth=0.9, label="Prior")
    future_handle = Line2D([], [], color="0.78", linewidth=0.8, label="Future")
    current_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=4.0,
        markerfacecolor="#d62728",
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="Current",
    )
    figure.legend(
        handles=[prior_handle, future_handle, current_handle],
        loc="lower left",
        bbox_to_anchor=(0.19, 0.805),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE - 1,
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    return figure, canvas, prior_line, future_line, current_point, annotation


def _save_selected_state_overview(
    displacement: np.ndarray,
    force: np.ndarray,
    frame_rows: np.ndarray,
    selected_states: Sequence[_SelectedStateLike],
    output_dir: Path,
    *,
    case_label: str,
    figure_dpi: int,
) -> None:
    """Save the full response curve with markers for the selected states."""

    state_colours = {
        "peak_load": "#D55E00",
        "unloading_75pct": "#0072B2",
        "unloading_50pct": "#009E73",
        "unloading_25pct": "#CC79A7",
    }
    figure = Figure(
        figsize=(SELECTED_STATES_SIZE_CM[0] / 2.54, SELECTED_STATES_SIZE_CM[1] / 2.54),
        dpi=figure_dpi,
        facecolor="white",
        frameon=True,
    )
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.2, 0.2, 0.75, 0.76))
    axis.plot(
        displacement,
        force,
        color="0.15",
        linewidth=0.8,
        solid_capstyle="round",
        zorder=1,
    )

    state_order = {
        "unloading_25pct": 0,
        "unloading_50pct": 1,
        "unloading_75pct": 2,
        "peak_load": 3,
    }
    ordered_states = sorted(
        selected_states,
        key=lambda state: state_order.get(state.name, len(state_order)),
    )
    legend_handles = []
    for state in ordered_states:
        if state.frame_index < 0 or state.frame_index >= len(frame_rows):
            raise IndexError(
                f"Selected state {state.name} has an invalid frame index "
                f"{state.frame_index}"
            )
        row = int(frame_rows[state.frame_index])
        colour = state_colours.get(state.name, "#0072B2")
        axis.plot(
            displacement[row],
            force[row],
            marker="o",
            linestyle="none",
            markersize=5.0,
            markerfacecolor=colour,
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=3,
        )
        display_name = {
            "peak_load": "Peak load",
            "unloading_75pct": r"$75\%$ remaining",
            "unloading_50pct": r"$50\%$ remaining",
            "unloading_25pct": r"$25\%$ remaining",
        }.get(state.name, state.display_name)
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.5,
                markerfacecolor=colour,
                markeredgecolor="black",
                markeredgewidth=0.45,
                label=display_name,
            )
        )

    x_locator = MaxNLocator(nbins=5)
    x_ticks = x_locator.tick_values(
        float(np.min(displacement)), float(np.max(displacement))
    )
    y_locator = MaxNLocator(nbins=5)
    y_ticks = y_locator.tick_values(float(np.min(force)), float(np.max(force)))
    axis.xaxis.set_major_locator(FixedLocator(x_ticks))
    axis.yaxis.set_major_locator(FixedLocator(y_ticks))
    axis.set_xlim(x_ticks[0], x_ticks[-1])
    axis.set_ylim(y_ticks[0], y_ticks[-1])
    axis.set_xlabel(
        r"$t\,\dot{\varepsilon}_{\text{ext}}\,L_y\;[\mathrm{mm}]$",
        fontsize=FONT_SIZE,
        labelpad=2,
    )
    axis.set_ylabel(
        r"$\overline{\sigma_{yy}}\;[\mathrm{MPa}]$",
        fontsize=FONT_SIZE,
        labelpad=2,
    )
    axis.tick_params(
        axis="both",
        which="both",
        labelsize=FONT_SIZE,
        length=2.0,
        width=0.5,
        pad=2.0,
    )
    axis.grid(True, color="0.88", linestyle=":", linewidth=0.45)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    axis.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=FONT_SIZE,
        borderpad=0.3,
        handletextpad=0.35,
        handlelength=1.0,
    )

    save_options = {
        "dpi": figure_dpi,
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    output_path = output_dir / "load_displacement_selected_states"
    figure.savefig(output_path.with_suffix(".png"), **save_options)
    figure.savefig(
        output_path.with_suffix(".pdf"),
        format="pdf",
        metadata={"CreationDate": None},
        **save_options,
    )
    figure.clear()
    print(f"Saved {output_path}.png and {output_path}.pdf")


def Plot_LoadDisplacement_Case(
    path_data: str | Path,
    frames: Sequence[_FrameLike],
    selected_states: Sequence[_SelectedStateLike],
    path_animations_output: str | Path,
    *,
    case_label: str = "",
    figure_dpi: int = FIGURE_DPI,
    video_size_px: tuple[int, int] = VIDEO_SIZE_PX,
    video_fps: int = VIDEO_FPS,
    history_strain_rate: float = 1.5e-9,
    history_height_m: float = 1.0,
) -> None:
    """Create response video, selected-state plots, and a marked overview."""

    if not frames:
        raise ValueError("At least one output frame is required")
    data_dir = Path(path_data)
    output_dir = Path(path_animations_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not np.isfinite(history_strain_rate) or history_strain_rate <= 0.0:
        raise ValueError("history_strain_rate must be finite and positive")
    if not np.isfinite(history_height_m) or history_height_m <= 0.0:
        raise ValueError("history_height_m must be finite and positive")

    time, displacement, force = _read_history(data_dir)
    frame_rows = _history_rows(frames, len(displacement))
    if np.any(np.diff(frame_rows) < 0):
        raise ValueError("Saved output steps are not in chronological order")

    # Match the sensitivity plots using every time-history row, including any
    # final solver increment that was not saved as an animation frame.
    overview_x_mm = time * history_strain_rate * history_height_m * 1.0e3
    # Matches Plot_Lines: the force history in newtons is scaled by 1e-6
    # and plotted as the average vertical stress in MPa.
    overview_stress_mpa = force * 1.0e-3

    # Restrict the video trajectory to the final saved output state.  This
    # makes its last frame fully black rather than leaving an unreachable
    # grey tail when time_data contains unsaved solver increments.
    history_end = int(frame_rows[-1]) + 1
    displacement = displacement[:history_end]
    force = force[:history_end]
    _save_selected_state_overview(
        overview_x_mm,
        overview_stress_mpa,
        frame_rows,
        selected_states,
        output_dir,
        case_label=case_label,
        figure_dpi=figure_dpi,
    )

    figure, canvas, prior_line, future_line, current_point, annotation = (
        _create_figure(displacement, force, video_size_px, case_label)
    )
    selected_by_index: dict[int, list[_SelectedStateLike]] = {}
    for state in selected_states:
        selected_by_index.setdefault(state.frame_index, []).append(state)

    video_path = output_dir / "load_displacement.mp4"
    writer = _open_video_writer(video_path, video_fps)
    try:
        for frame_index, (frame, row) in enumerate(zip(frames, frame_rows)):
            row = int(row)
            prior_line.set_data(displacement[: row + 1], force[: row + 1])
            future_line.set_data(displacement[row:], force[row:])
            current_point.set_data([displacement[row]], [force[row]])
            annotation.set_text(
                _annotation(frame, displacement[row], force[row])
            )
            canvas.draw()
            video_frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[
                ..., :3
            ].copy()
            writer.append_data(video_frame)

            for state in selected_by_index.get(frame_index, []):
                annotation.set_text(
                    _annotation(
                        frame,
                        displacement[row],
                        force[row],
                        state.display_name,
                    )
                )
                snapshot_path = output_dir / f"load_displacement_{state.name}"
                save_options = {
                    "dpi": figure_dpi,
                    "facecolor": "white",
                    "edgecolor": "white",
                    "transparent": False,
                }
                figure.savefig(snapshot_path.with_suffix(".png"), **save_options)
                figure.savefig(
                    snapshot_path.with_suffix(".pdf"),
                    format="pdf",
                    metadata={"CreationDate": None},
                    **save_options,
                )
                annotation.set_text(
                    _annotation(frame, displacement[row], force[row])
                )
    finally:
        writer.close()
        figure.clear()
    print(f"Saved {video_path} ({len(frames)} frames)")
