"""Render a compact principal-stress point-cloud history for one case.

The complete cell-average stress cloud is plotted in every frame, coloured
by the cell-average phase field, alongside the material failure surface
``F = 0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import h5py
import imageio
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


STRESS_FIELDS = ("sigma_xx", "sigma_yy", "sigma_xy")
PHASE_FIELD = "phasefield"
STRESS_SCALE = 1.0e6
STRESS_UNIT = r"\mathrm{MPa}"
POINT_SIZE = 0.45
LOW_PHASE_THRESHOLD = 0.2
LOW_PHASE_ALPHA = 0.5
FONT_SIZE = 7
TITLE_FONT_SIZE = 8
CASE_LABEL_FONT_SIZE = 5
FIGURE_SIZE_CM = (8.0, 8.0)
# Manuscript stills: final printed size, no title, case label or annotation.
STILL_SIZE_CM = (6.8, 5.4)
FIGURE_DPI = 300
VIDEO_SIZE_PX = (800, 800)
VIDEO_FPS = 10
FAILURE_SURFACE_PATH = Path(__file__).resolve().parents[1] / "FailureSurf.hdf5"
FAILURE_SURFACE_GROUP = "true_2d"


class _FrameLike(Protocol):
    path: Path
    time: float


class _SelectedStateLike(Protocol):
    name: str
    display_name: str
    frame_index: int


def _cell_average(values: np.ndarray) -> np.ndarray:
    """Return one scalar per cell from a scalar DG output table."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim < 1:
        raise ValueError("A cell field must have at least one dimension")
    return values.reshape(values.shape[0], -1).mean(axis=1)


def _principal_stresses(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered in-plane principal stresses, ``sigma_1 >= sigma_2``."""

    mean_stress = 0.5 * (sigma_xx + sigma_yy)
    radius = np.sqrt((0.5 * (sigma_xx - sigma_yy)) ** 2 + sigma_xy**2)
    return mean_stress + radius, mean_stress - radius


def _read_stress_snapshot(
    output_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(output_file, "r") as h5_file:
        if "Fields" not in h5_file:
            raise KeyError(f"{output_file} does not contain a Fields group")
        fields = h5_file["Fields"]
        required_fields = STRESS_FIELDS + (PHASE_FIELD,)
        missing = [name for name in required_fields if name not in fields]
        if missing:
            raise KeyError(
                f"{output_file} is missing field(s): {', '.join(missing)}"
            )
        sigma_xx, sigma_yy, sigma_xy = (
            _cell_average(fields[name][...]) for name in STRESS_FIELDS
        )
        phasefield = _cell_average(fields[PHASE_FIELD][...])

    if not (sigma_xx.size == sigma_yy.size == sigma_xy.size == phasefield.size):
        raise ValueError(f"Fields in {output_file} have different cell counts")

    sigma_1, sigma_2 = _principal_stresses(sigma_xx, sigma_yy, sigma_xy)
    finite = (
        np.isfinite(sigma_1)
        & np.isfinite(sigma_2)
        & np.isfinite(phasefield)
    )
    if not np.all(finite):
        sigma_1 = sigma_1[finite]
        sigma_2 = sigma_2[finite]
        phasefield = phasefield[finite]
    if sigma_1.size == 0:
        raise ValueError(f"No finite stress states in {output_file}")
    return sigma_1, sigma_2, phasefield


def _read_failure_surface(
    failure_surface_path: Path,
    surface_group: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not failure_surface_path.is_file():
        raise FileNotFoundError(
            f"Failure-surface file does not exist: {failure_surface_path}"
        )
    with h5py.File(failure_surface_path, "r") as h5_file:
        if surface_group not in h5_file:
            raise KeyError(
                f"{failure_surface_path} does not contain group {surface_group!r}"
            )
        group = h5_file[surface_group]
        required = ("sigma_1", "sigma_2", "failure_criterion")
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(
                f"{failure_surface_path}:{surface_group} is missing "
                f"{', '.join(missing)}"
            )
        sigma_1 = np.asarray(group["sigma_1"], dtype=float)
        sigma_2 = np.asarray(group["sigma_2"], dtype=float)
        failure_criterion = np.asarray(group["failure_criterion"], dtype=float)

    if not (sigma_1.shape == sigma_2.shape == failure_criterion.shape):
        raise ValueError(
            f"Failure-surface arrays in {failure_surface_path}:{surface_group} "
            "have different shapes"
        )
    return sigma_1, sigma_2, failure_criterion


def _time_label(time: float) -> str:
    magnitude = abs(time)
    if magnitude != 0.0 and (magnitude >= 1.0e3 or magnitude < 1.0e-2):
        exponent = int(np.floor(np.log10(magnitude)))
        mantissa = time / 10.0**exponent
        return rf"{mantissa:.3f} \times 10^{{{exponent}}}"
    return f"{time:.4g}"


def _annotation(frame: _FrameLike, state_name: str | None = None) -> str:
    prefix = f"{state_name}; " if state_name else ""
    return rf"{prefix}$t = {_time_label(frame.time)}\,\mathrm{{s}}$"


def _point_render_data(
    sigma_1: np.ndarray,
    sigma_2: np.ndarray,
    phasefield: np.ndarray,
    phase_norm: Normalize,
) -> tuple[np.ndarray, np.ndarray]:
    """Return phase-ordered positions and per-point RGBA colours.

    Low-phase points are drawn first and made translucent, ensuring that they
    cannot obscure the less numerous damaged states plotted on top of them.
    """

    order = np.argsort(phasefield, kind="stable")
    ordered_phasefield = phasefield[order]
    offsets = np.column_stack((sigma_1[order], sigma_2[order])) / STRESS_SCALE
    colours = np.asarray(
        colormaps["inferno"](phase_norm(ordered_phasefield)),
        dtype=float,
    )
    colours[:, 3] = np.where(
        ordered_phasefield < LOW_PHASE_THRESHOLD,
        LOW_PHASE_ALPHA,
        1.0,
    )
    return offsets, colours


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
    first_sigma_1: np.ndarray,
    first_sigma_2: np.ndarray,
    first_phasefield: np.ndarray,
    surface_sigma_1: np.ndarray,
    surface_sigma_2: np.ndarray,
    failure_criterion: np.ndarray,
    video_size_px: tuple[int, int],
    case_label: str,
    paper: bool = False,
) -> tuple[Figure, FigureCanvasAgg, object, object]:
    """Build the figure; ``paper`` gives the manuscript still layout."""

    if paper:
        figure_width = STILL_SIZE_CM[0] / 2.54
        figure_height = STILL_SIZE_CM[1] / 2.54
        figure_dpi = FIGURE_DPI
    else:
        figure_width = FIGURE_SIZE_CM[0] / 2.54
        figure_height = FIGURE_SIZE_CM[1] / 2.54
        figure_dpi = video_size_px[0] / figure_width
    figure = Figure(
        figsize=(figure_width, figure_height),
        dpi=figure_dpi,
        facecolor="white",
        frameon=True,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes(
        (0.215, 0.165, 0.56, 0.79) if paper else (0.17, 0.19, 0.62, 0.62)
    )
    tick_font_size = FONT_SIZE if paper else FONT_SIZE - 1

    surface_x = surface_sigma_1 / STRESS_SCALE
    surface_y = surface_sigma_2 / STRESS_SCALE
    limit_min = -0.5  # MPa.
    limit_max = 0.5  # MPa.

    invalid_region = Polygon(
        [(limit_min, limit_min), (limit_min, limit_max), (limit_max, limit_max)],
        closed=True,
        facecolor="0.92",
        edgecolor="none",
        zorder=0,
    )
    axis.add_patch(invalid_region)
    axis.plot(
        [limit_min, limit_max],
        [limit_min, limit_max],
        color="0.70",
        linewidth=0.5,
        zorder=1,
    )
    axis.text(
        limit_min + 0.22 * (limit_max - limit_min),
        limit_min + 0.78 * (limit_max - limit_min),
        r"$\sigma_2 > \sigma_1$",
        color="0.45",
        fontsize=FONT_SIZE - 1,
        ha="center",
        va="center",
    )

    valid_surface = np.ma.masked_where(
        surface_sigma_2 > surface_sigma_1,
        failure_criterion,
    )
    axis.contour(
        surface_x,
        surface_y,
        valid_surface,
        levels=[0.0],
        colors="black",
        linewidths=0.8,
        zorder=2,
    )

    phase_norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    first_offsets, first_colours = _point_render_data(
        first_sigma_1,
        first_sigma_2,
        first_phasefield,
        phase_norm,
    )
    scatter = axis.scatter(
        first_offsets[:, 0],
        first_offsets[:, 1],
        c=first_colours,
        s=POINT_SIZE,
        linewidths=0.0,
        rasterized=True,
        zorder=2.5,
    )
    axis.set_xlim(limit_min, limit_max)
    axis.set_ylim(limit_min, limit_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(rf"$\sigma_1\;[{STRESS_UNIT}]$", fontsize=FONT_SIZE, labelpad=2)
    axis.set_ylabel(rf"$\sigma_2\;[{STRESS_UNIT}]$", fontsize=FONT_SIZE, labelpad=2)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(
        axis="both",
        which="both",
        labelsize=tick_font_size,
        length=2.0,
        width=0.5,
        pad=2.0,
    )
    axis.grid(True, color="0.85", linestyle=":", linewidth=0.45)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)

    if not paper:
        figure.text(
            0.025,
            0.975,
            "Principal-stress evolution",
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
    )
    if paper:
        colorbar_x = 0.835
        colorbar_width = 0.022
        colorbar_axis = figure.add_axes((colorbar_x, 0.20, colorbar_width, 0.55))
        colorbar_title_y = 0.78
    else:
        colorbar_x = 0.85
        colorbar_width = 0.018
        colorbar_axis = figure.add_axes((colorbar_x, 0.22, colorbar_width, 0.55))
        colorbar_title_y = 0.815
    colorbar = ColorbarBase(
        colorbar_axis,
        cmap=colormaps["inferno"],
        norm=phase_norm,
        ticks=np.linspace(0.0, 1.0, 5),
        orientation="vertical",
    )
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(
        axis="y",
        which="both",
        labelsize=tick_font_size,
        length=2.0,
        width=0.5,
        pad=2.0,
    )
    figure.text(
        colorbar_x + 0.5 * colorbar_width,
        colorbar_title_y,
        r"$\phi$ [-]",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    return figure, canvas, scatter, annotation


def Plot_PrincipalStress_Case(
    frames: Sequence[_FrameLike],
    selected_states: Sequence[_SelectedStateLike],
    path_animations_output: str | Path,
    *,
    case_label: str = "",
    failure_surface_path: str | Path = FAILURE_SURFACE_PATH,
    surface_group: str = FAILURE_SURFACE_GROUP,
    figure_dpi: int = FIGURE_DPI,
    video_size_px: tuple[int, int] = VIDEO_SIZE_PX,
    video_fps: int = VIDEO_FPS,
) -> None:
    """Create a principal-stress MP4 and selected-state PNG/PDF figures."""

    if not frames:
        raise ValueError("At least one output frame is required")
    output_dir = Path(path_animations_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_sigma_1, surface_sigma_2, failure_criterion = _read_failure_surface(
        Path(failure_surface_path),
        surface_group,
    )
    first_sigma_1, first_sigma_2, first_phasefield = _read_stress_snapshot(
        Path(frames[0].path)
    )
    figure, canvas, scatter, annotation = _create_figure(
        first_sigma_1,
        first_sigma_2,
        first_phasefield,
        surface_sigma_1,
        surface_sigma_2,
        failure_criterion,
        video_size_px,
        case_label,
    )
    paper_figure, _, paper_scatter, _ = _create_figure(
        first_sigma_1,
        first_sigma_2,
        first_phasefield,
        surface_sigma_1,
        surface_sigma_2,
        failure_criterion,
        video_size_px,
        case_label,
        paper=True,
    )
    selected_by_index: dict[int, list[_SelectedStateLike]] = {}
    for state in selected_states:
        selected_by_index.setdefault(state.frame_index, []).append(state)
    phase_norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

    video_path = output_dir / "principal_stress.mp4"
    writer = _open_video_writer(video_path, video_fps)
    try:
        for frame_index, frame in enumerate(frames):
            if frame_index == 0:
                sigma_1, sigma_2, phasefield = (
                    first_sigma_1,
                    first_sigma_2,
                    first_phasefield,
                )
            else:
                sigma_1, sigma_2, phasefield = _read_stress_snapshot(
                    Path(frame.path)
                )
            point_offsets, point_colours = _point_render_data(
                sigma_1,
                sigma_2,
                phasefield,
                phase_norm,
            )
            scatter.set_offsets(point_offsets)
            scatter.set_facecolors(point_colours)
            annotation.set_text(_annotation(frame))
            canvas.draw()
            video_frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[
                ..., :3
            ].copy()
            writer.append_data(video_frame)

            for state in selected_by_index.get(frame_index, []):
                # Manuscript stills come from the title-free paper figure.
                paper_scatter.set_offsets(point_offsets)
                paper_scatter.set_facecolors(point_colours)
                snapshot_path = output_dir / f"principal_stress_{state.name}"
                save_options = {
                    "dpi": figure_dpi,
                    "facecolor": "white",
                    "edgecolor": "white",
                    "transparent": False,
                }
                paper_figure.savefig(snapshot_path.with_suffix(".png"), **save_options)

                # The manuscript figure draws the points opaque; the PDF keeps
                # the rasterized scatter layer on a transparent background, so the
                # failure-region fill remains visible underneath.
                original_facecolors = paper_scatter.get_facecolors().copy()
                opaque_facecolors = original_facecolors.copy()
                opaque_facecolors[:, 3] = 1.0
                paper_scatter.set_facecolors(opaque_facecolors)
                try:
                    paper_figure.savefig(
                        snapshot_path.with_suffix(".pdf"),
                        format="pdf",
                        metadata={"CreationDate": None},
                        **save_options,
                    )
                finally:
                    paper_scatter.set_facecolors(original_facecolors)
    finally:
        writer.close()
        figure.clear()
        paper_figure.clear()
    print(f"Saved {video_path} ({len(frames)} frames)")
