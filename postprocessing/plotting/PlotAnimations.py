"""Create paper-quality field, mesh, stress, and response animations.

The IceCubes HDF5 output format stores one discontinuous field table per
output step and one immutable mesh file per mesh epoch.  This module reads
those files directly, keeps only one state in memory at a time, and renders
with PyVista/VTK.  Scalar fields are drawn as filled surfaces.  Quadratic
displacement elements are split into four rendering triangles, so both the
colour field and the magnified deformation retain the saved P2 detail.  The
dedicated mesh-evolution view remains an outline-only plot.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import warnings

import h5py
import imageio
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
from PIL import Image
import pyvista as pv

from .PlotLoadDisplacement import Plot_LoadDisplacement_Case
from .PlotPrincipalStress import Plot_PrincipalStress_Case


OUTPUT_PATTERN = re.compile(r"^Outputs_(?P<step>\d+)\.hdf5$")
MESH_PATTERN = re.compile(r"^mesh_epoch_(?P<epoch>\d+)\.hdf5$")

DEFORMATION_SCALE = 1000.0
DISPLACEMENT_TO_MM = 1.0e3
FONT_SIZE = 7
TITLE_FONT_SIZE = 8
CASE_LABEL_FONT_SIZE = 5
FIGURE_SIZE_CM = (8.0, 8.0)
FIGURE_DPI = 600
PAPER_DPI = 300
# Manuscript stills: saved at their final printed size without title, case
# label or state annotation (the caption carries that information).  The
# view keeps the plate at the left and the colourbar in the right margin.
STILL_SIZE_CM = (6.8, 5.4)
STILL_BOUNDS = (-0.04, 1.345, -0.05, 1.05, -0.01, 0.01)
VIDEO_SIZE_PX = (1200, 1200)
VIDEO_FPS = 5
PLOT_BOUNDS = (-0.1, 1.3, -0.1, 1.3, -0.01, 0.01)
STRESS_FIELDS = ("sigma_xx", "sigma_yy", "sigma_xy")
STRESS_SCALE = 1.0e6

_TRIANGLE_PATTERNS = {
    1: np.asarray([[0, 1, 2]], dtype=np.int64),
    # Basix P2 triangle order: vertices 0,1,2 followed by edge midpoints
    # (1,2), (0,2), (0,1).
    2: np.asarray(
        [[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]],
        dtype=np.int64,
    ),
}


@dataclass(frozen=True)
class _Frame:
    step: int
    path: Path
    time: float
    load: float
    external_displacement: float  # BC_Top_u, metres.
    epoch: int


@dataclass(frozen=True)
class _SelectedState:
    name: str
    display_name: str
    frame_index: int


def _numbered_files(
    folder: Path,
    pattern: re.Pattern[str],
    group: str,
) -> list[tuple[int, Path]]:
    numbered: list[tuple[int, Path]] = []
    for path in folder.iterdir():
        match = pattern.match(path.name)
        if match is not None:
            numbered.append((int(match.group(group)), path))
    return sorted(numbered, key=lambda item: item[0])


def _decode_names(raw_names: np.ndarray) -> list[str]:
    return [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_names
    ]


def _read_time_history(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_path = data_dir / "time_data.hdf5"
    if not time_path.exists():
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

    time_column = names.index("time") if "time" in names else 0
    if "BC_Top_F" in names:
        load_column = names.index("BC_Top_F")
    elif data.shape[1] > 4:
        warnings.warn(
            f"{time_path} has no BC_Top_F name; using legacy column 4",
            stacklevel=2,
        )
        load_column = 4
    else:
        raise KeyError(f"{time_path} does not contain a BC_Top_F column")

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

    return (
        data[:, time_column],
        data[:, load_column],
        data[:, displacement_column],
    )


def _time_row_indices(steps: np.ndarray, history_length: int) -> np.ndarray:
    # IceCubes numbers the first completed solve as step 1, while the first
    # time-history row has zero-based index 0.
    indices = steps - 1
    if np.all((indices >= 0) & (indices < history_length)):
        return indices
    if history_length == len(steps):
        return np.arange(len(steps), dtype=int)
    raise ValueError(
        "Could not map Outputs_<step>.hdf5 files to time_data.hdf5 rows: "
        f"steps span {steps[0]}..{steps[-1]}, but the history has "
        f"{history_length} rows"
    )


def _index_frames(data_dir: Path) -> list[_Frame]:
    output_files = _numbered_files(data_dir, OUTPUT_PATTERN, "step")
    if not output_files:
        raise FileNotFoundError(f"No Outputs_<step>.hdf5 files found in {data_dir}")

    times, loads, external_displacements = _read_time_history(data_dir)
    steps = np.asarray([step for step, _ in output_files], dtype=int)
    time_rows = _time_row_indices(steps, len(times))

    frames: list[_Frame] = []
    for (step, path), row in zip(output_files, time_rows):
        with h5py.File(path, "r") as h5_file:
            if "epoch_tag" not in h5_file:
                raise KeyError(f"{path} does not contain an epoch_tag")
            epoch = int(h5_file["epoch_tag"][()])
        frames.append(
            _Frame(
                step=step,
                path=path,
                time=float(times[row]),
                load=float(loads[row]),
                external_displacement=float(external_displacements[row]),
                epoch=epoch,
            )
        )
    return frames


def _select_states(frames: list[_Frame]) -> list[_SelectedState]:
    load_magnitudes = np.abs(np.asarray([frame.load for frame in frames]))
    if not np.all(np.isfinite(load_magnitudes)):
        raise ValueError("BC_Top_F contains non-finite values at saved output steps")

    peak_index = int(np.argmax(load_magnitudes))
    peak_load = float(load_magnitudes[peak_index])
    if peak_load <= 0.0:
        raise ValueError("The saved output states do not contain a non-zero load")

    selected = [_SelectedState("peak_load", "Peak load", peak_index)]
    post_peak = np.arange(peak_index, len(frames), dtype=int)
    post_peak_loads = load_magnitudes[post_peak]
    for percentage in (75, 50, 25):
        target = percentage / 100.0 * peak_load
        relative_index = int(np.argmin(np.abs(post_peak_loads - target)))
        frame_index = int(post_peak[relative_index])
        actual_fraction = load_magnitudes[frame_index] / peak_load
        if np.min(post_peak_loads) > target:
            warnings.warn(
                f"The saved unloading path does not reach {percentage}% of peak "
                f"load; using its closest state ({actual_fraction:.1%})",
                stacklevel=2,
            )
        selected.append(
            _SelectedState(
                f"unloading_{percentage}pct",
                rf"Unloading (${percentage}\%$ remaining)",
                frame_index,
            )
        )
    return selected


def _write_state_table(
    output_dir: Path,
    frames: list[_Frame],
    selected_states: list[_SelectedState],
) -> None:
    peak_load = abs(frames[selected_states[0].frame_index].load)
    with (output_dir / "selected_states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "state",
                "step",
                "time_s",
                "BC_Top_F",
                "absolute_load_fraction_of_peak",
                "mesh_epoch",
                "output_file",
            ]
        )
        for state in selected_states:
            frame = frames[state.frame_index]
            writer.writerow(
                [
                    state.name,
                    frame.step,
                    f"{frame.time:.16g}",
                    f"{frame.load:.16g}",
                    f"{abs(frame.load) / peak_load:.16g}",
                    frame.epoch,
                    frame.path.name,
                ]
            )


def _mesh_file(data_dir: Path, epoch: int) -> Path:
    path = data_dir / f"mesh_epoch_{epoch:03d}.hdf5"
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh for epoch {epoch}: {path}")
    return path


@lru_cache(maxsize=4)
def _read_coordinates(mesh_file: Path, degree: int) -> np.ndarray:
    group_name = f"Mesh_DG{degree}"
    with h5py.File(mesh_file, "r") as h5_file:
        if group_name not in h5_file:
            raise KeyError(f"{mesh_file} does not contain {group_name}")
        group = h5_file[group_name]
        if "X" not in group or "Y" not in group:
            raise KeyError(f"{mesh_file}:{group_name} requires X and Y datasets")
        x = np.asarray(group["X"], dtype=float)
        y = np.asarray(group["Y"], dtype=float)

    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Invalid coordinate tables in {mesh_file}:{group_name}")
    return np.stack((x, y), axis=-1)


def _read_field(output_file: Path, field_name: str) -> tuple[np.ndarray, int]:
    with h5py.File(output_file, "r") as h5_file:
        dataset_name = f"Fields/{field_name}"
        if dataset_name not in h5_file:
            raise KeyError(f"{output_file} does not contain {dataset_name}")
        dataset = h5_file[dataset_name]
        values = np.asarray(dataset, dtype=float)
        if "dg_degree" not in dataset.attrs:
            raise KeyError(f"{output_file}:{dataset_name} has no dg_degree attribute")
        degree = int(dataset.attrs["dg_degree"])
    if values.ndim != 2:
        raise ValueError(f"{output_file}:{dataset_name} must be a 2-D cell table")
    return values, degree


def _read_displacement(output_file: Path) -> tuple[np.ndarray, np.ndarray, int]:
    u_x, degree_x = _read_field(output_file, "u_x")
    u_y, degree_y = _read_field(output_file, "u_y")
    if degree_x != degree_y or u_x.shape != u_y.shape:
        raise ValueError(f"u_x and u_y are incompatible in {output_file}")
    return u_x, u_y, degree_x


def _scan_displacement_limits(frames: list[_Frame]) -> tuple[float, float]:
    """Return symmetric and zero-based y-displacement limits in millimetres."""

    max_abs_y = 0.0
    max_positive_y = 0.0
    for frame in frames:
        u_x, u_y, _ = _read_displacement(frame.path)
        if not np.all(np.isfinite(u_x)) or not np.all(np.isfinite(u_y)):
            raise ValueError(f"Non-finite displacement values in {frame.path}")
        max_abs_y = max(max_abs_y, float(np.max(np.abs(u_y), initial=0.0)))
        max_positive_y = max(max_positive_y, float(np.max(u_y, initial=0.0)))
    if max_abs_y == 0.0:
        # VTK needs a non-degenerate range for an all-zero history.
        max_abs_y = np.finfo(float).eps
    if max_positive_y == 0.0:
        # Keep the zero-based range non-degenerate if no positive value occurs.
        max_positive_y = np.finfo(float).eps
    return (
        max_abs_y * DISPLACEMENT_TO_MM,
        max_positive_y * DISPLACEMENT_TO_MM,
    )


def _scan_stress_limit(frames: list[_Frame]) -> float:
    """Return one symmetric MPa limit shared by all stress components."""

    max_abs_stress = 0.0
    for frame in frames:
        with h5py.File(frame.path, "r") as h5_file:
            for field_name in STRESS_FIELDS:
                dataset_name = f"Fields/{field_name}"
                if dataset_name not in h5_file:
                    raise KeyError(f"{frame.path} does not contain {dataset_name}")
                dataset = h5_file[dataset_name]
                if int(dataset.attrs.get("dg_degree", -1)) != 0:
                    raise ValueError(
                        f"{frame.path}:{dataset_name} must be a DG0 cell field"
                    )
                values = np.asarray(dataset, dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite {field_name} values in {frame.path}"
                    )
                max_abs_stress = max(
                    max_abs_stress,
                    float(np.max(np.abs(values), initial=0.0)),
                )
    if max_abs_stress == 0.0:
        max_abs_stress = np.finfo(float).eps
    return max_abs_stress / STRESS_SCALE


def _field_surface_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
    degree: int,
) -> pv.PolyData:
    if degree not in _TRIANGLE_PATTERNS:
        raise NotImplementedError(
            f"Only degree-1 and degree-2 triangular output is supported, got degree {degree}"
        )
    if coordinates.shape[:2] != values.shape:
        raise ValueError(
            f"Coordinate shape {coordinates.shape[:2]} does not match field shape {values.shape}"
        )

    node_count = coordinates.shape[1]
    expected_nodes = 3 if degree == 1 else 6
    if node_count != expected_nodes:
        raise ValueError(
            f"A degree-{degree} triangle requires {expected_nodes} nodes, got {node_count}"
        )

    points_2d = coordinates.reshape(-1, 2)
    points = np.column_stack((points_2d, np.zeros(len(points_2d))))
    offsets = np.arange(coordinates.shape[0], dtype=np.int64)[:, None, None] * node_count
    triangles = (offsets + _TRIANGLE_PATTERNS[degree][None, :, :]).reshape(-1, 3)
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    grid = pv.PolyData(points, faces)
    grid.point_data["values"] = values.reshape(-1)
    return grid


def _cell_field_surface_grid(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> pv.PolyData:
    """Create filled triangles carrying one constant scalar per element."""

    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 2):
        raise ValueError(
            "A DG0 cell field requires degree-1 triangular mesh coordinates"
        )
    flattened_values = np.asarray(values, dtype=float).reshape(len(values), -1)
    if len(coordinates) != len(flattened_values) or flattened_values.shape[1] != 1:
        raise ValueError(
            f"Mesh has {len(coordinates)} cells but the field has shape {values.shape}"
        )

    points_2d = coordinates.reshape(-1, 2)
    points = np.column_stack((points_2d, np.zeros(len(points_2d))))
    triangles = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    grid = pv.PolyData(points, faces)
    grid.cell_data["values"] = flattened_values[:, 0]
    return grid


def _field_grid(data_dir: Path, frame: _Frame, field_name: str) -> pv.PolyData:
    mesh_file = _mesh_file(data_dir, frame.epoch)
    if field_name in {"u_x", "u_y"}:
        u_x, u_y, degree = _read_displacement(frame.path)
        coordinates = _read_coordinates(mesh_file, degree)
        deformed = coordinates.copy()
        deformed[..., 0] += DEFORMATION_SCALE * u_x
        deformed[..., 1] += DEFORMATION_SCALE * u_y
        values = (u_x if field_name == "u_x" else u_y) * DISPLACEMENT_TO_MM
        return _field_surface_grid(deformed, values, degree)

    if field_name in STRESS_FIELDS:
        values, degree = _read_field(frame.path, field_name)
        if degree != 0:
            raise ValueError(f"{frame.path}:Fields/{field_name} is not DG0")
        coordinates = _read_coordinates(mesh_file, degree=1)
        return _cell_field_surface_grid(coordinates, values / STRESS_SCALE)

    values, degree = _read_field(frame.path, field_name)
    coordinates = _read_coordinates(mesh_file, degree)
    return _field_surface_grid(coordinates, values, degree)


def _configure_plotter(
    grid: pv.DataSet,
    *,
    cmap: str,
    clim: tuple[float, float],
    bounds: tuple[float, ...],
    preference: str = "point",
    line_width: float | None = None,
    log_scale: bool = False,
) -> tuple[pv.Plotter, object]:
    plotter = pv.Plotter(off_screen=True, window_size=VIDEO_SIZE_PX, border=False)
    plotter.set_background("white")
    mesh_options: dict[str, object] = {
        "scalars": "values",
        "preference": preference,
        "cmap": cmap,
        "clim": clim,
        "lighting": False,
        "show_scalar_bar": False,
        "log_scale": log_scale,
    }
    if line_width is not None:
        mesh_options["line_width"] = line_width
        mesh_options["render_lines_as_tubes"] = False
    actor = plotter.add_mesh(grid, **mesh_options)
    plotter._fixed_plot_bounds = bounds
    _set_fixed_camera(plotter, bounds)
    return plotter, actor


def _set_fixed_camera(plotter: pv.Plotter, bounds: tuple[float, ...]) -> None:
    """Apply one deterministic square, parallel view to every plot."""

    x_min, x_max, y_min, y_max, _, _ = bounds
    centre_x = 0.5 * (x_min + x_max)
    centre_y = 0.5 * (y_min + y_max)
    plotter.camera_position = [
        (centre_x, centre_y, 1.0),
        (centre_x, centre_y, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 0.5 * (y_max - y_min)


def _state_annotation(
    frame: _Frame,
    state_name: str | None = None,
) -> str:
    prefix = f"{state_name}; " if state_name else ""
    time_magnitude = abs(frame.time)
    if time_magnitude != 0.0 and (time_magnitude >= 1.0e3 or time_magnitude < 1.0e-2):
        exponent = int(np.floor(np.log10(time_magnitude)))
        mantissa = frame.time / 10.0**exponent
        time_label = rf"{mantissa:.3f} \times 10^{{{exponent}}}"
    else:
        time_label = f"{frame.time:.4g}"
    return rf"{prefix}$t = {time_label}\,\mathrm{{s}}$"


def _colourbar_ticks(clim: tuple[float, float], log_scale: bool) -> np.ndarray:
    if log_scale:
        return np.geomspace(clim[0], clim[1], 5)
    return np.linspace(clim[0], clim[1], 5)


def _compose_frame(
    raw_image: np.ndarray,
    *,
    title: str,
    case_label: str,
    annotation: str,
    colorbar_title: str,
    cmap: str,
    clim: tuple[float, float],
    log_scale: bool,
    paper: bool = False,
) -> np.ndarray:
    """Add consistently sized MathText labels and a borderless colourbar.

    With ``paper`` the frame is composed at ``STILL_SIZE_CM`` without the
    title, case label and annotation.
    """

    height, width = raw_image.shape[:2]
    size_cm = STILL_SIZE_CM if paper else FIGURE_SIZE_CM
    figure_width_in = size_cm[0] / 2.54
    figure_height_in = size_cm[1] / 2.54
    dpi = width / figure_width_in
    figure = Figure(
        figsize=(figure_width_in, figure_height_in),
        dpi=dpi,
        facecolor="white",
        frameon=True,
    )
    canvas = FigureCanvasAgg(figure)

    image_axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
    image_axis.imshow(raw_image, interpolation="none")
    image_axis.set_axis_off()

    if not paper:
        figure.text(
            0.025,
            0.975,
            title,
            ha="left",
            va="top",
            fontsize=TITLE_FONT_SIZE,
            color="black",
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
        figure.text(
            0.025,
            0.925,
            annotation,
            ha="left",
            va="top",
            fontsize=FONT_SIZE,
            color="black",
            linespacing=1.15,
        )

    # The specimen occupies x <= 1.0 within the fixed [-0.2, 1.2] view.  Put
    # the scale in the resulting right-hand margin, and leave dedicated space
    # above it for the quantity label and (where needed) its power of ten.
    if paper:
        colorbar_x = 0.79
        colorbar_width = 0.022
        colorbar_text_x = 0.84
        colorbar_axis = figure.add_axes((colorbar_x, 0.12, colorbar_width, 0.60))
        title_y, exponent_y = 0.80, 0.745
    else:
        colorbar_x = 0.875
        colorbar_width = 0.018
        colorbar_text_x = 0.91
        colorbar_axis = figure.add_axes((colorbar_x, 0.18, colorbar_width, 0.56))
        title_y, exponent_y = 0.805, 0.765
    norm = LogNorm(vmin=clim[0], vmax=clim[1]) if log_scale else Normalize(*clim)
    colorbar = ColorbarBase(
        colorbar_axis,
        cmap=colormaps[cmap],
        norm=norm,
        ticks=_colourbar_ticks(clim, log_scale),
        orientation="vertical",
    )
    colorbar.outline.set_visible(False)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    colorbar.ax.tick_params(
        axis="y",
        which="both",
        labelsize=FONT_SIZE,
        length=2.0,
        width=0.5,
        color="black",
    )
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.tick_params(axis="y", labelleft=False, labelright=True, pad=2.0)
    figure.text(
        colorbar_text_x,
        title_y,
        colorbar_title,
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        color="black",
    )
    if log_scale:
        colorbar.minorticks_off()
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: rf"${value:.2g}$")
        )
    elif clim == (0.0, 1.0):
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        max_magnitude = max(abs(clim[0]), abs(clim[1]))
        use_scientific = max_magnitude >= 1.0e3 or max_magnitude < 1.0e-1
        if use_scientific:
            exponent = int(np.floor(np.log10(max_magnitude)))
            scale = 10.0**exponent
            colorbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: rf"${value / scale:.2g}$")
            )
            figure.text(
                colorbar_text_x,
                exponent_y,
                rf"$\times 10^{{{exponent}}}$",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE,
                color="black",
            )
        else:
            colorbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: rf"${value:.2g}$")
            )

    canvas.draw()
    composed = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    figure.clear()
    return composed


def _set_actor_grid(actor: object, grid: pv.DataSet, clim: tuple[float, float]) -> None:
    actor.mapper.dataset = grid
    actor.mapper.scalar_range = clim


def _open_video_writer(path: Path):
    return imageio.get_writer(
        str(path),
        fps=VIDEO_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    )


def _capture_frame(
    plotter: pv.Plotter,
    *,
    window_size: tuple[int, int],
    title: str,
    case_label: str,
    annotation: str,
    colorbar_title: str,
    cmap: str,
    clim: tuple[float, float],
    log_scale: bool,
    paper: bool = False,
) -> np.ndarray:
    _set_fixed_camera(plotter, STILL_BOUNDS if paper else plotter._fixed_plot_bounds)
    raw_image = np.asarray(
        plotter.screenshot(return_img=True, window_size=window_size),
        dtype=np.uint8,
    ).copy()
    return _compose_frame(
        raw_image,
        title=title,
        case_label=case_label,
        annotation=annotation,
        colorbar_title=colorbar_title,
        cmap=cmap,
        clim=clim,
        log_scale=log_scale,
        paper=paper,
    )


def _save_still(
    plotter: pv.Plotter,
    path: Path,
    *,
    title: str,
    case_label: str,
    annotation: str,
    colorbar_title: str,
    cmap: str,
    clim: tuple[float, float],
    log_scale: bool = False,
    paper: bool = True,
) -> None:
    """Save a PNG and a PDF still; ``paper`` gives the manuscript layout."""

    size_cm = STILL_SIZE_CM if paper else FIGURE_SIZE_CM
    width = int(round(size_cm[0] / 2.54 * FIGURE_DPI))
    height = int(round(size_cm[1] / 2.54 * FIGURE_DPI))
    image = _capture_frame(
        plotter,
        window_size=(width, height),
        title=title,
        case_label=case_label,
        annotation=annotation,
        colorbar_title=colorbar_title,
        cmap=cmap,
        clim=clim,
        log_scale=log_scale,
        paper=paper,
    )
    image = np.asarray(image, dtype=np.uint8)[..., :3]
    Image.fromarray(image).save(
        path,
        dpi=(FIGURE_DPI, FIGURE_DPI),
        compress_level=4,
    )

    paper_size = (
        int(round(size_cm[0] / 2.54 * PAPER_DPI)),
        int(round(size_cm[1] / 2.54 * PAPER_DPI)),
    )
    paper_image = Image.fromarray(image).resize(
        paper_size,
        resample=getattr(Image, "Resampling", Image).LANCZOS,
    )
    figure = Figure(
        figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54),
        dpi=PAPER_DPI,
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    image_axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), frameon=False)
    image_axis.imshow(np.asarray(paper_image), interpolation="none", aspect="auto")
    image_axis.set_axis_off()
    figure.savefig(
        path.with_suffix(".pdf"),
        format="pdf",
        metadata={"CreationDate": None},
        dpi=PAPER_DPI,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        pad_inches=0,
    )
    figure.clear()


def _field_style(
    field_name: str,
    displacement_limit: float,
    positive_y_limit: float,
    stress_limit: float,
) -> tuple[str, str, str, tuple[float, float]]:
    if field_name == "u_x":
        return (
            rf"$u_x$: deformed $\times {DEFORMATION_SCALE:g}$",
            r"$u_x$" + "\n" + r"$[\mathrm{mm}]$",
            "coolwarm",
            (-displacement_limit, displacement_limit),
        )
    if field_name == "u_y":
        return (
            rf"$u_y$: deformed $\times {DEFORMATION_SCALE:g}$",
            r"$u_y$" + "\n" + r"$[\mathrm{mm}]$",
            "coolwarm",
            (0.0, positive_y_limit),
        )
    if field_name == "phasefield":
        return (
            r"Phase field $\phi$: undeformed",
            r"$\phi$ [-]",
            "inferno",
            (0.0, 1.0),
        )
    if field_name in STRESS_FIELDS:
        component = field_name.removeprefix("sigma_")
        return (
            rf"$\sigma_{{{component}}}$: undeformed",
            rf"$\sigma_{{{component}}}$" + "\n" + r"$[\mathrm{MPa}]$",
            "coolwarm",
            (-stress_limit, stress_limit),
        )
    raise ValueError(f"Unsupported field: {field_name}")


def _render_field(
    data_dir: Path,
    output_dir: Path,
    frames: list[_Frame],
    selected_states: list[_SelectedState],
    case_label: str,
    field_name: str,
    displacement_limit: float,
    positive_y_limit: float,
    snapshot_positive_y_limit: float,
    stress_limit: float,
    bounds: tuple[float, ...],
) -> None:
    title, colorbar_title, cmap, clim = _field_style(
        field_name,
        displacement_limit,
        positive_y_limit,
        stress_limit,
    )
    if field_name == "u_y":
        if (
            not np.isfinite(snapshot_positive_y_limit)
            or snapshot_positive_y_limit <= 0.0
        ):
            raise ValueError("The snapshot u_y limit must be finite and positive")
        snapshot_clim = (0.0, snapshot_positive_y_limit)
    else:
        snapshot_clim = clim

    first_grid = _field_grid(data_dir, frames[0], field_name)
    plotter, actor = _configure_plotter(
        first_grid,
        cmap=cmap,
        clim=clim,
        bounds=bounds,
        preference="cell" if field_name in STRESS_FIELDS else "point",
    )
    selected_by_index: dict[int, list[_SelectedState]] = {}
    for state in selected_states:
        selected_by_index.setdefault(state.frame_index, []).append(state)
    video_path = output_dir / f"{field_name}.mp4"
    writer = _open_video_writer(video_path)
    try:
        for frame_index, frame in enumerate(frames):
            grid = first_grid if frame_index == 0 else _field_grid(data_dir, frame, field_name)
            if frame_index != 0:
                _set_actor_grid(actor, grid, clim)
            plotter.render()
            video_frame = _capture_frame(
                plotter,
                window_size=VIDEO_SIZE_PX,
                title=title,
                case_label=case_label,
                annotation=_state_annotation(frame),
                colorbar_title=colorbar_title,
                cmap=cmap,
                clim=clim,
                log_scale=False,
            )
            writer.append_data(video_frame)

            for state in selected_by_index.get(frame_index, []):
                if snapshot_clim != clim:
                    actor.mapper.scalar_range = snapshot_clim
                    plotter.render()
                try:
                    _save_still(
                        plotter,
                        output_dir / f"{field_name}_{state.name}.png",
                        title=title,
                        case_label=case_label,
                        annotation=_state_annotation(
                            frame,
                            state.display_name,
                        ),
                        colorbar_title=colorbar_title,
                        cmap=cmap,
                        clim=snapshot_clim,
                    )
                finally:
                    if snapshot_clim != clim:
                        actor.mapper.scalar_range = clim
                        plotter.render()
    finally:
        writer.close()
        plotter.close()
    print(f"Saved {video_path} ({len(frames)} frames)")


def _mesh_grid(mesh_file: Path) -> tuple[pv.PolyData, np.ndarray]:
    coordinates = _read_coordinates(mesh_file, degree=1)
    if coordinates.shape[1] != 3:
        raise ValueError(f"{mesh_file}: Mesh_DG1 is not triangular")

    edge_01 = np.linalg.norm(coordinates[:, 1] - coordinates[:, 0], axis=1)
    edge_12 = np.linalg.norm(coordinates[:, 2] - coordinates[:, 1], axis=1)
    edge_20 = np.linalg.norm(coordinates[:, 0] - coordinates[:, 2], axis=1)
    characteristic_size = np.maximum.reduce((edge_01, edge_12, edge_20))
    if np.any(~np.isfinite(characteristic_size)) or np.any(characteristic_size <= 0.0):
        raise ValueError(f"{mesh_file} contains degenerate or non-finite elements")

    points_2d = coordinates.reshape(-1, 2)
    points = np.column_stack((points_2d, np.zeros(len(points_2d))))
    base = np.arange(coordinates.shape[0], dtype=np.int64)[:, None] * 3
    edge_pattern = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    edges = (base[:, None, :] + edge_pattern[None, :, :]).reshape(-1, 2)
    lines = np.column_stack((np.full(len(edges), 2, dtype=np.int64), edges)).ravel()
    # Construct an empty PolyData before assigning points.  ``PolyData(points)``
    # would create one vertex cell per point in addition to the line cells,
    # which would make per-edge characteristic-size data ambiguous.
    grid = pv.PolyData()
    grid.points = points
    grid.lines = lines
    grid.cell_data["values"] = np.repeat(characteristic_size, 3)
    return grid, characteristic_size


def _mesh_frames(data_dir: Path, frames: list[_Frame]) -> list[tuple[_Frame, Path]]:
    first_frame_by_epoch: dict[int, _Frame] = {}
    for frame in frames:
        first_frame_by_epoch.setdefault(frame.epoch, frame)

    mesh_files = _numbered_files(data_dir, MESH_PATTERN, "epoch")
    referenced_epochs = set(first_frame_by_epoch)
    available_epochs = {epoch for epoch, _ in mesh_files}
    missing = referenced_epochs - available_epochs
    if missing:
        raise FileNotFoundError(
            f"Missing mesh epoch files referenced by outputs: {sorted(missing)}"
        )
    return [
        (first_frame_by_epoch[epoch], path)
        for epoch, path in mesh_files
        if epoch in referenced_epochs
    ]


def _mesh_size_range(mesh_frames: list[tuple[_Frame, Path]]) -> tuple[float, float]:
    h_min, h_max = np.inf, -np.inf
    for _, mesh_file in mesh_frames:
        coordinates = _read_coordinates(mesh_file, degree=1)
        edge_01 = np.linalg.norm(coordinates[:, 1] - coordinates[:, 0], axis=1)
        edge_12 = np.linalg.norm(coordinates[:, 2] - coordinates[:, 1], axis=1)
        edge_20 = np.linalg.norm(coordinates[:, 0] - coordinates[:, 2], axis=1)
        characteristic_size = np.maximum.reduce((edge_01, edge_12, edge_20))
        h_min = min(h_min, float(np.min(characteristic_size)))
        h_max = max(h_max, float(np.max(characteristic_size)))
    if not np.isfinite(h_min) or not np.isfinite(h_max):
        raise ValueError("Could not determine a finite characteristic mesh-size range")
    if h_min == h_max:
        h_min *= 1.0 - 1.0e-6
        h_max *= 1.0 + 1.0e-6
    return h_min, h_max


def _mesh_annotation(
    frame: _Frame,
    element_count: int,
    state_name: str | None = None,
) -> str:
    """Format load/time and adaptive-mesh information on separate lines."""

    count_label = f"{element_count:,}".replace(",", r"{,}")
    mesh_information = (
        rf"Mesh epoch {frame.epoch}; $N_{{\mathrm{{el}}}} = {count_label}$"
    )
    return f"{_state_annotation(frame, state_name)}\n{mesh_information}"


def _render_mesh(
    data_dir: Path,
    output_dir: Path,
    frames: list[_Frame],
    selected_states: list[_SelectedState],
    case_label: str,
    bounds: tuple[float, ...],
) -> None:
    mesh_frames = _mesh_frames(data_dir, frames)
    clim = _mesh_size_range(mesh_frames)
    title = r"Mesh outline coloured by $h^{\mathrm{el}}$"
    colorbar_title = r"$h^{\mathrm{el}}$ [m]"
    cmap = "viridis"
    first_grid, first_characteristic_size = _mesh_grid(mesh_frames[0][1])
    plotter, actor = _configure_plotter(
        first_grid,
        cmap=cmap,
        clim=clim,
        bounds=bounds,
        preference="cell",
        line_width=0.7,
        log_scale=True,
    )
    video_path = output_dir / "mesh_evolution.mp4"
    writer = _open_video_writer(video_path)
    try:
        for mesh_index, (frame, mesh_file) in enumerate(mesh_frames):
            if mesh_index == 0:
                grid = first_grid
                characteristic_size = first_characteristic_size
            else:
                grid, characteristic_size = _mesh_grid(mesh_file)
            if mesh_index != 0:
                _set_actor_grid(actor, grid, clim)
            plotter.render()
            video_frame = _capture_frame(
                plotter,
                window_size=VIDEO_SIZE_PX,
                title=title,
                case_label=case_label,
                annotation=_mesh_annotation(frame, len(characteristic_size)),
                colorbar_title=colorbar_title,
                cmap=cmap,
                clim=clim,
                log_scale=True,
            )
            writer.append_data(video_frame)
    finally:
        writer.close()
        plotter.close()
    print(f"Saved {video_path} ({len(mesh_frames)} mesh epochs)")

    # The mesh at a selected loading state can share its topology with several
    # output frames, so render these four load annotations separately.
    for state in selected_states:
        frame = frames[state.frame_index]
        grid, characteristic_size = _mesh_grid(_mesh_file(data_dir, frame.epoch))
        still_plotter, _ = _configure_plotter(
            grid,
            cmap=cmap,
            clim=clim,
            bounds=bounds,
            preference="cell",
            line_width=0.7,
            log_scale=True,
        )
        try:
            _save_still(
                still_plotter,
                output_dir / f"mesh_{state.name}.png",
                title=title,
                case_label=case_label,
                annotation=_mesh_annotation(
                    frame,
                    len(characteristic_size),
                    state.display_name,
                ),
                colorbar_title=colorbar_title,
                cmap=cmap,
                clim=clim,
                log_scale=True,
            )
        finally:
            still_plotter.close()


def Plot_Animations_Case(
    path_data: str | Path,
    path_animations_output: str | Path,
    *,
    history_strain_rate: float = 1.5e-9,
    history_height_m: float = 1.0,
) -> None:
    """Render displacement, phase-field, stress, and mesh outputs for one case.

    Parameters
    ----------
    path_data:
        The case's ``Data`` directory.  Passing the parent case directory is
        also accepted when it contains a directory named ``Data``.
    path_animations_output:
        Case-specific output directory.  Videos of u_y, the phase field,
        sigma_yy, the mesh, the principal stresses and the load-displacement
        curve, four loading-state figures per plotted quantity, and a CSV
        recording the selected states are written here.

    Notes
    -----
    The ``u_y`` animation uses ``[0, max(u_y)]`` over the
    full history, while its selected-state snapshots use the external
    displacement at the 25% unloading state as their shared upper limit.
    Limits are displayed in millimetres.
    Phase-field colours are fixed to
    ``[0, 1]``.  The sigma_yy colours use the symmetric MPa colour
    range evaluated over all three fields and all frames.  Displacement
    geometry is magnified by
    ``DEFORMATION_SCALE``; phase-field, stress, and mesh geometry are not.
    The displacement, phase, and stress fields are shown as filled surfaces,
    while the dedicated mesh-evolution view shows coloured element outlines.
    These spatial outputs omit coordinate axes and use the same fixed
    ``[-0.2, 1.2]`` view in both directions.  The principal-stress view shows
    the full cell cloud against the ``true_2d`` failure surface, with points
    coloured by cell-average phase field.  Every frame includes the two-level
    case path in small type at the upper-right for identification.
    The load--displacement view uses ``BC_Top_u`` in millimetres and
    ``BC_Top_F`` in kilonewtons; its black history, light-grey future, and red
    current-state marker advance at the same saved-output cadence and frame
    rate as the spatial-field videos.
    """

    data_dir = Path(path_data)
    if (data_dir / "Data").is_dir():
        data_dir = data_dir / "Data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    output_dir = Path(path_animations_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _index_frames(data_dir)
    selected_states = _select_states(frames)
    case_dir = data_dir.parent
    case_label = f"{case_dir.parent.name}/{case_dir.name}"
    _write_state_table(output_dir, frames, selected_states)
    unloading_25pct = next(
        state for state in selected_states if state.name == "unloading_25pct"
    )
    snapshot_positive_y_limit = (
        frames[unloading_25pct.frame_index].external_displacement
        * DISPLACEMENT_TO_MM
    )
    if (
        not np.isfinite(snapshot_positive_y_limit)
        or snapshot_positive_y_limit <= 0.0
    ):
        raise ValueError(
            "BC_Top_u at the 25% unloading state must be finite and positive"
        )

    displacement_limit, positive_y_limit = _scan_displacement_limits(frames)
    stress_limit = _scan_stress_limit(frames)

    print(
        f"Rendering {len(frames)} saved states from {data_dir} to {output_dir}; "
        f"u_x colour limit = +/-{displacement_limit:.6g} mm; "
        f"u_y animation colour limit = [0, {positive_y_limit:.6g}] mm; "
        f"u_y snapshot colour limit = [0, {snapshot_positive_y_limit:.6g}] mm; "
        f"stress colour limit = +/-{stress_limit:.6g} MPa"
    )
    # The fields of Figs. 6 and 7. The stress colour range above is shared by all three
    # stress components, as in the paper figures.
    for field_name in ("u_y", "phasefield", "sigma_yy"):
        _render_field(
            data_dir,
            output_dir,
            frames,
            selected_states,
            case_label,
            field_name,
            displacement_limit,
            positive_y_limit,
            snapshot_positive_y_limit,
            stress_limit,
            PLOT_BOUNDS,
        )
    _render_mesh(
        data_dir,
        output_dir,
        frames,
        selected_states,
        case_label,
        PLOT_BOUNDS,
    )
    Plot_LoadDisplacement_Case(
        data_dir,
        frames,
        selected_states,
        output_dir,
        case_label=case_label,
        figure_dpi=FIGURE_DPI,
        video_size_px=VIDEO_SIZE_PX,
        video_fps=VIDEO_FPS,
        history_strain_rate=history_strain_rate,
        history_height_m=history_height_m,
    )
    Plot_PrincipalStress_Case(
        frames,
        selected_states,
        output_dir,
        case_label=case_label,
        figure_dpi=FIGURE_DPI,
        video_size_px=VIDEO_SIZE_PX,
        video_fps=VIDEO_FPS,
    )
