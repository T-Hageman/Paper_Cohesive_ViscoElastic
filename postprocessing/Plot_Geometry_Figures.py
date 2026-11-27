"""Geometry sketches of the manuscript in one common style.

Writes, in figures/ (repository root):
    PlateWithHole.pdf          plate with a central hole (Section 3), 6 x 7 cm
    SeaIce_Geometry.pdf        sea-ice floe with an off-centre defect (Section 4.1), 8 x 6 cm
    Cliff_Geometry.pdf         grounded ice cliff with surface notches (Section 4.2), 8 x 4 cm
    GcCorrection_Geometry.pdf  double-edge-notched bar (Appendix B), 8 x 6 cm

Style (shared by both): grey domain with a black outline, hatched fixed edge,
a thick bar on the loaded edge with the prescribed displacement drawn as
arrows, dimension lines with filled arrowheads, 7 pt labels (PlotLines.FONTSIZE).
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "simulations"))   # Calibrate_Gc_Correction
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402

from plotting.PlotLines_Appendix import FONTSIZE  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"   # figures/ in the repository root
FIGURES.mkdir(exist_ok=True)

# Section 4 geometries (keep in step with the case set-ups once they exist).
FLOE_SIDE = 100.0            # m
FLOE_DEFECT_RADIUS = 0.5     # m
FLOE_DEFECT_CENTRE = (50.0, 50.0)  # m, central so that the cracks branch rather than run to an edge
CLIFF_LENGTH = 500.0         # m, Cliff_Params.py
CLIFF_HEIGHT = 200.0         # m
CLIFF_WATER_LEVEL = 182.0    # m, water level at the start of the lowering
CLIFF_NOTCHES = tuple(25.0 + 50.0 * i for i in range(10))  # m, positions of the surface notches
CLIFF_NOTCH_DEPTH = 10.0     # m

DOMAIN_FILL = "0.78"
FINE_FILL = "0.62"
OUTLINE_WIDTH = 1.2
THIN_WIDTH = 0.6
ARROW_SCALE = 6.0


def _save(fig, name, size_cm):
    fig.set_size_inches(size_cm[0] / 2.54, size_cm[1] / 2.54)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_options = {"dpi": 1200, "facecolor": "white", "edgecolor": "white", "transparent": False}
    fig.savefig(str(FIGURES / f"{name}.png"), **save_options)
    fig.savefig(str(FIGURES / f"{name}.pdf"), metadata={"CreationDate": None}, **save_options)
    plt.close(fig)
    print(f"Saved {FIGURES / name}.pdf")


def hatched_edge(ax, start, end, depth, count, side=-1.0):
    """Thick edge line with hatch strokes on the outside (fixed boundary)."""

    start, end = np.asarray(start, float), np.asarray(end, float)
    tangent = (end - start) / np.linalg.norm(end - start)
    normal = side * np.array([-tangent[1], tangent[0]])
    ax.plot(*zip(start, end), color="k", linewidth=OUTLINE_WIDTH, solid_capstyle="butt", zorder=4)
    for s in np.linspace(0.0, 1.0, count):
        foot = start + s * (end - start)
        tip = foot + depth * (normal - tangent)  # 45-degree stroke
        ax.plot(*zip(foot, tip), color="k", linewidth=THIN_WIDTH, zorder=4)


def loaded_edge(ax, start, end, gap, thickness, outward):
    """Thick bar just outside the loaded edge (rigid loading platen)."""

    start, end = np.asarray(start, float), np.asarray(end, float)
    outward = np.asarray(outward, float)
    corners = [start + gap * outward, end + gap * outward,
               end + (gap + thickness) * outward, start + (gap + thickness) * outward]
    ax.add_patch(Polygon(corners, closed=True, facecolor="k", edgecolor="none", zorder=4))


def arrow(ax, start, end, text=None, text_offset=(0.0, 0.0), width=THIN_WIDTH, color="k", **text_kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ARROW_SCALE,
                                 linewidth=width, color=color, shrinkA=0, shrinkB=0, zorder=5))
    if text is not None:
        mid = 0.5 * (np.asarray(start) + np.asarray(end)) + np.asarray(text_offset)
        ax.text(*mid, text, fontsize=FONTSIZE, ha="center", va="center", color=color, **text_kw)


def dimension(ax, start, end, text, text_offset, rotation=0.0, tick=0.0, tick_direction=(0.0, 1.0)):
    """Double-headed dimension line with optional extension ticks."""

    start, end = np.asarray(start, float), np.asarray(end, float)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>", mutation_scale=ARROW_SCALE,
                                 linewidth=THIN_WIDTH, color="k", shrinkA=0, shrinkB=0, zorder=5))
    if tick > 0.0:
        tick_direction = np.asarray(tick_direction, float)
        for point in (start, end):
            ax.plot(*zip(point - tick * tick_direction, point + 0.3 * tick * tick_direction),
                    color="k", linewidth=0.4, zorder=3)
    mid = 0.5 * (start + end) + np.asarray(text_offset)
    ax.text(*mid, text, fontsize=FONTSIZE, ha="center", va="center", rotation=rotation,
            rotation_mode="anchor", zorder=6)


def plate_with_hole():
    """Section 3 geometry: 1 m square plate, central hole of radius 0.1 m."""

    fig, ax = plt.subplots()
    side, radius = 1.0, 0.1
    ax.add_patch(Polygon([(0, 0), (side, 0), (side, side), (0, side)], closed=True,
                         facecolor=DOMAIN_FILL, edgecolor="k", linewidth=OUTLINE_WIDTH, zorder=2))
    ax.add_patch(Circle((side / 2, side / 2), radius, facecolor="white", edgecolor="k",
                        linewidth=OUTLINE_WIDTH, zorder=3))
    # fixed bottom edge, loaded top edge
    hatched_edge(ax, (0, 0), (side, 0), depth=0.045, count=17, side=-1.0)
    loaded_edge(ax, (0, side), (side, side), gap=0.012, thickness=0.02, outward=(0, 1))
    # prescribed strain rate: the whole top surface moves with the loading bar
    for x in np.linspace(0.1, side - 0.1, 5):
        arrow(ax, (x, side + 0.032), (x, side + 0.12))
    ax.text(-0.03, side + 0.076, r"$\dot{\varepsilon}_\mathrm{ext}$", fontsize=FONTSIZE,
            ha="right", va="center")
    # dimensions
    dimension(ax, (0, side + 0.17), (side, side + 0.17), r"$1\;\mathrm{m}$", (0.0, 0.05),
              tick=0.03, tick_direction=(0, 1))
    dimension(ax, (side + 0.08, 0), (side + 0.08, side), r"$1\;\mathrm{m}$", (0.06, 0.0),
              rotation=-90, tick=0.03, tick_direction=(1, 0))
    d = radius / np.sqrt(2.0)
    dimension(ax, (side / 2 - d, side / 2 - d), (side / 2 + d, side / 2 + d), "", (0, 0))
    ax.text(side / 2 - 0.125, side / 2 + 0.125, r"$0.2\;\mathrm{m}$", fontsize=FONTSIZE,
            ha="center", va="center", rotation=45, rotation_mode="anchor", zorder=6)
    ax.set_xlim(-0.16, side + 0.2)
    ax.set_ylim(-0.08, side + 0.29)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "PlateWithHole", (6.0, 7.0))


def notched_bar(lx, ly, notch_width, notch_depth, centre_width, stations, extract_label=r"$\sigma,\;\delta$"):
    """Appendix B geometry: double-edge-notched bar with the extraction line."""

    fig, ax = plt.subplots()
    outline = [(0, 0), (lx / 2 - notch_width / 2, 0), (lx / 2 - notch_width / 2, notch_depth),
               (lx / 2 + notch_width / 2, notch_depth), (lx / 2 + notch_width / 2, 0), (lx, 0),
               (lx, ly), (lx / 2 + notch_width / 2, ly), (lx / 2 + notch_width / 2, ly - notch_depth),
               (lx / 2 - notch_width / 2, ly - notch_depth), (lx / 2 - notch_width / 2, ly), (0, ly)]
    ax.add_patch(Polygon(outline, closed=True, facecolor=DOMAIN_FILL, edgecolor="none", zorder=2))
    fine = Polygon([(lx / 2 - centre_width / 2, 0), (lx / 2 + centre_width / 2, 0),
                    (lx / 2 + centre_width / 2, ly), (lx / 2 - centre_width / 2, ly)],
                   closed=True, facecolor=FINE_FILL, edgecolor="none", zorder=2.5)
    fine.set_clip_path(Polygon(outline, closed=True, transform=ax.transData))
    ax.add_patch(fine)
    ax.add_patch(Polygon(outline, closed=True, facecolor="none", edgecolor="k",
                         linewidth=OUTLINE_WIDTH, zorder=3))
    # clamped left edge, loaded right edge
    hatched_edge(ax, (0, ly), (0, 0), depth=0.018, count=9, side=-1.0)
    loaded_edge(ax, (lx, 0), (lx, ly), gap=0.005, thickness=0.008, outward=(1, 0))
    for y in np.linspace(0.03, ly - 0.03, 4):
        arrow(ax, (lx + 0.018, y), (lx + 0.055, y))
    ax.text(lx + 0.062, ly / 2, r"$u$", fontsize=FONTSIZE, va="center", ha="left")
    # extraction line and stations
    ax.plot([lx / 2, lx / 2], [notch_depth, ly - notch_depth], color="C3", linestyle=":",
            linewidth=0.8, zorder=4)
    ax.plot([lx / 2, lx / 2], list(stations), color="C3", linestyle="-", linewidth=1.5, zorder=4)
    ax.text(lx / 2 + 0.012, ly / 2, extract_label, fontsize=FONTSIZE, color="C3", va="center")
    # dimensions and annotations
    dimension(ax, (0, -0.04), (lx, -0.04), r"$0.4\;\mathrm{m}$", (0.0, -0.02),
              tick=0.01, tick_direction=(0, -1))
    dimension(ax, (-0.045, 0), (-0.045, ly), r"$0.2\;\mathrm{m}$", (-0.02, 0.0), rotation=90,
              tick=0.01, tick_direction=(-1, 0))
    dimension(ax, (lx / 2 - centre_width / 2, ly + 0.02), (lx / 2 + centre_width / 2, ly + 0.02),
              r"$6\ell,\;h^{\mathrm{el}}=\ell/2$" "\u2013" r"$\ell/8$", (0.0, 0.02), tick=0.01,
              tick_direction=(0, 1))
    ax.annotate(r"notches $0.01\;\mathrm{m}$", xy=(lx / 2 + notch_width / 2, notch_depth / 2),
                xytext=(lx / 2 + 0.09, -0.017), fontsize=FONTSIZE, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"), zorder=6)
    ax.set_xlim(-0.075, lx + 0.085)
    ax.set_ylim(-0.075, ly + 0.065)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "GcCorrection_Geometry", (8.0, 6.0))


def roller_edge(ax, start, end, radius, count, side=-1.0, wall_side=1.0):
    """Rollers against a hatched wall: zero normal displacement, free tangential slip."""

    start, end = np.asarray(start, float), np.asarray(end, float)
    tangent = (end - start) / np.linalg.norm(end - start)
    normal = side * np.array([-tangent[1], tangent[0]])
    ax.plot(*zip(start, end), color="k", linewidth=OUTLINE_WIDTH, zorder=4)
    for s_ in np.linspace(0.12, 0.88, count):
        centre = start + s_ * (end - start) + radius * normal
        ax.add_patch(Circle(centre, radius, facecolor="white", edgecolor="k",
                            linewidth=THIN_WIDTH, zorder=4))
    wall_start, wall_end = start + 2 * radius * normal, end + 2 * radius * normal
    hatched_edge(ax, wall_start, wall_end, depth=1.6 * radius, count=count + 4, side=wall_side)


def sea_ice_floe():
    """Section 4.1 geometry: square floe, central defect, far-field strain rate, and the
    symmetry line of the half model used for the fragmentation case."""

    fig, ax = plt.subplots()
    L = FLOE_SIDE
    cx, cy = FLOE_DEFECT_CENTRE
    ax.add_patch(Polygon([(0, 0), (L, 0), (L, L), (0, L)], closed=True, facecolor=DOMAIN_FILL,
                         edgecolor="k", linewidth=OUTLINE_WIDTH, zorder=2))
    ax.plot([0, L], [cy, cy], color="k", linewidth=THIN_WIDTH, linestyle=(0, (8, 3, 1.5, 3)), zorder=2.5)
    # the defect is far too small to draw to scale: marker plus magnified inset
    ax.add_patch(Circle((cx, cy), 1.5, facecolor="white", edgecolor="k", linewidth=THIN_WIDTH, zorder=3))
    inset_centre, inset_radius = (cx + 24.0, cy - 22.0), 9.0
    ax.add_patch(Circle(inset_centre, inset_radius, facecolor="white", edgecolor="k",
                        linewidth=THIN_WIDTH, zorder=3))
    ax.add_patch(Circle(inset_centre, 0.45 * inset_radius, facecolor=DOMAIN_FILL, edgecolor="k",
                        linewidth=OUTLINE_WIDTH, zorder=3.5))
    ax.add_patch(Circle(inset_centre, 0.45 * inset_radius, facecolor="white", edgecolor="k",
                        linewidth=OUTLINE_WIDTH, zorder=3.6))
    d = 0.45 * inset_radius / np.sqrt(2.0)
    arrow(ax, inset_centre, (inset_centre[0] + d, inset_centre[1] + d), width=0.5)
    ax.text(inset_centre[0], inset_centre[1] - inset_radius - 3.0, r"$r=0.5\;\mathrm{m}$",
            fontsize=FONTSIZE, ha="center", va="top", zorder=6)
    # connector lines: the two outer common tangents of the marker and the inset circle,
    # touching both at the same outward normal angle alpha, cos(alpha - theta) = (r1 - r2)/D
    r_marker = 1.5
    gap = np.subtract(inset_centre, (cx, cy))
    theta, D = np.arctan2(gap[1], gap[0]), np.hypot(*gap)
    for sign in (1.0, -1.0):
        alpha = theta + sign * np.arccos((r_marker - inset_radius) / D)
        n = np.array([np.cos(alpha), np.sin(alpha)])
        start, end = np.array((cx, cy)) + r_marker * n, np.array(inset_centre) + inset_radius * n
        ax.plot([start[0], end[0]], [start[1], end[1]], color="0.4", linewidth=0.4, zorder=3)
    # loading bars and far-field strain rate on the left and right edges
    loaded_edge(ax, (0, 0), (0, L), gap=1.2, thickness=2.0, outward=(-1, 0))
    loaded_edge(ax, (L, 0), (L, L), gap=1.2, thickness=2.0, outward=(1, 0))
    for y in np.linspace(10.0, L - 10.0, 5):
        arrow(ax, (L + 3.2, y), (L + 12.0, y))
        arrow(ax, (-3.2, y), (-12.0, y))
    ax.text(L + 13.5, L / 2, r"$\dot{\varepsilon}_\mathrm{ext}$", fontsize=FONTSIZE, ha="left", va="center")
    ax.text(-13.5, L / 2, r"$\dot{\varepsilon}_\mathrm{ext}$", fontsize=FONTSIZE, ha="right", va="center")
    # dimensions and defect position
    dimension(ax, (0, L + 8.0), (L, L + 8.0), r"$100\;\mathrm{m}$", (0.0, 5.0), tick=3.0,
              tick_direction=(0, 1))
    dimension(ax, (L + 24.0, 0), (L + 24.0, L), r"$100\;\mathrm{m}$", (6.0, 0.0), rotation=-90,
              tick=3.0, tick_direction=(1, 0))
    # the defect is central: no position dimensions needed
    ax.set_xlim(-30.0, L + 34.0)
    ax.set_ylim(-6.0, L + 18.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "SeaIce_Geometry", (8.0, 6.0))


def ice_cliff():
    """Section 4.2 geometry: grounded cliff on a free-slip bed with surface notches, sea water against the face."""

    fig, ax = plt.subplots()
    L, H, hw = CLIFF_LENGTH, CLIFF_HEIGHT, CLIFF_WATER_LEVEL
    water_fill = "#cfe0ee"
    ax.add_patch(Polygon([(L, 0), (L + 75.0, 0), (L + 75.0, hw), (L, hw)], closed=True, facecolor=water_fill,
                         edgecolor="none", zorder=1))
    ax.plot([L, L + 75.0], [hw, hw], color="k", linewidth=THIN_WIDTH, zorder=3)
    ax.add_patch(Polygon([(0, 0), (L, 0), (L, H), (0, H)], closed=True, facecolor=DOMAIN_FILL,
                         edgecolor="none", zorder=2))
    ax.plot([0, L, L, 0], [H, H, 0, 0], color="k", linewidth=OUTLINE_WIDTH, zorder=3)
    for x in CLIFF_NOTCHES:
        ax.plot([x, x], [H, H - CLIFF_NOTCH_DEPTH], color="k", linewidth=0.8, solid_capstyle="butt", zorder=3)

    # far-field edge and bed: rollers (zero normal displacement, free slip)
    roller_edge(ax, (0, 0), (0, H), radius=5.0, count=5, side=1.0)
    roller_edge(ax, (0, 0), (L, 0), radius=5.0, count=12, side=-1.0, wall_side=-1.0)
    # gravity
    arrow(ax, (0.30 * L, 0.72 * H), (0.30 * L, 0.36 * H), width=0.8)
    ax.text(0.30 * L + 10.0, 0.54 * H, r"$\rho_\mathrm{i}\,\mathbf{g}$", fontsize=FONTSIZE, ha="left", va="center")
    # water pressure on the face, growing with depth below the water level
    for y in (0.15 * hw, 0.5 * hw, 0.85 * hw):
        arrow(ax, (L + 8.0 + 52.0 * (1.0 - y / hw), y), (L + 2.0, y))
    dimension(ax, (L + 92.0, 0), (L + 92.0, hw), r"$h_\mathrm{w}(t)$", (16.0, 0.0), rotation=90, tick=6.0,
              tick_direction=(1, 0))
    # dimensions
    dimension(ax, (-52.0, 0), (-52.0, H), r"$200\;\mathrm{m}$", (-15.0, 0.0), rotation=90,
              tick=6.0, tick_direction=(-1, 0))
    dimension(ax, (0, H + 62.0), (L, H + 62.0), r"$500\;\mathrm{m}$", (0.0, 15.0), tick=6.0,
              tick_direction=(0, 1))
    dimension(ax, (CLIFF_NOTCHES[2], H + 22.0), (CLIFF_NOTCHES[3], H + 22.0), r"$50\;\mathrm{m}$", (0.0, 15.0),
              tick=6.0, tick_direction=(0, 1))
    ax.annotate(r"notches, $10\;\mathrm{m}$ deep", xy=(CLIFF_NOTCHES[6], H - 0.6 * CLIFF_NOTCH_DEPTH),
                xytext=(CLIFF_NOTCHES[6] + 20.0, H - 52.0), fontsize=FONTSIZE, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", linewidth=0.4, shrinkA=1, shrinkB=0), zorder=6)
    ax.set_xlim(-88.0, L + 124.0)
    ax.set_ylim(-34.0, H + 92.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "Cliff_Geometry", (8.0, 4.0))


def main():
    import Calibrate_Gc_Correction as C
    from Plot_GcCalibration_Figure import STATIONS

    plate_with_hole()
    sea_ice_floe()
    ice_cliff()
    notched_bar(C.LX, C.LY, C.NOTCH_WIDTH, C.NOTCH_DEPTH, C.LX_CENTRE, STATIONS)


if __name__ == "__main__":
    main()
