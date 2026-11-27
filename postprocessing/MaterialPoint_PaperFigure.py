"""Generate the Section 2 monotonic material-response figure and source data.

    python MaterialPoint_PaperFigure.py

Uniaxial stress (sigma_xx=sigma_xy=0) permits free lateral contraction.
The intrinsic-2D constitutive equations reduce exactly to
    eps_yy = sigma_yy/E + eps_v_yy + eta_yy,
    delta eps_v_yy = dt*A_Glen*sigma_yy^3/8,
    sigma_yy <= d(phi)*f_t, eta_yy >= 0,
with complementarity between the last two conditions. The homogeneous AT2
history equation is solved with the existing bracketed scalar phase update.
These are constitutive-law illustrations, evaluated directly without the
production kernel's activation tolerances or artificial local stiffness.
There is no spatial crack gradient or finite-element calculation.

The output contains PDF/PNG/EPS, CSV histories, and a half-increment check.
Each panel uses the mesh-study FormatAndSave helper: 8 cm width, 7 pt text,
and 0.5 pt lines. The plotting areas are approximately 3 cm high;
axial-strain labels are on the lower row only.
LaTeX supplies subfigure labels. Gc=1.0 J/m2 and loading
to 0.001 axial strain illustrate near-complete loss of cohesion. The paper
figure displays axial strains up to 0.2e-3 on linear axes. Fine uniform increments
resolve initiation; geometrically growing increments cover the long tail.
Only monotonic loading is included.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from MaterialPoint_Verification import (
    DEV, VOL, SQRT2, LocalResponse, Material, advance_point, degradation,
    initial_state, _cubic_stress,
)


RATES = (1e-9, 1.5e-9, 2.5e-9, 5e-9, 7.5e-9, 1e-8, 1.5e-8, 2.5e-8, 5e-8)
COLUMNS = ("time_s", "strain_yy", "strain_xx", "stress_yy_Pa", "creep_yy",
           "fracture_yy", "phi", "degradation", "history_J_m3")


class UniaxialStressPoint:
    """Exact tensile reduction of the unregularized intrinsic-2D material law."""

    def __init__(self, material=None):
        self.material = material or Material(gc=1.0, cohesive_regularization_factor=0.0)

    def solve(self, strain_yy, dt, old, phi):
        m = self.material
        remaining = strain_yy - old.local.creep[1]
        if remaining < -1e-15:
            raise ValueError("This monotonic tensile reduction requires tensile remaining strain")
        coefficient = dt * m.glen_a / 8.0
        trial = _cubic_stress(max(0.0, remaining), 1.0 / m.youngs, coefficient)
        stress_yy = min(trial, degradation(phi, m) * m.tensile)
        creep_yy = old.local.creep[1] + coefficient * stress_yy**3
        fracture_yy = (max(0.0, strain_yy - creep_yy - stress_yy / m.youngs)
                       if trial > degradation(phi, m) * m.tensile else 0.0)
        a, b = m.strength_weights
        lambdas = SQRT2 * fracture_yy * np.array([b, a]) / (a + b)
        stress = np.zeros(6)
        stress[1] = stress_yy
        creep = np.zeros(6)
        creep[0], creep[1] = -creep_yy, creep_yy
        return LocalResponse(stress, creep, lambdas[0] * VOL + lambdas[1] * DEV,
                             lambdas, m.tensile * fracture_yy,
                             stress_yy**2 / (2.0 * m.youngs),
                             SQRT2 * abs(creep_yy - old.local.creep[1]))


def tensile_history(rate, increment=3e-9, maximum_strain=1e-4, material=None,
                    relative_increment=None):
    """Return a monotonic history with optional growing steps beyond strain 1e-4."""
    if rate <= 0 or increment <= 0 or maximum_strain <= 0:
        raise ValueError("Positive loading rate, increment and maximum strain required")
    point = UniaxialStressPoint(material)
    m = point.material
    state = initial_state()
    rows = [[0.0] * len(COLUMNS)]
    rows[0][COLUMNS.index("degradation")] = 1.0
    previous = 0.0
    if relative_increment is not None and not 0 < relative_increment <= 0.1:
        raise ValueError("relative_increment must be in (0, 0.1]")
    while previous < maximum_strain:
        step = increment
        if relative_increment is not None and previous >= 1e-4:
            step = max(increment, relative_increment * previous)
        strain = min(previous + step, maximum_strain)
        if relative_increment is not None and previous < 1e-4:
            strain = min(strain, 1e-4)
        state = advance_point(point, strain, (strain - previous) / rate, state, evolve=True)
        local = state.local
        lateral = -m.poisson * local.stress[1] / m.youngs + local.creep[0] + local.eta[0]
        rows.append([strain / rate, strain, lateral, local.stress[1], local.creep[1],
                     local.eta[1], state.phi, degradation(state.phi, m), state.history])
        previous = strain
    return np.asarray(rows)


def plot_figure(histories, output, maximum_strain=2e-4, material=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from plotting.PlotLines import FormatAndSave, FONTSIZE, LINEWIDTH

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    material = material or UniaxialStressPoint().material
    rates = sorted(histories)
    norm = LogNorm(vmin=rates[0], vmax=rates[-1])
    colors = {rate: plt.get_cmap("viridis")(0.05 + 0.85 * norm(rate)) for rate in rates}
    creep_limits = {rate: (8.0 * rate / material.glen_a)**(1.0 / 3.0) for rate in rates}
    fields = (("stress_yy_Pa", 1e-6, r"$\sigma_{yy}\;[\mathrm{MPa}]$", "Stress"),
              ("phi", 1.0, r"$\phi\;[-]$", "PhaseField"),
              ("creep_yy", 1e3, r"$\varepsilon_{\mathrm{v},yy}\;[10^{-3}]$", "CreepStrain"),
              ("fracture_yy", 1e3, r"$\eta_{yy}\;[10^{-3}]$", "FractureStrain"))
    maximum = min(maximum_strain, max(history[-1, 1] for history in histories.values()))
    for field, scale, label, filename in fields:
        fig, ax = plt.subplots()
        if field == "stress_yy_Pa":
            for rate in rates:
                ax.axhline(creep_limits[rate] * scale, color=colors[rate],
                           linestyle=":", lw=LINEWIDTH, zorder=1)
            ax.axhline(material.tensile * scale, color="black", linestyle=":",
                       lw=LINEWIDTH, zorder=3)
        for rate in rates:
            history = histories[rate]
            values = history[:, COLUMNS.index(field)] * scale
            ax.plot(history[:, 1] * 1e3, values, color=colors[rate], lw=LINEWIDTH)
        lower_row = field in {"creep_yy", "fracture_yy"}
        ax.set(xlim=(0, maximum * 1e3),
               xlabel=r"$\varepsilon_{yy}\;[10^{-3}]$" if lower_row else "",
               ylabel=label)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if field in {"creep_yy", "fracture_yy"}:
            ax.set_ylim(0, maximum * 1e3)
        else:
            stress_maximum = math.ceil(max(material.tensile, *creep_limits.values()) / 1e5) * 0.1
            ax.set_ylim(0, 1.0 if field == "phi" else stress_maximum)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        stem = str(output / f"MaterialPoint_{filename}")
        FormatAndSave(fig, ax, stem, fontsize=FONTSIZE,
                      figure_size_cm=(8, 4.2 if lower_row else 3.65),
                      top=0.97, bottom=0.27 if lower_row else 0.16,
                      left=0.20, right=0.97)
        fig.savefig(stem + ".pdf", facecolor="white", edgecolor="white", transparent=False)
        plt.close(fig)

    handles = []
    for rate in rates:
        exponent = int(math.floor(math.log10(rate)))
        coefficient = rate / 10.0**exponent
        handles.append(Line2D([], [], color=colors[rate], lw=LINEWIDTH,
                              label=rf"$\dot{{\varepsilon}}_0={coefficient:g}\cdot10^{{{exponent}}}\;\mathrm{{s}}^{{-1}}$"))
    handles.extend([
        Line2D([], [], color="0.3", lw=LINEWIDTH, label="Response"),
        Line2D([], [], color="0.3", lw=LINEWIDTH, linestyle=":", label="Steady creep limit"),
        Line2D([], [], color="black", lw=LINEWIDTH, linestyle=":", label=r"Tensile strength $f_\mathrm{t}$"),
    ])
    # Matplotlib fills legend columns first; reorder so rates increase along rows.
    columns = 3
    rows = math.ceil(len(handles) / columns)
    order = [r * columns + c for c in range(columns) for r in range(rows)
             if r * columns + c < len(handles)]
    fig = plt.figure(figsize=(16 / 2.54, 1.85 / 2.54))
    fig.legend(handles=[handles[i] for i in order], loc="center", ncol=columns,
               frameon=False, fontsize=FONTSIZE, columnspacing=2.5,
               handletextpad=0.5, handlelength=2.7, labelspacing=0.3)
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(output / f"MaterialPoint_Legend.{suffix}", dpi=1200,
                    facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "MaterialPoint_PaperResults_Gc1.0_Rates")
    parser.add_argument("--maximum-strain", type=float, default=0.001)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a fresh --output-dir; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    m = Material(gc=1.0, cohesive_regularization_factor=0.0)
    histories, summaries = {}, {}
    for rate in RATES:
        history = tensile_history(rate, 3e-9, args.maximum_strain, material=m, relative_increment=0.002)
        finer = tensile_history(rate, 1.5e-9, args.maximum_strain, material=m, relative_increment=0.001)
        np.savetxt(output / f"rate_{rate:g}.csv", history, delimiter=",",
                   header=",".join(COLUMNS), comments="", fmt="%.12g")
        comparison = np.interp(history[:, 1], finer[:, 1], finer[:, 3])
        summaries[f"{rate:g}"] = {"peak_stress_Pa": float(history[:, 3].max()),
                                  "final_phi": float(history[-1, 6]),
                                  "final_degradation": float(history[-1, 7]),
                                  "final_creep_yy": float(history[-1, 4]),
                                  "final_fracture_yy": float(history[-1, 5]),
                                  "end_time_s": float(history[-1, 0]),
                                  "half_increment_max_stress_difference_Pa": float(np.max(abs(history[:, 3] - comparison)))}
        histories[rate] = history
        print(rate, summaries[f"{rate:g}"], flush=True)
    plot_figure(histories, output, material=m)
    parameters = asdict(m)
    parameters["strength_norm_smoothing"] = 0.0
    (output / "settings.json").write_text(json.dumps({
        "material": parameters, "loading": "uniaxial stress; free lateral strain",
        "rates_s_inv": RATES, "strain_increment": 3e-9, "check_increment": 1.5e-9,
        "maximum_strain": args.maximum_strain, "relative_tail_increment": 0.002,
        "plot_maximum_strain": 2e-4,
        "colormap": "viridis", "color_normalization": "logarithmic strain rate",
        "colormap_range": [0.05, 0.9],
        "stress_reference_lines": {
            "tensile_strength_Pa": m.tensile,
            "steady_creep_stress_Pa": {f"{rate:g}": (8 * rate / m.glen_a)**(1 / 3) for rate in RATES},
        },
        "creep_fracture_transition_rate_s_inv": m.glen_a * m.tensile**3 / 8,
        "solver": "exact tensile constitutive reduction and homogeneous AT2 scalar balance",
        "summaries": summaries,
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
