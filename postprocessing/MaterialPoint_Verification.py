"""Verify the paper's intrinsic-2D material point, including homogeneous AT2.

Run with the FEMCubes Python environment, on ONE process/thread:
    python MaterialPoint_Verification.py
    python MaterialPoint_Verification.py --check-reference
    python MaterialPoint_Verification.py --plot-only
    python MaterialPoint_Verification.py --reference-only
    python MaterialPoint_Verification.py --protocol fixed --reference-step 1e-8

The fixed protocol prescribes phi=0.5, eps_xx=eps_xy=0,
eps_yy=2.5e-9*t to 1e-4. The evolving protocol starts intact and solves the
spatially homogeneous AT2/history equation together with the local update.
The cycle protocol adds a hold, unload and reload to test irreversibility.

Each production history is also checked against an independent scalar solver
at the SAME increments. Temporal errors are measured against a separate fine
scalar history (default strain increment <=3e-9). The scalar solver uses
the two invariant equilibrium equations and a bracketed potential-norm solve;
it calls neither the C++ residual/Jacobian nor the production update.

In the homogeneous problem there is no grad(phi) term:
    H=max(H_old,F_d(eta)), phi=2*(1-kappa)*H/(Gc/ell+2*(1-kappa)*H).
This is a material-point verification, not a localized crack or an ell-
independence test: its damage response naturally depends on Gc/ell.

The default local coefficient is 1e-8*kappa*K, as in the current C++ source,
NOT the kappa*K written in the draft. The 1e-6 Pa strength-norm smoothing is
included in the scalar reference. Tiny sub-strength eigenstrains from this
smoothing can differ from the production activation tolerance.

Results: CSV histories, JSON convergence metrics/settings, and PNG/PDF plots.
Existing output is preserved; choose a fresh --output-dir for a repeat run.
No finite-element jobs, meshes, or shared-library rebuilds are launched.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq


SQRT2 = math.sqrt(2.0)
VOL = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) / SQRT2
DEV = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0]) / SQRT2


@dataclass(frozen=True)
class Material:
    """Paper material parameters; the scalar reduction assumes Glen exponent 3."""

    youngs: float = 9.5e9
    poisson: float = 0.35
    glen_a: float = 1.0e-24
    glen_n: float = 3.0
    tensile: float = 0.4e6
    compressive: float = 0.6e6
    kappa: float = 1.0e-3
    gc: float = 5.0
    ell: float = 0.02
    cohesive_regularization_factor: float = 1.0e-8
    strength_norm_smoothing: float = 1.0e-6

    def __post_init__(self):
        if self.glen_n != 3.0:
            raise ValueError("This independent cubic reduction requires glen_n=3")
        if not (self.youngs > 0 and -1 < self.poisson < 1 and self.glen_a >= 0
                and 0 < self.tensile < self.compressive and 0 < self.kappa < 1
                and self.gc > 0 and self.ell > 0
                and self.cohesive_regularization_factor >= 0
                and self.strength_norm_smoothing > 0):
            raise ValueError("Invalid material parameters")

    @property
    def shear(self):
        return self.youngs / (2.0 * (1.0 + self.poisson))

    @property
    def lame(self):
        return self.youngs * self.poisson / (1.0 - self.poisson**2)

    @property
    def bulk(self):
        return self.lame + self.shear

    @property
    def creep_a(self):
        return 2.0 * self.glen_a / 2.0 ** ((self.glen_n + 1.0) / 2.0)

    @property
    def strength_weights(self):
        # F_d=sqrt(a*lambda_vol^2+b*lambda_dev^2+eps^2)-eps.
        ft_star = self.tensile / (2.0 * math.sqrt(1.0 - (self.tensile / self.compressive)**2))
        return 2.0 * ft_star**2, self.compressive**2 / 2.0

    @property
    def local_regularization(self):
        return self.cohesive_regularization_factor * self.kappa * self.bulk


@dataclass
class LocalResponse:
    """Current constitutive state, using the production six-component storage."""

    stress: np.ndarray
    creep: np.ndarray
    eta: np.ndarray
    multipliers: np.ndarray
    drive: float
    elastic_energy: float
    creep_increment: float


@dataclass
class PointState:
    local: LocalResponse
    phi: float
    history: float


def degradation(phi, material):
    """Return the strength degradation, including the residual strength."""
    return (1.0 - material.kappa) * (1.0 - phi)**2 + material.kappa


def initial_state(phi=0.0):
    """Start with zero stresses/internal strains and the prescribed phase field."""
    return PointState(LocalResponse(np.zeros(6), np.zeros(6), np.zeros(6),
                                    np.zeros(2), 0.0, 0.0, 0.0), phi, 0.0)


def _cubic_stress(remaining_strain, compliance, creep_coefficient):
    if creep_coefficient == 0.0:
        return remaining_strain / compliance
    scale = math.sqrt(compliance / (3.0 * creep_coefficient))
    argument = 1.5 * remaining_strain / (compliance * scale)
    return 2.0 * scale * math.sinh(math.asinh(argument) / 3.0)


class ScalarReference:
    """Independent update for nonnegative eps_yy, eps_xx=eps_xy=0 in true 2-D.

    Let v=eps_v:DEV, e=eps_yy/sqrt(2), and Q=F_d+norm_smoothing.
    Setting z=Q/d(phi) eliminates both fracture multipliers:
      lambda_vol/z = 2K*e / ((2K+r)*z+a),
      lambda_dev/z = |q| / (r*z+b),
      |q|*(1/(2mu)+z/(r*z+b)) + A*dt*|q|^3 = |e-v_old|.
    The remaining scalar equation is
      a*(lambda_vol/z)^2+b*(lambda_dev/z)^2+(eps/z)^2=d(phi)^2.
    It has a bracketed root and needs no Newton tangent or active-set code.
    """

    def __init__(self, material=Material()):
        self.material = material

    def solve(self, strain_yy, dt, old, phi):
        if strain_yy < 0 or dt <= 0:
            raise ValueError("Require nonnegative total strain and positive dt")
        m = self.material
        a, b = m.strength_weights
        r = m.local_regularization
        d = degradation(phi, m)
        e = strain_yy / SQRT2
        previous_v = float(old.local.creep @ DEV)
        remaining = e - previous_v
        direction = math.copysign(1.0, remaining)
        magnitude = abs(remaining)
        c = m.creep_a * dt

        def components(z):
            vol_over_z = 2.0 * m.bulk * e / ((2.0 * m.bulk + r) * z + a)
            compliance = 1.0 / (2.0 * m.shear) + z / (r * z + b)
            q = _cubic_stress(magnitude, compliance, c)
            return vol_over_z, q / (r * z + b), q

        def residual(z):
            l1_over_z, l2_over_z, _ = components(z)
            return math.hypot(math.sqrt(a) * l1_over_z, math.sqrt(b) * l2_over_z,
                              m.strength_norm_smoothing / z) / d - 1.0

        lower = m.strength_norm_smoothing / d
        upper = max(2.0 * lower, math.sqrt(a * e**2 + b * magnitude**2
                                         + m.strength_norm_smoothing**2) / d)
        z = lower if residual(lower) <= 1e-14 else brentq(
            residual, lower, upper, xtol=1e-14, rtol=1e-12)
        l1_over_z, l2_over_z, q = components(z)
        l1, l2 = l1_over_z * z, l2_over_z * z
        p = 2.0 * m.bulk * (e - l1)
        dv = direction * c * q**3
        eta = l1 * VOL + direction * l2 * DEV
        drive = max(0.0, math.hypot(math.sqrt(a) * l1, math.sqrt(b) * l2,
                                   m.strength_norm_smoothing) - m.strength_norm_smoothing)
        return LocalResponse(p * VOL + direction * q * DEV,
                             (previous_v + dv) * DEV, eta, np.array([l1, l2]), drive,
                             p**2 / (4.0 * m.bulk) + q**2 / (4.0 * m.shear), abs(dv))


class ProductionKernel:
    """Call the C++ kernel of FEMCubes (Solid:Cohesive_ViscoElastic_Kernel) without instantiating finite elements."""

    def __init__(self, material=Material()):
        from femcubes.solids.cpp_kernels.cohesive_viscoelastic import _cohesive_viscoelastic_kernels_cpp as cpp

        if material.cohesive_regularization_factor != 1e-8 or material.strength_norm_smoothing != 1e-6:
            raise ValueError("Production comparison requires its current hard-coded regularizations")
        self.material = material
        self.library = str(Path(cpp.__file__).resolve())
        viscosity = (1.0 / material.creep_a)**(1.0 / material.glen_n) if material.creep_a else 0.0
        self.kernel = cpp.CohesiveViscoElasticKernels(
            material.bulk, material.lame, material.shear, viscosity,
            material.glen_n if material.creep_a else 1.0,
            material.tensile, material.compressive, 1.0, 0.0, material.compressive,
            material.kappa, "r2_2L", False, True)

    def solve(self, strain_yy, dt, old, phi):
        strain = np.zeros(6)
        strain[1] = strain_yy
        stress = np.zeros(6)
        creep = np.zeros(6)
        eta = old.local.eta.copy()
        multipliers = old.local.multipliers.copy()
        elastic, drive, dv = np.zeros(1), np.zeros(1), np.zeros(1)
        self.kernel(strain, dt, np.array([degradation(phi, self.material)]), stress,
                    eta, old.local.eta.copy(), creep, old.local.creep.copy(),
                    np.zeros(36), elastic, drive, dv, multipliers,
                    old.local.multipliers.copy())
        if not all(np.isfinite(x).all() for x in (stress, creep, eta, multipliers, drive)):
            raise RuntimeError(f"Production update failed at eps_yy={strain_yy:g}, phi={phi:g}, dt={dt:g}")
        return LocalResponse(stress, creep, eta, multipliers, float(drive[0]),
                             float(elastic[0]), float(dv[0]))


def advance_point(backend, strain_yy, dt, old, evolve=False):
    """Advance one increment; every phase-field trial uses the SAME old creep."""
    if not evolve:
        local = backend.solve(strain_yy, dt, old, old.phi)
        return PointState(local, old.phi, max(old.history, local.drive))
    m = backend.material

    def residual(phi):
        local = backend.solve(strain_yy, dt, old, phi)
        history = max(old.history, local.drive)
        return m.gc / m.ell * phi - 2.0 * (1.0 - m.kappa) * (1.0 - phi) * history

    lower_residual = residual(old.phi)
    phi = old.phi if lower_residual >= -1e-12 else brentq(
        residual, old.phi, 1.0, xtol=1e-12, rtol=1e-12)
    local = backend.solve(strain_yy, dt, old, phi)
    return PointState(local, phi, max(old.history, local.drive))


COLUMNS = (
    "time_s", "eps_yy", "sigma_xx_Pa", "sigma_yy_Pa", "creep_xx", "creep_yy",
    "lambda_vol", "lambda_dev", "eta_xx", "eta_yy", "phi", "degradation",
    "history_J_m3", "drive_J_m3", "elastic_energy_J_m3", "creep_power_W_m3",
    "phase_residual_Pa", "strain_balance_error",
)


@dataclass
class History:
    protocol: str
    backend: str
    increment: float
    data: np.ndarray

    def column(self, name):
        return self.data[:, COLUMNS.index(name)]


def loading_segments(protocol, rate=2.5e-9):
    """Return (end time, end eps_yy, label), with all strains nonnegative."""
    if protocol not in {"fixed", "evolving", "cycle"}:
        raise ValueError(f"Unknown protocol {protocol!r}")
    segments = [(1e-4 / rate, 1e-4, "load")]
    if protocol == "cycle":
        end = segments[-1][0] + 10000.0
        segments += [(end, 1e-4, "hold"),
                     (end + 8e-5 / rate, 2e-5, "unload"),
                     (end + 18e-5 / rate, 1.2e-4, "reload")]
    return segments


def run_history(backend, protocol, increment, rate=2.5e-9):
    """Integrate a prescribed history with uniform maximum increments per segment."""
    if not (math.isfinite(increment) and increment > 0 and rate > 0):
        raise ValueError("Positive finite increment and rate required")
    m = backend.material
    state = initial_state(0.5 if protocol == "fixed" else 0.0)
    rows = []

    def record(t, strain):
        local = state.local
        p, q = float(local.stress @ VOL), float(local.stress @ DEV)
        elastic_yy = (p / (2.0 * m.bulk) + q / (2.0 * m.shear)) / SQRT2
        phase_residual = (m.gc / m.ell * state.phi
                          - 2.0 * (1.0 - m.kappa) * (1.0 - state.phi) * state.history)
        rows.append([
            t, strain, local.stress[0], local.stress[1], local.creep[0], local.creep[1],
            *local.multipliers, local.eta[0], local.eta[1], state.phi,
            degradation(state.phi, m), state.history, local.drive, local.elastic_energy,
            m.creep_a * abs(q)**4, phase_residual,
            strain - elastic_yy - local.creep[1] - local.eta[1],
        ])

    record(0.0, 0.0)
    previous_time, previous_strain = 0.0, 0.0
    dt_max = increment / rate
    for end_time, end_strain, _ in loading_segments(protocol, rate):
        start_time, start_strain = previous_time, previous_strain
        duration = end_time - start_time
        count = int(math.ceil(duration / dt_max - 1e-10))
        for i in range(1, count + 1):
            t = start_time + min(i * dt_max, duration)
            fraction = (t - start_time) / duration
            strain = start_strain + fraction * (end_strain - start_strain)
            state = advance_point(backend, strain, t - previous_time, state,
                                  evolve=protocol != "fixed")
            record(t, strain)
            previous_time, previous_strain = t, strain
    return History(protocol, type(backend).__name__, increment, np.asarray(rows))


ERROR_FIELDS = ("sigma_yy_Pa", "creep_yy", "lambda_vol", "lambda_dev", "phi")


def compare_histories(history, reference):
    """Compare at the tested history's times, interpolating only the reference."""
    metrics = {}
    for field in ERROR_FIELDS:
        expected = np.interp(history.column("time_s"), reference.column("time_s"),
                             reference.column(field))
        error = float(np.max(np.abs(history.column(field) - expected)))
        scale = float(np.max(np.abs(reference.column(field))))
        metrics[field] = {"max_absolute_error": error,
                          "relative_to_reference_peak": error / scale if scale else None}
    return metrics


def _save_history(path, history):
    np.savetxt(path, history.data, delimiter=",", header=",".join(COLUMNS), comments="", fmt="%.12g")


def _history_checks(history):
    phi = history.column("phi")
    return {
        "max_phi": float(np.max(phi)),
        "min_degradation": float(np.min(history.column("degradation"))),
        "max_abs_strain_balance_error": float(np.max(np.abs(history.column("strain_balance_error")))),
        "minimum_phase_increment": float(np.min(np.diff(phi))),
        "max_abs_phase_residual_Pa": (None if history.protocol == "fixed" else
                                      float(np.max(np.abs(history.column("phase_residual_Pa"))))),
        "max_creep_yy": float(np.max(history.column("creep_yy"))),
        "max_lambda_vol": float(np.max(history.column("lambda_vol"))),
        "max_lambda_dev": float(np.max(history.column("lambda_dev"))),
        "minimum_creep_power_W_m3": float(np.min(history.column("creep_power_W_m3"))),
    }


def _constitutive_checks(history, material):
    a, b = material.strength_weights
    p = (history.column("sigma_xx_Pa") + history.column("sigma_yy_Pa")) / SQRT2
    q = (history.column("sigma_yy_Pa") - history.column("sigma_xx_Pa")) / SQRT2
    utilization = np.hypot(p / math.sqrt(a), q / math.sqrt(b)) / history.column("degradation")
    expected_dv = (np.diff(history.column("time_s")) * material.creep_a
                   * np.abs(q[1:])**2 * q[1:])
    actual_dv = SQRT2 * np.diff(history.column("creep_yy"))
    omitted = (np.abs(expected_dv) > 1e-12) & (np.abs(actual_dv) < 1e-16)
    return {
        "max_strength_utilization": float(np.max(utilization)),
        "max_backward_euler_creep_defect": float(np.max(np.abs(actual_dv - expected_dv))),
        "increments_with_omitted_creep_above_1e_minus12": int(np.count_nonzero(omitted)),
    }


def _plot_results(output, material, results, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")

    def save(figure, name):
        figure.savefig(output / f"{name}.png", bbox_inches="tight")
        figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    def curves(axis, reference, histories, x, y, scale=1.0):
        axis.plot(reference.column(x), reference.column(y) * scale,
                  color="black", lw=1.4, label="Independent fine reference")
        for index, history in enumerate(histories):
            axis.plot(history.column(x), history.column(y) * scale, color=colors[index % len(colors)],
                      lw=0.85, linestyle=("--", "-.", ":")[index % 3],
                      label=rf"$\Delta\varepsilon_{{\max}}={history.increment:g}$")

    for protocol, (reference, histories) in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.5), constrained_layout=True)
        if protocol == "fixed":
            fields = [("sigma_yy_Pa", r"$\sigma_{yy}$ [MPa]", 1e-6),
                      ("creep_yy", r"$\varepsilon^v_{yy}$", 1),
                      ("lambda_vol", r"$\lambda_1$ (volumetric)", 1),
                      ("lambda_dev", r"$\lambda_2$ (deviatoric)", 1)]
            for ax, (field, label, scale) in zip(axes.flat, fields):
                curves(ax, reference, histories, "eps_yy", field, scale)
                ax.set(xlabel=r"$\varepsilon_{yy}$", ylabel=label)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            fig.suptitle(r"Fixed $\phi=0.5$: coupled creep and fracture update")
        elif protocol == "evolving":
            fields = [("sigma_yy_Pa", r"$\sigma_{yy}$ [MPa]", 1e-6),
                      ("phi", r"$\phi$", 1),
                      ("degradation", r"$d(\phi)$", 1),
                      ("creep_yy", r"$\varepsilon^v_{yy}$", 1)]
            for ax, (field, label, scale) in zip(axes.flat, fields):
                curves(ax, reference, histories, "eps_yy", field, scale)
                ax.set(xlabel=r"$\varepsilon_{yy}$", ylabel=label)
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            fig.suptitle("Homogeneous AT2: evolving loss of cohesion "
                         + rf"($G_c/\ell={material.gc / material.ell:g}$ Pa)")
        else:
            hours = reference.column("time_s") / 3600.0
            axes[0, 0].plot(hours, reference.column("eps_yy"), color="black")
            axes[0, 0].set(xlabel="Time [h]", ylabel=r"Prescribed $\varepsilon_{yy}$")
            curves(axes[0, 1], reference, histories, "eps_yy", "sigma_yy_Pa", 1e-6)
            axes[0, 1].set(xlabel=r"$\varepsilon_{yy}$", ylabel=r"$\sigma_{yy}$ [MPa]")
            axes[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axes[1, 0].plot(hours, reference.column("phi"), color="black")
            for index, history in enumerate(histories):
                axes[1, 0].plot(history.column("time_s") / 3600, history.column("phi"),
                                color=colors[index % len(colors)], ls="--", lw=0.85)
            axes[1, 0].set(xlabel="Time [h]", ylabel=r"Irreversible $\phi$")
            for field, label, color in [("creep_yy", r"Creep $\varepsilon^v_{yy}$", colors[0]),
                                         ("eta_yy", r"Fracture $\eta_{yy}$", colors[1])]:
                axes[1, 1].plot(hours, reference.column(field), label=label, color=color)
            axes[1, 1].set(xlabel="Time [h]", ylabel="Internal strain (fine reference)")
            axes[1, 1].legend(fontsize=8)
            fig.suptitle("Homogeneous AT2: load, hold, unload and reload")
        axes[0, 1].legend(fontsize=7)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        save(fig, f"MaterialPoint_{protocol}")

    fig, axes = plt.subplots(1, len(results), figsize=(4.0 * len(results), 3.4),
                             squeeze=False, constrained_layout=True)
    for ax, (protocol, entries) in zip(axes.flat, metrics.items()):
        for index, field in enumerate(ERROR_FIELDS):
            if field == "phi" and protocol == "fixed":
                continue
            values = [entry["temporal_error"][field]["relative_to_reference_peak"] for entry in entries]
            if any(v is None or v <= 0 for v in values):
                continue
            ax.loglog([entry["strain_increment"] for entry in entries], values, "o-",
                      ms=4, color=colors[index % len(colors)], label=field.replace("_Pa", ""))
        ax.set(xlabel=r"Maximum $\Delta\varepsilon$", ylabel="Maximum error / reference peak",
               title=protocol)
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7)
    save(fig, "MaterialPoint_convergence")

    if "fixed" in results and "evolving" in results:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), constrained_layout=True)
        reference, histories = results["fixed"]
        curves(axes[0, 0], reference, histories, "eps_yy", "sigma_yy_Pa", 1e-6)
        axes[0, 0].set(xlim=(7e-6, 1.1e-5), ylim=(0.075, 0.105),
                       xlabel=r"$\varepsilon_{yy}$", ylabel=r"$\sigma_{yy}$ [MPa]",
                       title="Fixed phase field: initiation")
        a, b = material.strength_weights
        for index, history in enumerate(histories):
            p = (history.column("sigma_xx_Pa") + history.column("sigma_yy_Pa")) / SQRT2
            q = (history.column("sigma_yy_Pa") - history.column("sigma_xx_Pa")) / SQRT2
            utilization = np.hypot(p / math.sqrt(a), q / math.sqrt(b)) / history.column("degradation")
            axes[0, 1].plot(history.column("eps_yy"), utilization, color=colors[index],
                            lw=0.9, label=rf"$\Delta\varepsilon={history.increment:g}$")
        axes[0, 1].axhline(1.0, color="black", ls="--", lw=1)
        axes[0, 1].set(xlim=(7e-6, 1.1e-5), ylim=(0.8, 1.12),
                       xlabel=r"$\varepsilon_{yy}$", ylabel="Strength-surface utilization",
                       title="Admissible stress requires utilization <= 1")
        reference, histories = results["evolving"]
        curves(axes[1, 0], reference, histories, "eps_yy", "creep_yy")
        axes[1, 0].set(xlim=(0, 2e-5), ylim=(0, 7e-7), xlabel=r"$\varepsilon_{yy}$",
                       ylabel=r"$\varepsilon^v_{yy}$", title="Early creep: delayed accumulation")
        for index, protocol in enumerate(("fixed", "evolving")):
            entries = metrics[protocol]
            axes[1, 1].loglog([e["strain_increment"] for e in entries],
                              [e["matched_increment_error"]["sigma_yy_Pa"]["max_absolute_error"]
                               for e in entries], "o-", color=colors[index], label=protocol)
        axes[1, 1].set(xlabel=r"Maximum $\Delta\varepsilon$",
                       ylabel="Maximum stress difference [Pa]",
                       title="Production vs independent solve at SAME increments")
        axes[0, 1].legend(fontsize=7)
        axes[1, 1].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        save(fig, "MaterialPoint_discrepancies")


def plot_saved_results(output):
    """Regenerate figures from saved CSVs without importing the production kernel."""
    settings = json.loads((output / "settings.json").read_text())
    report = json.loads((output / "verification.json").read_text())
    results = {}
    for protocol in settings["protocols"]:
        reference = History(protocol, "ScalarReference", settings["reference_step"],
                            np.loadtxt(output / f"{protocol}_reference.csv", delimiter=",", skiprows=1))
        histories = [History(protocol, "saved", step,
                              np.loadtxt(output / f"{protocol}_production_{step:g}.csv",
                                         delimiter=",", skiprows=1))
                     for step in settings["increments"]]
        results[protocol] = (reference, histories)
    _plot_results(output, Material(**settings["material"]), results, report["metrics"])
    (output / "plot_settings.json").write_text(json.dumps({
        "postprocessing_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }, indent=2) + "\n")


def main(argv=None):
    """Run and save the requested material-point verification protocols."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", nargs="+", choices=("fixed", "evolving", "cycle"),
                        default=["fixed", "evolving", "cycle"])
    parser.add_argument("--increments", nargs="+", type=float, default=[3e-7, 1.5e-7, 7.5e-8])
    parser.add_argument("--reference-step", type=float, default=3e-9)
    parser.add_argument("--reference-only", action="store_true",
                        help="use scalar updates throughout; requires no FEMCubes import")
    parser.add_argument("--check-reference", action="store_true",
                        help="also halve the fine reference step to assess reference resolution")
    parser.add_argument("--plot-only", action="store_true",
                        help="regenerate figures from an existing output directory without FEMCubes")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "MaterialPoint_Results")
    args = parser.parse_args(argv)
    if (not all(math.isfinite(s) and s > 0 for s in args.increments + [args.reference_step])
            or args.reference_step >= min(args.increments)):
        parser.error("require positive finite steps, with reference-step smaller than all increments")
    protocols = list(dict.fromkeys(args.protocol))
    increments = sorted(set(args.increments), reverse=True)
    output = args.output_dir.expanduser().resolve()
    if args.plot_only:
        plot_saved_results(output)
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output is not empty: {output}; choose a fresh --output-dir")
    material = Material()
    reference_backend = ScalarReference(material)
    production = reference_backend if args.reference_only else ProductionKernel(material)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"material": asdict(material), "protocols": protocols, "increments": increments,
                "reference_step": args.reference_step, "reference_only": args.reference_only,
                "check_reference": args.check_reference,
                "rate_s_inv": 2.5e-9,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "kernel_library": getattr(production, "library", None)}
    if settings["kernel_library"]:
        settings["kernel_sha256"] = hashlib.sha256(Path(settings["kernel_library"]).read_bytes()).hexdigest()
    with (output / "settings.json").open("x") as file:
        json.dump(settings, file, indent=2)
    started = time.perf_counter()
    results, metrics, checks, reference_refinement = {}, {}, {}, {}
    for protocol in protocols:
        print(f"{protocol}: independent fine reference, step <= {args.reference_step:g}", flush=True)
        reference = run_history(reference_backend, protocol, args.reference_step)
        _save_history(output / f"{protocol}_reference.csv", reference)
        if args.check_reference:
            print(f"{protocol}: reference resolution check, step <= {args.reference_step / 2:g}", flush=True)
            finer_reference = run_history(reference_backend, protocol, args.reference_step / 2)
            reference_refinement[protocol] = compare_histories(reference, finer_reference)
            _save_history(output / f"{protocol}_finer_reference.csv", finer_reference)
        histories, metrics[protocol] = [], []
        checks[protocol] = {"reference": {**_history_checks(reference),
                                          **_constitutive_checks(reference, material)}}
        for increment in increments:
            print(f"{protocol}: {type(production).__name__}, step <= {increment:g}", flush=True)
            history = run_history(production, protocol, increment)
            matched = (history if args.reference_only else
                       run_history(reference_backend, protocol, increment))
            tag = f"{increment:g}"
            _save_history(output / f"{protocol}_production_{tag}.csv", history)
            if not args.reference_only:
                _save_history(output / f"{protocol}_matched_reference_{tag}.csv", matched)
            metrics[protocol].append({"strain_increment": increment,
                                      "temporal_error": compare_histories(history, reference),
                                      "independent_temporal_error": compare_histories(matched, reference),
                                      "matched_increment_error": compare_histories(history, matched)})
            checks[protocol][tag] = {**_history_checks(history), **_constitutive_checks(history, material)}
            histories.append(history)
        results[protocol] = (reference, histories)
    _plot_results(output, material, results, metrics)
    inaccurate = any(c["max_strength_utilization"] > 1.001
                     or c["increments_with_omitted_creep_above_1e_minus12"] > 0
                     for protocol in checks.values() for key, c in protocol.items() if key != "reference")
    report = {"elapsed_seconds": time.perf_counter() - started,
              "verification_status": "discrepancies_detected" if inaccurate else "no_flagged_discrepancies",
              "reference_refinement": reference_refinement,
              "metrics": metrics, "checks": checks,
              "notes": ["Fixed-phi is externally constrained; its AT2 residual need not vanish.",
                        "Evolving/cycle solve homogeneous AT2 to scalar-root tolerance.",
                        "Finite ell enters this homogeneous response through Gc/ell.",
                        "The reference includes norm smoothing; production activation tolerances can differ.",
                        "Phase-field trials never commit additional creep within an increment."]}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Material-point verification of the paper's intrinsic-2D formulation\n\n"
        "MaterialPoint_fixed: original phi=0.5 verification.\n"
        "MaterialPoint_evolving: coupled spatially homogeneous AT2/history response.\n"
        "MaterialPoint_cycle: loading, hold, unloading and reloading; damage cannot heal.\n"
        "MaterialPoint_convergence: production histories against independent fine references.\n"
        "MaterialPoint_discrepancies: initiation overshoot and omitted early creep.\n\n"
        "verification.json separates production-vs-fine, production-vs-independent at\n"
        "identical increments, and independent-solver temporal errors. Optional\n"
        "reference_refinement compares the fine reference against half its increment.\n\n"
        f"Status: {report['verification_status']}\n"
        "In the tested installed kernel, fixed-phi stresses near initiation can exceed\n"
        "the combined strength surface. This also occurs with creep disabled.\n"
        "The 1e-8 absolute viscous-multiplier tolerance can omit small creep increments;\n"
        "reducing dt can therefore delay creep accumulation. These are separate from\n"
        "the finite-element ten-pass staggered warnings. No constitutive code was changed.\n\n"
        "Homogeneous phase-field degradation depends on Gc/ell. It does not represent\n"
        "a localized crack, mesh convergence, or phase-field length-scale independence.\n"
    )
    print(f"Saved material-point results to {output} ({report['elapsed_seconds']:.1f} s)", flush=True)
    print(f"Verification status: {report['verification_status']}", flush=True)
    for protocol in protocols:
        c = checks[protocol][f"{increments[-1]:g}"]
        print(f"{protocol}: max phi={c['max_phi']:.6f}, min d={c['min_degradation']:.6f}; "
              f"strain balance <= {c['max_abs_strain_balance_error']:.3g}")
    return 0


if __name__ == "__main__":
    # Small dense operations should share the workstation with the running plate jobs.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    raise SystemExit(main())
