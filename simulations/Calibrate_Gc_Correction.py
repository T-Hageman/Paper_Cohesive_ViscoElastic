"""Calibrate / verify the Feng-style mesh correction of Gc for the CVE model.

Double-edge-notched bar in quasi-static tension (no creep, no inertia),
discretised exactly as the manuscript cases (P2 displacement, P1 phase
field, quadrature degree 3, intrinsic 2-D, AT2, r2_2L strength surface).
The ligament breaks completely, so the dissipated energy per unit ligament
length D = W_ext / L_lig (W_ext = integral of F du) measures the numerical
fracture energy Gc_num(h). Feng, Ren & Lu (CMAME 2026) show that for this
class of models Gc_num = Gc (1 + c h/ell) + O(h^2), with the shape of the
cohesive law preserved, so a prescribed Gc/(1 + c h/ell) cancels the
leading error. ``--analyse`` fits c from the uncorrected runs; corrected
runs (``--coefficient c``) should then give a mesh-independent D.

Run from simulations/ in the FenicsX environment, e.g.
    python Calibrate_Gc_Correction.py --dxratios 2,3,4,5,8          # uncorrected
    python Calibrate_Gc_Correction.py --dxratios 2,3,5 --coefficient 0.5
    python Calibrate_Gc_Correction.py --analyse
Each case runs with ``--ranks`` MPI ranks; ``--jobs`` cases run at once.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "Results_GcCorrection"
ELL = 0.02          # phase-field length [m], as in the manuscript
GC = 30.0           # J/m^2: l_ch = E Gc / f_t^2 = 1.8 m
LX, LY = 0.4, 0.2   # bar [m]; no snap-back since 2 f_t^2 Lx / (E Gc) = 0.45 < 1
LX_CENTRE = 0.12    # finely meshed centre region (6 ell)
NOTCH_DEPTH = 0.01  # each side -> ligament 0.18 m
NOTCH_WIDTH = 0.01
LIGAMENT = LY - 2 * NOTCH_DEPTH
END_DISPLACEMENT = 5.0e-4  # m; traction < 2% f_t at delta = 0.46 mm
COMPLETION_FILE = "simulation_time.txt"


def case_name(dxratio, coefficient):
    tag = "uncorrected" if coefficient is None else f"c{coefficient:g}"
    return f"{tag}/dx_{dxratio}"


def build_params(output_folder, dxratio, coefficient, rate=1.0e-6):
    from femcubes import LinearSolver, ParamsTemplate
    from femcubes.core.utils.mpi_utils import mprint

    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        Quadrature_Degree: int = 3

        MeshType: str = "NotchedTensileBar"
        dim: int = 2
        Lx: float = LX
        Ly: float = LY
        Lx_centre: float = LX_CENTRE
        Notch_Width: float = NOTCH_WIDTH
        Notch_Depth: float = NOTCH_DEPTH
        Notch_Sides: str = "both"
        dx: float = 0.02
        dx_centre: float = ELL / dxratio

        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,
            "Youngs": 9.5e9,
            "Poisson": 0.35,
            "Visc_Type": "None",     # no creep: D = W_ext exactly
            "ell": ELL,
            "f_t": 0.4e6,
            "f_s": 0.6e6,
            "Gc": GC,
            "PF_formulation": "AT2",
            "FailureType": "r2_2L",
            "Inertia": False,
            "kappa": 1.0e-3,
            "LE_Damping": 0.0,
        }
        # intrinsic 2-D constants, as in Small_Reference_Params
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Youngs"] * Materials["Ice"]["Poisson"] / (1.0 - Materials["Ice"]["Poisson"]**2)
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Lame"] + Materials["Ice"]["Shear"]
        gravity: float = 0.0

        Plot_Fields: list = []
        Plot_Measures: list = [("t", "BC_Right_F")]

        Solution_Steps = {"u": 0, "phasefield": 1}
        Step_Solvers = {0: "Newton", 1: "Newton"}
        Field_Orders = {"u": 2, "phasefield": 1}

        Models = {}
        ModelNames = ["Solid", "pf", "BC_Left_x", "BC_Left_y", "BC_Right_y", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:Cohesive_ViscoElastic_Kernel",
            "Material": "Ice",
            "plane_stress": False,
            "true_2d": True,
            "consistent_stiffness": True,
        }
        Models["pf"] = {
            "Type": "Solid:PhaseField",
            "Material": "Ice",
            "CDF_field": "PsiField",
            "Irreversible_Method": "Hist",
        }
        if coefficient is not None:
            Models["pf"]["Gc_Mesh_Correction"] = {
                "Coefficient": coefficient,
                "Source_Field": "etaNorm",
                "Source_Threshold": 1.0e-8,
                "Search_Radius": 2.0,
                "Reduction": "mean",
            }
        Models["BC_Left_x"] = {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0}
        Models["BC_Left_y"] = {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_y", "value": 0.0}
        Models["BC_Right_y"] = {"Type": "Boundary:PrescribedValue", "boundary": "right", "field": "u_y", "value": 0.0}
        Models["BC_Right"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "right",
            "field": "u_x",
            "rate": rate * LX,
            "Dummy": 1e12,
            "F_Cutoff": 0.02,
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes_force: bool = False
        max_it: int = 20
        n_passes: int = 10
        line_search_lims: list = [1e-1, 1.0]
        line_search: bool = True
        rel_tol: float = 1e-6
        abs_tol: float = 0.001

        output_dir: str = str(Path(output_folder) / "Data")
        output_interval: int = 10
        write_vtk: bool = False
        write_hdf5: bool = True
        save_Mesh_every_step: bool = False
        output_fields: list = ["u_x", "u_y", "phasefield", "sigma_xx", "sigma_yy", "sigma_xy"] + (
            ["Gc_eff"] if coefficient is not None else [])

        end_time: float = END_DISPLACEMENT / (rate * LX)
        dt: float = 2.0e-6 / (rate * LX)          # 2 um of end displacement per step
        start_time: float = 0.0
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5
        TimeStepControl = {
            "Type": "Measure",
            "time_list": [-1e99, 0],
            "dt_list": [60, 2.0e-6 / (rate * LX)],
            "dt_min": 1.0e-14,
            "dt_max": 2.0e-6 / (rate * LX),
            "Measure": "Crack_Length_Change",
            "Target_Change": 0.01 / 4,
        }

    mprint(f"Bar case: dx_centre = {ELL / dxratio:.4e} m (ell/h = {dxratio}), "
           f"Gc correction coefficient = {coefficient}, end displacement = {END_DISPLACEMENT:.2e} m")
    return Params()


def run_case(output_folder, dxratio, coefficient):
    """Run one bar case (MPI worker)."""
    from femcubes import Mesh, Physics, TimeSolver
    from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank
    import femcubes.boundary  # noqa: F401
    import femcubes.solids  # noqa: F401

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    isolate_jit_cache()
    start = time.time()
    params = build_params(output_folder, dxratio, coefficient)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()
    n_cells = mesh.mesh.topology.index_map(2).size_global
    if rank == 0:
        elapsed = time.time() - start
        mprint(f"Total simulation time: {elapsed:.2f} seconds")
        with open(output_folder / COMPLETION_FILE, "w") as f:
            f.write(f"Total simulation time: {elapsed:.2f} seconds\n")
            f.write(f"Number of elements: {n_cells}\n")


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------

def _mean_centre_diameter(case_dir):
    """Mean cell diameter (longest edge) of the cells in the centre region."""
    import h5py

    with h5py.File(case_dir / "Data/mesh_epoch_000.hdf5") as h:
        X = np.asarray(h["Mesh_DG1/X"]).reshape(-1, 3)
        Y = np.asarray(h["Mesh_DG1/Y"]).reshape(-1, 3)
    xc = X.mean(axis=1)
    centre = np.abs(xc - LX / 2) < LX_CENTRE / 2 - 0.5 * ELL
    edges = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edges.append(np.hypot(X[:, i] - X[:, j], Y[:, i] - Y[:, j]))
    diameter = np.max(edges, axis=0)
    return float(diameter[centre].mean()), int(centre.sum())


def _local_cohesive_law(case_dir, u_target, stations=(0.07, 0.09, 0.11, 0.13)):
    """Effective Gc' of the local traction-separation law at mid-height of the ligament.

    At the output step closest to the end displacement ``u_target``, the
    opening delta (jump of u_x across the crack line from linear fits of u_x
    on 1.5-3.5 ell either side) and the crack-line traction sigma_xx (cell
    average within one element of the line) are read at several y stations.
    For the Dirac-limit law sigma = f_t (Gc'/(Gc' + f_t delta))^2 this gives
    Gc' = f_t delta / (sqrt(f_t/sigma) - 1); the mesh error of this model
    class appears as Gc'/Gc = 1 + c h/ell (Feng et al. 2026).
    """
    import h5py
    import re

    f_t = 0.4e6
    x_crack = LX / 2
    with h5py.File(case_dir / "Data/time_data.hdf5") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["data_names"][()]]
        data = np.asarray(h["data"])
    u_hist = np.abs(data[:, names.index("BC_Right_u")])
    steps = sorted(int(re.match(r"Outputs_(\d+)", p.name).group(1))
                   for p in (case_dir / "Data").glob("Outputs_*.hdf5"))
    steps = [s for s in steps if 0 <= s - 1 < len(u_hist)]
    step = min(steps, key=lambda s: abs(u_hist[s - 1] - u_target))
    with h5py.File(case_dir / "Data/mesh_epoch_000.hdf5") as h:
        X0, Y0 = h["Mesh_DG0/X"][()][:, 0], h["Mesh_DG0/Y"][()][:, 0]
        X2, Y2 = h["Mesh_DG2/X"][()], h["Mesh_DG2/Y"][()]
    with h5py.File(case_dir / f"Data/Outputs_{step}.hdf5") as h:
        ux = h["Fields/u_x"][()]
        sxx = h["Fields/sigma_xx"][()][:, 0]
    h_mean, _ = _mean_centre_diameter(case_dir)
    tol = 0.6 * h_mean
    values = []
    for y0 in stations:
        m2 = np.abs(Y2 - y0) < tol
        x2, u2 = X2[m2], ux[m2]

        def fit(sel):
            return np.polyfit(x2[sel] - x_crack, u2[sel], 1)[1]

        delta = (fit((x2 > x_crack + 1.5 * ELL) & (x2 < x_crack + 3.5 * ELL))
                 - fit((x2 < x_crack - 1.5 * ELL) & (x2 > x_crack - 3.5 * ELL)))
        mc = (np.abs(Y0 - y0) < tol) & (np.abs(X0 - x_crack) < h_mean)
        sigma = sxx[mc].mean()
        if 0.0 < sigma < f_t and delta > 0.0:
            values.append(f_t * delta / (np.sqrt(f_t / sigma) - 1.0))
    return float(np.mean(values)) if values else float("nan"), float(u_hist[step - 1])


def analyse_case(case_dir):
    import h5py

    with h5py.File(case_dir / "Data/time_data.hdf5") as h:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h["data_names"][()]]
        data = np.asarray(h["data"])
    col = {k: i for i, k in enumerate(names)}
    u = data[:, col["BC_Right_u"]]
    F = data[:, col["BC_Right_F"]]
    order = np.argsort(u)
    work = float(np.trapz(F[order], u[order]))
    h_mean, n_centre = _mean_centre_diameter(case_dir)
    gc_local = {}
    for u_target in (1.2e-4, 2.0e-4):
        gc_prime, u_actual = _local_cohesive_law(case_dir, u_target)
        gc_local[f"Gc_local_{u_target * 1e6:.0f}um"] = gc_prime / GC
    return {
        **gc_local,
        "peak_F": float(np.max(np.abs(F))),
        "F_end": float(F[-1]),
        "u_end": float(u[-1]),
        "W_ext": work,
        "D_per_length": work / LIGAMENT,
        "h_mean": h_mean,
        "n_centre_cells": n_centre,
        "steps": int(len(u)),
    }


def analyse(root):
    rows = []
    for completion in sorted(root.glob("*/dx_*/" + COMPLETION_FILE)):
        case_dir = completion.parent
        tag, dx_name = case_dir.parent.name, case_dir.name
        dxratio = int(dx_name.split("_")[1])
        coefficient = None if tag == "uncorrected" else float(tag[1:])
        result = analyse_case(case_dir)
        result.update({"tag": tag, "dxratio": dxratio, "coefficient": coefficient})
        rows.append(result)
    if not rows:
        print("No completed cases found.")
        return
    print(f"{'case':>20} {'ell/h':>6} {'h_mean/ell':>10} {'peak F':>10} {'F_end/peak':>10} "
          f"{'D/Gc':>8} {'Gc\'/Gc@120um':>12} {'Gc\'/Gc@200um':>12} {'steps':>6}")
    for r in rows:
        print(f"{r['tag'] + '/dx_' + str(r['dxratio']):>20} {r['dxratio']:6d} {r['h_mean'] / ELL:10.4f} "
              f"{r['peak_F']:10.4e} {abs(r['F_end']) / r['peak_F']:10.4f} {r['D_per_length'] / GC:8.4f} "
              f"{r['Gc_local_120um']:12.4f} {r['Gc_local_200um']:12.4f} {r['steps']:6d}")
    uncorrected = [r for r in rows if r["coefficient"] is None]
    if len(uncorrected) >= 2:
        s = np.array([r["h_mean"] / ELL for r in uncorrected])
        for key in ("Gc_local_120um", "Gc_local_200um"):
            g = np.array([r[key] for r in uncorrected])
            slope, intercept = np.polyfit(s, g, 1)
            print(f"Uncorrected fit {key}: Gc'/Gc = {intercept:.4f} + {slope:.4f} (h_mean/ell)  ->  "
                  f"c = slope/intercept = {slope / intercept:.3f}")
    (root / "analysis.json").write_text(json.dumps(rows, indent=2) + "\n")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dxratios", default="2,3,4,5,8", help="comma-separated ell/h values")
    parser.add_argument("--coefficient", type=float, default=None, help="Gc correction coefficient c (default: off)")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("DXRATIO", "COEFF"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()

    if args.worker:
        dxratio = int(args.worker[0])
        coefficient = None if args.worker[1] == "None" else float(args.worker[1])
        run_case(root / case_name(dxratio, coefficient), dxratio, coefficient)
        return 0
    if args.analyse:
        analyse(root)
        return 0

    pending = [int(x) for x in args.dxratios.split(",")]
    running = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg", "OMPI_MCA_hwloc_base_binding_policy": "none"})
    while pending or running:
        while pending and len(running) < args.jobs:
            dxratio = pending.pop(0)
            folder = root / case_name(dxratio, args.coefficient)
            if (folder / COMPLETION_FILE).is_file():
                print(f"Skipping complete {folder.relative_to(root)}", flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "run.log").open("w")
            command = ["mpirun", "-n", str(args.ranks), sys.executable, "-u", str(Path(__file__).resolve()),
                       "--output-root", str(root), "--worker", str(dxratio), str(args.coefficient)]
            process = subprocess.Popen(command, cwd=SCRIPT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((folder, process, log, time.time()))
            print(f"Started {folder.relative_to(root)}", flush=True)
        for job in running[:]:
            folder, process, log, start = job
            if process.poll() is None:
                continue
            log.close()
            running.remove(job)
            status = "Finished" if process.returncode == 0 and (folder / COMPLETION_FILE).is_file() else "FAILED"
            print(f"{status} {folder.relative_to(root)} ({(time.time() - start) / 60:.1f} min)", flush=True)
        time.sleep(2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
