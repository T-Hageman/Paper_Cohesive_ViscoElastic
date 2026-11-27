"""Example run of the plate-with-hole benchmark (Section 3 of the paper).

A 1 x 1 m ice plate with a central hole of radius 0.1 m, clamped at the bottom and pulled
at the top at the medium nominal strain rate of the paper (1.5e-9 1/s). The ice creeps by
Glen's law and fractures through the strength-based cohesive phase-field model. The
parameters come from simulations/Small_Reference_Params.py with the settings of the sweep
driver (simulations/Run_Plate_Sweeps.py: element-size correction of Gc, Newton solver,
end of the run), for Gc = 5 J/m^2 and the largest length scale of the paper, ell = 0.1 m.

The mesh is that of the length-scale study of the paper: a background mesh of 0.03 m, refined
initially to 0.002 m around the hole, where the creep relaxes the stress concentration before
the cracks nucleate, and refined adaptively to ell/4 wherever the cohesive strain appears, so
that the mesh follows the cracks as they grow from the hole (mesh.png and mesh_refinement.mp4
show the refined mesh).

Run with

    mpirun -n 8 python Example_run.py

The load-displacement curve, the final phase field, sigma_yy, the refined mesh and an
animation of the refinement are written to results/Example_run/.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulations"))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from femcubes import Mesh, Physics, TimeSolver  # noqa: E402
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank  # noqa: E402

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: E402, F401
import femcubes.solids  # noqa: E402, F401

from Small_Reference_Params import _build_small_params  # noqa: E402

OUTPUT_FOLDER = ROOT / "results" / "Example_run"

# Case of the length-scale study (Run_Plate_Sweeps.py, case Results_lSweep_LO_Gc5.0_GcCorr0.33/l_0.1)
ELL = 0.1                   # phase-field length scale (m)
GC = 5.0                    # fracture energy (J/m^2)
RATE = 1.5e-9               # nominal strain rate (1/s)
GC_CORRECTION = 0.33        # coefficient c of Gc / (1 + c h / ell)
END_DISPLACEMENT = 1.0e-4   # top displacement at the end of the run (m)

DXRATIO = 4                 # adaptive refinement to h = ell / DXRATIO where the cohesive strain appears

# Coarsening for the example (values of the ell = 0.1 m case of Fig. 11 in brackets)
COARSE = {
    "model": "Solid:Cohesive_ViscoElastic",  # reduced return mapping [Solid:Cohesive_ViscoElastic_Kernel]
    "Target_Change": ELL / 4,  # largest crack growth per time step (m) [0.0025]
}


def build_params():
    """Parameters of the paper case with the coarsening of COARSE."""
    # MeshRef=True: the mesh of the length-scale study, a background mesh of 0.03 m with 0.002 m at the
    # hole, refined adaptively to ell / DXRATIO where the cohesive strain etaNorm appears
    params = _build_small_params(l=ELL, dxratio=DXRATIO, rate=RATE, include_plots=True,
                                 OutputFolder=str(OUTPUT_FOLDER), MeshRef=True)
    # Settings of the sweep driver (Run_Plate_Sweeps.run_tensile_ice_small)
    params.Materials["Ice"]["Gc"] = GC
    params.Models["pf"]["Gc_Mesh_Correction"] = {
        "Coefficient": GC_CORRECTION,
        "Source_Field": "etaNorm",
        "Source_Threshold": 1.0e-8,
        "Search_Radius": 2.0,
        "Reduction": "mean",
    }
    params.Step_Solvers = {**params.Step_Solvers, 0: "Newton"}
    params.end_time = END_DISPLACEMENT / (RATE * params.Ly)
    # Coarsening
    params.Models["Solid"]["Type"] = COARSE["model"]
    params.TimeStepControl = {**params.TimeStepControl, "Target_Change": COARSE["Target_Change"]}
    # Plots: phase field and vertical stress at the end, load-displacement curve
    params.Plot_Fields = ["phasefield", "sigma_yy", "u_y"]
    params.Plot_Measures = [("BC_Top_u", "BC_Top_F")]
    params.record_movie = False
    return params


def plot_load_displacement(params):
    """Nominal stress against top displacement, with the paper curve if its results are present."""
    def read(path):
        with h5py.File(path, "r") as h:
            names = [n.decode() for n in h["data_names"][:]]
            data = h["data"][:]
        return 1e3 * abs(data[:, names.index("BC_Top_u")]), abs(data[:, names.index("BC_Top_F")]) / params.Lx / 1e6

    fig, ax = plt.subplots(figsize=(8 / 2.54, 6 / 2.54), layout="constrained")
    ax.plot(*read(OUTPUT_FOLDER / "Data" / "time_data.hdf5"), "k-", lw=1.0, label="this run")
    case = Path("Results_lSweep_LO_Gc5.0_GcCorr0.33") / "l_0.1" / "Data" / "time_data.hdf5"
    paper = next((root / case for root in (ROOT / "results", ROOT / "data") if (root / case).is_file()), None)
    if paper is not None:
        ax.plot(*read(paper), "C0--", lw=1.0, label="paper run")
    ax.set_xlabel(r"$u\;[\mathrm{mm}]$")
    ax.set_ylabel(r"$\overline{\sigma}\;[\mathrm{MPa}]$")
    ax.set_xlim(0.0, 1e3 * END_DISPLACEMENT)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUTPUT_FOLDER / "load_displacement.png", dpi=300)
    plt.close(fig)


def main():
    start = time.time()
    isolate_jit_cache()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    params = build_params()
    mprint(f"Plate with hole: ell = {ELL} m, Gc = {GC} J/m^2, rate = {RATE:.1e} 1/s, coarse settings {COARSE}")

    mesh = Mesh(params)
    mesh.plot_mesh()
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    mesh.SavePlotsToFolder(str(OUTPUT_FOLDER))
    mesh.Close_Plots()
    if rank == 0:
        plot_load_displacement(params)
        mprint(f"Done in {time.time() - start:.0f} s; figures in {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
