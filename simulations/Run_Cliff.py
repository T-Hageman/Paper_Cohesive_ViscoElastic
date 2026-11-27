"""Run one ice-cliff simulation (manuscript Section 4.2) with the set-up of Cliff_Params.py.

MPI worker and command line in one; a case is one call, for example the elastic cliff
of Fig. 17 with the opening-only criterion:

    mpirun -n 16 python -u Run_Cliff.py \
        --name new_wl_elastic_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_csf20 \
        --water-level 182 --length 500 --damping 1e6 --end-time 4200 --f-s 1.5e6 \
        --notch-spacing 50 --notch-width 0.5 --stagnation-passes 0 \
        --rheology elastic --failure-type r2_2L_PSD --crack-step-factor 20

writes to results/Results_Cliff/<name>/ (Data/, Movies/, Monitor/) in the repository
root. The README lists the command lines of all cliff runs of the paper.

Set OMP_NUM_THREADS=1 (and the other thread counts) before starting.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

COMPLETION_FILE = "simulation_complete.txt"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "results" / "Results_Cliff"


def run_cliff(OutputFolder, snapshot_steps: int = 10, **options) -> None:
    """Run one cliff simulation; ``options`` are passed to ``build_cliff_params``."""
    import femcubes
    from femcubes import Mesh, Physics, TimeSolver
    from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank
    import femcubes.boundary  # noqa: F401
    import femcubes.solids  # noqa: F401

    from Cliff_Params import build_cliff_params, summarize

    out = str(OutputFolder)
    Path(out).mkdir(parents=True, exist_ok=True)
    isolate_jit_cache()
    start = time.time()

    params = build_cliff_params(OutputFolder=out, **options)
    mprint(f"Cliff case {Path(out).name}")
    mprint(f"FEMCubes {femcubes.__version__} from {Path(femcubes.__file__).parent}")
    mprint(summarize(params))

    mesh = Mesh(params)
    mesh.plot_mesh()
    n_elems_before = mesh.mesh.topology.index_map(2).size_global
    install_plot_snapshots(mesh, params, out + "/Monitor", every_steps=snapshot_steps, every_seconds=600.0)

    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()

    mesh.SavePlotsToFolder(out)
    mesh.Close_Plots()
    n_dofs = mesh.get_total_dofcount()
    n_elems_after = mesh.mesh.topology.index_map(2).size_global
    if rank == 0:
        elapsed = time.time() - start
        mprint(f"Total simulation time: {elapsed:.2f} seconds")
        # Written last: its presence marks the case as complete.
        with open(f"{out}/{COMPLETION_FILE}", "w") as f:
            f.write(f"Total simulation time: {elapsed:.2f} seconds\n")
            f.write(f"Number of elements before refinement: {n_elems_before}\n")
            f.write(f"Number of elements after refinement: {n_elems_after}\n")
            f.write(f"Total number of degrees of freedom: {n_dofs}\n")


def install_plot_snapshots(mesh, params, folder: str, every_steps: int = 10,
                           every_seconds: float = 600.0) -> None:
    """Save all open plots as PNG during the run, for monitoring a case.

    Wraps ``mesh.Plot_Measures`` (the last plotting call of every time step in
    ``TimeSolver.run``) so that every ``every_steps`` steps, or whenever
    ``every_seconds`` have passed, the mesh, field and measure plots are written
    to ``folder`` (overwriting the previous snapshot) together with ``status.txt``
    holding the current step, time, time step, cell count and all global measures.
    Call on every rank after the mesh exists; only rank 0 writes files.
    """
    import os
    import time

    from femcubes.core.utils.mpi_utils import rank

    os.makedirs(folder, exist_ok=True)
    original = mesh.Plot_Measures
    state = {"step": 0, "last": time.time(), "start": time.time()}

    def plot_measures_and_snapshot():
        original()
        state["step"] += 1
        now = time.time()
        due = state["step"] % every_steps == 0 or now - state["last"] >= every_seconds
        if not due:
            return
        state["last"] = now
        mesh.SavePlotsToFolder(folder)  # rank 0 only inside
        if rank == 0:
            lines = [
                f"updated {time.strftime('%Y-%m-%d %H:%M:%S')} after {(now - state['start']) / 3600:.2f} h wall time",
                f"step {state['step']}, t = {params.t:.6e} s, dt = {params.dt:.6e} s, end_time = {params.end_time:.6e} s",
                f"cells = {mesh.mesh.topology.index_map(2).size_global}, remesh epoch {getattr(mesh, 'remesh_epoch', 0)}",
            ]
            lines += [f"{name} = {value:.6e}" for name, value in sorted(params.Global_Measures.items())
                      if np.isscalar(value)]
            with open(os.path.join(folder, "status.txt"), "w") as handle:
                handle.write("\n".join(lines) + "\n")

    mesh.Plot_Measures = plot_measures_and_snapshot


def main(argv=None) -> int:
    import Cliff_Params as cliff

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="case folder below the output root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--rheology", default="viscoelastic", choices=cliff.RHEOLOGIES)
    parser.add_argument("--water-level", type=float, default=None,
                        help="water level before t = 0 [m]; flotation by default")
    parser.add_argument("--length", type=float, default=cliff.CLIFF_LENGTH)
    parser.add_argument("--ell", type=float, default=cliff.ELL)
    parser.add_argument("--f-s", type=float, default=cliff.F_S, help="compressive strength [Pa]")
    parser.add_argument("--failure-type", default="r2_2L", choices=cliff.FAILURE_TYPES,
                        help="r2_2L: two-multiplier criterion; r2_2L_PSD: opening-only criterion")
    parser.add_argument("--damping", type=float, default=cliff.DAMPING)
    parser.add_argument("--end-time", type=float, default=cliff.END_TIME)
    parser.add_argument("--stagnation-passes", type=int, default=3,
                        help="passes without 10 %% residual reduction after which a step is rejected; 0: never")
    parser.add_argument("--crack-step-factor", type=float, default=cliff.CRACK_STEP_FACTOR,
                        help="crack advance per step <= this factor times the target element size")
    parser.add_argument("--notch-spacing", type=float, default=0.0,
                        help="spacing of notches in the upper surface [m]; 0 for none")
    parser.add_argument("--notch-width", type=float, default=cliff.NOTCH_WIDTH)
    args = parser.parse_args(argv)

    run_cliff(
        Path(args.output_root) / args.name,
        rheology=args.rheology, water_level=args.water_level, length=args.length, ell=args.ell,
        f_s=args.f_s, failure_type=args.failure_type, damping=args.damping, end_time=args.end_time,
        stagnation_passes=args.stagnation_passes, crack_step_factor=args.crack_step_factor,
        notch_spacing=args.notch_spacing, notch_width=args.notch_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
