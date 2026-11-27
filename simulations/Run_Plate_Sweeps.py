"""Run the plate-with-hole sweeps behind Figs. 5 to 11 and Table 2 of the paper (Section 3).

Run from simulations/ in the FenicsX environment:
    python Run_Plate_Sweeps.py --list-cases
    python Run_Plate_Sweeps.py --dry-run
    python Run_Plate_Sweeps.py
    python Run_Plate_Sweeps.py --figure 8          # one figure only
    python Run_Plate_Sweeps.py --figure 8 --figure 9
    python Run_Plate_Sweeps.py --case-ids 0-4,15

Defaults: two concurrent cases with 26 MPI ranks each (52 cores total).

90 cases, each at strain rates {1.0, 1.5, 2.5}e-9 /s:
  Figs. 8/9 (IDs 0-29): element-size sweep at ell = 0.02 m, ell/h = 1-5, on
    prescribed uniform meshes WITHOUT adaptive refinement; Gc = 1 (Fig. 8)
    and Gc = 10 J/m2 (Fig. 9).
  Fig. 11 (IDs 30-89): length-scale sweep, ell = {0.1, 0.05, 0.02, 0.01,
    0.005} m, Gc = {1, 2.5, 5, 10} J/m2, WITH adaptive refinement to a target
    ell/h = 4 (background 0.03 m, 0.002 m near the hole). The Gc = 10 cases
    (IDs 75-89, added 5 Nov 2026) link the length-scale study to the Gc = 10
    element-size study and supply the eta-activation measures for the
    rate/energy table; they are appended so that IDs 0-74 are unchanged.
The --figure numbers are those of the paper (Figs. 8, 9 and 11); Figs. 5-7,
10 and Table 2 use cases of these sweeps (see the README).
All other settings are those of the earlier runs of the paper:
Small_Reference_Params: intrinsic 2-D, AT2 with P2 displacement / P1 phase
field, Glen creep, damping 1e9. Movie recording is disabled, which does not
affect the solution.

The simulated interval is shortened from the previous 1 mm top displacement
to END_DISPLACEMENT_GC10 (0.35 mm) for Gc=10 and END_DISPLACEMENT (0.1 mm)
otherwise, leaving headroom beyond the plotted 0.25 mm / 0.05 mm windows.
The existing F_Cutoff early stop (50 steps after F < 0.1 F_max) still applies.
Peak loads of cases that do not fracture are the maximum within this interval.

The displacement step uses standard Newton with line search (MECH_SOLVER)
instead of Small_Reference_Params' trust region; both gave identical results
and cost on a test case. The paper results were computed with IceCubes
(commit ce4763f and later), the predecessor of FEMCubes; FEMCubes v0.1.0 has the
same kernel (Solid:Cohesive_ViscoElastic_Kernel) with the 1e-12 CVE viscous
tolerance, without which post-peak steps stall.

Output folders, below results/ in the repository root, match those read by
postprocessing/Process_All_Data.py:
    Results_dxSweep_LO{_LowRate,,_HighRate}/dx_N              (Fig. 8)
    Results_dxSweep_LO{_LowRate,,_HighRate}_Gc10/dx_N         (Fig. 9)
    Results_lSweep_LO{_LowRate,,_HighRate}{,_Gc2.5,_Gc5.0,_Gc10}/l_L (Fig. 11)

A case is complete once simulation_time.txt exists; complete cases are
skipped, so the sweep can be restarted after an interruption. An incomplete
folder from an earlier attempt is renamed to <case>.incomplete-<timestamp>
rather than overwritten. Each case writes local_run.log, case_settings.json
(including the FEMCubes version) and the normal Data/ HDF5 outputs.
"""

import argparse
from collections import deque
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
LOADING_RATES = (("_LowRate", 1.0e-9), ("", 1.5e-9), ("_HighRate", 2.5e-9))
# Figs. 8 and 9: element-size sweep on prescribed meshes.
DX_LENGTH_SCALE = 0.02
DX_RATIOS = (1, 2, 3, 4, 5)
DX_FRACTURE_ENERGIES = ((8, 1.0, ""), (9, 10.0, "_Gc10"))  # (figure, Gc, suffix)
# Fig. 11: length-scale sweep with adaptive refinement to ell/h = 4.
L_SWEEP_LENGTH_SCALES = (0.1, 0.05, 0.02, 0.01, 0.005)
L_SWEEP_FRACTURE_ENERGIES = ((1.0, ""), (2.5, "_Gc2.5"), (5.0, "_Gc5.0"), (10.0, "_Gc10"))
L_SWEEP_DXRATIO = 4
FIGURES = (8, 9, 11)
# Top displacement [m] at which each run ends (previously 1e-3 m for all).
END_DISPLACEMENT = 1.0e-4
END_DISPLACEMENT_GC10 = 3.5e-4
MECH_SOLVER = "Newton"  # nonlinear solver for the displacement step
COMPLETION_FILE = "simulation_time.txt"


def _case(output_root, figure, name, **parameters):
    return {
        "figure": figure,
        "name": name,
        "output_folder": Path(output_root) / name,
        "parameters": {
            "end_displacement": (END_DISPLACEMENT_GC10 if parameters["Gc"] == 10.0
                                 else END_DISPLACEMENT),
            **parameters,
        },
    }


def build_sweep_cases(output_root, gc_correction=None):
    """Return all 90 cases: Fig. 8, then Fig. 9, then Fig. 11.

    With ``gc_correction`` (the coefficient c of the Feng-style mesh correction
    Gc/(1 + c h/ell), see ``Gc_Mesh_Correction`` in the FEMCubes phase-field
    model), the results go to ``..._GcCorr<c>`` study folders.
    """
    corr_suffix = "" if gc_correction is None else f"_GcCorr{gc_correction:g}"
    cases = []
    for figure, fracture_energy, gc_suffix in DX_FRACTURE_ENERGIES:
        for rate_suffix, rate in LOADING_RATES:
            study = f"Results_dxSweep_LO{rate_suffix}{gc_suffix}{corr_suffix}"
            for dxratio in DX_RATIOS:
                cases.append(_case(output_root, figure, f"{study}/dx_{dxratio}",
                                   l=DX_LENGTH_SCALE, dxratio=dxratio, Gc=fracture_energy,
                                   rate=rate, MeshRef=False, gc_correction=gc_correction))
    for fracture_energy, gc_suffix in L_SWEEP_FRACTURE_ENERGIES:
        for rate_suffix, rate in LOADING_RATES:
            study = f"Results_lSweep_LO{rate_suffix}{gc_suffix}{corr_suffix}"
            for length_scale in L_SWEEP_LENGTH_SCALES:
                cases.append(_case(output_root, 11, f"{study}/l_{length_scale}",
                                   l=length_scale, dxratio=L_SWEEP_DXRATIO,
                                   Gc=fracture_energy, rate=rate, MeshRef=True,
                                   gc_correction=gc_correction))
    return cases


def run_tensile_ice_small(OutputFolder, l=0.02, dxratio=4, Gc=1.0, rate=1.5e-9, MeshRef=False,
                          end_displacement=None, gc_correction=None):
    """Run one plate-with-hole simulation (MPI worker only)."""
    from femcubes import Mesh, Physics, TimeSolver
    from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank

    # Import these packages to register their models with Physics.
    import femcubes.boundary  # noqa: F401
    import femcubes.glaciology  # noqa: F401
    import femcubes.solids  # noqa: F401

    from Small_Reference_Params import _build_small_params

    output_folder = Path(OutputFolder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_folder = str(output_folder)

    isolate_jit_cache()
    start_time = time.time()

    mprint(f"Case parameters: l={l}, dxratio={dxratio}, Gc={Gc}, rate={rate:.2e}, "
           f"MeshRef={MeshRef}, end_displacement={end_displacement}")
    params = _build_small_params(
        l=l,
        dxratio=dxratio,
        rate=rate,
        include_plots=False,
        OutputFolder=output_folder,
        MeshRef=MeshRef,
    )
    params.Materials["Ice"]["Gc"] = Gc
    if gc_correction is not None:
        # Feng-style non-local mesh correction of Gc (Gc_Mesh_Correction of Solid:PhaseField).
        params.Models["pf"]["Gc_Mesh_Correction"] = {
            "Coefficient": gc_correction,
            "Source_Field": "etaNorm",
            "Source_Threshold": 1.0e-8,
            "Search_Radius": 2.0,
            "Reduction": "mean",
        }
        params.output_fields = list(params.output_fields) + ["Gc_eff"]
    mprint(f"Gc mesh correction: {params.Models['pf'].get('Gc_Mesh_Correction')}")
    # Nonlinear solver for the displacement step (step 0).
    params.Step_Solvers = {**params.Step_Solvers, 0: MECH_SOLVER}
    mprint(f"Step solvers: {params.Step_Solvers}")
    if end_displacement is not None:
        # The top moves at rate * Ly, so this ends the run at end_displacement.
        params.end_time = end_displacement / (abs(rate) * params.Ly)
    mprint(f"End time = {params.end_time:.3e} s "
           f"(top displacement {abs(rate) * params.Ly * params.end_time * 1e3:.3f} mm)")

    ell_phys = (params.Materials["Ice"]["Youngs"] * params.Materials["Ice"]["Gc"]
                / params.Materials["Ice"]["f_t"] ** 2)
    mprint(f"Physical length scale  = {ell_phys:.3e} m")
    mprint(f"Phasefield length scale  = {params.Materials['Ice']['ell']:.3e} m")

    mesh = Mesh(params)
    mesh.plot_mesh()
    n_elems_before = mesh.mesh.topology.index_map(2).size_global

    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    solver = TimeSolver(params, physics)
    solver.run()
    mesh.SavePlotsToFolder(output_folder)
    mesh.Close_Plots()

    n_dofs = mesh.get_total_dofcount()
    n_elems_after = mesh.mesh.topology.index_map(2).size_global

    if rank == 0:
        elapsed = time.time() - start_time
        mprint(f"Total simulation time: {elapsed:.2f} seconds")
        # Written last: its presence marks the case as complete.
        with open(f"{output_folder}/{COMPLETION_FILE}", "w") as f:
            f.write(f"Total simulation time: {elapsed:.2f} seconds\n")
            f.write(f"Number of elements before refinement: {n_elems_before}\n")
            f.write(f"Number of elements after refinement: {n_elems_after}\n")
            f.write(f"Total number of degrees of freedom: {n_dofs}\n")


def _femcubes_commit():
    """Return the FEMCubes git commit (with a dirty flag) of a source install, else its
    installed version, or None."""
    import importlib.metadata
    import importlib.util

    try:
        spec = importlib.util.find_spec("femcubes")
        repo = Path(spec.origin).resolve().parent.parent
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (AttributeError, OSError, subprocess.CalledProcessError):
        try:
            return "v" + importlib.metadata.version("femcubes")
        except importlib.metadata.PackageNotFoundError:
            return None


def _parse_case_ids(text, n_cases):
    ids = []
    for part in text.split(","):
        start, _, stop = part.partition("-")
        ids.extend(range(int(start), int(stop or start) + 1))
    if any(not 0 <= i < n_cases for i in ids):
        raise argparse.ArgumentTypeError(f"case IDs must be between 0 and {n_cases - 1}")
    return sorted(set(ids))


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _worker_command(launcher, ranks, task_id, output_root, gc_correction=None):
    command = [
        launcher, "-n", str(ranks), sys.executable, "-u", str(Path(__file__).resolve()),
        "--task-id", str(task_id), "--output-root", str(output_root),
    ]
    if gc_correction is not None:
        command += ["--gc-correction", str(gc_correction)]
    return command


def _worker_environment(output_folder):
    env = os.environ.copy()
    # One computational thread per MPI rank: 2 cases x 26 ranks = 52 CPUs.
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"
    env["OMP_DYNAMIC"] = "FALSE"
    env["OMP_PROC_BIND"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    # Independent launchers should not all pin their ranks onto the same CPUs.
    env["I_MPI_PIN"] = "0"
    env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
    env["MPLBACKEND"] = "Agg"
    cache = output_folder / ".cache"
    env["XDG_CACHE_HOME"] = str(cache)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    env["NUMBA_CACHE_DIR"] = str(cache / "numba")
    for directory in (cache, cache / "matplotlib", cache / "numba"):
        directory.mkdir(parents=True, exist_ok=True)
    return env


def _prepare_output_folder(case, task_id, commit, launcher, ranks):
    """Set aside an incomplete earlier attempt, then record the case settings."""
    folder = case["output_folder"]
    if folder.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = folder.with_name(f"{folder.name}.incomplete-{stamp}")
        folder.rename(aside)
        print(f"Moved incomplete [{task_id}] {case['name']} -> {aside.name}", flush=True)
    folder.mkdir(parents=True)
    settings = {
        "task_id": task_id,
        "name": case["name"],
        "figure": case["figure"],
        "parameters": case["parameters"],
        "femcubes_commit": commit,
        "mpi_launcher": launcher,
        "mpi_ranks": ranks,
        "python": sys.executable,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "case_settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def _stop_running(running):
    """Stop MPI process groups and close their logs when interrupted."""
    for job in running:
        try:
            os.killpg(job["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for job in running:
        process = job["process"]
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            job["log"].close()


def run_parallel_cases(cases, task_ids, launcher, jobs, ranks, output_root, gc_correction=None):
    """Run the selected cases, at most ``jobs`` at a time."""
    commit = _femcubes_commit()
    if commit is None or commit.endswith("-dirty"):
        print(f"WARNING: FEMCubes commit is {commit}; results may not be reproducible.",
              flush=True)
    pending = deque((i, cases[i]) for i in task_ids)
    running = []
    failed = []
    sweep_start = time.time()
    print(f"Running {len(task_ids)} cases: up to {jobs} at once, {ranks} MPI ranks each. "
          f"FEMCubes commit: {commit}", flush=True)
    try:
        while pending or running:
            while pending and len(running) < jobs:
                task_id, case = pending.popleft()
                _prepare_output_folder(case, task_id, commit, launcher, ranks)
                output_folder = case["output_folder"]
                env = _worker_environment(output_folder)
                log_path = output_folder / "local_run.log"
                log = log_path.open("w")
                command = _worker_command(launcher, ranks, task_id, output_root, gc_correction)
                try:
                    process = subprocess.Popen(
                        command, cwd=SCRIPT_DIR, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                except OSError:
                    log.close()
                    raise
                running.append({"task_id": task_id, "case": case, "process": process,
                                "log": log, "log_path": log_path, "start": time.time()})
                print(f"Started  [{task_id}] {case['name']}", flush=True)

            for job in running[:]:
                return_code = job["process"].poll()
                if return_code is None:
                    continue
                job["log"].close()
                running.remove(job)
                minutes = (time.time() - job["start"]) / 60
                complete = (job["case"]["output_folder"] / COMPLETION_FILE).is_file()
                if return_code or not complete:
                    failed.append(job)
                    print(f"FAILED   [{job['task_id']}] {job['case']['name']} "
                          f"(exit {return_code}, {minutes:.1f} min); see {job['log_path']}",
                          flush=True)
                else:
                    print(f"Finished [{job['task_id']}] {job['case']['name']} "
                          f"({minutes:.1f} min)", flush=True)
            if running:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted; stopping active MPI runs.", flush=True)
        return 130
    except OSError as error:
        print(f"Could not launch a case: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        _stop_running(running)

    hours = (time.time() - sweep_start) / 3600
    if failed:
        print("Failed task IDs: " + ",".join(str(job["task_id"]) for job in failed),
              flush=True)
        return 1
    print(f"Completed all {len(task_ids)} cases in {hours:.2f} h.", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=_positive_int, default=2,
                        help="maximum concurrent cases (default: 2)")
    parser.add_argument("--ranks", type=_positive_int, default=26,
                        help="MPI ranks per case, one thread each (default: 26)")
    parser.add_argument("--mpi-launcher", default="mpirun",
                        help="MPI launcher executable or path (default: mpirun)")
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIR,
                        help="results directory (default: results/ in the repository root)")
    parser.add_argument("--figure", type=int, choices=FIGURES, action="append",
                        help="run only the cases for this manuscript figure (repeatable)")
    parser.add_argument("--case-ids", help="comma-separated IDs or ranges, e.g. 0-4,15")
    parser.add_argument("--list-cases", action="store_true",
                        help="list case IDs, folders and status without running")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the MPI commands that would be launched")
    parser.add_argument("--gc-correction", type=float, default=None, metavar="C",
                        help="run with the Feng-style Gc mesh correction Gc/(1 + C h/ell) "
                             "(results go to *_GcCorr<C> folders)")
    parser.add_argument("--task-id", type=int, help=argparse.SUPPRESS)  # MPI workers
    args = parser.parse_args(argv)

    output_root = args.output_root.expanduser().resolve()
    cases = build_sweep_cases(output_root, args.gc_correction)

    if args.task_id is not None:
        # Only MPI workers import FEMCubes, so the supervisor never initializes MPI.
        case = cases[args.task_id]
        run_tensile_ice_small(OutputFolder=str(case["output_folder"]), **case["parameters"])
        return 0

    selected = (list(range(len(cases))) if args.case_ids is None
                else _parse_case_ids(args.case_ids, len(cases)))
    if args.figure:
        selected = [i for i in selected if cases[i]["figure"] in args.figure]

    def is_complete(i):
        return (cases[i]["output_folder"] / COMPLETION_FILE).is_file()

    if args.list_cases:
        for i in selected:
            status = "done" if is_complete(i) else (
                "incomplete" if cases[i]["output_folder"].exists() else "pending")
            print(f"{i}\tFig. {cases[i]['figure']}\t{status}\t{cases[i]['name']}")
        return 0

    to_run = [i for i in selected if not is_complete(i)]
    skipped = len(selected) - len(to_run)
    if skipped:
        print(f"Skipping {skipped} completed case(s).", flush=True)
    if args.dry_run:
        print(f"Up to {args.jobs} concurrent cases x {args.ranks} MPI ranks each.")
        for i in to_run:
            print(shlex.join(_worker_command(args.mpi_launcher, args.ranks, i, output_root,
                                             args.gc_correction)))
        return 0
    if not to_run:
        print("Nothing to run.")
        return 0

    launcher = shutil.which(args.mpi_launcher)
    if launcher is None:
        parser.error(f"MPI launcher not found: {args.mpi_launcher}")
    available = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                 else (os.cpu_count() or 1))
    required = min(args.jobs, len(to_run)) * args.ranks
    if required > available:
        parser.error(f"{required} CPUs requested, but only {available} are available; "
                     "reduce --jobs or --ranks")
    return run_parallel_cases(cases, to_run, launcher, args.jobs, args.ranks, output_root,
                              args.gc_correction)


if __name__ == "__main__":
    raise SystemExit(main())
