"""Run the sea-ice floe cases of manuscript Section 4.1 (SeaIce_Floe_Params.py).

Run from simulations/ in the FenicsX environment:
    python Run_Floe_Cases.py --list-cases
    python Run_Floe_Cases.py --dry-run
    python Run_Floe_Cases.py --case-ids 0 --ranks 52
    python Run_Floe_Cases.py --case-ids 1-3 --ranks 8

Defaults: one case at a time on 56 MPI ranks (one thread each).

Cases (upper half of the floe, plane stress, Gc = 5 J/m^2, ell = 0.25 m, settings of
SeaIce_Floe_Params.build_floe_params):
  0  fast          1.0e-8 /s: fragmentation (Figs. 12b, 13, 14). In the paper run on
     52 ranks and stopped by hand after about 68 h, before the cracks reached the
     floe edge.
  1  intermediate  2.5e-9 /s (Fig. 13)
  2  slow          1.0e-9 /s (Fig. 13); does not fracture.
  3  fast          1.0e-8 /s, creep switched off: the elastic reference (Fig. 13)
  Cases 1 and 3 fracture right after nucleation; in the paper they were stopped
  by hand once the crack was 1 m long, as only the response up to nucleation is
  used. Cases 1-3 have no long dynamic stage; 8 ranks each are enough.

Output folders: results/Results_Floe/<rate>_l0.25_sym (suffix _elastic for the case
without creep). A case is complete once simulation_time.txt exists; complete cases
are skipped, so the driver can be restarted after an interruption. An incomplete
folder from an earlier attempt is renamed to <case>.incomplete-<timestamp>. Each case
writes local_run.log, case_settings.json (including the FEMCubes version), the Data/
HDF5 outputs, Movies/ and the Monitor/ snapshots (PNGs + status.txt every 10 steps or
10 min).
"""

import argparse
from collections import deque
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from Run_Plate_Sweeps import (COMPLETION_FILE, _femcubes_commit, _positive_int,
                                   _stop_running, _worker_environment)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_ROOT = "Results_Floe"
# Copy of SeaIce_Floe_Params.RATES so that the supervisor never imports FEMCubes (and
# MPI); the worker checks it against the module.
RATES = {"slow": 1.0e-9, "intermediate": 2.5e-9, "fast": 1.0e-8}
# The cases of the paper: (rate name, rheology).
CASES = (("fast", "viscoelastic"), ("intermediate", "viscoelastic"), ("slow", "viscoelastic"),
         ("fast", "elastic"))
DEFAULT_RANKS = 56


def build_floe_cases(output_root):
    cases = []
    for rate_name, rheology in CASES:
        name = f"{RESULTS_ROOT}/{rate_name}_l0.25_sym" + ("_elastic" if rheology == "elastic" else "")
        cases.append({
            "name": name,
            "output_folder": Path(output_root) / name,
            "parameters": {"rate_name": rate_name, "strain_rate": RATES[rate_name], "rheology": rheology},
        })
    return cases


def run_floe_case(OutputFolder, rate_name, strain_rate, rheology):
    """Run one floe simulation (MPI worker only)."""
    from femcubes import Mesh, Physics, TimeSolver
    from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank
    import femcubes.boundary  # noqa: F401
    import femcubes.glaciology  # noqa: F401
    import femcubes.solids  # noqa: F401

    import SeaIce_Floe_Params as floe
    from SeaIce_Floe_Params import build_floe_params, install_plot_snapshots, summarize

    if floe.RATES != RATES:
        raise RuntimeError("Run_Floe_Cases.RATES differs from SeaIce_Floe_Params.RATES; "
                           "update the copy in the driver.")
    out = str(OutputFolder)
    Path(out).mkdir(parents=True, exist_ok=True)
    isolate_jit_cache()
    start = time.time()

    params = build_floe_params(strain_rate=strain_rate, OutputFolder=out, rheology=rheology)
    mprint(f"Floe case {rate_name}, {rheology}: strain rate {strain_rate:.2e} /s, ell = {floe.ELL} m, "
           f"Gc = {floe.GC} J/m^2")
    mprint(summarize(params))

    mesh = Mesh(params)
    mesh.plot_mesh()
    n_elems_before = mesh.mesh.topology.index_map(2).size_global
    install_plot_snapshots(mesh, params, out + "/Monitor", every_steps=10, every_seconds=600.0)

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


def _parse_case_ids(text, n_cases):
    ids = []
    for part in text.split(","):
        start, _, stop = part.partition("-")
        ids.extend(range(int(start), int(stop or start) + 1))
    if any(not 0 <= i < n_cases for i in ids):
        raise argparse.ArgumentTypeError(f"case IDs must be between 0 and {n_cases - 1}")
    return sorted(set(ids))


def _worker_command(launcher, ranks, task_id, output_root):
    return [
        launcher, "-n", str(ranks), sys.executable, "-u", str(Path(__file__).resolve()),
        "--task-id", str(task_id), "--output-root", str(output_root),
    ]


def _prepare_output_folder(case, task_id, commit, launcher, ranks):
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
        "parameters": case["parameters"],
        "femcubes_commit": commit,
        "mpi_launcher": launcher,
        "mpi_ranks": ranks,
        "python": sys.executable,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "case_settings.json").write_text(json.dumps(settings, indent=2) + "\n")


def run_cases(cases, task_ids, launcher, jobs, ranks, output_root):
    commit = _femcubes_commit()
    if commit is None or commit.endswith("-dirty"):
        print(f"WARNING: FEMCubes commit is {commit}; results may not be reproducible.",
              flush=True)
    pending = deque((i, cases[i]) for i in task_ids)
    running = []
    failed = []
    sweep_start = time.time()
    print(f"Running {len(task_ids)} floe cases: up to {jobs} at once, {ranks} MPI ranks "
          f"each. FEMCubes commit: {commit}", flush=True)
    try:
        while pending or running:
            while pending and len(running) < jobs:
                task_id, case = pending.popleft()
                _prepare_output_folder(case, task_id, commit, launcher, ranks)
                output_folder = case["output_folder"]
                env = _worker_environment(output_folder)
                log_path = output_folder / "local_run.log"
                log = log_path.open("w")
                command = _worker_command(launcher, ranks, task_id, output_root)
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
                print(f"Started  [{task_id}] {case['name']} at "
                      f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

            for job in running[:]:
                return_code = job["process"].poll()
                if return_code is None:
                    continue
                job["log"].close()
                running.remove(job)
                hours = (time.time() - job["start"]) / 3600
                complete = (job["case"]["output_folder"] / COMPLETION_FILE).is_file()
                if return_code or not complete:
                    failed.append(job)
                    print(f"FAILED   [{job['task_id']}] {job['case']['name']} "
                          f"(exit {return_code}, {hours:.2f} h); see {job['log_path']}",
                          flush=True)
                else:
                    print(f"Finished [{job['task_id']}] {job['case']['name']} "
                          f"({hours:.2f} h)", flush=True)
            if running:
                time.sleep(5.0)
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
    parser.add_argument("--jobs", type=_positive_int, default=1,
                        help="maximum concurrent cases (default: 1)")
    parser.add_argument("--ranks", type=_positive_int, default=DEFAULT_RANKS,
                        help=f"MPI ranks per case, one thread each (default: {DEFAULT_RANKS})")
    parser.add_argument("--mpi-launcher", default="mpirun",
                        help="MPI launcher executable or path (default: mpirun)")
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIR,
                        help="results directory (default: results/ in the repository root)")
    parser.add_argument("--case-ids", help="comma-separated IDs or ranges, e.g. 0-2")
    parser.add_argument("--list-cases", action="store_true",
                        help="list case IDs, folders and status without running")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the MPI commands that would be launched")
    parser.add_argument("--task-id", type=int, help=argparse.SUPPRESS)  # MPI workers
    args = parser.parse_args(argv)

    output_root = args.output_root.expanduser().resolve()
    cases = build_floe_cases(output_root)

    if args.task_id is not None:
        # Only MPI workers import FEMCubes, so the supervisor never initializes MPI.
        case = cases[args.task_id]
        run_floe_case(OutputFolder=case["output_folder"], **case["parameters"])
        return 0

    selected = (list(range(len(cases))) if args.case_ids is None
                else _parse_case_ids(args.case_ids, len(cases)))

    def is_complete(i):
        return (cases[i]["output_folder"] / COMPLETION_FILE).is_file()

    if args.list_cases:
        for i in selected:
            status = "done" if is_complete(i) else (
                "incomplete" if cases[i]["output_folder"].exists() else "pending")
            p = cases[i]["parameters"]
            print(f"{i}\t{status}\t{cases[i]['name']}\trate {p['strain_rate']:.1e} /s, {p['rheology']}")
        return 0

    to_run = [i for i in selected if not is_complete(i)]
    skipped = len(selected) - len(to_run)
    if skipped:
        print(f"Skipping {skipped} completed case(s).", flush=True)
    if args.dry_run:
        print(f"Up to {args.jobs} concurrent cases x {args.ranks} MPI ranks each.")
        for i in to_run:
            print(shlex.join(_worker_command(args.mpi_launcher, args.ranks, i, output_root)))
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
    return run_cases(cases, to_run, launcher, args.jobs, args.ranks, output_root)


if __name__ == "__main__":
    raise SystemExit(main())
