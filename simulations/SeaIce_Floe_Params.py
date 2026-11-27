"""Parameters of the sea-ice floe case (manuscript Section 4.1, fig:seaice_*).

A square floe of FLOE_SIDE x FLOE_SIDE m in plane stress with a circular
defect of radius DEFECT_RADIUS at its centre. A horizontal far-field strain
rate is imposed by pulling the left and right edges apart at equal and
opposite velocities (penalty displacement control, which also returns the
reaction force); the top and bottom edges are free. The material, failure
surface, creep law, phase-field settings, Gc element-size correction, solver
and time-step control follow Section 3 (Small_Reference_Params.py and
Run_Plate_Sweeps.py); the differences are listed under "Section 4.1
choices" below.

Only the upper half (y >= FLOE_SIDE/2) is simulated (Upper_Half_Plate_Mesh.py):
u_y = 0 on the horizontal centreline, the phase field is left free there, and
the edge reactions are those of half the floe.

Mesh: a coarse far-field mesh (FAR_FIELD_DX), pre-refined to the adaptive
target size ell/ELL_OVER_H inside PREREFINE_RADIUS of the defect and graded
back to FAR_FIELD_DX by 2*PREREFINE_RADIUS, with the same etaNorm-triggered
adaptive refinement as the Section 3 length-scale study taking over along the
cracks.

Usage from a driver (MPI worker), see Run_Floe_Cases.py:

    from SeaIce_Floe_Params import build_floe_params, install_plot_snapshots, RATES
    params = build_floe_params(strain_rate=RATES["fast"], OutputFolder=folder)
    mesh = Mesh(params); mesh.plot_mesh()
    install_plot_snapshots(mesh, params, folder + "/Monitor")   # PNGs + status.txt during the run
    ...
    mesh.SavePlotsToFolder(folder); mesh.Close_Plots()

Monitoring while a case runs: <case>/Monitor/{mesh,phasefield,u_x,u_y}.png, the
load-displacement, load-time and crack-length plots, and status.txt (step, t,
dt, reaction force, strain, crack length, Eta_Max, cell count), refreshed every
10 steps or 10 minutes. The MP4 movies in <case>/Movies are finalised at the end.
"""

from __future__ import annotations

from femcubes import LinearSolver, ParamsTemplate
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.mpi_utils import mprint

import Upper_Half_Plate_Mesh  # noqa: F401  (registers MeshType UpperHalfPlateWithHole)

# ----------------------------------------------------------------------------
# Geometry (keep in step with Plot_Geometry_Figures.py: FLOE_SIDE, FLOE_DEFECT_RADIUS)
# ----------------------------------------------------------------------------
FLOE_SIDE = 100.0          # m, square floe
DEFECT_RADIUS = 0.5        # m, circular defect at the centre (user decision 6 Nov 2026)
THICKNESS = 1.0            # m; plane stress, enters only the reported force per unit width

# ----------------------------------------------------------------------------
# Section 4.1 choices
# ----------------------------------------------------------------------------
# Far-field strain rates [1/s]. Section 3 used 1.0, 1.5 and 2.5e-9 /s for the same
# material and the same stress concentration (circular hole, factor 3), where 1e-9 /s
# reached a creep plateau without fracture. The fast rate is a decade above the
# Section 3 range so that creep is negligible before fracture.
RATES = {"slow": 1.0e-9, "intermediate": 2.5e-9, "fast": 1.0e-8}
# Regularisation length [m]: ell = 0.25 m (<= DEFECT_RADIUS/2 and just below the cohesive
# length ell_phys = E Gc/f_t^2 = 0.3 m at GC = 5).
ELL = 0.25
ELL_OVER_H = 4             # adaptive target ell/h^el (as the Section 3 length-scale study)
# Fracture energy [J/m^2]. Gc = 1 (K_Ic = 0.1 MPa m^1/2) drove the crack so far beyond the
# Griffith condition (critical half-length 0.13 m at the nucleation stress, smaller than the
# defect) that the damage band widened to several ell and split continuously ("palm tree",
# run of 11-12 Nov 2026, kept in Results_Floe/fast_l0.25_Gc1_abandoned). Gc = 10
# (K_Ic = 0.31 MPa m^1/2, upper range of sea-ice measurements) keeps the crack stable until
# ~0.22 MPa nominal (user decision 12 Nov 2026). The Gc = 10 fast case ran a single
# straight crack to ~8 m from the defect before its first (symmetric) split (kept in
# Results_Floe/fast_l0.25_Gc10_late-split): too late to be ice-like, so Gc = 5
# (K_Ic = 0.22 MPa m^1/2) is used instead (user decision 12 Nov 2026, evening).
GC = 5.0
GC_CORRECTION = 0.33       # coefficient c of Gc/(1 + c h/ell) (Section 3, Appendix B)
# Bulk damping c_d [Pa s/m^2] (body force -c_d v). Section 3 used 1e9 to suppress
# branching. Here it only has to damp the stress waves after fragmentation:
# decay time rho/c_d ~ 0.1 s, i.e. ~3 transits of a P wave across the floe
# (user choice 10 Nov 2026); the drag at the loading velocities (< 1e-6 m/s)
# is < 1e-2 Pa and irrelevant.
DAMPING = 1.0e4
# ----------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------
# Far-field element size. The refinement bisects every edge longer than target_h, so
# the refined size is (parent edge)/2^k <= target_h. With 2.0 m the unstructured
# far-field edges (1.8-2.4 m) straddled 32 x target_h: 28% of the parents needed a
# sixth bisection and gave ell/8 elements next to ell/4 ones (run of 12-13 Nov 2026,
# kept as Results_Floe/fast_l0.25_Gc5_dx2_buffer8_multipass). With 3.0 m all parents
# (2.6-3.8 m) take exactly six bisections: refined longest edges ell/4.2-ell/6.2,
# median ell/5.3, no ell/8 elements (user asked for 2.5 m, which gives ell/5.4-ell/7.6,
# median ell/6.4; 3.0 m chosen on 13 Nov 2026 to stay closer to ell/4).
FAR_FIELD_DX = 3.0         # m, initial element size away from the defect
PREREFINE_RADIUS = 1.0     # m from the defect boundary meshed at ell/ELL_OVER_H, graded to FAR_FIELD_DX by 2x
# Adaptive refinement: FAR_FIELD_DX / target_h = 48 at ELL = 0.25 m -> 6 bisection levels.
# Crack-length change allowed per step (time-step control): CRACK_STEP_FACTOR target
# elements of total crack-length measure (2 ell at 8), halved for the upper-half model,
# which carries half the cracks. The buffer is 4 layers (ell); the cohesive fracture
# strain that triggers refinement activates ahead of the phase field, so the indicator
# itself already leads the damage.
CRACK_STEP_FACTOR = 8
BUFFER_LAYERS = 4
# Every refinement event bisects the marked cells once and re-solves the increment; one
# event per step suffices: cells entering the buffer gain a level every step and reach
# target_h before the eigenstrain does (6 levels in < 0.25 m of advance).
MAX_LEVELS_PER_STEP = 1

# ----------------------------------------------------------------------------
# Time stepping
# ----------------------------------------------------------------------------
END_STRAIN = 1.0e-4        # far-field strain at which a non-fracturing run ends (Section 3: 0.1 mm / 1 m)
STRAIN_PER_STEP = 3.0e-7   # dt_max = STRAIN_PER_STEP / strain_rate (Section 3: dt = 3e-7/rate)
DT_MIN = 1.0e-6            # s, floor of the time-step control (see the TimeStepControl comment)
F_CUTOFF = 0.1             # end 50 steps after the reaction force drops below this fraction of its peak
DUMMY = 1.0e12             # penalty stiffness of the displacement control [Pa/m]


def build_floe_params(strain_rate: float, OutputFolder: str, rheology: str = "viscoelastic"):
    """Return the FEMCubes parameter object of one floe simulation.

    ``strain_rate`` is the far-field strain rate (1/s): the right edge moves at
    +strain_rate*FLOE_SIDE/2 and the left edge at -strain_rate*FLOE_SIDE/2.
    ``rheology`` is "viscoelastic" (Glen creep) or "elastic" (the same model
    with the creep switched off), the reference for the effect of creep.
    """
    if rheology not in ("viscoelastic", "elastic"):
        raise ValueError("rheology must be 'viscoelastic' or 'elastic'")
    strain_rate = abs(float(strain_rate))
    target_h = ELL / ELL_OVER_H
    edge_velocity = 0.5 * strain_rate * FLOE_SIDE
    L = FLOE_SIDE

    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        Quadrature_Degree: int = 3

        ## Mesh: upper half of the plate with a central hole, pre-refined around the hole
        MeshType: str = "UpperHalfPlateWithHole"
        dim: int = 2
        Lx: float = L
        Ly: float = L
        R: float = DEFECT_RADIUS
        dx: float = FAR_FIELD_DX
        dxmin: float = target_h
        R_Refine: float = PREREFINE_RADIUS

        MeshRefinement = RefinementConfig(
            enabled=True,
            indicator=FieldThresholdIndicatorConfig(
                field="etaNorm",
                threshold=1e-12,
                reduction="cell_max",
                comparison="greater_equal",
            ),
            target_h=target_h,
            buffer_layers=BUFFER_LAYERS,
            check_interval=1,
            max_levels_per_step=MAX_LEVELS_PER_STEP,
            max_events=int(1e99),
            repartition=True,
            repartition_imbalance_threshold=1.1,
            replot_on_refinement=True,
            save_remesh_animations=True,
        )

        ## Material (Section 3, tab:parameters), plane stress with 3-D elastic constants
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,           # kg/m^3
            "Youngs": 9.5e9,            # Pa
            "Poisson": 0.35,
            "Visc_Type": "Glen" if rheology == "viscoelastic" else "None",
            "Glen_n": 3.0,
            "Glen_A": 1e-24,            # Pa^-n s^-1
            "ell": ELL,                 # m
            "f_t": 0.4e6,               # Pa
            "f_s": 0.6e6,               # Pa
            "Gc": GC,                   # J/m^2
            "PF_formulation": "AT2",
            "FailureType": "r2_2L",
            "Inertia": True,
            "kappa": 1.0e-3,
            "LE_Damping": DAMPING,      # Pa s/m^2
        }
        # The plane-stress kernel condenses the 3-D law (sigma_zz = 0 solved locally),
        # so the 3-D constants are passed, unlike the intrinsic-2D plate of Section 3.
        _E, _nu = Materials["Ice"]["Youngs"], Materials["Ice"]["Poisson"]
        Materials["Ice"]["Shear"] = _E / (2.0 * (1.0 + _nu))
        Materials["Ice"]["Bulk"] = _E / (3.0 * (1.0 - 2.0 * _nu))
        Materials["Ice"]["Lame"] = _E * _nu / ((1.0 + _nu) * (1.0 - 2.0 * _nu))

        gravity: float = 0.0            # horizontal floe, in-plane problem

        ## Monitoring plots (off-screen PyVista windows on rank 0, updated every step).
        # Field plots are recorded to one MP4 each in movie_folder; the latest frame of
        # every plot (fields, mesh, measures) is also saved as PNG by install_plot_snapshots.
        Plot_Fields: list = ["phasefield", "u_x", "u_y"]
        Plot_Measures: list = [("BC_Right_u", "BC_Right_F"), ("t", "BC_Right_F"), ("t", "Crack_Length")]
        record_movie: bool = True
        movie_fps: int = 10
        movie_folder: str = OutputFolder + "/Movies"

        ## Fields and solution steps (staggered, Newton for both as in the Section 3 sweeps)
        Solution_Steps = {"u": 0, "phasefield": 1}
        Step_Solvers = {0: "Newton", 1: "Newton"}
        Field_Orders = {"u": 2, "phasefield": 1}

        Models = {}
        ModelNames = ["Solid", "pf", "BC_Right", "BC_Left"]

        Models["Solid"] = {
            "Type": "Solid:Cohesive_ViscoElastic_Kernel",
            "Material": "Ice",
            "plane_stress": True,
            "true_2d": False,
            "consistent_stiffness": True,
        }
        Models["pf"] = {
            "Type": "Solid:PhaseField",
            "Material": "Ice",
            "CDF_field": "PsiField",
            "Irreversible_Method": "Hist",
        }
        Models["pf"]["Gc_Mesh_Correction"] = {
            "Coefficient": GC_CORRECTION,
            "Source_Field": "etaNorm",
            "Source_Threshold": 1.0e-8,
            "Search_Radius": 2.0,
            "Reduction": "mean",
        }

        # Far-field strain rate: equal and opposite edge velocities. The reaction force
        # BC_Right_F divided by (FLOE_SIDE * THICKNESS) is the nominal far-field stress.
        Models["BC_Right"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "right",
            "field": "u_x",
            "rate": edge_velocity,
            "Dummy": DUMMY,
            "F_Cutoff": F_CUTOFF,
        }
        Models["BC_Left"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "left",
            "field": "u_x",
            "rate": -edge_velocity,
            "Dummy": DUMMY,
        }

        # Symmetry about the horizontal centreline; also removes the vertical rigid-body mode.
        ModelNames.append("BC_Sym_y")
        Models["BC_Sym_y"] = {"Type": "Boundary:PrescribedValue", "boundary": "symmetry",
                              "field": "u_y", "value": 0.0}

        ## Nonlinear and linear solvers (Section 3 sweep settings)
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes_force: bool = False
        max_it: int = 20
        # One staggered pass per step since 13 Nov 2026 (user): u then phase field once.
        # In the dynamic stage each tip moves less than an element per step, so the
        # one-step lag of the phase field is below the mesh resolution; the multi-pass
        # Gc = 5 run (Results_Floe/fast_l0.25_Gc5_dx2_buffer8_multipass) is the reference.
        # The staggered divergence/stagnation aborts below need > 1 pass and are inactive.
        n_passes: int = 1
        # Abort a staggered iteration whose post-pass residual grows for three passes in
        # a row (factor 1.5), so that the time step is reset at once. At nucleation on
        # the floe (10 Nov 2026 checks) the staggered loop otherwise deteriorated over
        # seven passes before the inner Newton failed, with the return map then
        # grinding through its iteration cap at every affected integration point.
        staggered_divergence_passes: int = 3
        staggered_divergence_factor: float = 1.5
        # 12 Nov 2026 (Gc = 10 run): at the Griffith instability of the hole crack the
        # quasi-static step has no solution until dt is at the microsecond scale, and
        # each failed attempt cost ~15 min on 28 ranks: the staggered residual stagnated
        # (no growth, so the check above never fired) while Newton walked uphill for 20
        # iterations with non-finite trial residuals at the full step. These opt-in
        # FEMCubes checks abort such an attempt within a few iterations and let the
        # time step fall by a decade per further failure.
        staggered_stagnation_passes: int = 3
        staggered_stagnation_factor: float = 0.9
        line_search_nonfinite_abort: int = 2
        reset_dt_factor: float = 4.0
        reset_dt_factor_repeated: float = 10.0
        line_search_lims: list = [1e-1, 1.0]
        line_search: bool = True
        rel_tol: float = 1e-5      # 1e-6 until 13 Nov 2026 (user: slightly more tolerant)
        abs_tol: float = 0.001 * L   # Section 3 used 1e-3 N on a 1 m edge; scaled with the edge length

        ## Output
        output_dir: str = OutputFolder + "/Data"
        output_interval: int = 10
        write_vtk: bool = False
        write_hdf5: bool = True
        save_Mesh_every_step: bool = False
        output_fields: list = ["u_x", "u_y", "phasefield", "sigma_xx", "sigma_yy", "sigma_xy", "Gc_eff"]

        ## Time: quasi-static loading with dt_max, collapsing during crack growth
        start_time: float = 0.0
        end_time: float = END_STRAIN / strain_rate
        dt: float = STRAIN_PER_STEP / strain_rate
        n_damp = 0.5
        Newmark_Beta: float = 0.25 * (1 + n_damp) ** 2
        Newmark_Gamma: float = 0.5 + n_damp
        TimeStepControl = {
            "Type": "Measure",
            "time_list": [-1e99, 0],
            "dt_list": [60, STRAIN_PER_STEP / strain_rate],
            # dt_min: at nucleation the phase field around the defect jumps to its new
            # equilibrium within one step whatever dt is (crack-length measure change
            # ~0.15 m at ell = 0.25 m, rate independent), so the crack-length control
            # cannot satisfy its target by halving dt and would spin down to the floor.
            # 1e-6 s is a decade below the propagation time step ell/4 / (km/s) and ends
            # that spin-down after a few halvings. (Section 3 used 1e-14.)
            "dt_min": DT_MIN,
            "dt_max": STRAIN_PER_STEP / strain_rate,
            "Measure": "Crack_Length_Change",
            # see CRACK_STEP_FACTOR (Section 3: 0.0025 m); half the cracks in the upper-half model
            "Target_Change": 0.5 * CRACK_STEP_FACTOR * target_h,
        }

    params = Params()
    mprint(f"Floe case: strain rate {strain_rate:.2e} /s, edge velocity +/-{edge_velocity:.3e} m/s, "
           f"{rheology}, ell = {ELL} m (target h = {target_h} m), Gc = {GC} J/m^2, damping {DAMPING:.1e} Pa s/m^2, "
           f"upper half (symmetry), dt_max = {params.dt:.1f} s, end_time = {params.end_time:.3e} s "
           f"({END_STRAIN:.0e} strain)")
    return params


def install_plot_snapshots(mesh, params, folder: str, every_steps: int = 10,
                           every_seconds: float = 600.0, edge_length: float = 0.5 * FLOE_SIDE) -> None:
    """Save all open plots as PNG during the run, for monitoring a long case.

    FEMCubes saves plot windows only when the driver calls ``SavePlotsToFolder``
    at the end, and the MP4 movies are not readable before they are finalised.
    This wraps ``mesh.Plot_Measures`` (the last plotting call of every time step
    in ``TimeSolver.run``) so that every ``every_steps`` steps, or whenever
    ``every_seconds`` have passed, the mesh, field and measure plots are written
    to ``folder`` (overwriting the previous snapshot) together with
    ``status.txt`` holding the current step, time, time step, reaction force,
    edge displacement, crack length and Eta_Max. Call on every rank after the
    mesh exists; only rank 0 writes files. ``edge_length`` is the length of the
    loaded edge in the model (FLOE_SIDE/2 for the upper half), used for the
    nominal stress.
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
            measures = params.Global_Measures
            lines = [
                f"updated {time.strftime('%Y-%m-%d %H:%M:%S')} after {(now - state['start']) / 3600:.2f} h wall time",
                f"step {state['step']}, t = {params.t:.6e} s, dt = {params.dt:.6e} s, end_time = {params.end_time:.6e} s",
                f"BC_Right_u = {measures.get('BC_Right_u', float('nan')):.6e} m, "
                f"far-field strain = {2.0 * measures.get('BC_Right_u', float('nan')) / FLOE_SIDE:.6e}",
                f"BC_Right_F = {measures.get('BC_Right_F', float('nan')):.6e} N "
                f"(peak {measures.get('BC_Right_F_max', float('nan')):.6e} N), nominal stress "
                f"{measures.get('BC_Right_F', float('nan')) / (edge_length * THICKNESS):.6e} Pa",
                f"Crack_Length = {measures.get('Crack_Length', float('nan')):.6e} m, "
                f"Eta_Max = {measures.get('Eta_Max', float('nan')):.6e}, "
                f"Eta_Active_Area = {measures.get('Eta_Active_Area', float('nan')):.6e} m^2",
                f"cells = {mesh.mesh.topology.index_map(2).size_global}, remesh epoch {getattr(mesh, 'remesh_epoch', 0)}",
            ]
            with open(os.path.join(folder, "status.txt"), "w") as handle:
                handle.write("\n".join(lines) + "\n")

    mesh.Plot_Measures = plot_measures_and_snapshot


def summarize(params) -> str:
    """One-screen summary of the settings of a case."""
    ice = params.Materials["Ice"]
    ell_phys = ice["Youngs"] * ice["Gc"] / ice["f_t"] ** 2
    lines = [
        f"domain {params.Lx} x {params.Ly} m (upper half simulated), defect radius {params.R} m, plane stress, thickness {THICKNESS} m",
        f"ell = {ice['ell']} m (ell_phys = {ell_phys:.3f} m, floe = {params.Lx / ice['ell']:.0f} ell), "
        f"Gc = {ice['Gc']} J/m^2, correction {params.Models['pf']['Gc_Mesh_Correction']['Coefficient']}",
        f"mesh: far field {params.dx} m, {params.dxmin} m within {params.R_Refine} m of the defect; "
        f"adaptive target {params.MeshRefinement.target_h} m, buffer {params.MeshRefinement.buffer_layers}, "
        f"<= {params.MeshRefinement.max_levels_per_step} levels/step",
        f"loading: edges at +/-{params.Models['BC_Right']['rate']:.3e} m/s, models {params.ModelNames}",
        f"damping {ice['LE_Damping']:.1e} Pa s/m^2, Newmark beta/gamma {params.Newmark_Beta}/{params.Newmark_Gamma}",
        f"dt_max {params.dt:.1f} s, end_time {params.end_time:.3e} s, crack advance/step <= "
        f"{params.TimeStepControl['Target_Change']} m",
    ]
    return "\n".join(lines)
