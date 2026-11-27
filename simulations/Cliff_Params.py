"""Parameters of the ice-cliff case (manuscript Section 4.2).

A grounded, land-terminating ice cliff of ``length`` x CLIFF_HEIGHT m, loaded by
its own weight and by water standing against its face. The far-field (left) edge
is on rollers, the bed is free-slip (u_y = 0), the surface is traction free.

The numerics are those of Section 3 and of the floe case (SeaIce_Floe_Params.py):
Solid:Cohesive_ViscoElastic (reduced return mapping) with the r2_2L strength
surface or the opening-only criterion r2_2L_PSD, AT2 with a linear phase field,
history irreversibility, kappa = 1e-3, the mesh correction of Gc, Newmark with
n_damp = 0.5, and adaptive refinement to ell/4 where the fracture eigenstrain is
active. The material is that of tab:parameters except for the compressive strength
f_s (see F_S below). The cliff is in plane strain (three-dimensional law and
constants).

Loading. Gravity acts in full from the start. During a spin-up of SPIN_UP_TIME
(t < 0) water stands against the face at ``water_level`` (flotation by default),
so that the prior state is that of a buttressed cliff with or without creep. From
t = 0 the level is lowered at a constant rate to zero, reached at t = LOWERING_TIME
(glaciology:LoweringOceanPressure of FEMCubes), so that the strength surface is reached
progressively. The
phase field is driven for t > 0 only.

Surface notches (``notch_spacing`` > 0, FEMCubes mesh type Rectangle_Notches).
Away from the face the surface stress of a cliff on a free-slip bed does not
vary along the cliff, so the surface reaches the strength surface everywhere at
once and degrades as a uniform layer. Rectangular notches of depth NOTCH_DEPTH
and width ``notch_width``, ``notch_spacing`` apart, stand for pre-existing
crevasses and give the cracks a place to start. The first notch lies half a
spacing from the left end: the left end acts as a symmetry plane, so that every
notch has neighbours at one spacing.

Usage from the MPI worker, see Run_Cliff.py.
"""

from __future__ import annotations

from femcubes import LinearSolver, ParamsTemplate
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig

import femcubes.glaciology  # noqa: F401  (registers glaciology:LoweringOceanPressure)

import Cliff_Measures  # noqa: F401  (registers Measure:PointDisplacement, Measure:QuadratureNormMax)

# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
CLIFF_LENGTH = 1000.0      # m
CLIFF_HEIGHT = 200.0       # m

# ----------------------------------------------------------------------------
# Material and regularisation (tab:parameters; ell/ell_phys as on the floe)
# ----------------------------------------------------------------------------
F_T = 0.4e6                # Pa, uniaxial tensile strength (tab:parameters)
# f_s is the uniaxial compressive strength of the r2_2L surface, which has no pressure
# strengthening. With the 0.6 MPa of tab:parameters the laterally confined elastic state
# under 200 m of ice is already beyond the surface below 146 m depth; 1-2 MPa is the usual
# range for ice cliffs. In plane strain an exposed column (sigma_xx = 0) reaches the
# surface at a depth of 0.879 rho g d = f_s: 127 m for 1 MPa.
F_S = 1.0e6                # Pa
# "r2_2L" is the two-multiplier strength surface of Section 2 of the paper, "r2_2L_PSD" the
# opening-only criterion of Section 2.1.4 (fig:cliff_criteria, fig:cliff_final).
FAILURE_TYPES = ("r2_2L", "r2_2L_PSD")
GC = 10.0                  # J/m^2; E Gc/f_t^2 = 0.59 m
ELL = 1.0                  # m; ell / (E Gc/f_t^2) = 1.7, the largest ratio of Section 3
ELL_OVER_H = 4             # adaptive target ell/h^el
GC_CORRECTION = 0.33       # coefficient c of Gc/(1 + c h/ell) (Section 3, Appendix B)
DAMPING = 1.0e4            # Pa s/m^2, as the floe
GRAVITY = -9.81            # m/s^2
WATER_DENSITY = 1025.0     # kg/m^3, sea water
LOWERING_TIME = 3600.0     # s over which the water level is lowered to zero

# ----------------------------------------------------------------------------
# Mesh, loading and output
# ----------------------------------------------------------------------------
FAR_FIELD_DX = 25.0        # m, initial element size (8 quadratic elements over the height; the refinement does the rest)
# Refinement where the fracture eigenstrain exceeds REFINE_THRESHOLD. The reduced return
# mapping is exact below the strength surface (etaNorm = 0 until yield), so the mesh is
# refined from the first yield on.
REFINE_THRESHOLD = 3.0e-12
NOTCH_DEPTH = 10.0         # m, depth of the surface notches (when notch_spacing > 0)
NOTCH_WIDTH = 2.0          # m, width of the notches at the surface
BUFFER_LAYERS = 4
MAX_LEVELS_PER_STEP = 6
CRACK_STEP_FACTOR = 8      # crack advance per step <= CRACK_STEP_FACTOR * target_h
SPIN_UP_TIME = 100 * 3600.0   # s, creep under gravity with the water against the face (t < 0)
SPIN_UP_DT = 3600.0           # s
END_TIME = 3600.0             # s after t = 0
RELEASE_DT = 1.0              # s, first time step after t = 0
RELEASE_WINDOW = 10.0         # s over which the time step stays at RELEASE_DT
DT_MAX = 30.0                 # s, time step afterwards (creep relaxation time at 1 MPa: ~100 s)
DT_MIN = 1.0e-6            # s
OUTPUT_INTERVAL = 10       # time steps between field outputs

RHEOLOGIES = ("viscoelastic", "elastic")


def build_cliff_params(OutputFolder: str, rheology: str = "viscoelastic", water_level: float | None = None,
                       length: float = CLIFF_LENGTH, ell: float = ELL, f_s: float = F_S,
                       failure_type: str = "r2_2L", damping: float = DAMPING, end_time: float = END_TIME,
                       stagnation_passes: int = 3, crack_step_factor: float = CRACK_STEP_FACTOR,
                       notch_spacing: float = 0.0, notch_width: float = NOTCH_WIDTH):
    """Return the FEMCubes parameter object of one cliff simulation.

    ``rheology`` is "viscoelastic" (Glen creep) or "elastic" (the same model with
    the creep switched off). Water stands against the face at ``water_level``
    (flotation if None) up to t = 0 and is lowered to zero over LOWERING_TIME.
    ``notch_spacing`` > 0 cuts notches of NOTCH_DEPTH by ``notch_width`` into the
    upper surface at that spacing, the first half a spacing from the left end and
    none within half a spacing of the face.

    ``stagnation_passes`` = 0 switches off the rejection of a step whose staggered
    iteration converges slowly (the step then keeps its iterate after ``n_passes``
    passes); the crack advance per step is then bounded by ``crack_step_factor``
    times the target element size only.
    """
    if stagnation_passes < 0 or crack_step_factor <= 0.0:
        raise ValueError("stagnation_passes must be >= 0 and crack_step_factor positive")
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"failure_type must be one of {FAILURE_TYPES}")
    if rheology not in RHEOLOGIES:
        raise ValueError(f"rheology must be one of {RHEOLOGIES}")
    height = CLIFF_HEIGHT
    if water_level is None:
        water_level = 910.0 / WATER_DENSITY * height      # flotation

    notch_positions = []
    if notch_spacing > 0.0:
        x_notch = 0.5 * notch_spacing
        while x_notch < length - 0.5 * notch_spacing + 1.0e-9:
            notch_positions.append(x_notch)
            x_notch += notch_spacing

    target_h = ell / ELL_OVER_H
    # dt targets: spin-up, the first steps after t = 0, afterwards. The table entry at
    # t = 0 also sets dt there.
    time_list = [-1e99, 0.0, RELEASE_WINDOW]
    dt_list = [SPIN_UP_DT, RELEASE_DT, DT_MAX]

    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        Quadrature_Degree: int = 3

        ## Mesh: unstructured rectangle, refined adaptively where the eigenstrain is active
        MeshType: str = "Rectangle_Notches" if notch_positions else "Rectangle"
        dim: int = 2
        Lx: float = length
        Ly: float = height
        dx: float = FAR_FIELD_DX
        Notch_Positions: tuple = tuple(notch_positions)
        Notch_Depth: float = NOTCH_DEPTH
        Notch_Width: float = notch_width

        MeshRefinement = RefinementConfig(
            enabled=True,
            indicator=FieldThresholdIndicatorConfig(
                field="etaNorm",
                threshold=REFINE_THRESHOLD,
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

        ## Material (Section 3, tab:parameters, except f_s), plane strain with the 3-D constants
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,           # kg/m^3
            "Youngs": 9.5e9,            # Pa
            "Poisson": 0.35,
            "Visc_Type": "Glen" if rheology == "viscoelastic" else "None",
            "Glen_n": 3.0,
            "Glen_A": 1e-24,            # Pa^-n s^-1
            "ell": ell,                 # m
            "f_t": F_T,                 # Pa
            "f_s": f_s,                 # Pa
            "Gc": GC,                   # J/m^2
            "PF_formulation": "AT2",
            "FailureType": failure_type,
            "Inertia": True,
            "kappa": 1.0e-3,
            "LE_Damping": damping,      # Pa s/m^2
        }
        _E, _nu = Materials["Ice"]["Youngs"], Materials["Ice"]["Poisson"]
        Materials["Ice"]["Shear"] = _E / (2.0 * (1.0 + _nu))
        Materials["Ice"]["Bulk"] = _E / (3.0 * (1.0 - 2.0 * _nu))
        Materials["Ice"]["Lame"] = _E * _nu / ((1.0 + _nu) * (1.0 - 2.0 * _nu))
        Materials["Water"] = {
            "Density": WATER_DENSITY,       # kg/m^3
            "WaterHeight": water_level,     # m above the bed, up to t = 0
        }

        gravity: float = GRAVITY        # m/s^2, in full from the start

        ## Monitoring plots (off-screen PyVista windows on rank 0, updated every step).
        Plot_Fields: list = ["phasefield", "u_x", "u_y"]
        Plot_Measures: list = [("t", "Crack_Length"), ("t", "CliffTop_u_x"), ("t", "Eta_Active_Area")]
        record_movie: bool = True
        movie_fps: int = 10
        movie_folder: str = OutputFolder + "/Movies"

        ## Fields and solution steps (staggered, Newton for both as in the Section 3 sweeps)
        Solution_Steps = {"u": 0, "phasefield": 1}
        Step_Solvers = {0: "Newton", 1: "Newton"}
        Field_Orders = {"u": 2, "phasefield": 1}

        Models = {}
        ModelNames = ["Solid", "pf", "BC_Left", "Water", "BC_Bottom", "CD", "CliffTop", "CreepMax"]

        Models["Solid"] = {
            "Type": "Solid:Cohesive_ViscoElastic",   # reduced return mapping
            "Material": "Ice",
            "plane_stress": False,
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
        Models["BC_Left"] = {"Type": "Boundary:PrescribedValue", "boundary": "left",
                             "field": "u_x", "value": 0.0}
        # Cliff face: water at water_level up to t = 0, then lowered to zero over LOWERING_TIME.
        Models["Water"] = {"Type": "glaciology:LoweringOceanPressure", "boundary": "right",
                           "LoweringTime": LOWERING_TIME, "FinalWaterHeight": 0.0}
        # Free-slip bed
        Models["BC_Bottom"] = {"Type": "Boundary:PrescribedValue", "boundary": "bottom",
                               "field": "u_y", "value": 0.0}

        ## Measures written to the per-step history (time_data.hdf5)
        Models["CD"] = {"Type": "Solid:CrackDepthMeasure"}
        Models["CliffTop"] = {
            "Type": "Measure:PointDisplacement",
            "field": "u",
            "points": {"CliffTop": [length, height], "CliffMid": [length, 0.5 * height]},
        }
        Models["CreepMax"] = {
            "Type": "Measure:QuadratureNormMax",
            "field": "Plastic_Strain",
            "measure": "Creep_Strain_Max",
        }

        ## Nonlinear and linear solvers (Section 3 sweep settings, as the floe)
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes_force: bool = False
        max_it: int = 20
        n_passes: int = 10
        staggered_divergence_passes: int = 3
        staggered_divergence_factor: float = 1.5
        staggered_stagnation_passes: int = stagnation_passes
        staggered_stagnation_factor: float = 0.9
        line_search_nonfinite_abort: int = 2
        reset_dt_factor: float = 4.0
        reset_dt_factor_repeated: float = 10.0
        line_search_lims: list = [1e-1, 1.0]
        line_search: bool = True
        rel_tol: float = 1e-5
        abs_tol: float = 0.1       # N per m thickness (floe: 0.001 N/m times the edge length)

        ## Output
        output_dir: str = OutputFolder + "/Data"
        output_interval: int = OUTPUT_INTERVAL
        write_vtk: bool = False
        write_hdf5: bool = True
        save_Mesh_every_step: bool = False
        output_fields: list = ["u_x", "u_y", "phasefield", "sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy",
                               "etaNorm", "Plastic_Strain_xx", "Plastic_Strain_yy", "Plastic_Strain_xy", "Gc_eff"]

        ## Time: spin-up, then steps limited by the crack growth
        start_time: float = -SPIN_UP_TIME
        end_time: float = END_TIME                  # overridden below
        dt: float = SPIN_UP_DT
        n_damp = 0.5
        Newmark_Beta: float = 0.25 * (1 + n_damp) ** 2
        Newmark_Gamma: float = 0.5 + n_damp
        TimeStepControl = {
            "Type": "Measure",
            "time_list": time_list,
            "dt_list": dt_list,
            "dt_min": DT_MIN,
            "dt_max": SPIN_UP_DT,
            "Measure": "Crack_Length_Change",
            "Target_Change": crack_step_factor * target_h,
        }

    params = Params()
    params.end_time = float(end_time)
    return params


def summarize(params) -> str:
    """One-screen summary of the settings of a case."""
    ice = params.Materials["Ice"]
    overburden = -ice["Density"] * params.gravity * params.Ly
    refinement = params.MeshRefinement
    lines = [
        f"cliff {params.Lx} x {params.Ly} m, base free-slip, plane strain",
        f"rheology: Visc_Type {ice['Visc_Type']} (Glen n = {ice['Glen_n']}, A = {ice['Glen_A']})",
        f"strength surface {ice['FailureType']}, f_t = {ice['f_t']:.2e} Pa, f_s = {ice['f_s']:.2e} Pa, "
        f"overburden at the bed {overburden:.2e} Pa",
        f"ell = {ice['ell']} m, Gc = {ice['Gc']} J/m^2 (E Gc/f_t^2 = {ice['Youngs'] * ice['Gc'] / ice['f_t'] ** 2:.2f} m), "
        f"{ice['PF_formulation']}, kappa = {ice['kappa']}, irreversibility {params.Models['pf']['Irreversible_Method']}, "
        f"Gc mesh correction {params.Models['pf']['Gc_Mesh_Correction']['Coefficient']}",
        (f"{len(params.Notch_Positions)} surface notches of {params.Notch_Depth} x {params.Notch_Width} m (depth x width) "
         f"at x = {params.Notch_Positions[0]:.0f} to {params.Notch_Positions[-1]:.0f} m" if params.Notch_Positions
         else "no surface notches"),
        f"mesh: {params.dx} m, adaptive target {refinement.target_h} m, buffer {refinement.buffer_layers}, "
        f"<= {refinement.max_levels_per_step} levels/step",
        f"damping {ice['LE_Damping']:.1e} Pa s/m^2, Newmark beta/gamma {params.Newmark_Beta}/{params.Newmark_Gamma}",
        f"time: {params.start_time:.0f} s to {params.end_time:.0f} s, water at {params.Materials['Water']['WaterHeight']:.1f} m "
        f"lowered to {params.Models['Water']['FinalWaterHeight']:.1f} m over {params.Models['Water']['LoweringTime']:.0f} s "
        f"from t = 0, damage driven for t > 0, dt {params.TimeStepControl['dt_list']} s "
        f"from t = {params.TimeStepControl['time_list'][1:]} s, crack advance/step <= "
        f"{params.TimeStepControl['Target_Change']} m, fields every {params.output_interval} steps",
    ]
    return "\n".join(lines)
