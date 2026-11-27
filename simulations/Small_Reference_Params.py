"""Parameters of the plate with a hole (Section 3 of the paper, Table 1).

A 1 x 1 m intrinsic two-dimensional ice plate with a central hole of radius 0.1 m,
clamped at the bottom and pulled at the top at a constant nominal strain rate. The
sweep driver Run_Plate_Sweeps.py sets the fracture energy, the damping, the
element-size correction of Gc and the end of the run of each case.
"""

from femcubes import LinearSolver, ParamsTemplate
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.mpi_utils import mprint


def _build_small_params(l: float = 1e-3, dxratio: int = 3, rate: float = 1e-4, include_plots: bool = False,
                        OutputFolder: str = None, MeshRef: bool = False):
    """Build the plate-with-hole problem.

    ``l`` is the phase-field length scale, ``dxratio`` the ratio ell/h, ``rate`` the nominal
    strain rate (1/s). ``MeshRef`` False gives a uniform mesh with h = l/dxratio (element-size
    study); True a background mesh of 0.03 m, 0.002 m at the hole, refined adaptively to
    h = l/dxratio where the cohesive strain appears (length-scale study).
    """
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        Quadrature_Degree: int = 3

        ## Mesh parameters
        MeshType: str = "FullPlateWithHole"
        dx: float = l / dxratio # Mesh resolution in meters
        dxmin: float = l / dxratio  # Mesh resolution in meters for the hole region
        dim: int = 2  # Spatial dimension
        Lx: float = 1.0
        Ly: float = 1.0
        R: float = 0.1
        R_Refine: float = 0.025

        if MeshRef:
            dx = 0.03
            dxmin = 0.01/5
            MeshRefinement = RefinementConfig(
                enabled=True,
                indicator=FieldThresholdIndicatorConfig(
                    field="etaNorm",  # A scalar quadrature field such as "PsiField" is also supported
                    threshold=1e-12,
                    reduction="cell_max",
                    comparison="greater_equal",
                ),
                target_h= l / dxratio,
                buffer_layers=6,
                check_interval=1,
                max_levels_per_step=int(1),
                max_events=int(1e99),
                repartition=True,  # Rebalance only after a complete refinement transfer
                repartition_imbalance_threshold=1.1,  # Trigger when max_owned / mean_owned exceeds this
                replot_on_refinement=True,  # Refresh the mesh plot after each successful refinement
                save_remesh_animations=True,  # Record mesh_refinement.mp4 without requiring interactive replotting
            )

        ## physics parameters
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,           # density of ice (kg/m^3)
            "Youngs": 9.5e9,            # Young's modulus (Pa)
            "Poisson": 0.35,            # Poisson's ratio
            "Visc_Type": "Glen",
            "Glen_n": 3.0,              # Glen's flow law exponent
            "Glen_A": 1e-24,            # Glen's flow law rate factor (Pa^-n s^-1)
            "ell": l,                   # Length scale parameter (m)
            "f_t": 0.4e6,               # Tensile strength (Pa)
            "f_s": 0.6e6,               # Shear strength (Pa)
            "Gc": 1.0,                  # Fracture energy (J/m^2)
            "PF_formulation": "AT2",
            "FailureType": "r2_2L",
            "Inertia": True,
            "kappa": 1.0e-3,
            "LE_Damping": 1e9
        }

        # Elastic constants of the intrinsic two-dimensional material
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Youngs"] * Materials["Ice"]["Poisson"] / (1.0 - Materials["Ice"]["Poisson"]**2)
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Lame"] + Materials["Ice"]["Shear"]

        gravity: float = 0.0  # no gravity (flat 2D plate)

        Plot_Fields: list = []
        Plot_Measures: list = [("t", "BC_Top_F")]
        if include_plots:
            record_movie: bool = True  # enable/disable movie recording
            movie_fps: int = 10  # frames per second
            movie_folder: str = OutputFolder+"/Movies"

        Solution_Steps = {"u": 0, "phasefield": 1}
        Step_Solvers = {0: "NewtonTrustRegion", 1: "Newton"}

        Field_Orders = {"u": 2, "phasefield": 1}

        Models = {}
        ModelNames = ["Solid", "pf", "BC_Bottom", "BC_Top", "BC_Bottom2", "NoPF"]

        Models["Solid"] = {
            "Type": "Solid:Cohesive_ViscoElastic_Kernel",
            "Material": "Ice",
            "plane_stress": False,
            "true_2d": True,
            "consistent_stiffness": True
        }
        Models["pf"] = {
            "Type": "Solid:PhaseField",
            "Material": "Ice",
            "CDF_field": "PsiField",
            "Irreversible_Method": "Hist"
        }

        Models["BC_Top"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "top",
            "field": "u_y",
            "rate": rate*Ly,
            "Dummy": 1e12,
            "F_Cutoff": 0.1,
        }
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0
        }
        Models["BC_Bottom2"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_x",
            "value": 0.0
        }
        Models["NoPF"] = {
            "Type": "Solid:No_PhaseField",
            "x_range": [-0.1, 1.1],
            "y_range": [-0.1, 0.1],
        }

        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU

        n_passes_force: bool = False
        max_it: int = 20
        n_passes: int = 10
        line_search_lims: list = [1e-1, 1.0]
        line_search: bool = True

        rel_tol: float = 1e-6
        abs_tol: float = 0.001

        # Output parameters
        output_dir: str = OutputFolder + "/Data"  # directory to save output files
        output_interval: int = 10  # interval (in time steps) to write output files
        write_vtk: bool = False  # whether to write VTK files for visualization
        write_hdf5: bool = True  # whether to write HDF5 files for data storage
        save_Mesh_every_step: bool = False  # whether to save mesh at every time step or only once
        output_fields: list = ["u_x", "u_y", "phasefield", "sigma_xx", "sigma_yy", "sigma_xy"]  # list of specific fields to output

        ## Time parameters
        end_time: float = 1e-3/abs(rate)  # end time (s)
        dt: float = 3e-7/abs(rate)  # time step (s)
        start_time: float = 0.0  # start time (s)
        n_damp = 0.5
        Newmark_Beta: float = 0.25*(1+n_damp)**2 # 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 0.5+n_damp # 1.0  # Newmark-gamma parameter
        TimeStepControl = {
            "Type": "Measure",
            "time_list": [-1e99, 0],
            "dt_list": [60, 3e-7/abs(rate)],
            "dt_min": 1.0e-14,
            "dt_max": 3e-7/abs(rate),
            "Measure": "Crack_Length_Change",
            "Target_Change": 0.01/4 # using smallest l used
        }

    mprint(f"Running with rate = {rate:.0e}, dt = {Params.dt:.2e} s, end_time = {Params.end_time:.2e} s")

    return Params()
