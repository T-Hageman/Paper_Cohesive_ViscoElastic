"""Global measures for the ice-cliff case (manuscript Section 4.2, fig:cliff_displacement).

Two measure-only models (no fields, no weak-form terms) that add entries to
``params.Global_Measures``, and thereby to the per-step history in
``time_data.hdf5``:

``Measure:PointDisplacement``
    Displacement of the mesh vertex nearest to each named point, e.g. the top of
    the cliff face:

        Models["Top"] = {
            "Type": "Measure:PointDisplacement",
            "field": "u",                                  # optional, default "u"
            "points": {"CliffTop": [2000.0, 200.0]},       # name -> coordinates
        }

    gives ``CliffTop_u_x`` and ``CliffTop_u_y``. The vertex is selected globally
    (MPI minimum of the distance) and again after every remesh.

``Measure:QuadratureNormMax``
    Largest norm of a quadrature-point field, e.g. the creep strain:

        Models["CreepMax"] = {
            "Type": "Measure:QuadratureNormMax",
            "field": "Plastic_Strain",                     # (6, 1) Mandel field or a scalar
            "measure": "Creep_Strain_Max",
        }

    For the (6, 1) Mandel fields of the cohesive models this is the Frobenius
    norm of the tensor, the same norm as ``etaNorm`` / ``Eta_Max``.

Subclassing ``BaseModel`` registers the model types, so importing this module in
the MPI worker is enough to use them from ``params.Models``.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.mpi_utils import comm

_COMPONENT_NAMES = ("x", "y", "z")


class PointDisplacementMeasure(BaseModel):
    """Record a vector field at the vertices nearest to named points."""

    MODEL_TYPE = "Measure:PointDisplacement"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        settings = params.Models[name]
        self.field_name = settings.get("field", "u")
        self.points = {key: np.asarray(point, dtype=float) for key, point in settings["points"].items()}
        if not self.points or not all(np.all(np.isfinite(point)) for point in self.points.values()):
            raise ValueError(f"{self.MODEL_TYPE} '{name}': 'points' must map names to finite coordinates.")
        self._dofs: dict | None = None
        self._dofs_epoch: int | None = None
        for key in self.points:
            for component in _COMPONENT_NAMES[: params.dim]:
                self.params.Global_Measures[f"{key}_{self.field_name}_{component}"] = 0.0

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Measure-only model: no weak-form contribution."""
        return None

    def Post_Remesh(self) -> None:
        """Drop the mesh-bound vertex selection; it is redone on demand."""
        self._dofs = None
        self._dofs_epoch = None

    def Update_Global_Measures(self) -> None:
        function = self.mesh.Get_Field(self.field_name)
        epoch = int(getattr(self.mesh, "remesh_epoch", 0))
        if self._dofs is None or self._dofs_epoch != epoch:
            self._dofs = {key: self._owned_vertex_dofs(function.function_space, point)
                          for key, point in self.points.items()}
            self._dofs_epoch = epoch

        block_size = function.function_space.dofmap.index_map_bs
        for key, dofs in self._dofs.items():
            count = comm.allreduce(int(dofs.size), op=MPI.SUM)
            for component in range(block_size):
                # one vertex in general; the mean if several are equally near
                local = float(np.sum(function.x.array[block_size * dofs + component]))
                value = comm.allreduce(local, op=MPI.SUM) / count
                self.params.Global_Measures[f"{key}_{self.field_name}_{_COMPONENT_NAMES[component]}"] = value

    def _owned_vertex_dofs(self, function_space, point) -> np.ndarray:
        """PRIVATE: Owned block DOFs of the vertex nearest to ``point`` (global nearest)."""
        domain = function_space.mesh
        gdim = domain.geometry.dim
        if point.size < gdim:
            raise ValueError(f"{self.MODEL_TYPE} '{self.name}': points need {gdim} coordinates.")
        point = point[:gdim]
        coordinates = domain.geometry.x[:, :gdim]
        extent = max(float(np.max(np.ptp(coordinates, axis=0))) if coordinates.size else 0.0, 1.0)
        local_min = np.min(np.linalg.norm(coordinates - point, axis=1)) if coordinates.size else np.inf
        radius = comm.allreduce(local_min, op=MPI.MIN) + 1.0e-8 * extent

        domain.topology.create_entities(0)
        domain.topology.create_connectivity(0, domain.topology.dim)
        vertices = dmesh.locate_entities(
            domain, 0, lambda x: np.linalg.norm(x[:gdim].T - point, axis=1) <= radius)
        dofs = fem.locate_dofs_topological(function_space, 0, vertices)
        dofs = dofs[dofs < function_space.dofmap.index_map.size_local].astype(np.int64)
        if comm.allreduce(int(dofs.size), op=MPI.SUM) == 0:
            raise RuntimeError(f"{self.MODEL_TYPE} '{self.name}': no vertex found near {point.tolist()}.")
        return dofs


class QuadratureNormMax(BaseModel):
    """Record the largest norm of a quadrature-point field."""

    MODEL_TYPE = "Measure:QuadratureNormMax"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        settings = params.Models[name]
        self.field_name = settings["field"]
        self.measure = settings["measure"]
        self.params.Global_Measures[self.measure] = 0.0

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Measure-only model: no weak-form contribution."""
        return None

    def Update_Global_Measures(self) -> None:
        function = self.mesh.Get_Field(self.field_name)
        dofmap = function.function_space.dofmap
        block_size = dofmap.index_map_bs
        owned = function.x.array[: dofmap.index_map.size_local * block_size].reshape(-1, block_size)
        local_max = float(np.max(np.linalg.norm(owned, axis=1))) if owned.size else 0.0
        self.params.Global_Measures[self.measure] = comm.allreduce(local_max, op=MPI.MAX)
