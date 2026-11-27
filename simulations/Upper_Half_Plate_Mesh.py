"""Upper half of the plate with a central hole, cut at its horizontal centreline.

``UpperHalfPlateWithHole`` meshes ``0 <= x <= Lx``, ``Ly/2 <= y <= Ly`` of the
``FullPlateWithHole`` geometry, with the upper half of the hole at
``(Lx/2, Ly/2)`` removed. ``Lx`` and ``Ly`` remain the dimensions of the *full*
plate and the coordinates are those of the full plate, so results of the two
can be overlaid. It is the symmetric counterpart for a plate pulled in x, whose
cracks grow from the hole in +-y (the FEMCubes ``HalfPlateWithHole`` cuts at the
vertical centreline instead, i.e. along that crack path).

Subclassing ``MeshGeneratorBase`` registers the mesh type, so importing this
module in the MPI worker is enough to use ``MeshType = "UpperHalfPlateWithHole"``.
Parameters as ``FullPlateWithHole`` (2-D, circular hole): ``Lx``, ``Ly``, ``R``,
``dx`` and optionally ``dxmin``, ``R_Refine`` (element size ``dxmin`` within
``R_Refine`` of the hole, graded to ``dx`` by ``2 R_Refine``).

Named boundaries: ``left``, ``right``, ``top``, ``hole`` and ``symmetry`` (both
straight segments at ``y = Ly/2``; the hole arc is excluded). For symmetric
mechanics prescribe ``u_y = 0`` on ``symmetry`` and leave ``u_x`` and the phase
field free there (natural zero-normal-gradient condition). Reactions on the
left and right edges are half-plate resultants.
"""

from __future__ import annotations

from typing import Any

import gmsh
import numpy as np

from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase, model_to_mesh
from femcubes.core.utils.mpi_utils import comm


class UpperHalfPlateWithHoleMeshGenerator(MeshGeneratorBase):
    """Generate the upper half of a 2-D plate with a centred circular hole (Gmsh)."""

    MESH_TYPE = "UpperHalfPlateWithHole"

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 2:
            raise NotImplementedError("UpperHalfPlateWithHole supports only dim=2.")
        LX, LY, R, dx = params.Lx, params.Ly, params.R, params.dx
        dx_small = getattr(params, "dxmin", dx)
        R_Refine = getattr(params, "R_Refine", 2.0 * R)
        for name, value in (("Lx", LX), ("Ly", LY), ("R", R), ("dx", dx), ("dxmin", dx_small),
                            ("R_Refine", R_Refine)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"UpperHalfPlateWithHole requires positive finite {name}.")
        if R >= 0.5 * min(LX, LY):
            raise ValueError("The hole must fit strictly inside the full plate.")
        y_start = 0.5 * LY

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("upper_half_plate_with_hole")

        plate = gmsh.model.occ.addRectangle(0, y_start, 0, LX, LY - y_start)
        hole = gmsh.model.occ.addDisk(0.5 * LX, 0.5 * LY, 0, R, R)
        cut = gmsh.model.occ.cut([(2, plate)], [(2, hole)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in cut[0] if dim == 2]
        if not surfaces:
            raise RuntimeError("Upper-half plate-with-hole mesh generation failed to create a surface.")
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary_groups = {"symmetry": [], "right": [], "top": [], "left": [], "hole": []}
        tol = max(1e-6 * max(LX, LY, R), 1e-9)
        for surf in surfaces:
            for _, tag in gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False):
                xmin, ymin, _, xmax, ymax, _ = self._exact_bounding_box(1, tag)
                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)
                if y_planar and np.isclose(ymin, y_start, atol=tol):
                    boundary_groups["symmetry"].append(tag)
                elif y_planar and np.isclose(ymin, LY, atol=tol):
                    boundary_groups["top"].append(tag)
                elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                    boundary_groups["left"].append(tag)
                elif x_planar and np.isclose(xmin, LX, atol=tol):
                    boundary_groups["right"].append(tag)
                else:
                    boundary_groups["hole"].append(tag)
        if len(boundary_groups["symmetry"]) != 2 or len(boundary_groups["hole"]) != 1:
            raise RuntimeError(f"Unexpected boundary classification: {boundary_groups}")

        markers = {"symmetry": 1, "right": 2, "top": 3, "left": 4, "hole": 5}
        for name, marker in markers.items():
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name.capitalize())

        # Mesh sizing as FullPlateWithHole: distance-based threshold field from the hole
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", boundary_groups["hole"])
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", dx_small)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", dx)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", R_Refine)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2 * R_Refine)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)
        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers=markers,
        )
        gmsh.finalize()
        return generated
