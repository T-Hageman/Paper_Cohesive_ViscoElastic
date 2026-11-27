# Fracture and creep as a single constitutive problem: input files

This repository contains the input files, post-processing scripts and figure
scripts of the paper

> T. Hageman, *Fracture and creep as a single constitutive problem:
> strength-based phase-field modelling of nucleation, propagation and creep
> relaxation*, submitted to Computer Methods in Applied Mechanics and
> Engineering (2026). arXiv: (to be added).

The simulations use [FEMCubes](https://github.com/T-Hageman/FEMCubes), a
finite-element framework built on FEniCSx. With these scripts every figure and
table of the paper can be regenerated, by re-running the simulations and then
the figure scripts. The time histories of the paper runs are included in
`data/`, so that the line plots and Table 2 can be regenerated without running
any simulation (see [Stored data](#stored-data)).

The paper results were computed with IceCubes, the predecessor of FEMCubes.
The scripts here are ported to FEMCubes v0.1.0, which contains the same
models: the first time steps of every simulation script reproduce the time
histories of the paper runs to round-off, and the figure scripts reproduce the
figures of the paper from the stored results.

## Requirements and installation

FEMCubes **v0.1.0** with its Conda environment (FEniCSx 0.10, PETSc, MPICH),
on Linux, macOS, or Windows through WSL2:

```bash
git clone --branch v0.1.0 https://github.com/T-Hageman/FEMCubes.git
cd FEMCubes
conda env create -f environment.yml
conda activate FenicsX
python -m pip install --no-build-isolation -e .
cd ..
git clone https://github.com/T-Hageman/Paper_Cohesive_ViscoElastic.git
cd Paper_Cohesive_ViscoElastic
```

The FEMCubes environment also provides the packages of the figure scripts
(matplotlib, PyVista, h5py, and imageio-ffmpeg for the movies). Use one
thread per MPI rank:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Figures 1 and 2 are drawn with MATLAB (R2025a).

## Quick start

`Example_run.py` runs the plate-with-hole benchmark of Section 3 (medium
loading rate, `Gc = 5 J/m^2`, `ell = 0.1 m`) with the model, settings and mesh
of the paper: a background mesh of 0.03 m, refined initially to 0.002 m around
the hole and adaptively to `ell/4` wherever the cohesive strain appears. Two
settings are coarser than in the paper: the reduced return mapping instead of
the C++ kernel, and a larger limit on the crack growth per time step. Its
load-displacement curve matches that of the paper run.

```bash
mpirun -n 8 python Example_run.py      # about 10 minutes on 8 cores
```

It writes the load-displacement curve, the final phase field, the vertical
stress, the refined mesh and an animation of the refinement to
`results/Example_run/`, with the curve of the corresponding paper run (from
`data/`) for comparison.

## Repository layout

| Folder | Content |
| --- | --- |
| `simulations/` | parameter files, run drivers and the additional FEMCubes models and mesh generator of the simulations |
| `postprocessing/` | figure and table scripts, and the material-point calculation of Fig. 3 |
| `postprocessing/plotting/` | plotting helpers (line plots, field stills, principal stresses, movies) |
| `matlab/` | strength-surface figures (Figs. 1 and 2) |
| `data/` | time histories of the paper runs, in the folder structure of `results/` |
| `results/` | written by the simulations, read by the figure scripts (not in the repository) |
| `figures/` | written by the figure scripts (not in the repository) |

The scripts can be run from any folder. Each case writes its own folder below
`results/`, for example `results/Results_Cliff/<case>/` with `Data/`
(the time history `time_data.hdf5`, the field outputs and the meshes),
`Movies/` and `Monitor/`. The figure scripts read each study folder
(`Results_Cliff`, `Results_lSweep_LO_GcCorr0.33`, ...) from `results/` when it
exists there, and from `data/` otherwise; `PAPER_RESULTS_DIR=<folder>` replaces
`results/` by another folder, for example a copy of the full paper results.
They write the files under the names included in the manuscript.

## Stored data

`data/` (15 MB) holds, for every simulation of the paper, the time history
`Data/time_data.hdf5` (reaction forces, displacements, crack length and depth,
cohesive-strain measures per time step), the completion or stop marker, and
`case_settings.json` with the case parameters and the commit of IceCubes, the
predecessor of FEMCubes, that ran it. `Results_GcCorrection/analysis.json`
holds the local cohesive law extracted from the field outputs of the
Appendix B bar. From `data/` alone, without running any simulation:

- Figs. 8, 9, 10, 11 and Table 2 (plate with a hole),
- Fig. 13 (`Plot_Floe_Figures.py --lines-only`),
- Fig. 18 (`Plot_Cliff_Figures.py --length-scale-only`),
- Fig. B.1 (`Plot_GcCalibration_Figure.py` and `Plot_Geometry_Figures.py`).

Figs. 1 to 4, 12a, 15 and B.2 need no simulation results. Figs. 5 to 7, 12b,
14, 16 and 17 show field outputs, which are part of the full result folders of
the paper runs (about 130 GB). These are not included; they are regenerated by
re-running the corresponding simulations.

Model names in FEMCubes: `Solid:Cohesive_ViscoElastic_Kernel` is the
integration-point update by the C++ kernel (Sections 3 and 4.1, Appendix B);
`Solid:Cohesive_ViscoElastic` is the reduced return mapping, used for the ice
cliff of Section 4.2. Both solve the same local problem.

## Reproducing the results

Run times are those of the paper runs, with the number of MPI ranks given.
The run drivers skip completed cases, so they can be restarted.

### Section 2: material point (Fig. 3)

```bash
python postprocessing/MaterialPoint_PaperFigure.py
```

Serial, under a minute; no finite elements. Data and figure panels go to
`results/MaterialPoint_PaperResults_Gc1.0_Rates/`.
`postprocessing/MaterialPoint_Verification.py` compares the scalar update
with the C++ kernel of FEMCubes.

### Section 3: plate with a hole (Figs. 4 to 11, Tables 1 and 2)

The material parameters (Table 1) and the set-up are in
`simulations/Small_Reference_Params.py`; `simulations/Run_Plate_Sweeps.py`
runs the sweeps (`--figure 8`, `9` or `11` selects the cases of that figure).

```bash
cd simulations
python Run_Plate_Sweeps.py --list-cases
python Run_Plate_Sweeps.py --figure 8                             # Figs. 8, 10b: Gc = 1, element size
python Run_Plate_Sweeps.py --case-ids 5-9 --gc-correction 0.33    # Fig. 10b: Gc = 1, medium rate, corrected Gc
python Run_Plate_Sweeps.py --figure 9                             # Fig. 10a: Gc = 10, element size
python Run_Plate_Sweeps.py --figure 9 --gc-correction 0.33        # Figs. 9, 10a: Gc = 10, corrected Gc
python Run_Plate_Sweeps.py --figure 11 --gc-correction 0.33       # Figs. 5-7, 11, Table 2: length scale
```

Defaults: two cases at a time on 26 MPI ranks each (`--jobs`, `--ranks`).
Per case on 26 ranks: element-size study 0.3 to 6 h; length-scale study 1.5 h
(`ell = 0.1 m`) to 13 h (`ell = 0.01 m`), and 15 to 56 h for `ell = 0.005 m`.
The cases with `Gc = 10` and `ell = 0.005 m` at the medium and high rate
(IDs 84 and 89) take several days; for the paper they were stopped after the
peak load, which is all that is used of them.

Figures and table, from the repository root:

```bash
python postprocessing/Plot_Geometry_Figures.py        # Figs. 4, 12a, 15, B.1a
python postprocessing/Process_All_Data.py             # Figs. 8, 9, 11
python postprocessing/Process_All_Data.py --detail    # Figs. 5-7 (figures/animations/...)
python postprocessing/Compare_GcCorrection.py         # Fig. 10a (figures/gc_correction/)
python postprocessing/Compare_GcCorrection.py --study Results_dxSweep_LO --xmax 0.035   # Fig. 10b
python postprocessing/Table2_Rate_Energy.py           # Table 2
```

### Section 4.1: sea-ice floe (Figs. 12 to 14)

```bash
cd simulations
python Run_Floe_Cases.py --list-cases
python Run_Floe_Cases.py --case-ids 0 --ranks 52      # fast rate, fragmentation
python Run_Floe_Cases.py --case-ids 1-3 --ranks 8     # intermediate and slow rate, elastic reference
cd ..
python postprocessing/Plot_Floe_Figures.py            # Figs. 12b, 13, 14
```

All cases use the upper half of the floe. The fragmentation case (ID 0) was
stopped by hand after about 68 h on 52 ranks, before the cracks reached the
floe edge; Fig. 14 uses its outputs 410, 1210 and 2060. Cases 1 and 3 fracture
right after nucleation and were stopped once the crack was 1 m long (about 1 h
on 8 ranks); only their response up to nucleation is used. The slow case
(ID 2) does not fracture and takes about 10 min on 8 ranks.

### Section 4.2: ice cliff (Figs. 15 to 18)

Each cliff run is one call of `simulations/Run_Cliff.py`. All runs share

```bash
COMMON="--water-level 182 --length 500 --damping 1e6 --end-time 4200 --f-s 1.5e6 \
        --notch-spacing 50 --notch-width 0.5 --stagnation-passes 0"
```

| Figure | Case (`--name`) | Further options | Ranks; wall time |
| --- | --- | --- | --- |
| 16a, 16b | `new_wl_{elastic,viscoelastic}_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_r2_fs1.5_nostag_csf1` | `--rheology {elastic,viscoelastic} --failure-type r2_2L --crack-step-factor 1` | 8 / 10; stopped by hand at t = 537 s / 350 s |
| 16c, 16d | `new_wl_{elastic,viscoelastic}_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_csf1` | `--rheology {elastic,viscoelastic} --failure-type r2_2L_PSD --crack-step-factor 1` | 4 / 16; stopped by hand at t = 1801 s / 960 s |
| 17, 18 | `new_wl_{elastic,viscoelastic}_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_csf20` | `--rheology {elastic,viscoelastic} --failure-type r2_2L_PSD --crack-step-factor 20` | 16 / 16; 20 h / 116 h |
| 18 | `new_wl_{elastic,viscoelastic}_FreeSlip_l2_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_adv5m` | `--rheology {elastic,viscoelastic} --failure-type r2_2L_PSD --ell 2 --crack-step-factor 10` | 8 / 12; 19 h / 36 h |
| 18 | `new_wl_{elastic,viscoelastic}_FreeSlip_l4_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_adv5m` | `--rheology {elastic,viscoelastic} --failure-type r2_2L_PSD --ell 4 --crack-step-factor 5` | 8 / 12; 7 h / 23 h |

For example:

```bash
mpirun -n 16 python -u simulations/Run_Cliff.py \
    --name new_wl_elastic_FreeSlip_l1_Gc10_L500_notch50h_w05_wl182_PSD_fs1.5_nostag_csf20 \
    $COMMON --rheology elastic --failure-type r2_2L_PSD --crack-step-factor 20
python postprocessing/Plot_Cliff_Figures.py           # Figs. 16, 17, 18
```

The free-slip bed, the lowering water level and the reduced return mapping are
fixed in `Cliff_Params.py`. The runs of Fig. 16 only need to reach t = 537 s
(elastic) and t = 350 s (viscoelastic).

### Appendix B: element-size correction of the fracture energy (Figs. B.1, B.2)

```bash
cd simulations
python Calibrate_Gc_Correction.py --dxratios 2,3,4,5,8                    # uncorrected
python Calibrate_Gc_Correction.py --dxratios 2,3,4,5,8 --coefficient 0.25
python Calibrate_Gc_Correction.py --dxratios 2,3,4,5,8 --coefficient 0.33
python Calibrate_Gc_Correction.py --dxratios 2,3,4,5,8 --coefficient 0.5
python Calibrate_Gc_Correction.py --analyse
cd ..
python postprocessing/Plot_GcCalibration_Figure.py   # Fig. B.1
python postprocessing/Plot_GcCorrection_Order.py     # Fig. B.2 (no simulations)
```

Each case runs on 4 ranks (three cases at a time) in 4 to 60 min.

### Appendix C: eigenstrain directions

`python simulations/Check_Eigenstrain_Directions.py` checks the statements
of Appendix C numerically (serial, a few seconds, no finite elements).

## Figures and tables

Result folders are below `results/` (or `data/`); the output files are below
`figures/` unless stated otherwise.

| Paper | Simulation inputs and driver (`simulations/`) | Result folders (`results/` or `data/`) | Figure script (`postprocessing/`) and output file | From `data/` |
| --- | --- | --- | --- | --- |
| Fig. 1 | (none) | (none) | `matlab/PlotFailSurfaces.m`: `failure_surface_{3d,2d}` (in `matlab/`) | no data needed |
| Fig. 2 | (none) | (none) | `matlab/PlotCliffFailSurfaces.m`: `cliff_failure_surface_{3d,2d}` (in `matlab/`) | no data needed |
| Fig. 3 | (none) | `MaterialPoint_PaperResults_Gc1.0_Rates` | `MaterialPoint_PaperFigure.py`: `MaterialPoint_{Stress,PhaseField,CreepStrain,FractureStrain,Legend}` (in the result folder) | no data needed |
| Fig. 4 | (geometry) | (none) | `Plot_Geometry_Figures.py`: `PlateWithHole` | no data needed |
| Table 1 | `Small_Reference_Params.py` | (none) | (none) | no data needed |
| Figs. 5, 6, 7 | `Small_Reference_Params.py`, `Run_Plate_Sweeps.py --figure 11 --gc-correction 0.33` | `Results_lSweep_LO_Gc5.0_GcCorr0.33/l_0.02` | `Process_All_Data.py --detail`: `animations/Results_lSweep_LO_Gc5.0_GcCorr0.33/l_0.02/{load_displacement_selected_states,mesh_unloading_75pct,u_y_*,phasefield_*,sigma_yy_*,principal_stress_*}` | no (field outputs) |
| Table 2 | as Figs. 5 to 7 | `Results_lSweep_LO{_LowRate,,_HighRate}{,_Gc5.0,_Gc10}_GcCorr0.33/l_0.02` | `Table2_Rate_Energy.py` (printed) | yes |
| Fig. 8 | `Run_Plate_Sweeps.py --figure 8` | `Results_dxSweep_LO{_LowRate,,_HighRate}/dx_{1..5}` | `Process_All_Data.py`: `MeshRefinement_Study{_PeakLoad,}` | yes |
| Fig. 9 | `Run_Plate_Sweeps.py --figure 9 --gc-correction 0.33` | `Results_dxSweep_LO{_LowRate,,_HighRate}_Gc10_GcCorr0.33/dx_{1..5}` | `Process_All_Data.py`: `MeshRefinement_Study_Gc10{_PeakLoad,}` | yes |
| Fig. 10a | `Run_Plate_Sweeps.py --figure 9`, with and without `--gc-correction 0.33` | `Results_dxSweep_LO_Gc10`, `Results_dxSweep_LO_Gc10_GcCorr0.33` | `Compare_GcCorrection.py`: `gc_correction/Results_dxSweep_LO_Gc10_GcCorr0.33_overlay` | yes |
| Fig. 10b | `Run_Plate_Sweeps.py --figure 8`, and `--case-ids 5-9 --gc-correction 0.33` | `Results_dxSweep_LO`, `Results_dxSweep_LO_GcCorr0.33` | `Compare_GcCorrection.py --study Results_dxSweep_LO --xmax 0.035`: `gc_correction/Results_dxSweep_LO_GcCorr0.33_overlay` | yes |
| Fig. 11 | `Run_Plate_Sweeps.py --figure 11 --gc-correction 0.33` | `Results_lSweep_LO{_LowRate,,_HighRate}{,_Gc2.5,_Gc5.0,_Gc10}_GcCorr0.33/l_*` | `Process_All_Data.py`: `lSweep_Study_AllCases_PeakLoad`, `lSweep_Study_Gc5.0` | yes |
| Fig. 12a | (geometry) | (none) | `Plot_Geometry_Figures.py`: `SeaIce_Geometry` | no data needed |
| Figs. 12b, 13, 14 | `SeaIce_Floe_Params.py`, `Upper_Half_Plate_Mesh.py`, `Run_Floe_Cases.py` (IDs 0 to 3) | `Results_Floe/{fast,intermediate,slow}_l0.25_sym`, `Results_Floe/fast_l0.25_sym_elastic` | `Plot_Floe_Figures.py`: `SeaIce_Mesh`, `SeaIce_Stress`, `SeaIce_Pattern_{1,2,3}_{phasefield,ux}` | Fig. 13 only |
| Fig. 15 | (geometry) | (none) | `Plot_Geometry_Figures.py`: `Cliff_Geometry` | no data needed |
| Fig. 16 | `Cliff_Params.py`, `Run_Cliff.py`, `Cliff_Measures.py` | `Results_Cliff/new_wl_*_l1_*_{r2,PSD}_fs1.5_nostag_csf1` | `Plot_Cliff_Figures.py`: `Cliff_Criterion_{Original,Opening}_{Elastic,Viscoelastic}` | no (field outputs) |
| Fig. 17 | as Fig. 16 | `Results_Cliff/new_wl_*_l1_*_PSD_fs1.5_nostag_csf20` | `Plot_Cliff_Figures.py`: `Cliff_Final_{Elastic,Viscoelastic}_{phasefield,ux}` | no (field outputs) |
| Fig. 18 | as Fig. 16 | `Results_Cliff/new_wl_*_l1_*_csf20`, `Results_Cliff/new_wl_*_l{2,4}_*_adv5m` | `Plot_Cliff_Figures.py`: `Cliff_LengthScale` | yes |
| Fig. B.1 | `Calibrate_Gc_Correction.py` | `Results_GcCorrection/{uncorrected,c0.25,c0.33,c0.5}/dx_{2,3,4,5,8}` | `Plot_GcCalibration_Figure.py`: `GcCorrection_{LoadDisp,Calibration}`; `Plot_Geometry_Figures.py`: `GcCorrection_Geometry` | yes |
| Fig. B.2 | (none) | (none) | `Plot_GcCorrection_Order.py`: `GcCorrection_Order_{Position,Size}` | no data needed |

## Citation and license

Please cite the paper above when using these files; `CITATION.cff` has the
details. The scripts are released under the MIT license (`LICENSE`).

## Use of AI

This repository, collecting the inputs, scripts and data of the paper, was assembled based on the research repository, and cleaned-up and commented with the help of AI (Claude Code, Anthropic). The author reviewed the repository and takes full responsibility for its content.
