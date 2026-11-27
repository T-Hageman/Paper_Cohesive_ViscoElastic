"""Values of tab:rate_energy (Table 2 of the paper) from the length-scale runs at ell = 0.02 m.

Reads Results_lSweep_LO{_LowRate,,_HighRate}{,_Gc5.0,_Gc10}_GcCorr0.33/l_0.02/Data/time_data.hdf5
(in results/, or data/ for the paper runs; written by simulations/Run_Plate_Sweeps.py --gc-correction 0.33) and prints, per loading rate and
fracture energy:
  u_act   top displacement at which the crack-length measure first exceeds CRACK_ONSET,
          i.e. the eigenstrain is active at the hole (the strength is reached first at the
          clamped bottom corners, where the No_PhaseField strip prevents a crack);
  u_peak  top displacement at the peak load;
  s_peak  peak nominal stress F / (Lx * 1 m) [MPa].
Run:  python postprocessing/Table2_Rate_Energy.py
"""
import os
from decimal import ROUND_HALF_UP, Decimal

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import numpy as np

from plotting.paths import study_folder

RATES = (("_LowRate", 1.0e-9), ("", 1.5e-9), ("_HighRate", 2.5e-9))
FRACTURE_ENERGIES = (("", 1.0), ("_Gc5.0", 5.0), ("_Gc10", 10.0))
CRACK_ONSET = 1.0e-14   # m; the crack-length integral is below 1e-16 before the hole activates
PLATE_WIDTH = 1.0       # m (Lx of Small_Reference_Params.py)


def rounded(value):
    """Three decimals, rounded half up from four as in the table (0.0255 -> 0.026)."""
    return str(Decimal(f"{value:.4f}").quantize(Decimal("0.001"), ROUND_HALF_UP))


def history(folder):
    with h5py.File(folder / "Data" / "time_data.hdf5", "r") as h:
        names = [n.decode() for n in h["data_names"][:]]
        data = h["data"][:]
    return {name: data[:, i] for i, name in enumerate(names)}


def main():
    print(f"{'rate [1/s]':>10} {'Gc':>4} {'u_act [mm]':>10} {'u_peak [mm]':>11} {'s_peak [MPa]':>12}  note")
    for rate_suffix, rate in RATES:
        for gc_suffix, gc in FRACTURE_ENERGIES:
            c = history(study_folder(f"Results_lSweep_LO{rate_suffix}{gc_suffix}_GcCorr0.33") / "l_0.02")
            u = np.abs(c["BC_Top_u"]) * 1e3
            force = np.abs(c["BC_Top_F"])
            crack = c["Crack_Length"]
            active = np.flatnonzero(crack > CRACK_ONSET)
            u_act = rounded(u[active[0]]) if active.size else "--"
            peak = int(np.argmax(force))
            # No load drop: the peak is the end of the loading interval
            no_drop = force[-1] >= 0.9 * force[peak]
            print(f"{rate:10.1e} {gc:4g} {u_act:>10} {rounded(u[peak]):>11} {rounded(force[peak] / PLATE_WIDTH / 1e6):>12}"
                  + ("  no load drop (value at the end)" if no_drop else ""))


if __name__ == "__main__":
    main()
