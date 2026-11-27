"""Element-size error of the discrete cohesive law for phase-field elements of order p (Appendix B).

One-dimensional bar with the eigenstrain confined to a single point x_q, a fraction xi
into an element of size h. The discrete phase-field problem
    (phi, w) + ell^2 (phi', w') = 2 (1 - phi_q) (f_t delta ell / Gc) w(x_q)
gives Gc_num/Gc = 1 / (2 g_q), with g_q the discrete Green's function at x_q (continuum: 1/2).
This is compared with the leading-order result Gc_num/Gc = 1 + c_p(xi) h/ell.

Writes, in figures/ (repository root):
    GcCorrection_Order_Position.pdf  (a) coefficient c_p against the position of the opened point
    GcCorrection_Order_Size.pdf      (b) position-averaged Gc_num/Gc against h/ell
Run:  python postprocessing/Plot_GcCorrection_Order.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

SCRIPT_DIR = Path(__file__).resolve().parent
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotting.PlotLines_Appendix import FONTSIZE, FormatAndSave  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"   # figures/ in the repository root
FIGURES.mkdir(exist_ok=True)
ORDERS = (1, 2, 3, 4)
MARKERS = ("o", "s", "^", "v")
HALF_LENGTH = 14.0      # bar half-length in units of ell: exp(-14) truncation of the profile
S_POSITION = 0.25       # h/ell of the numerical points in panel (a)


def shape_functions(p):
    """Lagrange shape functions of order p on [0, 1] with equispaced nodes, and their derivatives."""

    coefficients = np.linalg.inv(np.vander(np.linspace(0, 1, p + 1), p + 1, increasing=True))
    powers = np.arange(p + 1)

    def N(x):
        return np.power.outer(np.asarray(x, dtype=float), powers) @ coefficients

    def dN(x):
        x = np.asarray(x, dtype=float)
        return (powers * np.power.outer(x, np.maximum(powers - 1, 0))) @ coefficients

    return N, dN


def gc_ratio(p, xi, s):
    """Gc_num/Gc of the discrete problem for order p, opened point at xi (array), and h/ell = s."""

    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    n_elements = 2 * int(round(HALF_LENGTH / s)) + 1
    N, dN = shape_functions(p)
    x_gauss, w_gauss = legendre.leggauss(p + 1)      # exact for the mass and gradient terms
    x_gauss, w_gauss = 0.5 * (x_gauss + 1), 0.5 * w_gauss
    Ne, dNe = N(x_gauss), dN(x_gauss)
    element_matrix = (Ne.T * w_gauss) @ Ne * s + (dNe.T * w_gauss) @ dNe / s   # lengths in units of ell
    n_dofs = n_elements * p + 1
    A = np.zeros((n_dofs, n_dofs))
    for e in range(n_elements):
        dofs = np.arange(e * p, e * p + p + 1)
        A[np.ix_(dofs, dofs)] += element_matrix
    dofs = np.arange((n_elements // 2) * p, (n_elements // 2) * p + p + 1)
    Nq = N(xi)                                       # (n_xi, p+1)
    rhs = np.zeros((n_dofs, len(xi)))
    rhs[dofs, :] = Nq.T
    g = np.linalg.solve(A, rhs)
    g_q = np.einsum("ij,ji->i", Nq, g[dofs, :])
    return 1.0 / (2.0 * g_q)


def c_analytical(p, xi):
    """Leading-order coefficient c_p(xi): L2 error of the degree p-1 projection of the gradient jump."""

    xi = np.asarray(xi, dtype=float)
    z = 2 * xi - 1
    total = np.zeros_like(xi)
    for k in range(1, p):
        dPk = legendre.legval(z, legendre.legder(np.eye(k + 1)[k]))
        total += (2 * k + 1) / (k * (k + 1)) ** 2 * dPk**2
    return 2 * xi * (1 - xi) * (1 - 4 * xi * (1 - xi) * total)


def c_mean(p):
    return p / (4 * p**2 - 1)


def position_panel():
    fig, ax = plt.subplots()
    xi_line = np.linspace(0, 1, 401)
    xi_points = np.linspace(0, 1, 21)
    for p, marker in zip(ORDERS, MARKERS):
        line, = ax.plot(xi_line, c_analytical(p, xi_line), label=rf"$p={p}$")
        ax.plot(xi_points, (gc_ratio(p, xi_points, S_POSITION) - 1) / S_POSITION, marker=marker,
                linestyle="none", markersize=2.5, markeredgewidth=0.5, markerfacecolor="none",
                color=line.get_color())
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$c_p$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=FONTSIZE, ncol=4, columnspacing=0.8, handlelength=1.5, loc="upper center")
    FormatAndSave(fig, ax, str(FIGURES / "GcCorrection_Order_Position"))
    plt.close(fig)


def size_panel():
    fig, ax = plt.subplots()
    s_line = np.array([0.0, 0.55])
    s_points = np.array([1 / 8, 1 / 5, 1 / 4, 1 / 3, 1 / 2])
    xi = (np.arange(200) + 0.5) / 200               # midpoint rule over the crack position
    for p, marker in zip(ORDERS, MARKERS):
        line, = ax.plot(s_line, 1 + c_mean(p) * s_line, label=rf"$p={p}$")
        ax.plot(s_points, [gc_ratio(p, xi, s).mean() for s in s_points], marker=marker,
                linestyle="none", markersize=3.5, markeredgewidth=0.5, color=line.get_color())
    ax.set_xlabel(r"$h/\ell$")
    ax.set_ylabel(r"$G_\text{c}^\text{num}/G_\text{c}$")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(1.0, 1.2)
    ax.set_yticks(np.arange(1.0, 1.2001, 0.05))
    ax.legend(fontsize=FONTSIZE, loc="upper left")
    FormatAndSave(fig, ax, str(FIGURES / "GcCorrection_Order_Size"))
    plt.close(fig)


if __name__ == "__main__":
    position_panel()
    size_panel()
    print("Figures written to", FIGURES)
