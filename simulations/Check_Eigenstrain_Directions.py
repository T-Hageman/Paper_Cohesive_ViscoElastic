"""Are the eigenstrain directions G_i of the paper consistent with the variational principle?

Internal variable: the tensor eta. Energy per point (Mandel 6-vectors, 3-D constants)
    W(eta; e, d) = (e - eta) C (e - eta) / 2 + d F_d(eta) + F_i(eta),      e = eps - eps_v
Checks, for the two-multiplier criterion (G_1 = +-I/sqrt 3, G_2 = dev e / |dev e|) and for the
opening-only criterion (G_i = n_i n_i of e, eta positive semidefinite):
 1. the multiplier solution satisfies the inequality for ALL admissible tensor variations
    (two-multiplier: every symmetric tensor; opening-only: the positive semidefinite cone);
 2. an independent minimisation over the full tensor finds the same eta;
 3. the stress C (e - eta) equals the derivative of the reduced energy min_eta W with respect to
    the strain (no term from the change of the directions is missing);
 4. counter-example: directions taken from another tensor (the total strain, with eps_v != 0)
    give a stress that is NOT that derivative; the difference equals lambda_2 R : dG_2/d eps;
 5. the kernels: C++ kernel stress against the equations of the paper; symmetry of the
    consistent tangents with and without creep; stress = derivative of the incremental potential
    with creep.
"""
import math

import numpy as np  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402

from femcubes.solids._principal_cve import PrincipalCVEUpdate  # noqa: E402
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (  # noqa: E402
    CohesiveViscoElasticKernels,
)

E, NU = 9.5e9, 0.35
MU = E / (2 * (1 + NU))
LAME = E * NU / ((1 + NU) * (1 - 2 * NU))
BULK = LAME + 2 * MU / 3
F_T, F_C = 0.4e6, 1.5e6
FT_S = F_T / (3 * math.sqrt(1 - (F_T / F_C) ** 2))
FS_S = math.sqrt(2 / 3) * F_C
PEN = 1.0e6 * FT_S
KAPPA = 1.0e-3
GLEN_A, GLEN_N = 1.0e-24, 3.0
VISC = (3 ** ((GLEN_N + 1) / 2) / (2 * GLEN_A)) ** (1 / GLEN_N)
SQRT2, SQRT3 = math.sqrt(2.0), math.sqrt(3.0)
I6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
PDEV = np.eye(6) - np.outer(I6, I6) / 3
C = 2 * MU * np.eye(6) + LAME * np.outer(I6, I6)
S0 = F_T / E                                       # strength strain


def mandel(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], SQRT2 * T[1, 2], SQRT2 * T[0, 2], SQRT2 * T[0, 1]])


def tensor(v):
    return np.array([[v[0], v[5] / SQRT2, v[4] / SQRT2], [v[5] / SQRT2, v[1], v[3] / SQRT2],
                     [v[4] / SQRT2, v[3] / SQRT2, v[2]]])


def F_two(eta, d):
    """d F_d + F_i of the two-multiplier criterion (eq:r2_2l_potentials)."""
    t = eta[:3].sum()
    q2 = (PDEV @ eta) @ eta
    return d * math.sqrt(FT_S ** 2 * max(t, 0.0) ** 2 + FS_S ** 2 * q2) + PEN * max(-t, 0.0)


def dF_two(eta, d, delta):
    """One-sided directional derivative of d F_d + F_i at eta in the direction delta (exact; a
    trace of eta at round-off is zero)."""
    t = eta[:3].sum()
    dev = PDEV @ eta
    td, q = delta[:3].sum(), np.linalg.norm(dev)
    if abs(t) < 1e-12 * S0:
        if q < 1e-12 * S0:
            return d * math.sqrt(FT_S ** 2 * max(td, 0.0) ** 2 + FS_S ** 2 * (PDEV @ delta) @ delta) \
                + PEN * max(-td, 0.0)
        return d * FS_S * (dev @ delta) / q + PEN * max(-td, 0.0)
    if t < 0.0:
        return d * FS_S * (dev @ delta) / q - PEN * td
    return d * (FT_S ** 2 * t * td + FS_S ** 2 * dev @ delta) / math.sqrt(FT_S ** 2 * t * t + FS_S ** 2 * q * q)


def F_open(eta, d):
    """d F_d of the opening-only criterion (eq:opening_potentials)."""
    t = eta[:3].sum()
    return d * math.sqrt(max(FT_S ** 2 * t * t + FS_S ** 2 * (PDEV @ eta) @ eta, 0.0))


def W(eta, e, d, F):
    el = e - eta
    return 0.5 * el @ C @ el + F(eta, d)


# ----------------------------------------------------------------------
# Two-multiplier criterion: the equations of the paper, solved exactly
# ----------------------------------------------------------------------
def two_multiplier(e, d, direction_strain=None):
    """eta = lambda_1 G_1 + lambda_2 G_2 from eq:fracture_multiplier_derivative. The directions
    follow from ``direction_strain`` (default e, as in the paper)."""
    g = e if direction_strain is None else direction_strain
    s = 1.0 if g[:3].sum() >= 0.0 else -1.0
    G1 = s * I6 / SQRT3
    dev = PDEV @ g
    G2 = dev / np.linalg.norm(dev)
    a, b = e @ G1, max(e @ G2, 0.0)
    if s < 0.0:                                    # compressive branch: F = P sqrt3 l1 + d fs* l2
        l1 = max(a - PEN / (SQRT3 * BULK), 0.0)
        l2 = max(b - d * FS_S / (2 * MU), 0.0)
    elif (3 * BULK * a) ** 2 / (3 * FT_S ** 2) + (2 * MU * b) ** 2 / FS_S ** 2 <= d * d:
        l1 = l2 = 0.0
    else:
        def g_of(Q):
            return 3 * FT_S ** 2 * (BULK * a / (BULK * Q + d * FT_S ** 2)) ** 2 \
                + FS_S ** 2 * (2 * MU * b / (2 * MU * Q + d * FS_S ** 2)) ** 2 - 1.0
        hi = FS_S * (abs(a) + abs(b)) * 10 + 1.0
        Q = brentq(g_of, 0.0, hi, xtol=1e-30, rtol=1e-15, maxiter=500)
        l1 = BULK * a * Q / (BULK * Q + d * FT_S ** 2)
        l2 = 2 * MU * b * Q / (2 * MU * Q + d * FS_S ** 2)
    return l1 * G1 + l2 * G2, (l1, l2), (G1, G2)


def residual_two(eta, e, d):
    """R = -sigma + d dF_d/d eta (smooth points: tr eta > 0)."""
    t = eta[:3].sum()
    dev = PDEV @ eta
    Fd = math.sqrt(FT_S ** 2 * t * t + FS_S ** 2 * dev @ dev)
    return -C @ (e - eta) + d * (FT_S ** 2 * t * I6 + FS_S ** 2 * dev) / Fd


# ----------------------------------------------------------------------
# Opening-only criterion: the prototype update (principal dyads of e)
# ----------------------------------------------------------------------
def run(update, eps, dam, dt=1.0, old=None):
    n = eps.shape[0]
    out = {k: np.zeros(n * m) for k, m in (("stress", 6), ("eta", 6), ("strain_v", 6), ("tangent", 36),
                                           ("energy", 1), ("psi", 1), ("lam_v", 1), ("lam_c", 2))}
    old = np.zeros(n * 6) if old is None else old.copy().ravel()
    update(eps.copy().ravel(), dt, dam.copy(), out["stress"], out["eta"], np.zeros(n * 6), out["strain_v"], old,
           out["tangent"], out["energy"], out["psi"], out["lam_v"], out["lam_c"], np.zeros(n * 2))
    return {k: v.reshape(n, -1) for k, v in out.items()}


def random_states(rng, n, lo=-0.5, hi=1.0):
    eps = np.empty((n, 6))
    for i in range(n):
        X = rng.normal(size=(3, 3))
        eps[i] = mandel(0.5 * (X + X.T)) * S0 * 10.0 ** rng.uniform(lo, hi)
    eps[:, :3] += rng.normal(size=(n, 1)) * S0 * 2.0          # tensile and compressive traces
    dam = np.where(rng.uniform(size=n) < 0.4, 1.0, 10.0 ** rng.uniform(-2.0, 0.0, size=n))
    return eps, dam


def fd_gradient(fun, x, h):
    g = np.empty(x.size)
    for k in range(x.size):
        dx = np.zeros(x.size)
        dx[k] = h
        g[k] = (fun(x + dx) - fun(x - dx)) / (2 * h)
    return g


def main():
    rng = np.random.default_rng(11)
    n = 300
    eps, dam = random_states(rng, n)
    psd = PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_C, kappa=KAPPA)
    kernel = CohesiveViscoElasticKernels(BULK, LAME, MU, 0.0, 1.0, F_T, F_C, 1.0, 0.0, F_C, KAPPA, "r2_2L",
                                         False, False)
    out_psd = run(psd, eps, dam)
    out_ker = run(kernel, eps, dam)

    # ---------------- 1. inequality for all admissible tensor variations ----------------
    print("1. Inequality of eq:fracture_weak for all admissible TENSOR variations at the multiplier solution")
    worst_two = worst_two_fd = 0.0
    count = {"elastic": 0, "tensile branch": 0, "compressive branch": 0}
    for e, d in zip(eps, dam):
        eta, lam, _ = two_multiplier(e, d)
        sig = C @ (e - eta)
        scale = d * F_T
        if not eta.any():
            count["elastic"] += 1
        elif eta[:3].sum() > 1e-14:
            count["tensile branch"] += 1
            worst_two = max(worst_two, np.abs(residual_two(eta, e, d)).max() / scale)     # all 6 components
        else:
            count["compressive branch"] += 1
        # one-sided directional derivatives in random tensor directions (all regimes, kinks included)
        for _ in range(200):
            delta = rng.normal(size=6)
            delta /= np.linalg.norm(delta)
            worst_two_fd = min(worst_two_fd, (-sig @ delta + dF_two(eta, d, delta)) / scale)
    print(f"   two-multiplier ({count}): largest |R| over all six components at smooth points "
          f"{worst_two:.1e} (relative to d f_t);")
    print(f"      smallest (exact, one-sided) directional derivative of W over 200 random tensor directions per "
          f"state {worst_two_fd:.1e} (relative; >= 0 required)")
    # opening-only: normal cone of the positive semidefinite cone: R >= 0 (psd) and R : eta = 0
    worst_min = worst_compl = 0.0
    n_open = 0
    worst_fd = 0.0
    for e, d, eta in zip(eps, dam, out_psd["eta"]):
        scale = d * F_T
        w0 = W(eta, e, d, F_open)
        for _ in range(200):                       # admissible directions: towards other psd tensors
            X = rng.normal(size=(3, 3))
            target = mandel(X @ X.T) * S0 * 10.0 ** rng.uniform(-2, 1)
            step = 1e-6 * (target - eta)
            worst_fd = min(worst_fd, (W(eta + step, e, d, F_open) - w0) / np.linalg.norm(step) / scale)
        if np.abs(eta).max() < 1e-13:
            continue
        n_open += 1
        t = eta[:3].sum()
        dev = PDEV @ eta
        R = -C @ (e - eta) + d * (FT_S ** 2 * t * I6 + FS_S ** 2 * dev) / math.sqrt(FT_S ** 2 * t * t
                                                                                    + FS_S ** 2 * dev @ dev)
        worst_min = min(worst_min, np.linalg.eigvalsh(tensor(R)).min() / scale)
        worst_compl = max(worst_compl, abs(R @ eta) / (scale * np.linalg.norm(eta)))
    print(f"   opening-only ({n_open} of {n} states opening): smallest eigenvalue of R {worst_min:.1e}, "
          f"|R : eta| {worst_compl:.1e} (relative);")
    print(f"      smallest directional derivative of W towards 200 random psd tensors per state {worst_fd:.1e}")

    # ---------------- 2. independent minimisation over the full tensor ----------------
    print("2. Independent minimisation over the full tensor eta (BFGS from random starts)")
    diff_two = diff_open = 0.0
    lower_two = lower_open = 0.0
    n_two = n_op = 0
    for i in range(n):
        e, d = eps[i], dam[i]
        eta, _, _ = two_multiplier(e, d)
        if eta[:3].sum() > 1e-8 and n_two < 40:    # smooth minimum: unconstrained, six components
            n_two += 1
            ref = W(eta, e, d, F_two)
            best = None
            for _ in range(3):
                x0 = (eta + rng.normal(size=6) * np.linalg.norm(eta) * 0.5) / S0
                res = minimize(lambda x: W(x * S0, e, d, F_two) / ref, x0, method="BFGS",
                               options={"gtol": 1e-12, "maxiter": 5000})
                if best is None or res.fun < best.fun:
                    best = res
            diff_two = max(diff_two, np.linalg.norm(best.x * S0 - eta) / np.linalg.norm(eta))
            lower_two = min(lower_two, best.fun - 1.0)
        eta = out_psd["eta"][i]
        if np.abs(eta).max() > 1e-8 and n_op < 40:  # eta = L L^T: positive semidefinite by construction
            n_op += 1
            ref = W(eta, e, d, F_open)

            def fun(x):
                L = np.array([[x[0], 0, 0], [x[1], x[2], 0], [x[3], x[4], x[5]]])
                return W(mandel(L @ L.T) * S0, e, d, F_open) / ref
            best = None
            for _ in range(6):
                res = minimize(fun, rng.normal(size=6) * math.sqrt(np.linalg.norm(eta) / S0), method="BFGS",
                               options={"gtol": 1e-12, "maxiter": 20000})
                if best is None or res.fun < best.fun:
                    best = res
            x = best.x
            L = np.array([[x[0], 0, 0], [x[1], x[2], 0], [x[3], x[4], x[5]]])
            diff_open = max(diff_open, np.linalg.norm(mandel(L @ L.T) * S0 - eta) / np.linalg.norm(eta))
            lower_open = min(lower_open, best.fun - 1.0)
    print(f"   two-multiplier, {n_two} states: largest |eta_min - eta_multiplier| / |eta| {diff_two:.1e}, "
          f"energy found below the multiplier solution: {lower_two:.1e} (relative)")
    print(f"   opening-only,   {n_op} states: largest |eta_min - eta_principal| / |eta| {diff_open:.1e}, "
          f"energy found below the principal solution: {lower_open:.1e} (relative)")

    # ---------------- 3. stress = derivative of the reduced energy ----------------
    print("3. Stress C (e - eta) against the derivative of the reduced energy min_eta W with respect to the strain")
    h = 2e-4 * S0
    err_two = err_open = 0.0
    for i in range(0, n, 3):
        e, d = eps[i], dam[i]
        eta = two_multiplier(e, d)[0]
        sig = C @ (e - eta)
        grad = fd_gradient(lambda x: W(two_multiplier(x, d)[0], x, d, F_two), e, h)
        err_two = max(err_two, np.abs(grad - sig).max() / max(np.abs(sig).max(), d * F_T))
        sig = out_psd["stress"][i]

        def reduced(x):
            return W(run(psd, x[None], np.array([d]))["eta"][0], x, d, F_open)
        grad = fd_gradient(reduced, e, h)
        err_open = max(err_open, np.abs(grad - sig).max() / max(np.abs(sig).max(), d * F_T))
    print(f"   two-multiplier: largest relative difference {err_two:.1e};  opening-only: {err_open:.1e}")

    # ---------------- 4. counter-example ----------------
    print("4. Counter-example: G_2 from the total strain, with a creep strain present (eps_v != 0)")
    err_bad = err_formula = 0.0
    k = 0
    for i in range(n):
        e, d = eps[i], dam[i]
        ev = PDEV @ (rng.normal(size=6) * S0)
        total = e + ev
        eta, lam, (G1, G2) = two_multiplier(e, d, direction_strain=total)
        if lam[1] <= 0.0 or lam[0] <= 0.0:
            continue
        k += 1
        sig = C @ (e - eta)
        grad = fd_gradient(lambda x: W(two_multiplier(x - ev, d, direction_strain=x)[0], x - ev, d, F_two),
                           total, h)
        R = residual_two(eta, e, d)
        missing = lam[1] * (PDEV - np.outer(G2, G2)) @ R / np.linalg.norm(PDEV @ total)
        err_bad = max(err_bad, np.abs(grad - sig).max() / np.abs(sig).max())
        err_formula = max(err_formula, np.abs(grad - sig - missing).max() / np.abs(sig).max())
        if k == 60:
            break
    print(f"   {k} states: |dW/d eps - C (e - eta)| up to {err_bad:.1e} of the stress; "
          f"after adding lambda_2 R : dG_2/d eps the difference is {err_formula:.1e}")

    # ---------------- 5. the kernels ----------------
    print("5. Kernels")
    ours = np.array([C @ (e - two_multiplier(e, d)[0]) for e, d in zip(eps, dam)])
    scale = np.maximum(np.abs(ours).max(axis=1), dam * F_T)
    print(f"   C++ kernel (r2_2L, no creep) against the equations of the paper: largest stress difference "
          f"{(np.abs(out_ker['stress'] - ours).max(axis=1) / scale).max():.1e} (relative; the kernel has the "
          f"residual stiffness kappa K lambda)")

    def asym(out):
        T = out["tangent"].reshape(-1, 6, 6)
        return (np.abs(T - T.transpose(0, 2, 1)).max(axis=(1, 2)) / np.abs(T).max(axis=(1, 2))).max()
    print(f"   tangent asymmetry |T - T^T| / |T| without creep: C++ kernel {asym(out_ker):.1e}, "
          f"opening-only prototype {asym(out_psd):.1e}")
    dt = 3600.0
    old = np.array([PDEV @ v for v in rng.normal(size=(n, 6)) * S0 * 0.3])
    kernel_v = CohesiveViscoElasticKernels(BULK, LAME, MU, VISC, GLEN_N, F_T, F_C, 1.0, 0.0, F_C, KAPPA, "r2_2L",
                                           False, False)
    psd_v = PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_C, kappa=KAPPA, viscosity=VISC, n_exp=GLEN_N)
    eps_v = eps * 3.0                                # stresses high enough for a creep increment that matters
    ker_v, opn_v = run(kernel_v, eps_v, dam, dt, old), run(psd_v, eps_v, dam, dt, old)
    ratio = np.linalg.norm(opn_v["strain_v"] - old, axis=1) / np.linalg.norm(eps_v - old, axis=1)
    print(f"   with creep (dt {dt:g} s, creep increment up to {ratio.max():.2f} of the trial strain): tangent "
          f"asymmetry C++ kernel {asym(ker_v):.1e}, opening-only prototype {asym(opn_v):.1e}")
    # incremental potential with creep: psi_el + d F(eta) + dt Phi_v(dv / dt), opening-only prototype
    dv = opn_v["strain_v"] - old
    s = (PDEV @ opn_v["stress"].T).T
    ns, nv = np.linalg.norm(s, axis=1), np.linalg.norm(dv, axis=1)
    c0 = nv / (dt * ns ** GLEN_N)                   # flow rule dv / dt = c0 |s|^(n - 1) s
    c0 = np.median(c0[nv > 1e-12])
    print(f"   creep law of the update: dv/dt = c0 |s|^(n-1) s with c0 = {c0:.4e} "
          f"(spread {np.ptp((nv / (dt * ns ** GLEN_N))[nv > 1e-9 * S0]) / c0:.1e})")

    def incremental(x, d, o):
        r = run(psd_v, x[None], np.array([d]), dt, o[None])
        v = r["strain_v"][0] - o
        rate = np.linalg.norm(v) / dt
        phi = GLEN_N / (GLEN_N + 1) * c0 ** (-1 / GLEN_N) * rate ** ((GLEN_N + 1) / GLEN_N)
        return W(r["eta"][0], x - o - v, d, F_open) + dt * phi
    err = 0.0
    for i in range(0, n, 5):
        grad = fd_gradient(lambda x: incremental(x, dam[i], old[i]), eps_v[i], h)
        sig = opn_v["stress"][i]
        err = max(err, np.abs(grad - sig).max() / max(np.abs(sig).max(), dam[i] * F_T))
    print(f"   opening-only with creep: stress against the derivative of the incremental potential: {err:.1e}")


if __name__ == "__main__":
    main()
