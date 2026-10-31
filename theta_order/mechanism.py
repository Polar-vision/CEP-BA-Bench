"""
Mechanism separation for the Theta-order paper.

Claim under test (Sec. 6.4): a chart changes the *path* of the iteration, not its
*limit*.  Two facts frame the experiment.

(i)  The linearised Gauss-Newton step is chart-independent:
         S (S'A'AS)^-1 S'A' = (A'A)^-1 A' ,
     so the tangent step in X-space is the same for every chart (verified below
     to 3e-15).  What differs is where the iterate *lands*, because the update is
     X(q + dq), not X + S dq: the charts differ at second order.

(ii) For a pinhole camera the residual is affine in inverse depth,
         e = f (x/z, y/z) = f rho (x, y),      rho = 1/z ,
     exactly.  Hence a chart whose scale coordinate is affine in rho solves the
     depth direction in a single Newton step, and a chart whose coordinate is
     Lambda does not.  This, not conditioning, is what separates the families at
     a fixed iteration budget.

Experiments
  A  tangent-step invariance and analytic-Jacobian check (double).
  B  one step from +50% depth, eight scenarios: which charts are exact, and does
     the outcome track the condition number?  (It does not.)
  C  error vs iteration budget, degenerate and well-conditioned scenes.
  D  single precision: same ordering, floor set by rounding.
  E  Monte Carlo with pixel noise at convergence: every chart reaches the CRB.
  F  advantage vs initialization quality.

Run:  "E:/zuo/Program Files/Python313/python.exe" mechanism.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = HERE if os.path.exists(os.path.join(HERE, "verify_scenarios.py")) \
    else os.path.join(HERE, "..", "New_GSIS", "theta_derivation")
sys.path.insert(0, os.path.abspath(_SRC))
import verify_scenarios as vs                                    # noqa: E402

F32, F64 = np.float32, np.float64
FAMS = vs.FAMS
SHORT = {"XYZ": "XYZ", "InvZ": "InvZ", "SphRange": "SphRng",
         "SphInvRange": "SphInv", "Parallax": "Parallax"}
BUDGETS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30]


# --------------------------------------------------------------- chart Jacobian
def S_analytic(sc, fam, q):
    """dX/dq in closed form.  Central differences are not accurate enough: for the
    parallax chart dLambda/domega ~ 4.5e3, and the truncation error, amplified by
    the block condition number, would masquerade as a chart effect."""
    if fam == "XYZ":
        return np.eye(3)
    a, b = float(q[0]), float(q[1])
    w = sc.u0 + a * sc.e1 + b * sc.e2
    n = np.linalg.norm(w)
    m = w / n
    dma = (sc.e1 - (sc.e1 @ m) * m) / n
    dmb = (sc.e2 - (sc.e2 @ m) * m) / n

    if fam == "SphRange":
        s = float(q[2])
        return np.column_stack([s * dma, s * dmb, m])
    if fam == "SphInvRange":
        s = float(q[2])
        return np.column_stack([dma / s, dmb / s, -m / s ** 2])
    if fam == "InvZ":
        rho = float(q[2])
        Rt = sc.Rs[0].T
        return np.column_stack([
            Rt @ np.array([1.0 / rho, 0.0, 0.0]),
            Rt @ np.array([0.0, 1.0 / rho, 0.0]),
            Rt @ np.array([-a / rho ** 2, -b / rho ** 2, -1.0 / rho ** 2])])
    if fam == "Parallax":
        B, bh = sc.B, sc.bhat
        om = float(q[2])
        beta = np.arccos(np.clip(m @ bh, -1.0, 1.0))
        Lam = B * np.sin(beta + om) / np.sin(om)
        kap = B * np.sin(beta)
        dLam_dom = -kap / np.sin(om) ** 2
        dLam_dbeta = B * np.cos(beta + om) / np.sin(om)
        db_a = -(dma @ bh) / np.sin(beta)
        db_b = -(dmb @ bh) / np.sin(beta)
        return np.column_stack([Lam * dma + dLam_dbeta * db_a * m,
                                Lam * dmb + dLam_dbeta * db_b * m,
                                dLam_dom * m])
    raise KeyError(fam)


# --------------------------------------------------------------------- solver
def depth_error(sc, X):
    return abs(np.linalg.norm(X - sc.Cm) - sc.Lam) / sc.Lam


def build(sc, dtype, obs=None):
    Rs = [np.array(R, dtype=dtype) for R in sc.Rs]
    Cs = [np.array(C, dtype=dtype) for C in sc.Cs]
    ob = [np.array(o, dtype=dtype) for o in
          (sc.obs if obs is None else obs)]
    f = dtype(sc.f)

    def resid(X):
        out = []
        for R, C, o in zip(Rs, Cs, ob):
            z = R @ (X - C)
            out.append(np.array([f * z[0] / z[2], f * z[1] / z[2]], dtype=dtype) - o)
        return np.concatenate(out)

    def jac(X):
        rows = []
        for R, C in zip(Rs, Cs):
            z = R @ (X - C)
            zz = z[2]
            rows.append(np.array([[f / zz, dtype(0), -f * z[0] / zz ** 2],
                                  [dtype(0), f / zz, -f * z[1] / zz ** 2]],
                                 dtype=dtype) @ R)
        return np.vstack(rows)

    return resid, jac


def gn(sc, fam, n_iter=30, dL=0.5, ang=2e-3, dtype=F64, obs=None):
    """Free Gauss-Newton, fixed budget, no tolerance exit.  Returns depth error
    after every iteration (index 0 = initial) and the first tangent step."""
    resid, jac = build(sc, dtype, obs)
    Xp = sc.X * (1.0 + dL) + ang * sc.Lam * sc.e1
    q = sc.q_of(fam, Xp).astype(dtype)
    errs = [depth_error(sc, np.array(sc.X_of(fam, q.astype(F64)), F64))]
    first = None
    for _ in range(n_iter):
        X = np.array(sc.X_of(fam, q.astype(F64)), dtype=dtype)
        r = resid(X)
        J = jac(X) @ np.array(S_analytic(sc, fam, q.astype(F64)), dtype=dtype)
        try:
            dq = -np.linalg.solve((J.T @ J).astype(dtype), (J.T @ r).astype(dtype))
        except np.linalg.LinAlgError:
            errs.append(np.inf)
            continue
        if not np.all(np.isfinite(dq)):
            errs.append(np.inf)
            continue
        if first is None:
            first = np.array(S_analytic(sc, fam, q.astype(F64)) @ dq.astype(F64))
        q = vs.clamp_q(fam, (q + dq).astype(dtype))
        errs.append(depth_error(sc, np.array(sc.X_of(fam, q.astype(F64)), F64)))
    return errs, first


def crb(sc):
    A = sc.A()
    cov = np.linalg.inv(A.T @ A) * sc.sigma ** 2
    return float(np.sqrt(sc.u0 @ cov @ sc.u0) / sc.Lam)


# ------------------------------------------------------------------ experiments
def exp_A(sc):
    print("\n[A] tangent step is chart-independent (double, free step)")
    worst_j, worst_dx = 0.0, 0.0
    for fam in FAMS:
        q = sc.q_of(fam, sc.X)
        Sa, Sf = S_analytic(sc, fam, q), sc.S(fam, q)
        worst_j = max(worst_j, np.abs(Sa - Sf).max() / np.abs(Sa).max())
    _, dX_ref = gn(sc, "XYZ", n_iter=1)
    for fam in FAMS:
        _, dX = gn(sc, fam, n_iter=1)
        worst_dx = max(worst_dx, np.linalg.norm(dX - dX_ref) / np.linalg.norm(dX_ref))
    print(f"    analytic vs central-difference Jacobian : {worst_j:.2e}")
    print(f"    first tangent step, max chart deviation : {worst_dx:.2e}")
    return dict(jac=float(worst_j), dX=float(worst_dx))


def exp_B():
    """One Newton step, all scenarios: which charts are exact in depth?"""
    print("\n[B] depth error after ONE step from +50% (double)")
    hdr = (f"    {'scenario':<22}{'omega[deg]':>11}{'cond_XYZ':>10}{'cond_Par':>9}"
           + "".join(f"{SHORT[f]:>10}" for f in FAMS))
    print(hdr)
    rows = []
    for name, Lam, B, beta, f, sig, need, _note in vs.SCEN:
        sc = vs.Scene(Lam, B, beta, f, n_c=4, sigma_px=sig)
        m = vs.metrics(sc, "XYZ")
        mp = vs.metrics(sc, "Parallax")
        e1 = {}
        for fam in FAMS:
            tr, _ = gn(sc, fam, n_iter=1)
            e1[fam] = tr[1]
        rows.append(dict(name=name, omega=float(sc.omega(sc.u0)),
                         cond_xyz=float(m["cond"]), cond_par=float(mp["cond"]),
                         err=e1, crb=crb(sc)))
        print(f"    {name:<22}{np.degrees(sc.omega(sc.u0)):>11.4f}"
              f"{m['cond']:>10.2e}{mp['cond']:>9.2f}"
              + "".join(f"{e1[fam]:>10.2e}" for fam in FAMS))
    xyz = [r["err"]["XYZ"] for r in rows]
    par = [r["err"]["Parallax"] for r in rows]
    print(f"    XYZ       : {min(xyz):.3e} .. {max(xyz):.3e}  (factor {max(xyz)/min(xyz):.1f})"
          f"   cond spans {max(r['cond_xyz'] for r in rows)/min(r['cond_xyz'] for r in rows):.1e}")
    print(f"    Parallax  : {min(par):.3e} .. {max(par):.3e}  (factor {max(par)/min(par):.1f})")
    return rows


def exp_C(name, sc):
    tr = {fam: gn(sc, fam, n_iter=30)[0] for fam in FAMS}
    print(f"\n[C] {name}: Lambda={sc.Lam} m, B={sc.B} m, "
          f"omega={np.degrees(sc.omega(sc.u0)):.3f} deg, "
          f"cond_XYZ={vs.metrics(sc, 'XYZ')['cond']:.2e}")
    print(f"    {'N':>3}" + "".join(f"{SHORT[f]:>11}" for f in FAMS) + f"{'spread':>10}")
    for k in BUDGETS:
        vals = [tr[fam][min(k, len(tr[fam]) - 1)] for fam in FAMS]
        pos = [v for v in vals if v > 1e-14]
        sp = (max(pos) / min(pos)) if len(pos) > 1 else float("nan")
        print(f"    {k:>3}" + "".join(f"{v:>11.2e}" for v in vals) + f"{sp:>10.1f}")
    return {f: tr[f] for f in FAMS}


def exp_D(sc):
    print("\n[D] single precision, same protocol")
    tr = {fam: gn(sc, fam, n_iter=30, dtype=F32)[0] for fam in FAMS}
    print(f"    {'N':>3}" + "".join(f"{SHORT[f]:>11}" for f in FAMS))
    for k in (1, 3, 10, 30):
        print(f"    {k:>3}" + "".join(f"{tr[fam][k]:>11.2e}" for fam in FAMS))
    return {f: tr[f] for f in FAMS}


def exp_E(sc, n_mc=200, budgets=(1, 2, 3, 5, 8, 12, 20, 40)):
    print(f"\n[E] Monte Carlo, sigma={sc.sigma} px: does every chart reach the CRB?")
    rng = np.random.default_rng(11)
    acc = {fam: {k: [] for k in budgets} for fam in FAMS}
    for _ in range(n_mc):
        noisy = [o + rng.normal(0.0, sc.sigma, 2) for o in sc.obs]
        for fam in FAMS:
            tr, _ = gn(sc, fam, n_iter=max(budgets), obs=noisy)
            for k in budgets:
                acc[fam][k].append(tr[min(k, len(tr) - 1)])
    rows = []
    for k in budgets:
        row = dict(budget=k)
        for fam in FAMS:
            v = np.array(acc[fam][k])
            v = v[np.isfinite(v)]
            row[fam] = float(np.sqrt(np.mean(v ** 2))) if len(v) else np.inf
        row["crb"] = crb(sc)
        rows.append(row)
    print(f"    {'N':>3}" + "".join(f"{SHORT[f]:>11}" for f in FAMS) + f"{'CRB':>11}")
    for row in rows:
        print(f"    {row['budget']:>3}" + "".join(f"{row[f]:>11.3e}" for f in FAMS)
              + f"{row['crb']:>11.3e}")
    return rows


def exp_F(sc, inits=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0)):
    print("\n[F] advantage at N=3 vs initialization quality")
    rows = []
    for dL in inits:
        e = {fam: gn(sc, fam, n_iter=3, dL=dL)[0][3] for fam in ("XYZ", "Parallax")}
        rows.append(dict(dL=dL, xyz=e["XYZ"], par=e["Parallax"],
                         ratio=float(e["XYZ"] / max(e["Parallax"], 1e-300))))
        print(f"    init +{dL*100:5.0f}%   XYZ={e['XYZ']:.3e}   "
              f"Parallax={e['Parallax']:.3e}   ratio={rows[-1]['ratio']:.3e}")
    return rows


def main():
    deg = dict(name="degenerate", Lam=15.0, B=5.0e-2, beta=90.0, f=3.64e4)
    tol = dict(name="well-conditioned", Lam=15.0, B=8.0e-1, beta=90.0, f=3.64e4)
    sc_d = vs.Scene(deg["Lam"], deg["B"], deg["beta"], deg["f"], n_c=4)
    sc_t = vs.Scene(tol["Lam"], tol["B"], tol["beta"], tol["f"], n_c=4)

    out = dict(deg=deg, tol=tol, crb=crb(sc_d))

    out["A"] = exp_A(sc_d)
    out["B"] = exp_B()
    out["C_deg"] = exp_C("degenerate", sc_d)
    out["C_tol"] = exp_C("well-conditioned", sc_t)
    out["D"] = exp_D(sc_d)
    out["E"] = exp_E(sc_d)
    out["F"] = exp_F(sc_d)

    with open(os.path.join(HERE, "mechanism.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote mechanism.json")


if __name__ == "__main__":
    main()
