"""
The two mechanisms that mechanism.py could not reach: local minima (basin
selection) and step acceptance (damping).

mechanism.py runs a point-block Gauss-Newton without step control, so both
mechanisms are outside its protocol by construction.  Here the camera block is
switched back on and the solver becomes a Levenberg--Marquardt with a gain-ratio
acceptance test, which is what a real bundle adjustment does.

Protocol
  A small rig of four cameras; the first two are held fixed, because they define
  the reference frame and the metric scale and therefore remove the seven-dof
  similarity gauge.  The remaining cameras and all points are optimised jointly.
  Observations are noiseless, so the global minimum is the truth with zero cost.

Experiments
  G  acceptance: iterations, accepted and rejected steps, final damping, gain
     ratios and final error for the five charts, in a short-baseline and in a
     well-conditioned scene, with Marquardt (diag) and with identity damping.
  H  basins: random starts, paired across charts (common random numbers);
     how many reach the true minimum, and among those that reach the same
     cost, how far apart the recovered depths are.

Run:  "E:/zuo/Program Files/Python313/python.exe" mechanism2.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = HERE if os.path.exists(os.path.join(HERE, "verify_scenarios.py")) \
    else os.path.join(HERE, "..", "New_GSIS", "theta_derivation")
sys.path.insert(0, os.path.abspath(_SRC))
sys.path.insert(0, HERE)
import verify_scenarios as vs                                    # noqa: E402
import mechanism as mech                                         # noqa: E402

FAMS = vs.FAMS
SHORT = {"XYZ": "XYZ", "InvZ": "InvZ", "SphRange": "SphRng",
         "SphInvRange": "SphInv", "Parallax": "Parallax"}


# ------------------------------------------------------------------ geometry
def so3_exp(w):
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def skew(z):
    return np.array([[0, -z[2], z[1]], [z[2], 0, -z[0]], [-z[1], z[0], 0]])


class BAProblem:
    """n_c cameras on the baseline, n_pt points in a cone; cameras 0 and 1 fixed."""

    def __init__(self, name, Lam, B, beta_deg, f, n_c=4, n_pt=8,
                 spread_deg=2.0, seed=3):
        self.name = name
        self.frame = vs.Scene(Lam, B, beta_deg, f, n_c=n_c, sigma_px=0.5)
        fr = self.frame
        self.Lam, self.B, self.f, self.n_c, self.n_pt = Lam, B, f, n_c, n_pt
        rng = np.random.default_rng(seed)
        P = []
        for _ in range(n_pt):
            a = np.radians(rng.uniform(-spread_deg, spread_deg))
            b = np.radians(rng.uniform(-spread_deg, spread_deg))
            d = Lam * float(np.exp(0.08 * rng.standard_normal()))
            P.append(fr.Cm + d * fr.mab(a, b))
        self.P = np.array(P)
        self.R = [R.copy() for R in fr.Rs]
        self.C = [C.copy() for C in fr.Cs]
        self.obs = np.array([[vs.pi(R @ (X - C), f) for X in self.P]
                             for R, C in zip(self.R, self.C)])
        self.fixed = [0, 1]
        self.nv = 6 * (n_c - len(self.fixed)) + 3 * n_pt

    # ---- chart maps for every point
    def Xs(self, fam, qs):
        return np.array([self.frame.X_of(fam, q) for q in qs])

    def qs_of(self, fam, X):
        return [self.frame.q_of(fam, x) for x in X]

    # ---- residuals and Jacobian
    def rj(self, fam, qs, R, C):
        f = self.f
        nC = self.n_c
        S = np.array([mech.S_analytic(self.frame, fam, q) for q in qs])
        X = self.Xs(fam, qs)
        nfree = nC - len(self.fixed)
        r = np.zeros(2 * nC * self.n_pt)
        J = np.zeros((2 * nC * self.n_pt, 6 * nfree + 3 * self.n_pt))
        k = 0
        col_c = {}
        c = 0
        for j in range(nC):
            if j not in self.fixed:
                col_c[j] = 6 * c
                c += 1
        base_p = 6 * nfree
        for j in range(nC):
            Rj, Cj = R[j], C[j]
            Z = (Rj @ (X - Cj).T).T                      # (P,3)
            zx, zy, zz = Z[:, 0], Z[:, 1], Z[:, 2]
            D = np.zeros((self.n_pt, 2, 3))
            D[:, 0, 0] = f / zz
            D[:, 0, 2] = -f * zx / zz ** 2
            D[:, 1, 1] = f / zz
            D[:, 1, 2] = -f * zy / zz ** 2
            e = np.stack([f * zx / zz, f * zy / zz], axis=1) - self.obs[j]
            if j not in self.fixed:
                Sx = np.zeros((self.n_pt, 3, 3))
                Sx[:, 0, 1] = -Z[:, 2]
                Sx[:, 0, 2] = Z[:, 1]
                Sx[:, 1, 0] = Z[:, 2]
                Sx[:, 1, 2] = -Z[:, 0]
                Sx[:, 2, 0] = -Z[:, 1]
                Sx[:, 2, 1] = Z[:, 0]
                Jth = -np.einsum('ipk,ikl->ipl', D, Sx)      # d e / d theta
                Jt = -np.einsum('ipk,kl->ipl', D, Rj)        # d e / d C
            Jp = np.einsum('ipk,kl,ilm->ipm', D, Rj, S)      # d e / d q
            for i in range(self.n_pt):
                r[k:k + 2] = e[i]
                if j not in self.fixed:
                    J[k:k + 2, col_c[j]:col_c[j] + 3] = Jth[i]
                    J[k:k + 2, col_c[j] + 3:col_c[j] + 6] = Jt[i]
                J[k:k + 2, base_p + 3 * i:base_p + 3 * i + 3] = Jp[i]
                k += 2
        return r, J

    def cost(self, fam, qs, R, C):
        r, _ = self.rj(fam, qs, R, C)
        return 0.5 * float(r @ r)

    def depth_err(self, fam, qs):
        X = self.Xs(fam, qs)
        d = np.linalg.norm(X - self.C[0], axis=1)
        dt = np.linalg.norm(self.P - self.C[0], axis=1)
        return float(np.median(np.abs(d - dt) / dt))

    def cam_err(self, R, C):
        out = []
        for j in range(self.n_c):
            if j in self.fixed:
                continue
            out.append(np.linalg.norm(C[j] - self.C[j]) / self.B)
            tr = np.clip((np.trace(R[j].T @ self.R[j]) - 1.0) / 2.0, -1.0, 1.0)
            out.append(np.arccos(tr))
        return float(np.mean(out)) if out else 0.0


# ------------------------------------------------------------------- solver
def step_state(prob, fam, qs, R, C, d):
    nfree = prob.n_c - len(prob.fixed)
    R2 = [r.copy() for r in R]
    C2 = [c.copy() for c in C]
    c = 0
    for j in range(prob.n_c):
        if j in prob.fixed:
            continue
        R2[j] = so3_exp(d[6 * c:6 * c + 3]) @ R[j]
        C2[j] = C[j] + d[6 * c + 3:6 * c + 6]
        c += 1
    q2 = []
    b = 6 * nfree
    for i in range(prob.n_pt):
        q2.append(vs.clamp_q(fam, qs[i] + d[b + 3 * i:b + 3 * i + 3]))
    return q2, R2, C2


def lm(prob, fam, q0, R0, C0, max_iter=200, damping="diag", gtol=1e-11,
       record=False):
    qs = [np.array(q, float) for q in q0]
    R = [r.copy() for r in R0]
    C = [c.copy() for c in C0]
    r, J = prob.rj(fam, qs, R, C)
    F = 0.5 * float(r @ r)
    lam, nu = 1e-4, 2.0
    g = J.T @ r
    g0 = float(np.max(np.abs(g)))
    st = dict(iter=0, acc=0, rej=0, rho=[], F=[F], lam=[lam], path=[],
              gnorm=[g0])
    if record:
        st["path"].append(prob.depth_err(fam, qs))
    for it in range(max_iter):
        H = J.T @ J
        g = J.T @ r
        if np.max(np.abs(g)) <= gtol * max(g0, 1e-12):
            break
        Dg = np.diag(np.diag(H)) if damping == "diag" else np.eye(H.shape[0])
        try:
            d = -np.linalg.solve(H + lam * Dg, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break
        pred = -(g @ d + 0.5 * d @ H @ d)
        q2, R2, C2 = step_state(prob, fam, qs, R, C, d)
        r2, J2 = prob.rj(fam, q2, R2, C2)
        F2 = 0.5 * float(r2 @ r2)
        rho = (F - F2) / pred if pred > 1e-300 else -1.0
        st["iter"] += 1
        if rho > 0 and np.isfinite(F2) and F2 < F:
            qs, R, C, r, J, F = q2, R2, C2, r2, J2, F2
            st["acc"] += 1
            st["rho"].append(float(rho))
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
        else:
            st["rej"] += 1
            lam *= nu
            nu *= 2.0
        st["F"].append(F)
        st["lam"].append(lam)
        if record:
            st["path"].append(prob.depth_err(fam, qs))
        if lam > 1e12:
            break
    st["F_final"] = F
    st["lam_final"] = lam
    st["q"] = qs
    st["R"], st["C"] = R, C
    st["depth"] = prob.depth_err(fam, qs)
    st["cam"] = prob.cam_err(R, C)
    st["mean_rho"] = float(np.mean(st["rho"])) if st["rho"] else float("nan")
    st["converged"] = bool(np.max(np.abs(J.T @ r)) <= gtol * max(g0, 1e-12))
    return st


def init_state(prob, fam, dL=0.35, ang=1.0e-2, camT=0.05, camR=2e-3, rng=None):
    """Perturb points and the free cameras; identical across charts."""
    if rng is None:
        rng = np.random.default_rng(7)
    X0 = []
    for X in prob.P:
        v = X - prob.C[0]
        L = np.linalg.norm(v)
        m = v / L
        m2 = m + ang * (rng.standard_normal() * prob.frame.e1
                        + rng.standard_normal() * prob.frame.e2)
        m2 /= np.linalg.norm(m2)
        X0.append(prob.C[0] + L * (1.0 + dL) * m2)
    q0 = prob.qs_of(fam, np.array(X0))
    R0, C0 = [], []
    for j in range(prob.n_c):
        if j in prob.fixed:
            R0.append(prob.R[j].copy())
            C0.append(prob.C[j].copy())
        else:
            w = camR * rng.standard_normal(3)
            R0.append(so3_exp(w) @ prob.R[j])
            C0.append(prob.C[j] + camT * prob.B * rng.standard_normal(3))
    return q0, R0, C0


# ---------------------------------------------------------------- experiments
REGIMES = dict(
    mild=dict(dL=(0.15, 0.45), ang=(3e-3, 1.0e-2),
              camT=(0.0, 0.03), camR=(0.0, 1.5e-3)),
    hard=dict(dL=(0.30, 3.00), ang=(5e-3, 4.0e-2),
              camT=(0.0, 0.20), camR=(0.0, 8.0e-3)),
)


def draw(rng, reg):
    r = REGIMES[reg]
    return dict(dL=float(rng.uniform(*r["dL"])), ang=float(rng.uniform(*r["ang"])),
                camT=float(rng.uniform(*r["camT"])),
                camR=float(rng.uniform(*r["camR"])))


def summarise(rows, fam, scene, regime, damping):
    it = np.array([r["it"] for r in rows], float)
    ac = np.array([r["acc"] for r in rows], float)
    rj = np.array([r["rej"] for r in rows], float)
    Fv = np.array([r["F"] for r in rows])
    d = np.array([r["depth"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    return dict(scene=scene, regime=regime, damping=damping, fam=fam,
                it=float(np.median(it)), acc=float(np.mean(ac)),
                rej=float(np.mean(rj)), rho=float(np.nanmean(rho)),
                medF=float(np.median(Fv)), med=float(np.median(d)),
                ok=float(np.mean(ok)), n=len(rows),
                runs=rows)


def one_run(prob, fam, start, regime, max_iter, damping="diag"):
    draw_rng = np.random.default_rng(9000 + start)
    cfg = draw(draw_rng, regime)
    q0, R0, C0 = init_state(prob, fam, cfg["dL"], cfg["ang"], cfg["camT"],
                            cfg["camR"], np.random.default_rng(9000 + start))
    st = lm(prob, fam, q0, R0, C0, max_iter=max_iter, damping=damping)
    X = prob.Xs(fam, st["q"])
    dt = np.linalg.norm(prob.P - prob.C[0], axis=1)
    dr = np.linalg.norm(X - prob.C[0], axis=1) / dt
    rec = dict(start=start, fam=fam, dL=cfg["dL"], ang=cfg["ang"],
               camT=cfg["camT"], camR=cfg["camR"], it=st["iter"],
               acc=st["acc"], rej=st["rej"], rho=st["mean_rho"],
               F=st["F_final"], lam=st["lam_final"], depth=st["depth"],
               dmax=float(dr.max()), conv=st["converged"])
    rec["ok"] = bool(rec["depth"] < 1e-3 and rec["F"] < 1e-8)
    rec["escape"] = bool(rec["dmax"] > 10.0)
    rec["stall"] = bool(rec["lam"] > 1e10)
    return rec, st


def exp_G(prob, damping="diag", regime="hard", n_start=20, max_iter=250):
    print(f"\n[G] acceptance: {prob.name}, damping={damping}, regime={regime}, "
          f"{n_start} starts")
    print(f"    {'chart':<9}{'it':>5}{'acc':>6}{'rej':>6}{'rho':>7}"
          f"{'medF':>10}{'depth':>10}{'ok':>5}")
    out = []
    per = {fam: [] for fam in FAMS}
    for s in range(n_start):
        for fam in FAMS:
            rec, _ = one_run(prob, fam, s, regime, max_iter, damping)
            per[fam].append(rec)
    for fam in FAMS:
        sm = summarise(per[fam], fam, prob.name, regime, damping)
        out.append(sm)
        print(f"    {SHORT[fam]:<9}{sm['it']:>5.0f}{sm['acc']:>6.1f}{sm['rej']:>6.1f}"
              f"{sm['rho']:>7.3f}{sm['medF']:>10.1e}{sm['med']:>10.1e}"
              f"{sm['ok']:>5.0%}")
    return out


def exp_H(prob, regime="hard", n_start=60, max_iter=250):
    print(f"\n[H] basins: {prob.name}, regime={regime}, {n_start} starts")
    print(f"    {'chart':<9}{'ok':>5}{'esc':>5}{'stall':>6}{'med d':>10}"
          f"{'p90 d':>10}{'medF':>10}{'med it':>7}")
    out, per = [], {fam: [] for fam in FAMS}
    for s in range(n_start):
        for fam in FAMS:
            rec, _ = one_run(prob, fam, s, regime, max_iter)
            per[fam].append(rec)
    for fam in FAMS:
        rows = per[fam]
        sm = summarise(rows, fam, prob.name, regime, "diag")
        sm["escape"] = float(np.mean([r["escape"] for r in rows]))
        sm["stall"] = float(np.mean([r["stall"] for r in rows]))
        sm["p90"] = float(np.percentile([r["depth"] for r in rows], 90))
        out.append(sm)
        print(f"    {SHORT[fam]:<9}{sm['ok']:>5.0%}{sm['escape']:>5.0%}"
              f"{sm['stall']:>6.0%}{sm['med']:>10.1e}{sm['p90']:>10.1e}"
              f"{sm['medF']:>10.1e}{sm['it']:>7.0f}")
    return out


def exp_traj(prob, start=6, regime="hard", max_iter=250):
    """Two trajectories from the same start, for the figure."""
    out = {}
    for fam in ("SphRange", "Parallax", "XYZ"):
        _, st = one_run(prob, fam, start, regime, max_iter)
        out[fam] = dict(F=[float(v) for v in st["F"]],
                        lam=[float(v) for v in st["lam"]],
                        depth=st["depth"], Fend=st["F_final"],
                        rej=st["rej"], it=st["iter"])
    return out


def main():
    t0 = time.time()
    deg = BAProblem("short baseline", 15.0, 5.0e-2, 90.0, 3.64e4)
    tol = BAProblem("well conditioned", 15.0, 8.0e-1, 90.0, 3.64e4)
    print(f"short baseline: state {deg.nv}, residuals {2*deg.n_c*deg.n_pt}, "
          f"omega={np.degrees(deg.frame.omega(deg.frame.u0)):.3f} deg, "
          f"cond_XYZ={vs.metrics(deg.frame,'XYZ')['cond']:.2e}")

    out = dict(nv=deg.nv, nr=2 * deg.n_c * deg.n_pt,
               omega_deg=float(np.degrees(deg.frame.omega(deg.frame.u0))),
               cond_xyz=float(vs.metrics(deg.frame, "XYZ")["cond"]),
               cond_par=float(vs.metrics(deg.frame, "Parallax")["cond"]),
               n_pt=deg.n_pt, n_c=deg.n_c)
    out["G_diag"] = (exp_G(deg, "diag", "hard", 20) + exp_G(tol, "diag", "hard", 20))
    out["G_eye"] = (exp_G(deg, "eye", "hard", 12) + exp_G(tol, "eye", "hard", 12))
    out["H_hard_deg"] = exp_H(deg, "hard", 60)
    out["H_mild_deg"] = exp_H(deg, "mild", 40)
    out["H_hard_tol"] = exp_H(tol, "hard", 60)
    out["H_mild_tol"] = exp_H(tol, "mild", 40)
    out["traj"] = dict(deg=exp_traj(deg, 6), tol=exp_traj(tol, 6))
    out["secs"] = time.time() - t0

    with open(os.path.join(HERE, "mechanism2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote mechanism2.json  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
