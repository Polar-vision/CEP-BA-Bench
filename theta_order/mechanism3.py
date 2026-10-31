"""
Robust kernel: does anything in the classification survive gross errors?

mechanism2.py establishes that basin selection and step acceptance are numerical
rather than informational, but it does so on noiseless observations with a plain
least-squares cost.  A reviewer's next question is whether the whole comparison
is an artefact of that cost: a bundle adjustment in practice minimises a robust
M-estimator on a contaminated sample, and the landscape of that cost is rugged by
construction.

Two things are measured.

(I)  Invariance under weighting.  A robust kernel acts on the image residual,
     e -> w(e) e, i.e. it is a left (row) scaling.  The chart enters only as a
     right factor, J_q = J_X S, so the reweighted normal matrix is
     S^T (J_X^T W J_X) S with the *same* W, and the change-of-variables theorem
     and the Cramer-Rao bound hold verbatim.  Checked numerically at a common
     state, where the weights are common too.

(J)  Estimation with gross errors.  Huber IRLS, 10% contamination of amplitude
     30--300 px, 0.5 px Gaussian noise on the inliers, five charts, twenty
     starts: depth bias, inlier reprojection RMS, how the weights fall on the
     true outliers and on the inliers, against a least-squares control.

(K)  Basins under contamination, 40 starts, contaminated and clean variants of
     the same scene with the same noise, paired across charts.

The solver is iteratively reweighted least squares: the weights are frozen and
the weighted least-squares subproblem is solved by Levenberg--Marquardt, then the
weights are updated.  Solving the robust problem in one pass instead -- taking
the direction from the weighted Gauss--Newton system and testing it on the robust
cost -- stalls in the tail, because the gradient of the robust cost is not the
frozen-weight gradient and the iteration ends up burning its trust region on
directions that are not descent directions of the cost it is testing.  The IRLS
form is what a bundle adjustment actually does, and it is monotone.

Run:  "E:/zuo/Program Files/Python313/python.exe" mechanism3.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mechanism2 as m2                                          # noqa: E402
import verify_scenarios as vs                                    # noqa: E402

FAMS = m2.FAMS
SHORT = m2.SHORT
HUBER_C = 1.345          # 95% efficiency at the Gaussian
TUKEY_C = 4.685          # 95% efficiency at the Gaussian


# ------------------------------------------------------------------- problem
class RobustProblem(m2.BAProblem):
    """Same rig as mechanism2, with noise on the inliers and gross errors."""

    def __init__(self, name, Lam, B, beta_deg, f, contam=0.0, sigma_in=0.5,
                 out_mag=(30.0, 300.0), seed=11, **kw):
        super().__init__(name, Lam, B, beta_deg, f, **kw)
        self.obs_clean = self.obs.copy()
        rng = np.random.default_rng(seed)
        flat = self.obs_clean.reshape(-1, 2).copy()
        n = flat.shape[0]
        k = int(round(contam * n))
        mask = np.zeros(n, bool)
        if k:
            idx = rng.choice(n, k, replace=False)
            mask[idx] = True
            mag = np.exp(rng.uniform(np.log(out_mag[0]), np.log(out_mag[1]), k))
            th = rng.uniform(0.0, 2.0 * np.pi, k)
            flat[idx, 0] += mag * np.cos(th)
            flat[idx, 1] += mag * np.sin(th)
        flat[~mask] += sigma_in * rng.standard_normal((int((~mask).sum()), 2))
        self.obs = flat.reshape(self.obs.shape)
        self.out_mask = mask.reshape(self.obs.shape[:2])           # (cams, pts)
        self.contam = contam
        self.sigma_in = sigma_in
        self.sigma_meas = sigma_in * np.sqrt(2.0)    # noise of the 2D residual

    def reproj(self, X, R=None, C=None):
        R = self.R if R is None else R
        C = self.C if C is None else C
        return np.array([[vs.pi(Rj @ (x - Cj), self.f) for x in X]
                         for Rj, Cj in zip(R, C)])

    def diag(self, fam, qs, R=None, C=None):
        """Per-point diagnostics.

        The median over points is used rather than a global RMS, because one
        point that has escaped to infinity would otherwise dominate the summary
        of an otherwise correct fit.  Residuals are taken against the *clean*
        observations, through the optimised cameras: the gross errors are not
        allowed to hide in the number that says how well the fit explains the
        data that were actually good.
        """
        X = self.Xs(fam, qs)
        e = self.reproj(X, R, C) - self.obs_clean
        inl = ~self.out_mask
        per_pt = np.sqrt((e ** 2).sum(-1))                       # (cams, pts)
        rms = [float(np.sqrt(np.mean(per_pt[inl[:, i], i] ** 2)))
               for i in range(self.n_pt) if inl[:, i].any()]
        d = np.linalg.norm(X - self.C[0], axis=1)
        dt = np.linalg.norm(self.P - self.C[0], axis=1)
        rel = np.abs(d - dt) / dt
        return dict(rms_pt=float(np.median(rms)) if rms else float("nan"),
                    frac_pts=float(np.mean(rel < 0.05)),
                    depth=float(np.median(rel)), dmax=float(rel.max()))


# ------------------------------------------------------------------- kernels
def weights(e, c, kernel):
    a = np.abs(e) / max(c, 1e-12)
    if kernel == "l2":
        return np.ones_like(e)
    if kernel == "huber":
        return np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))
    return np.clip(1.0 - a ** 2, 0.0, None) ** 2                       # Tukey


def rho(e, c, kernel):
    a = np.abs(e)
    if kernel == "l2":
        return 0.5 * a ** 2
    if kernel == "huber":
        return np.where(a <= c, 0.5 * a ** 2, c * (a - 0.5 * c))
    t = np.clip(1.0 - (a / c) ** 2, 0.0, None)
    return (c ** 2 / 6.0) * (1.0 - t ** 3)          # constant beyond c


def kernel_c(kernel, sigma):
    return (HUBER_C if kernel == "huber" else TUKEY_C) * sigma


def meas_weights(r, c, kernel):
    """The loss acts on the norm of the 2D image residual, so the two components
    of a measurement share one weight -- the convention of every BA library."""
    en = np.linalg.norm(r.reshape(-1, 2), axis=1)
    wm = weights(en, c, kernel)
    return wm, np.repeat(wm, 2), en


# -------------------------------------------------------------------- solver
def lm_ls(prob, fam, q0, R0, C0, w, max_iter=80, gtol=1e-8, step_tol=1e-9):
    """Levenberg--Marquardt on 0.5 sum w_i e_i^2, weights frozen."""
    qs = [np.array(q, float) for q in q0]
    R = [r.copy() for r in R0]
    C = [c.copy() for c in C0]
    r, J = prob.rj(fam, qs, R, C)
    F = 0.5 * float(np.sum(w * r * r))
    lam, nu = 1e-4, 2.0
    g = J.T @ (w * r)
    g0 = float(np.max(np.abs(g)))
    st = dict(it=0, acc=0, rej=0, F=[F], lam=[lam], rho=[], rel=1.0)
    for _ in range(max_iter):
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        if np.max(np.abs(g)) <= gtol * max(g0, 1e-12):
            break
        dg = np.maximum(np.diag(H), 1e-9 * max(np.max(np.diag(H)), 1e-30))
        try:
            d = -np.linalg.solve(H + lam * np.diag(dg), g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break
        pred = -(g @ d + 0.5 * d @ H @ d)
        q2, R2, C2 = m2.step_state(prob, fam, qs, R, C, d)
        r2, J2 = prob.rj(fam, q2, R2, C2)
        F2 = 0.5 * float(np.sum(w * r2 * r2))
        gr = (F - F2) / pred if pred > 1e-300 else -1.0
        st["it"] += 1
        done = False
        if gr > 0 and np.isfinite(F2) and F2 < F:
            # a step is not comparable across charts, the displacement it
            # produces in space is
            Xa = prob.Xs(fam, qs)
            Xb = prob.Xs(fam, q2)
            rel = float(np.max(np.linalg.norm(Xb - Xa, axis=1)
                               / np.linalg.norm(Xa - prob.C[0], axis=1)))
            qs, R, C, r, J, F = q2, R2, C2, r2, J2, F2
            st["acc"] += 1
            st["rho"].append(float(gr))
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * gr - 1.0) ** 3)
            nu = 2.0
            st["rel"] = rel
            done = rel < step_tol
        else:
            st["rej"] += 1
            lam *= nu
            nu *= 2.0
        st["F"].append(F)
        st["lam"].append(lam)
        if lam > 1e12 or done:
            break
    st.update(F_final=F, lam_final=lam, q=qs, R=R, C=C,
              mean_rho=float(np.mean(st["rho"])) if st["rho"] else float("nan"))
    return st


def lm_robust(prob, fam, q0, R0, C0, kernel="huber", outer=4, inner=50,
              wtol=1e-3, sigma=None, warm=True, c=None):
    """IRLS: freeze the weights, solve the weighted least squares, repeat.

    The first pass is a plain least-squares one when `warm` is set.  Without it
    the first weight vector is 1/|e| for every measurement -- with a poor start
    the residuals are enormous and the Huber weight degenerates to the L1 weight,
    whose IRLS convergence is linear and slow.  A least-squares warm start
    followed by a few reweighting passes is what an implementation does.

    Convergence is on the weights: the estimate has stopped moving when the
    weights have.  The iterations and rejected steps of all passes are summed, so
    the counts are comparable with the least-squares run, which is the same loop
    with kernel='l2'.
    """
    sig = prob.sigma_meas if sigma is None else sigma
    if c is None:
        c = kernel_c(kernel, sig)
    q, R, C = q0, R0, C0
    wm_prev, tot_it, tot_acc, tot_rej, gacc = None, 0, 0, 0, []
    k, st = 0, None
    for k in range(outer):
        r, _ = prob.rj(fam, q, R, C)
        wm, w, _ = meas_weights(r, c, kernel)
        if warm and k == 0:
            w = np.ones_like(w)
        st = lm_ls(prob, fam, q, R, C, w, max_iter=inner)
        q, R, C = st["q"], st["R"], st["C"]
        tot_it += st["it"]
        tot_acc += st["acc"]
        tot_rej += st["rej"]
        gacc.append(st["mean_rho"])
        if wm_prev is not None and float(np.mean(np.abs(wm - wm_prev))) < wtol:
            break
        wm_prev = wm
    r, J = prob.rj(fam, q, R, C)
    wm, _, en = meas_weights(r, c, kernel)
    F = float(np.sum(rho(en, c, kernel)))
    out = dict(iter=tot_it, acc=tot_acc, rej=tot_rej, outer=k + 1,
               F_final=F, lam_final=st["lam_final"], q=q, R=R, C=C,
               w_meas=wm.reshape(prob.n_c, prob.n_pt),
               mean_rho=float(np.nanmean(gacc)))
    out.update(prob.diag(fam, q, R, C))
    return out


def lm_gnc(prob, fam, q0, R0, C0, kernel="tukey", schedule=None, passes=2,
           inner=50):
    """Graduated non-convexity on the kernel scale, then plain IRLS.

    A robust cost is minimised from an initialisation that is already in the
    wrong valley: with c at its final value the first weight vector is 1/|e| for
    every measurement, the cost becomes an L1 cost with an enormous leverage for
    the worst residuals, and the iteration settles in a compromise that fits
    nothing.  Widening the band first and shrinking it in stages keeps the
    iteration inside the valley that the wide (nearly quadratic) cost selects.
    """
    sig = prob.sigma_meas
    if schedule is None:
        schedule = ((20.0, 10.0, 6.0, TUKEY_C) if kernel == "tukey"
                    else (12.0, 6.0, 3.0, HUBER_C))
    q, R, C = q0, R0, C0
    tot = dict(iter=0, acc=0, rej=0, outer=0)
    last = None
    for i, s in enumerate(schedule):
        last = lm_robust(prob, fam, q, R, C, kernel=kernel, outer=passes,
                         inner=inner, warm=(i == 0), c=s * sig)
        q, R, C = last["q"], last["R"], last["C"]
        for k in ("iter", "acc", "rej", "outer"):
            tot[k] += last[k]
    last = lm_robust(prob, fam, q, R, C, kernel=kernel, outer=4, inner=inner,
                     warm=False, c=kernel_c(kernel, sig))
    for k in ("iter", "acc", "rej", "outer"):
        tot[k] += last[k]
    last.update(it_total=tot["iter"], rej_total=tot["rej"],
                passes=tot["outer"])
    return last


# ------------------------------------------------- weighted CRB invariance (I)
def weighted_cov(prob, fam, qs, R, C, kernel="huber", pts=(0, 3, 7)):
    """Spatial covariance of selected points from the *weighted* normal matrix."""
    r, J = prob.rj(fam, qs, R, C)
    c = kernel_c(kernel, prob.sigma_meas)
    _, w, _ = meas_weights(r, c, kernel)
    H = J.T @ (J * w[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    nfree = prob.n_c - len(prob.fixed)
    base = 6 * nfree
    out = {}
    for i in pts:
        S = np.array([m2.mech.S_analytic(prob.frame, fam, qs[i])])
        blk = cov[base + 3 * i:base + 3 * i + 3, base + 3 * i:base + 3 * i + 3]
        out[i] = S[0] @ blk @ S[0].T          # back to spatial coordinates
    return out


def exp_I(prob, kernel="huber", pts=(0, 3, 7)):
    print("\n[I] weighted CRB: same state, same weights, five charts")
    covs = {fam: weighted_cov(prob, fam, prob.qs_of(fam, prob.P), prob.R,
                              prob.C, kernel, pts) for fam in FAMS}
    ref = covs["XYZ"]
    md = np.array([(prob.P[i] - prob.C[0]) / np.linalg.norm(prob.P[i] - prob.C[0])
                   for i in pts])
    L = np.array([np.linalg.norm(prob.P[i] - prob.C[0]) for i in pts])
    dev, sig_d = {}, {}
    for fam in FAMS:
        dev[fam] = max(float(np.max(np.abs(covs[fam][i] - ref[i]))
                             / max(np.max(np.abs(ref[i])), 1e-30)) for i in pts)
        sig_d[fam] = float(np.mean([np.sqrt(max(md[k] @ covs[fam][i] @ md[k], 0.0))
                                    / L[k] for k, i in enumerate(pts)]))
        print(f"    {SHORT[fam]:<9} max rel deviation of the 3x3 blocks "
              f"{dev[fam]:.2e}, radial sigma {sig_d[fam]:.4f} m")
    return dict(dev=dev, sigma_depth=sig_d)


# ---------------------------------------------------------- estimation (J, K)
def one_run(prob, fam, start, regime, kernel, inner=50):
    cfg = m2.draw(np.random.default_rng(9000 + start), regime)
    q0, R0, C0 = m2.init_state(prob, fam, cfg["dL"], cfg["ang"], cfg["camT"],
                               cfg["camR"], np.random.default_rng(9000 + start))
    st = (lm_gnc(prob, fam, q0, R0, C0, kernel=kernel, inner=inner)
          if kernel == "tukey" else
          lm_robust(prob, fam, q0, R0, C0, kernel=kernel))
    wm = st["w_meas"]
    e = prob.reproj(prob.Xs(fam, st["q"]), st["R"], st["C"]) - prob.obs
    flagged = np.linalg.norm(e, axis=2) > 3.0 * prob.sigma_meas
    inl = ~prob.out_mask
    tpr = (float(np.mean(flagged[prob.out_mask])) if prob.out_mask.any()
           else float("nan"))
    return dict(start=start, fam=fam, depth=st["depth"], dmax=st["dmax"],
                frac=st["frac_pts"], rms_pt=st["rms_pt"], F=st["F_final"],
                it=st["iter"], tpr=tpr, fpr=float(np.mean(flagged[inl])),
                w_out=(float(np.mean(wm[prob.out_mask]))
                       if prob.out_mask.any() else float("nan")),
                w_in=float(np.mean(wm[inl]))), st


def summarise(rows, fam, scene, tag):
    def a(k):
        return np.array([r[k] for r in rows], float)
    d, rms, F, fr = a("depth"), a("rms_pt"), a("F"), a("frac")
    tpr, fpr = a("tpr"), a("fpr")
    tpr = tpr[np.isfinite(tpr)]
    o = dict(scene=scene, tag=tag, fam=fam, med=float(np.median(d)),
             p90=float(np.percentile(d, 90)), rms=float(np.median(rms)),
             medF=float(np.median(F)), pts=float(np.mean(fr)),
             tpr=float(np.mean(tpr)) if tpr.size else float("nan"),
             fpr=float(np.mean(fpr)), w_out=float(np.nanmean(a("w_out"))),
             w_in=float(np.nanmean(a("w_in"))), n=len(rows),
             depths=[float(v) for v in d], costs=[float(v) for v in F],
             runs=rows)
    print(f"    {SHORT[fam]:<9}{o['med']:>9.2%}{o['p90']:>9.2%}{o['rms']:>8.2f}"
          f"{o['medF']:>10.3f}{o['pts']:>7.0%}{o['tpr']:>7.0%}{o['fpr']:>7.1%}"
          f"{o['w_out']:>8.3f}{o['w_in']:>7.3f}")
    return o


HDR = (f"    {'chart':<9}{'med d':>9}{'p90 d':>9}{'inRMS':>8}{'F':>10}{'pts':>7}"
       f"{'TPR':>7}{'FPR':>7}{'<w>out':>8}{'<w>in':>7}")


def exp_J(prob, regime="hard", n_start=20, kernels=("l2", "tukey"), inner=50):
    print(f"\n[J] estimation, {prob.contam:.0%} gross errors: {prob.name}, "
          f"regime={regime}, {n_start} starts")
    out = {}
    for kernel in kernels:
        print(f"  -- {kernel}")
        print(HDR)
        per = {fam: [] for fam in FAMS}
        for s in range(n_start):
            for fam in FAMS:
                rec, _ = one_run(prob, fam, s, regime, kernel, inner)
                per[fam].append(rec)
        out[kernel] = [summarise(per[fam], fam, prob.name, kernel)
                       for fam in FAMS]
    return out


def exp_K(prob, regime="hard", n_start=40, kernel="tukey", rel=0.01):
    """Basin selection under a robust cost, measured on the *cost*.

    A redescending kernel is flat wherever a point has lost all its support, so
    two runs can sit at different depths with the same cost and neither is
    wrong.  What distinguishes a basin from a plateau is therefore the cost, and
    the criterion is paired: within one start, a chart has reached the good
    solution when its cost is within `rel` of the lowest cost any chart reached
    from that start.
    """
    print(f"\n[K] basins, {prob.name}, {prob.contam:.0%} gross errors, "
          f"{kernel}, {n_start} starts (cost criterion, rel={rel:.0%})")
    print(HDR)
    per = {fam: [] for fam in FAMS}
    hit = {fam: [] for fam in FAMS}
    for s in range(n_start):
        recs = {}
        for fam in FAMS:
            rec, _ = one_run(prob, fam, s, regime, kernel)
            recs[fam] = rec
        best = min(r["F"] for r in recs.values())
        for fam in FAMS:
            per[fam].append(recs[fam])
            hit[fam].append(recs[fam]["F"] <= best * (1.0 + rel))
    out = []
    for fam in FAMS:
        o = summarise(per[fam], fam, prob.name, f"c{prob.contam:.2f}")
        o["ok"] = float(np.mean(hit[fam]))
        out.append(o)
    print("    reached the lowest-cost solution: " + ", ".join(
        f"{SHORT[f]:} {100*o['ok']:.0f}%" for f, o in zip(FAMS, out)))
    return out


def main():
    t0 = time.time()
    kw = dict(contam=0.10, out_mag=(30.0, 300.0))
    deg = RobustProblem("short baseline", 15.0, 5.0e-2, 90.0, 3.64e4, **kw)
    deg0 = RobustProblem("short baseline", 15.0, 5.0e-2, 90.0, 3.64e4,
                         contam=0.0)
    tol = RobustProblem("well conditioned", 15.0, 8.0e-1, 90.0, 3.64e4, **kw)
    tol0 = RobustProblem("well conditioned", 15.0, 8.0e-1, 90.0, 3.64e4,
                         contam=0.0)
    print(f"short baseline: {deg.out_mask.sum()} of {deg.out_mask.size} "
          f"measurements corrupted, omega="
          f"{np.degrees(deg.frame.omega(deg.frame.u0)):.3f} deg")

    out = dict(n_pt=deg.n_pt, n_c=deg.n_c, contam=deg.contam,
               n_out=int(deg.out_mask.sum()), sigma_in=deg.sigma_in,
               omega_deg=float(np.degrees(deg.frame.omega(deg.frame.u0))))
    out["I"] = exp_I(deg)
    out["J_deg"] = exp_J(deg)
    out["J_tol"] = exp_J(tol)
    out["K_deg_clean"] = exp_K(deg0)
    out["K_deg_contam"] = exp_K(deg)
    out["K_tol_clean"] = exp_K(tol0)
    out["K_tol_contam"] = exp_K(tol)

    out["secs"] = time.time() - t0
    with open(os.path.join(HERE, "mechanism3.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote mechanism3.json  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
