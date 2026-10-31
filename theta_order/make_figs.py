"""Generate publication figures for the Theta-order manuscript.

Only numpy + matplotlib. Reuses the Scene class from verify_scenarios.py so that
every number in the figures is produced by the same code path as the tables.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
# In the released package verify_scenarios.py sits next to this file; inside the
# authoring tree it lives one level up.  Prefer the local copy so the package is
# self-contained.
SRC = HERE if os.path.exists(os.path.join(HERE, "verify_scenarios.py")) \
    else os.path.join(HERE, "..", "New_GSIS", "theta_derivation")
sys.path.insert(0, os.path.abspath(SRC))
import verify_scenarios as vs  # noqa: E402

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.linewidth": 0.8,
    "axes.labelsize": 10.5,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C = {
    "XYZ":        "#4d4d4d",
    "InvZ":       "#1f77b4",
    "SphRange":   "#9467bd",
    "SphInvRange": "#2ca02c",
    "Parallax":   "#d62728",
}
M = {"XYZ": "o", "InvZ": "s", "SphRange": "^", "SphInvRange": "v", "Parallax": "D"}
LBL = {
    "XYZ": "Cartesian XYZ",
    "InvZ": r"Inverse-$Z$",
    "SphRange": "Spherical range",
    "SphInvRange": "Spherical inv. range",
    "Parallax": "Two-anchor parallax",
}
FAMS = ["XYZ", "InvZ", "SphRange", "SphInvRange", "Parallax"]
SHORT_LBL = {"XYZ": "XYZ", "InvZ": r"Inv-$Z$", "SphRange": "Sph. range",
             "SphInvRange": "Sph. inv.", "Parallax": "Parallax"}

SCEN_EN = {
    "卫星视频立体(弱交会)": "video satellite",
    "高空无人机长焦悬停": "high-altitude UAV",
    "电力/风电长焦巡检": "power-line inspection",
    "桥梁/大坝变形监测": "dam monitoring",
    "手机长焦多摄测距": "phone tele-camera",
    "AR 眼镜双目": "AR glasses",
    "腹腔镜双目": "laparoscope",
    "水下 ROV 双目": "underwater ROV",
}

records = {}


# ------------------------------------------------------------------ Figure 1
def fig1_two_regimes():
    """Same omega, two origins: whether kappa travels with omega."""
    B0, beta = 1.0, np.pi / 2
    kap0 = B0 * np.sin(beta)

    # regime A: distant point, kappa fixed
    Lam = np.logspace(0, 6, 200)
    om_far = np.arctan2(kap0, Lam - B0 * np.cos(beta))
    kap_far = np.full_like(Lam, kap0)

    # regime B: short baseline, Lambda fixed
    Bs = np.logspace(0, -6, 200)
    Lam_s = 25.0
    kap_near = Bs * np.sin(beta)
    om_near = np.arctan2(kap_near, Lam_s - Bs * np.cos(beta))

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.loglog(np.degrees(om_far), kap_far, "-", color="#d62728", lw=1.6,
              label=r"distant point: $B$ fixed, $\Lambda\to\infty$")
    ax.loglog(np.degrees(om_near), kap_near, "--", color="#1f77b4", lw=1.6,
              label=r"short baseline: $\Lambda$ fixed, $B\to0$")

    ax.set_xlabel(r"parallax angle $\omega$  [deg]")
    ax.set_ylabel(r"$\kappa=B\sin\beta$  [m]")
    ax.set_xlim(1e-5, 5)
    ax.set_ylim(1e-7, 1e1)
    ax.grid(True, which="major", lw=0.35, alpha=0.45)
    ax.grid(True, which="minor", lw=0.2, alpha=0.25)
    ax.legend(loc="lower left", frameon=False, handlelength=1.8)

    ax.annotate(r"$\kappa$ constant" "\n" r"$\Rightarrow$ information retained",
                xy=(2e-3, 1.0), xytext=(3e-5, 6e-2), fontsize=7.4, color="#d62728",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#d62728"))
    ax.annotate(r"$\kappa\to0$" "\n" r"$\Rightarrow$ information vanishes",
                xy=(2e-4, 2e-4), xytext=(1.2e-5, 3e-5), fontsize=7.4, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#1f77b4"))
    fig.savefig(os.path.join(FIG, "fig1_two_regimes.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2
def fig2_classification():
    """Order exponents of the two block metrics in the two small-parallax regimes.

    Each cell is the exponent p in Theta(Lambda^p) (left two columns) or
    Theta(B^p) (right two columns); 0 means Theta(1), i.e. no degradation.
    """
    fams = ["SphRange", "XYZ", "InvZ", "SphInvRange", "Parallax"]
    cols = [r"$\mathrm{anis}$" "\n" r"$\Lambda$", r"$s_{\mathrm{blk}}$" "\n" r"$\Lambda$",
            r"$\mathrm{anis}$" "\n" r"$B$", r"$s_{\mathrm{blk}}$" "\n" r"$B$"]
    # exponents from Table 3 (theory)
    E = np.array([
        [+2, -2, -1, +1],   # spherical range
        [+1, -2, -1, +1],   # Cartesian XYZ
        [0,   0, -1, +1],   # inverse-Z
        [0,   0, -1, +1],   # spherical inverse range
        [0,   0,  0,  0],   # two-anchor parallax
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(3.55, 2.30))
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    cmap = LinearSegmentedColormap.from_list(
        "ord", ["#c0392b", "#e8a87c", "#f5f0e1", "#8fbf8f", "#2e7d32"])
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
                        cmap.N)
    ax.imshow(E, cmap=cmap, norm=norm, aspect="auto")

    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            v = int(E[i, j])
            txt = r"$\Theta(1)$" if v == 0 else f"${v:+d}$"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                    color="white" if abs(v) >= 2 else "0.12")

    ax.set_xticks(range(4))
    ax.set_xticklabels(cols, fontsize=7.0)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([LBL[f] for f in fams], fontsize=7.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)

    fig.savefig(os.path.join(FIG, "fig2_classification.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3
def _cond_curve(kind):
    if kind == "far":
        base = dict(B=1.0, beta=90.0, f=1000.0)
        vals = np.logspace(0, 6, 13)
        make = lambda v: vs.Scene(v, base["B"], base["beta"], base["f"], n_c=4)
    else:
        base = dict(Lam=25.0, beta=90.0, f=9091.0)
        vals = np.logspace(0, -6, 13)
        make = lambda v: vs.Scene(base["Lam"], v, base["beta"], base["f"], n_c=4)

    out = {fam: [] for fam in FAMS}
    for v in vals:
        sc = make(v)
        for fam in FAMS:
            m = vs.metrics(sc, fam)
            out[fam].append(m["cond"])
    return vals, out


def fig3_conditioning():
    vals_far, far = _cond_curve("far")
    vals_near, near = _cond_curve("near")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    order = ["SphRange", "XYZ", "InvZ", "SphInvRange", "Parallax"]
    lw = {"SphRange": 1.1, "XYZ": 1.3, "InvZ": 1.0, "SphInvRange": 1.0,
          "Parallax": 2.2}
    ms = {"SphRange": 3.0, "XYZ": 3.4, "InvZ": 2.6, "SphInvRange": 2.6,
          "Parallax": 3.6}
    alpha = {"SphRange": 1.0, "XYZ": 1.0, "InvZ": 0.95, "SphInvRange": 0.95,
             "Parallax": 1.0}

    for ax, vals, dat, xl, ttl in (
        (axes[0], vals_far, far, r"object distance  $\Lambda$  [m]",
         r"(a) distant-point regime"),
        (axes[1], vals_near, near, r"baseline  $B$  [m]",
         r"(b) short-baseline regime"),
    ):
        for fam in order:
            ax.loglog(vals, dat[fam], marker=M[fam], ms=ms[fam], lw=lw[fam],
                      color=C[fam], label=LBL[fam], mfc="none",
                      mew=1.0, alpha=alpha[fam],
                      zorder=5 if fam == "Parallax" else 2)
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$\mathrm{cond}(J_q^{\top}J_q)$")
        ax.set_title(ttl, fontsize=8.5, pad=4)
        ax.grid(True, which="major", lw=0.35, alpha=0.4)
        ax.set_ylim(1e-1, 1e16)
        if ax is axes[1]:
            ax.invert_xaxis()

    axes[0].legend(loc="upper left", frameon=False, handlelength=1.5,
                   borderaxespad=0.3)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig3_conditioning.pdf"))
    plt.close(fig)

    records["cond_far"] = {"vals": vals_far.tolist(),
                           **{k: v for k, v in far.items()}}
    records["cond_near"] = {"vals": vals_near.tolist(),
                            **{k: v for k, v in near.items()}}


# ------------------------------------------------------------------ Figure 4
def fig4_ceiling():
    """Is the coordinate-free bound tight? Measured sigma_L against sigma_px/(f||r||)."""
    scenes = vs.SCEN

    fig, ax = plt.subplots(figsize=(3.5, 2.55))

    rows = []
    for (name, Lam, B, beta, f, sig, req, _note) in scenes:
        sc = vs.Scene(Lam, B, beta, f, n_c=3)
        A = sc.A()
        cov = np.linalg.inv(A.T @ A)
        u = sc.u0

        # vector of radial sensitivities r_j = kappa_j / ||z_j||^2
        rr = []
        for Cj in sc.Cs:
            zj = sc.X - Cj
            kap_j = np.linalg.norm(np.cross(Cj - sc.Cm, sc.u0))
            rr.append(kap_j / (zj @ zj))
        rr = np.array(rr)
        lb = sig / (f * np.linalg.norm(rr))        # bound on sigma_Lambda [m]
        meas = sig * np.sqrt(u @ cov @ u)          # linearised sigma_Lambda [m]
        rows.append(dict(name=name, meas=float(meas), bound=float(lb),
                         ratio=float(meas / lb), req=req,
                         fom=float(f * np.linalg.norm(rr) * Lam)))

    lo_b = min(r["bound"] for r in rows) * 0.5
    hi_b = max(r["bound"] for r in rows) * 2.0
    x = np.logspace(np.log10(lo_b), np.log10(hi_b), 200)
    ax.loglog(x, x, "-", color="0.25", lw=1.1, zorder=1,
              label=r"$\sigma_\Lambda=\sigma_{\mathrm{px}}/(f\|\mathbf{r}\|)$")

    for r in rows:
        ax.plot(r["bound"], r["meas"], "o", ms=5.2, mfc="white",
                mec=C["Parallax"], mew=1.2, zorder=3)

    ax.set_xlabel(r"lower bound  $\sigma_{\mathrm{px}}/(f\|\mathbf{r}\|)$  [m]")
    ax.set_ylabel(r"linearised  $\sigma_\Lambda$  [m]")
    ax.set_xlim(lo_b, hi_b)
    ax.set_ylim(lo_b, hi_b)
    ax.grid(True, which="major", lw=0.35, alpha=0.4)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)

    lo = min(r["ratio"] for r in rows)
    hi = max(r["ratio"] for r in rows)
    ax.text(hi_b * 0.55, lo_b * 1.5,
            "measured / bound\n" + f"{lo:.2f} – {hi:.2f}",
            fontsize=7.0, color="0.3", ha="right", va="bottom")

    fig.savefig(os.path.join(FIG, "fig4_ceiling.pdf"))
    plt.close(fig)

    records["scenarios"] = rows


# ------------------------------------------------------------------ Figure 5
def fig5_invariance():
    """Five parameterizations, one covariance: the accuracy invariant."""
    Lam, B, beta, f = 1.0e3, 1.0, 90.0, 2000.0
    omegas = np.logspace(-4, -1.2, 10)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))

    rows = []
    sig_curves = {fam: [] for fam in FAMS}
    dev_curves = {fam: [] for fam in FAMS}

    for om in omegas:
        # baseline that produces this parallax angle at the given distance
        Bv = Lam * np.tan(om)
        sc = vs.Scene(Lam, Bv, beta, f, n_c=6)
        A = sc.A()
        cov0 = np.linalg.inv(A.T @ A)
        u = sc.u0
        sig0 = np.sqrt(u @ cov0 @ u)

        for fam in FAMS:
            q = sc.q_of(fam, sc.X)
            S = sc.S(fam, q)
            J = A @ S
            cov = np.linalg.inv(J.T @ J)
            covX = S @ cov @ S.T
            sig = np.sqrt(u @ covX @ u)
            dev = np.abs(covX - cov0).max() / np.abs(cov0).max()
            sig_curves[fam].append(sig)
            dev_curves[fam].append(max(dev, 1e-17))
        rows.append((om, sig0, {fam: sig_curves[fam][-1] for fam in FAMS}))

    ax = axes[0]
    for fam in FAMS:
        ax.loglog(omegas, sig_curves[fam], marker=M[fam], ms=3.0, lw=1.0,
                  color=C[fam], mfc="none", mew=0.9, label=LBL[fam])
    ax.set_xlabel(r"parallax angle  $\omega$  [rad]")
    ax.set_ylabel(r"depth std.\ $\sigma_\Lambda$  [m]")
    ax.set_title(r"(a) five coordinates, one curve", fontsize=9.5, pad=4)
    ax.grid(True, which="major", lw=0.35, alpha=0.4)
    ax.legend(loc="lower left", frameon=False, handlelength=1.4,
              borderaxespad=0.3)

    ax = axes[1]
    for fam in FAMS:
        ax.loglog(omegas, dev_curves[fam], marker=M[fam], ms=3.0, lw=1.0,
                  color=C[fam], mfc="none", mew=0.9)
    ax.axhline(2.2e-16, color="0.3", lw=0.8, ls="--")
    ax.text(1.2e-4, 3.2e-15, r"machine precision $\varepsilon_{\mathrm{f64}}$",
            fontsize=6.8, color="0.3")
    ax.set_xlabel(r"parallax angle  $\omega$  [rad]")
    ax.set_ylabel(r"rel.\ deviation of $\mathrm{cov}(\mathbf{X})$")
    ax.set_title(r"(b) deviation from the XYZ covariance", fontsize=9.5, pad=4)
    ax.set_ylim(1e-17, 1e-12)
    ax.grid(True, which="major", lw=0.35, alpha=0.4)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig5_invariance.pdf"))
    plt.close(fig)

    records["invariance"] = [
        dict(omega=float(r[0]), sigma_L=float(r[1]),
             spread=float(max(r[2].values()) - min(r[2].values())))
        for r in rows
    ]


# ------------------------------------------------------------------ Figure 6
def fig6_mechanism():
    """Where the iteration advantage comes from: the chart's nonlinearity, not
    its condition number."""
    d = json.load(open(os.path.join(HERE, "mechanism.json")))

    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.25))

    # ---- (a) depth error vs iteration; second scene as hollow markers only
    ax = axes[0]
    N = list(range(0, 11))
    handles = []
    for fam in FAMS:
        y = [max(d["C_deg"][fam][k], 1e-18) for k in N]
        h, = ax.semilogy(N, y, "-", color=C[fam], lw=1.3, label=LBL[fam])
        handles.append(h)
        y2 = [max(d["C_tol"][fam][k], 1e-18) for k in N]
        ax.semilogy(N, y2, "none", marker=M[fam], ms=4.2, mfc="white",
                    mec=C[fam], mew=0.85)
    ax.set_xlabel("iteration")
    ax.set_ylabel("depth error")
    ax.set_ylim(1e-18, 3.0)
    ax.set_xlim(-0.3, 10.3)
    ax.grid(True, which="major", lw=0.3, alpha=0.35)
    ax.annotate(r"$\delta_{k+1}=\delta_k^{2}$", xy=(2.6, 2.5e-3),
                xytext=(4.3, 2.0e-2), fontsize=8.6, color="0.2",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))


    # ---- (b) Monte Carlo: every chart meets the CRB
    ax = axes[1]
    rows = d["E"]
    nb = [r["budget"] for r in rows]
    for fam in FAMS:
        ax.semilogy(nb, [max(r[fam], 1e-18) for r in rows], "-",
                    color=C[fam], lw=1.25, marker=M[fam], ms=2.8,
                    mfc="none", mew=0.8)
    crb = rows[0]["crb"]
    ax.axhline(crb, color="0.15", lw=0.9, ls=":")
    ax.text(nb[-1] * 0.96, crb * 1.12, "CRB", fontsize=7.4, color="0.15",
            ha="right", va="bottom")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMSE of depth")
    ax.set_xscale("log")
    ax.set_ylim(2e-3, 1.2)
    ax.grid(True, which="major", lw=0.3, alpha=0.35)

    # ---- (c) one step only: flat in the condition number
    ax = axes[2]
    rows = d["B"]
    cx = [r["cond_xyz"] for r in rows]
    for fam in ("XYZ", "InvZ", "Parallax"):
        ax.loglog(cx, [r["err"][fam] for r in rows], "o-", color=C[fam],
                  lw=1.1, ms=3.4, mfc="white", mew=0.9)
    ax.set_xlabel("condition number")
    ax.set_ylabel("one-step error")
    ax.set_ylim(3e-9, 3.0)
    ax.set_xlim(5e2, 5e7)
    ax.grid(True, which="major", lw=0.3, alpha=0.35)


    fig.legend(handles, [LBL[f] for f in FAMS], loc="lower center", ncol=5,
               frameon=False, handlelength=1.8, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(FIG, "fig6_mechanism.pdf"))
    plt.close(fig)

    records["mechanism"] = dict(
        one_step=[dict(name=r["name"], cond=r["cond_xyz"], **r["err"])
                  for r in d["B"]],
        crb=d["E"][0]["crb"],
    )


# ------------------------------------------------------------------ Figure 7
def fig7_joint():
    """The two mechanisms the point-block protocol excludes: step acceptance and
    basin selection, measured in a joint adjustment with a trust region."""
    d = json.load(open(os.path.join(HERE, "mechanism2.json")))

    fig, axes = plt.subplots(2, 2, figsize=(6.95, 4.7))

    # ---- (a) accepted and rejected steps, short baseline, Marquardt damping
    ax = axes[0, 0]
    rows = [r for r in d["G_diag"] if r["scene"] == "short baseline"]
    ypos = np.arange(len(rows))[::-1]
    for y, r in zip(ypos, rows):
        ax.barh(y, r["acc"], height=0.62, color=C[r["fam"]], alpha=0.85,
                edgecolor="white", lw=0.6)
        ax.barh(y, r["rej"], left=r["acc"], height=0.62, color="white",
                edgecolor=C[r["fam"]], lw=1.2, hatch="////")
        ax.text(r["acc"] + r["rej"] + 0.8, y, f"{r['rej']:.1f}".rstrip("0").rstrip("."),
                va="center", fontsize=8.2, color=C[r["fam"]])
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_LBL[r["fam"]] for r in rows], fontsize=8.6)
    ax.set_xlabel("steps (mean of 20 starts)")
    ax.set_xlim(0, 46)
    ax.grid(True, axis="x", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.06, "hatched: rejected", transform=ax.transAxes,
            ha="right", fontsize=8.0, color="0.35")

    # ---- (b) cost and (c) damping along the same start
    tr = d["traj"]["deg"]
    for key, axx, lab, lim, thr in (("F", axes[0, 1], "cost", (1e-14, 1e7), None),
                                    ("lam", axes[1, 0], r"damping $\lambda$",
                                     (1e-6, 1e15), 1e12)):
        for fam in ("SphRange", "XYZ", "Parallax"):
            y = np.maximum(np.array(tr[fam][key]), 1e-30)
            axx.semilogy(np.arange(len(y)), y, "-", color=C[fam], lw=1.3,
                         label=SHORT_LBL[fam])
            axx.semilogy([len(y) - 1], [y[-1]], M[fam], ms=3.6, color=C[fam])
        axx.set_xlabel("iteration")
        axx.set_ylabel(lab)
        axx.set_ylim(*lim)
        axx.grid(True, which="major", lw=0.3, alpha=0.35)
        if thr is not None:
            axx.axhline(thr, color="0.4", ls=":", lw=0.9)
    axes[0, 1].set_xlim(-0.5, 30)
    axes[0, 1].legend(loc="lower right", frameon=False, fontsize=8.0,
                      handlelength=1.6, borderaxespad=0.25)
    axes[1, 0].set_xlim(-0.5, 30)
    axes[1, 0].text(0.5, 3e12, r"$\lambda>10^{12}$: no descent step found",
                    fontsize=7.6, color="0.35")

    # ---- (d) how often the true minimum is reached
    ax = axes[1, 1]
    rows = d["H_hard_deg"]
    xs = np.arange(len(rows))
    vals = [100 * r["ok"] for r in rows]
    ax.bar(xs, vals, width=0.62, color=[C[r["fam"]] for r in rows],
           alpha=0.92, edgecolor="white", lw=0.6)
    for x, v, r in zip(xs, vals, rows):
        ax.text(x, v + 3.5, f"{v:.0f}", ha="center", fontsize=8.0,
                color=C[r["fam"]] if v < 100 else "0.35")
    ax.axhline(100, ls=":", color="0.45", lw=0.9)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(np.arange(len(FAMS)))
    ax.set_xticklabels(["XYZ", r"Inv-$Z$", "Sph.\nrng", "Sph.\ninv.", "Par."],
                       fontsize=7.8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("reaching the true\nminimum [%]", fontsize=9.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.9, h_pad=1.1)
    fig.savefig(os.path.join(FIG, "fig7_joint.pdf"))
    plt.close(fig)

    records["joint"] = dict(
        accept=[dict(fam=r["fam"], scene=r["scene"], acc=r["acc"], rej=r["rej"],
                     rho=r["rho"], ok=r["ok"]) for r in d["G_diag"]],
        accept_eye=[dict(fam=r["fam"], scene=r["scene"], acc=r["acc"],
                         rej=r["rej"]) for r in d["G_eye"]],
        basin=[dict(regime=k, fam=r["fam"], ok=r["ok"], escape=r["escape"])
               for k in ("H_hard_deg", "H_mild_deg", "H_hard_tol", "H_mild_tol")
               for r in d[k]],
    )



# ------------------------------------------------------------------ Figure 8
def fig8_robust():
    """Robust cost: the estimator is chart-blind, the basin is not.

    Designed at the printed width (4.8 in), so the point sizes below are the
    sizes that appear on the page.
    """
    d = json.load(open(os.path.join(HERE, "mechanism3.json")))
    J = {r["fam"]: {k: next(q for q in d["J_deg"][k] if q["fam"] == r["fam"])
                    for k in ("l2", "tukey")} for r in d["J_deg"]["l2"]}
    Kc = {r["fam"]: r for r in d["K_deg_clean"]}
    Kx = {r["fam"]: r for r in d["K_deg_contam"]}
    TICK = ["XYZ", "InvZ", "S. rng", "S. inv", "Par."]
    xs = np.arange(len(FAMS))
    TS = 6.2

    fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.15))

    # ---- (a) cost reached from every start: only XYZ misses the low-cost set
    ax = axes[0]
    best = min(min(r["F"] for r in J[f]["tukey"]["runs"]) for f in FAMS)
    for k, fam in enumerate(FAMS):
        F = np.array([r["F"] for r in J[fam]["tukey"]["runs"]])
        x = k + np.linspace(-0.15, 0.15, F.size)
        ax.plot(x, F, M[fam], ms=2.0, mfc="none", mec=C[fam], mew=0.6, ls="none")
        ax.plot([k - 0.26, k + 0.26], [np.median(F)] * 2, color=C[fam], lw=1.3)
    ax.axhline(best, ls=":", color="0.45", lw=0.8)
    ax.text(4.45, 12.55, f"lowest {best:.2f}", fontsize=TS, color="0.35",
            ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(TICK, fontsize=TS, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("cost at the solution", fontsize=7.2)
    ax.set_yticks([13, 14, 15, 16, 17, 18])
    ax.tick_params(axis="y", labelsize=TS)
    ax.set_ylim(12.4, 18.4)
    ax.set_xlim(-0.55, 4.55)
    ax.text(0.5, 0.97, "20 starts", transform=ax.transAxes, ha="center",
            va="top", fontsize=TS, color="0.35")

    # ---- (b) depth error, least squares against Tukey
    ax = axes[1]
    w = 0.36
    for k, (key, lab, hatch, alpha) in enumerate(
            (("l2", "least squares", "\\\\\\\\", 0.30),
             ("tukey", "Tukey", None, 0.94))):
        vals = [100 * J[f][key]["med"] for f in FAMS]
        ax.bar(xs + (k - 0.5) * w, vals, width=w, color=[C[f] for f in FAMS],
               alpha=alpha, edgecolor="white", lw=0.4, hatch=hatch, label=lab)
    for f, v in zip(FAMS, [100 * J[f]["l2"]["med"] for f in FAMS]):
        ax.text(xs[FAMS.index(f)] - 0.5 * w, v * 1.14, f"{v:.1f}", ha="center",
                va="bottom", fontsize=5.6, color="0.35")
    for f, v in zip(FAMS, [100 * J[f]["tukey"]["med"] for f in FAMS]):
        ax.text(xs[FAMS.index(f)] + 0.5 * w, v * 1.14, f"{v:.1f}", ha="center",
                va="bottom", fontsize=5.6, color=C[f])
    ax.set_yscale("log")
    ax.set_ylim(0.4, 60)
    ax.set_yticks([1, 10])
    ax.set_yticklabels(["1", "10"], fontsize=TS)
    ax.set_xticks(xs)
    ax.set_xticklabels(TICK, fontsize=TS, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("depth error [%]", fontsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=5.8, handlelength=1.1, columnspacing=0.8,
              borderaxespad=0.0, handletextpad=0.4)

    # ---- (c) basins, clean against contaminated
    ax = axes[2]
    for k, (rows, lab, hatch, alpha) in enumerate(
            ((Kc, "clean", None, 0.94), (Kx, "10% gross", "////", 0.34))):
        vals = [100 * rows[f]["ok"] for f in FAMS]
        ax.bar(xs + (k - 0.5) * w, vals, width=w, color=[C[f] for f in FAMS],
               alpha=alpha, edgecolor="white", lw=0.4, hatch=hatch, label=lab)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=TS)
    ax.set_xticks(xs)
    ax.set_xticklabels(TICK, fontsize=TS, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("lowest cost reached [%]", fontsize=7.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=5.8, handlelength=1.1, columnspacing=0.8,
              borderaxespad=0.0, handletextpad=0.4)

    fig.tight_layout(w_pad=0.55)
    fig.savefig(os.path.join(FIG, "fig8_robust.pdf"))
    plt.close(fig)

    records["robust"] = dict(
        cost={f: [float(r["F"]) for r in J[f]["tukey"]["runs"]] for f in FAMS},
        depth_l2={f: J[f]["l2"]["med"] for f in FAMS},
        depth_tukey={f: J[f]["tukey"]["med"] for f in FAMS},
        basin_clean={f: Kc[f]["ok"] for f in FAMS},
        basin_contam={f: Kx[f]["ok"] for f in FAMS},
    )


if __name__ == "__main__":
    fig1_two_regimes()
    fig2_classification()
    fig3_conditioning()
    fig4_ceiling()
    fig5_invariance()
    fig6_mechanism()
    fig7_joint()
    fig8_robust()
    with open(os.path.join(HERE, "figdata.json"), "w") as fh:
        json.dump(records, fh, indent=1)
    print("figures written to", FIG)
    for r in records.get("invariance", []):
        print(f"  omega={r['omega']:.2e}  sigma_L={r['sigma_L']:.4f}  "
              f"across-family spread={r['spread']:.2e}")
