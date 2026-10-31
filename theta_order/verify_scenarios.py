"""
场景级验证：把候选应用场景落到 (Lam, B, beta, f_px) 上，
1) 算出 omega / kappa / 五族的 anis, s_blk, cond 与精度天花板 sigma_L/Lam；
2) 在 float64 与 float32 下实跑 GN 三角化，看 Theta(1) 是否变成可测的收益；
3) 扫 omega，找出每族在 float32 下的断裂点。

只用 numpy。运行： "E:/zuo/Program Files/Python313/python.exe" verify_scenarios.py
"""
import numpy as np

np.set_printoptions(precision=5, suppress=True)
F32 = np.float32
F64 = np.float64


# ---------------------------------------------------------------- 几何基元
def look_at(C, X):
    z = X - C
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def dpi(z, f):
    """D Pi(z), 2x3"""
    zx, zy, zz = z
    return np.array([[f / zz, 0.0, -f * zx / zz**2],
                     [0.0, f / zz, -f * zy / zz**2]])


def pi(z, f):
    return np.array([f * z[0] / z[2], f * z[1] / z[2]])


class Scene:
    """主锚在原点，主射线 u=+z；其余相机沿基线 t*B*bh 分布（与 verify_ratio.py 一致）。"""

    def __init__(self, Lam, B, beta_deg, f, n_c=4, sigma_px=0.5, tilt_deg=0.0):
        self.Lam, self.B, self.beta = Lam, B, np.radians(beta_deg)
        self.f, self.n_c, self.sigma = f, n_c, sigma_px
        u = np.array([0.0, 0.0, 1.0])
        bh = np.array([np.sin(self.beta), 0.0, np.cos(self.beta)])
        self.bhat, self.u = bh, u
        self.Cs = [np.zeros(3) + t * B * bh for t in np.linspace(0.0, 1.0, n_c)]
        self.Cm = self.Cs[0]
        X = Lam * u
        if tilt_deg:
            th = np.radians(tilt_deg)
            X = Lam * np.array([np.sin(th), 0.0, np.cos(th)])
        self.X = X
        self.Rs = [look_at(C, X) for C in self.Cs]
        self.obs = [pi(R @ (X - C), f) for R, C in zip(self.Rs, self.Cs)]
        # 局部切平面基
        u0 = (X - self.Cm) / np.linalg.norm(X - self.Cm)
        self.u0 = u0
        e1 = np.cross(u0, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.array([1.0, 0.0, 0.0])
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(u0, self.e1)

    def mab(self, a, b):
        return (self.u0 + a * self.e1 + b * self.e2) / np.linalg.norm(
            self.u0 + a * self.e1 + b * self.e2)

    def ab(self, m):
        den = m @ self.u0
        return (m @ self.e1) / den, (m @ self.e2) / den

    def kappa(self, m, B=None, bhat=None):
        B = self.B if B is None else B
        bhat = self.bhat if bhat is None else bhat
        return B * np.linalg.norm(np.cross(bhat, m))

    def omega(self, m, Lam=None, B=None, bhat=None):
        B = self.B if B is None else B
        bhat = self.bhat if bhat is None else bhat
        Lam = self.Lam if Lam is None else Lam
        kap = self.kappa(m, B, bhat)
        cosb = m @ bhat
        return np.arctan2(kap, Lam - B * cosb)

    # ---- 各族的 chart：q -> X
    def X_of(self, fam, q):
        if fam == "XYZ":
            return np.asarray(q, dtype=float)
        a, b, s = q
        m = self.mab(a, b)
        if fam == "SphRange":
            return self.Cm + s * m
        if fam == "SphInvRange":
            return self.Cm + m / s
        if fam == "InvZ":
            Xf = np.array([a / s, b / s, 1.0 / s])
            return self.Cm + self.Rs[0].T @ Xf
        if fam == "Parallax":
            B, bh = self.B, self.bhat
            s = min(max(float(s), 1e-12), np.pi - 1e-12)
            cosb = np.clip(m @ bh, -1.0, 1.0)
            beta = np.arccos(cosb)
            Lam = B * np.sin(beta + s) / np.sin(s)
            return self.Cm + Lam * m
        raise KeyError(fam)

    def q_of(self, fam, X):
        if fam == "XYZ":
            return np.asarray(X, dtype=float)
        v = X - self.Cm
        Lam = np.linalg.norm(v)
        m = v / Lam
        if fam == "SphRange":
            a, b = self.ab(m)
            return np.array([a, b, Lam])
        if fam == "SphInvRange":
            a, b = self.ab(m)
            return np.array([a, b, 1.0 / Lam])
        if fam == "InvZ":
            Xf = self.Rs[0] @ v
            rho = 1.0 / Xf[2]
            return np.array([Xf[0] * rho, Xf[1] * rho, rho])
        if fam == "Parallax":
            a, b = self.ab(m)
            om = self.omega(m, Lam=Lam)
            return np.array([a, b, om])
        raise KeyError(fam)

    def residual(self, fam, q):
        X = self.X_of(fam, q)
        r = []
        for R, C, o in zip(self.Rs, self.Cs, self.obs):
            r.append(pi(R @ (X - C), self.f) - o)
        return np.concatenate(r)

    def S(self, fam, q, h=1e-6):
        base = np.asarray(q, dtype=float)
        S = np.zeros((3, 3))
        for k in range(3):
            step = h * max(abs(base[k]), 1e-3)
            qp, qm = base.copy(), base.copy()
            qp[k] += step
            qm[k] -= step
            S[:, k] = (self.X_of(fam, qp) - self.X_of(fam, qm)) / (2 * step)
        return S

    def A(self):
        """d e / d X_world = DPi(z_cam) @ R   （漏了 R 会让整块恒为秩 2）"""
        return np.vstack([dpi(R @ (self.X - C), self.f) @ R
                          for R, C in zip(self.Rs, self.Cs)])


FAMS = ["XYZ", "InvZ", "SphRange", "SphInvRange", "Parallax"]


def metrics(sc, fam):
    q = sc.q_of(fam, sc.X)
    A = sc.A()
    S = sc.S(fam, q)
    J = A @ S
    sv = np.linalg.svd(J, compute_uv=False)
    scale = np.linalg.norm(A @ S[:, 2])
    bear = np.sqrt(np.linalg.norm(A @ S[:, 0]) * np.linalg.norm(A @ S[:, 1]))
    anis = bear / scale
    cond = (sv[0] / sv[2]) ** 2
    return dict(q=q, sv=sv, scale=scale, bear=bear, anis=anis,
                s_blk=scale / sc.f, cond=cond)


def gn(sc, fam, dtype, init_dL=0.20, init_ang=2e-3, iters=30, tol=1e-3):
    """只用点块做 GN（相机固定已知），测收敛与实际深度误差。"""
    q = sc.q_of(fam, sc.X).astype(dtype)
    # 扰动：深度 +20%、方向挪 2e-3 rad
    Xp = sc.X * (1 + init_dL)
    Xp = Xp + init_ang * sc.Lam * typecast(sc.e1, dtype)
    q = sc.q_of(fam, Xp).astype(dtype)
    A32 = sc.A().astype(dtype)
    Cs = [typecast(C, dtype) for C in sc.Cs]
    Rs = [typecast(R, dtype) for R in sc.Rs]
    obs = [typecast(o, dtype) for o in sc.obs]
    f = dtype(sc.f)
    hist = []
    for it in range(iters):
        X = typecast(sc.X_of(fam, q), dtype)
        r, Js = [], []
        for R, C, o in zip(Rs, Cs, obs):
            z = R @ (X - C)
            zz = z[2]
            r.append(np.array([f * z[0] / zz, f * z[1] / zz], dtype=dtype) - o)
            A = np.array([[f / zz, dtype(0), -f * z[0] / zz**2],
                          [dtype(0), f / zz, -f * z[1] / zz**2]], dtype=dtype)
            Js.append(A @ R)
        r = np.concatenate(r)
        J = np.vstack(Js).astype(dtype) @ typecast(sc.S(fam, q.astype(float)), dtype)
        rn = float(np.linalg.norm(r))
        hist.append(rn)
        try:
            dq = -np.linalg.solve((J.T @ J).astype(dtype), (J.T @ r).astype(dtype))
        except np.linalg.LinAlgError:
            return None, np.inf, hist
        if not np.all(np.isfinite(dq)):
            return None, np.inf, hist
        # 阻尼：步长过大时折半
        lam = dtype(1.0)
        for _ in range(8):
            qn = (q + lam * dq).astype(dtype)
            try:
                Xn = typecast(sc.X_of(fam, qn.astype(float)), dtype)
                rn2 = []
                for R, C, o in zip(Rs, Cs, obs):
                    z = R @ (Xn - C)
                    rn2.append(np.array([f * z[0] / z[2], f * z[1] / z[2]], dtype=dtype) - o)
                if float(np.linalg.norm(np.concatenate(rn2))) < rn:
                    break
            except Exception:
                pass
            lam = lam * dtype(0.5)
        q = clamp_q(fam, (q + lam * dq).astype(dtype))
        Xe = sc.X_of(fam, q.astype(float))
        if np.linalg.norm(Xe - sc.X) / sc.Lam < 1e-7:
            break
    Xe = sc.X_of(fam, q.astype(float))
    err = np.linalg.norm(Xe - sc.X) / sc.Lam
    return q, err, hist


def typecast(M, dtype):
    return np.array(M, dtype=dtype)


def clamp_q(fam, q):
    """把参数留在 chart 的定义域内（omega>0，rho>0）。"""
    if fam == "Parallax":
        q[2] = min(max(q[2], 1e-12), np.pi - 1e-12)
    if fam in ("SphInvRange", "InvZ"):
        q[2] = max(q[2], 1e-12)
    if fam == "SphRange":
        q[2] = max(q[2], 1e-12)
    return q


# ---------------------------------------------------------------- 场景表
SCEN = [
    # 名称, Lam[m], B[m], beta[deg], f_px, sigma_px, 需求相对精度, 说明
    ("卫星视频立体(弱交会)", 5.0e5, 1.5e4, 90.0, 1.66e6, 0.5, 1e-5, "H=500km,f=13.3mm,pix=8um"),
    ("高空无人机长焦悬停", 3.0e2, 2.0e-1, 90.0, 1.45e4, 0.5, 3e-3, "H=300m,50mm,pix=3.45um"),
    ("电力/风电长焦巡检", 1.5e1, 5.0e-2, 90.0, 3.64e4, 0.5, 1e-3, "15m,200mm,pix=5.5um"),
    ("桥梁/大坝变形监测", 2.5e1, 3.0e-1, 90.0, 9.09e3, 0.2, 4e-5, "25m,50mm,pix=5.5um,0.2px"),
    ("手机长焦多摄测距", 5.0, 1.5e-2, 90.0, 7.70e4, 0.5, 2e-3, "5m,77mm,pix=1.0um"),
    ("AR 眼镜双目", 2.0, 6.4e-2, 90.0, 1.0e3, 0.5, 1e-2, "2m,基线64mm"),
    ("腹腔镜双目", 3.0e-2, 1.0e-3, 90.0, 8.0e2, 0.5, 2e-2, "30mm,基线1mm"),
    ("水下 ROV 双目", 3.0, 2.0e-1, 90.0, 1.2e3, 1.0, 5e-3, "3m,基线200mm,浑浊"),
]


def main():
    print("=" * 132)
    print("表 1  场景量化：omega / kappa / 精度天花板 / 五族条件数（float64）")
    print("=" * 132)
    hdr = (f"{'场景':<22}{'Lam':>9}{'B':>9}{'omega[deg]':>11}{'kappa':>9}"
           f"{'f*omega':>9}{'sigL/Lam':>10}{'需求':>8}{'够?':>4}"
           f"{'cond_XYZ':>10}{'cond_InvZ':>10}{'cond_SphInv':>12}{'cond_Par':>9}{'a_InvZ':>8}")
    print(hdr)
    print("-" * 132)
    rows = []
    for name, Lam, B, beta, f, sig, need, note in SCEN:
        sc = Scene(Lam, B, beta, f, n_c=4, sigma_px=sig)
        m = sc.u0
        om = sc.omega(m)
        kap = sc.kappa(m)
        M = {fm: metrics(sc, fm) for fm in FAMS}
        sigL = sig / (f * om)  # 相对深度精度天花板
        ok = "Y" if sigL <= need else "N"
        print(f"{name:<22}{Lam:>9.3g}{B:>9.3g}{np.degrees(om):>11.4f}{kap:>9.3g}"
              f"{f*om:>9.3g}{sigL:>10.3g}{need:>8.1g}{ok:>4}"
              f"{M['XYZ']['cond']:>10.3g}{M['InvZ']['cond']:>10.3g}"
              f"{M['SphInvRange']['cond']:>12.3g}{M['Parallax']['cond']:>9.3g}"
              f"{M['InvZ']['anis']:>8.3g}")
        rows.append((name, sc, M, om, kap, sigL, need))

    print()
    print("=" * 132)
    print("表 2  float32 下实跑 GN（相机已知、只优化点块），初始深度误差 +20%，方向偏 2e-3 rad")
    print("=" * 132)
    print(f"{'场景':<22}{'':<3}" + "".join(f"{fm:>22}" for fm in FAMS))
    print(f"{'':<25}" + "".join(f"{'err/iter':>22}" for _ in FAMS))
    print("-" * 132)
    for name, sc, M, om, kap, sigL, need in rows:
        line = f"{name:<25}"
        for fm in FAMS:
            q, err, _ = gn(sc, fm, F32)
            ok = "" if err < 1e-3 else "R"
            line += f"{err:>18.2e}{ok:>4}"
        print(line)
    print("  R = 失败（相对深度误差 > 1e-3）")

    print()
    print("=" * 132)
    print("表 3  float64 基线对照")
    print("=" * 132)
    print(f"{'场景':<25}" + "".join(f"{fm:>20}" for fm in FAMS))
    for name, sc, M, om, kap, sigL, need in rows:
        line = f"{name:<25}"
        for fm in FAMS:
            q, err, _ = gn(sc, fm, F64)
            line += f"{err:>20.2e}"
        print(line)

    print()
    print("=" * 132)
    print("表 4  omega 扫描：短基线（Lam 固定）下每族在 float32 中断裂的 omega")
    print("=" * 132)
    for Lam, f in [(25.0, 9.09e3), (5.0, 7.7e4), (300.0, 1.45e4)]:
        print(f"\n  Lam={Lam:g} m, f={f:.3g} px   （B 从 1e-1 扫到 1e-6 m）")
        print(f"    {'B':>10}{'omega[rad]':>12}" + "".join(f"{fm:>18}" for fm in FAMS))
        for B in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            sc = Scene(Lam, B, 90.0, f, n_c=4)
            om = sc.omega(sc.u0)
            line = f"    {B:>10.1e}{om:>12.2e}"
            for fm in FAMS:
                q, err, _ = gn(sc, fm, F32)
                line += f"{err:>18.2e}"
            print(line)

    print()
    print("=" * 132)
    print("表 5  解析自检：‖A c_omega‖ 与 f*nu*theta*kappa_j/kappa 是否一致（式 72）")
    print("=" * 132)
    for Lam, B, beta, f in [(1e2, 1.0, 90.0, 800.0), (1e4, 1.0, 90.0, 800.0),
                            (25.0, 0.3, 90.0, 9091.0), (25.0, 1e-3, 90.0, 9091.0)]:
        sc = Scene(Lam, B, beta, f, n_c=4)
        M = metrics(sc, "Parallax")
        om = sc.omega(sc.u0)
        kap = sc.kappa(sc.u0)
        # 逐相机：f * nu_j * theta_j * kap_j / kap，theta_j in [1,nu_j]，取 1 得下界
        vals = []
        for j, (R, C) in enumerate(zip(sc.Rs, sc.Cs)):
            z = R @ (sc.X - C)
            nu = np.linalg.norm(z) / z[2]
            kj = np.linalg.norm(np.cross(C - sc.Cm, sc.u0)) if j else 0.0
            vals.append(f * nu * kj / kap)
        stacked = np.sqrt(np.sum(np.array(vals) ** 2))
        print(f"  Lam={Lam:<8g} B={B:<8g} omega={np.degrees(om):>9.5f}deg  "
              f"实测‖Ac_w‖={M['scale']:>12.5g}  式72下界={stacked:>12.5g}  "
              f"比值={M['scale']/max(stacked,1e-30):>8.4f}")


if __name__ == "__main__":
    main()
