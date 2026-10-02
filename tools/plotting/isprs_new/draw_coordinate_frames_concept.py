import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

out_dir = Path(r"E:\zuo\projects\ISPRS\New\figures")
out_png = out_dir / "coordinate_frames_concept_2panel.png"
out_pdf = out_dir / "coordinate_frames_concept_2panel.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
})

COL_W = "#9A9A9A"
COL_CAM = "#1F77B4"
COL_WP = "#2CA02C"
COL_RAY = "#D62728"
COL_BASE = "#FF7F0E"
COL_ANG = "#7B3294"
COL_TXT = "#222222"


def arrow(ax, origin, vec, color, label=None, lw=1.8, ls='-', alpha=1.0, text_offset=(0,0), ms=10):
    ox, oy = origin
    vx, vy = vec
    ax.annotate(
        "", xy=(ox+vx, oy+vy), xytext=(ox, oy),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=0, shrinkB=0, mutation_scale=ms, alpha=alpha)
    )
    if label:
        ax.text(ox+vx+text_offset[0], oy+vy+text_offset[1], label,
                color=color, fontsize=10, ha='center', va='center')


def axes2d(ax, origin, angle_deg, color, name, scale=0.9, lw=2.0, alpha=1.0, label_axes=True):
    a = math.radians(angle_deg)
    ex = (math.cos(a)*scale, math.sin(a)*scale)
    ez = (-math.sin(a)*scale, math.cos(a)*scale)
    arrow(ax, origin, ex, color, rf"$X_{{{name}}}$" if label_axes else None,
          lw=lw, alpha=alpha, text_offset=(0.08, 0.02))
    arrow(ax, origin, ez, color, rf"$Z_{{{name}}}$" if label_axes else None,
          lw=lw, alpha=alpha, text_offset=(0.02, 0.08))
    ax.text(origin[0]-0.08, origin[1]-0.16, rf"${name}$", color=color,
            fontsize=11, fontweight='bold', ha='right', va='top')


def point(ax, xy, label, color=COL_TXT, r=0.045, offset=(0.08,0.08), size=10):
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor='white', lw=0.7, zorder=5))
    ax.text(xy[0]+offset[0], xy[1]+offset[1], label, fontsize=size,
            color=color, ha='left', va='bottom')


def ray(ax, p0, p1, color=COL_RAY, lw=1.8, ls='-'):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls, zorder=2)


def setup(ax, title):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.0, 4.9)
    ax.set_ylim(-0.7, 3.4)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=8)

fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)

# Panel a: one-anchor frames
ax = axs[0]
setup(ax, "(a) One-anchor frames")
W0 = (-0.45, -0.35)
Ca = (1.25, 0.85)
Xk = (3.95, 2.75)

axes2d(ax, W0, 0, COL_W, "W", scale=0.75, lw=1.4, alpha=0.75)
point(ax, Ca, r"$C_a$", color=COL_CAM, offset=(-0.02,-0.32))
point(ax, Xk, r"$X_k$", color=COL_RAY, offset=(0.08,0.03))
ray(ax, Ca, Xk, COL_RAY, lw=1.9)

# Anchor camera frame: rotated, blue
axes2d(ax, Ca, 25, COL_CAM, r"A_c", scale=0.9, lw=2.1)
# Anchor world-parallel frame: parallel to W, green, shifted slightly at same origin visually
axes2d(ax, Ca, 0, COL_WP, r"A_w", scale=0.82, lw=2.0)

ax.text(2.1, 0.28, r"$A_c$: centered at $C_a$, camera-aligned", color=COL_CAM, fontsize=9)
ax.text(2.1, 0.05, r"$A_w$: centered at $C_a$, parallel to $W$", color=COL_WP, fontsize=9)

# Panel b: two-anchor parallax frames
ax = axs[1]
setup(ax, "(b) Two-anchor parallax frames")
W0 = (-0.45, -0.35)
Cm = (1.05, 0.85)
Cs = (3.0, 0.75)
Xk = (3.85, 2.75)

axes2d(ax, W0, 0, COL_W, "W", scale=0.75, lw=1.4, alpha=0.75)
point(ax, Cm, r"$C_m$", color=COL_CAM, offset=(-0.02,-0.32))
point(ax, Cs, r"$C_s$", color=COL_BASE, offset=(0.04,-0.3))
point(ax, Xk, r"$X_k$", color=COL_RAY, offset=(0.08,0.03))

ray(ax, Cm, Cs, COL_BASE, lw=2.0)
ray(ax, Cm, Xk, COL_RAY, lw=1.9)
ray(ax, Cs, Xk, COL_RAY, lw=1.4, ls='--')

axes2d(ax, Cm, 25, COL_CAM, r"M_c", scale=0.9, lw=2.1)
axes2d(ax, Cm, 0, COL_WP, r"M_w", scale=0.82, lw=2.0)

# parallax angle arc near X_k, between rays X->Cm and X->Cs
v1 = (Cm[0]-Xk[0], Cm[1]-Xk[1])
v2 = (Cs[0]-Xk[0], Cs[1]-Xk[1])
a1 = math.degrees(math.atan2(v1[1], v1[0]))
a2 = math.degrees(math.atan2(v2[1], v2[0]))
arc = Arc(Xk, width=0.75, height=0.75, angle=0, theta1=min(a1,a2), theta2=max(a1,a2), color=COL_ANG, lw=1.8)
ax.add_patch(arc)
ax.text(Xk[0]-0.52, Xk[1]-0.05, r"$\omega_k$", color=COL_ANG, fontsize=11)

ax.text(1.95, 0.28, r"$M_c$: main-anchor camera frame", color=COL_CAM, fontsize=9)
ax.text(1.95, 0.05, r"$M_w$: main-anchor world-parallel frame", color=COL_WP, fontsize=9)
ax.text(1.75, 0.95, "baseline", color=COL_BASE, fontsize=9, rotation=-3)

fig.suptitle("Coordinate frames used for object-point parameterizations", fontsize=13, fontweight='bold', y=1.02)
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(out_png)
print(out_pdf)
