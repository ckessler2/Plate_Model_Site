# render_plates_scene.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def plate_scene_xz(
    t, x, z, theta,
    n: int,
    L: float,
    W: float,
    *,
    add_trajectory: bool = False,
    plate_color: str = "royalblue",
    opacity: float = 0.85,
    camera_eye: tuple[float, float, float] = (1.6, 0.6, -1.6),
    projection: str = "perspective", 
) -> go.Figure:
    t     = np.asarray(t,     dtype=float)
    x     = np.asarray(x,     dtype=float)
    z     = np.asarray(z,     dtype=float)
    theta = np.asarray(-theta, dtype=float)

    if not (len(t) == len(x) == len(z) == len(theta)):
        raise ValueError("t, x, z, theta must have the same length.")
    if n < 1:
        raise ValueError("n must be >= 1")

    idx = np.linspace(0, len(t) - 1, n).round().astype(int)

    # Plate stands upright in XY plane; rotating about Y tilts it into Z
    local_3d = np.array([
        [-L / 2, -W / 2,  0.0],
        [ L / 2, -W / 2,  0.0],
        [ L / 2,  W / 2,  0.0],
        [-L / 2,  W / 2,  0.0],
    ], dtype=float)

    Xs, Ys, Zs = [], [], []
    I, J, K    = [], [], []

    for p, k in enumerate(idx):
        ang = theta[k]
        c, s = np.cos(ang), np.sin(ang)

        # Rotation about +Y axis
        R3 = np.array([[ c,  0.0,  s],
                       [0.0,  1.0, 0.0],
                       [-s,  0.0,  c]], dtype=float)

        corners = local_3d @ R3.T   # (4, 3) — free in all 3 dims
        corners[:, 0] += x[k]      # translate along trajectory X
        corners[:, 2] += z[k]      # translate along trajectory Z
        # Y is untouched — plate extends freely in ±Y

        base = 4 * p
        Xs.extend(corners[:, 0])
        Ys.extend(corners[:, 1])
        Zs.extend(corners[:, 2])

        I.extend([base + 0, base + 0])
        J.extend([base + 1, base + 2])
        K.extend([base + 2, base + 3])

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=Xs, y=Ys, z=Zs,
        i=I, j=J, k=K,
        color=plate_color,
        opacity=opacity,
        flatshading=True,
        name="plates",
    ))

    if add_trajectory:
        fig.add_trace(go.Scatter3d(
            x=x, y=np.zeros_like(x), z=z,
            mode="lines",
            line=dict(color="black", width=5),
            name="trajectory",
        ))

    # Auto limits with padding
    pad = 0.1
    x_all = np.array(Xs); y_all = np.array(Ys); z_all = np.array(Zs)
    def padded(arr):
        lo, hi = arr.min(), arr.max()
        d = max(hi - lo, 1e-6)
        return [lo - pad * d, hi + pad * d]

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="x",
                range=padded(x_all),
                showgrid=False, zeroline=False,
                showbackground=False,
                showline=True, linecolor="black", linewidth=2,
                showticklabels=True,
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
            ),
            yaxis=dict(
                title="y",
                range=padded(y_all),
                showgrid=False, zeroline=False,
                showbackground=False,
                showline=True, linecolor="black", linewidth=2,
                showticklabels=True,
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
            ),
            zaxis=dict(
                title="z",
                range=padded(z_all),
                showgrid=False, zeroline=False,
                showbackground=False,
                showline=True, linecolor="black", linewidth=2,
                showticklabels=True,
                tickfont=dict(color="black"),
                title_font=dict(color="black"),
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
                projection=dict(type=projection),
            ),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        showlegend=False,
    )

    return fig