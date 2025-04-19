import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from .utils import *
import plotly.graph_objects as go
from matplotlib.patches import Polygon


def plot_ellipsoid_mpl(a, b, c, yaw, pitch, roll, points, r=0.62, vmin=None, vmax=None, show_gt=True):
    u = torch.linspace(0, 2 * torch.pi, 80)
    v = torch.linspace(0, torch.pi, 40)
    u, v = torch.meshgrid(u, v, indexing="ij")

    x = a * torch.cos(u) * torch.sin(v)
    y = b * torch.sin(u) * torch.sin(v)
    z = c * torch.cos(v)
    ellipsoid = torch.stack((x, y, z), dim=-1).reshape(-1, 3).T  # (3, N)

    angles_rad = torch.deg2rad(torch.tensor([yaw, pitch, roll]))
    R = rotation_matrix_3d(angles_rad)
    rotated = (R @ ellipsoid.double()).T.reshape(x.shape + (3,))
    ellipsoid_xyz = rotated.reshape(-1, 3).T
    surface_dist = torch.norm(ellipsoid_xyz, dim=0) - r
    residuals = surface_dist.reshape(rotated.shape[:2])

    # Normalize residuals for coloring
    if vmin is None or vmax is None:
        vmin, vmax = residuals.min().item(), residuals.max().item()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["RdBu_r"]
    facecolors = cmap(norm(residuals))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        rotated[..., 0], rotated[..., 1], rotated[..., 2],
        facecolors=facecolors,
        rstride=1, cstride=1,
        antialiased=True, linewidth=0,
        alpha=0.8, shade=False
    )

    # Rotate points to align with ellipsoid
    points = (R @ points.double()).detach()
    ax.scatter(points[0], points[1], points[2], s=1, alpha=0.3, color="black")
    ax.set_axis_off()
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Residual from sphere (r=0.62)")
    if show_gt:
        u_gt = torch.linspace(0, 2 * torch.pi, 40)
        v_gt = torch.linspace(0, torch.pi, 20)
        u_gt, v_gt = torch.meshgrid(u_gt, v_gt, indexing="ij")
        x_gt = r * torch.cos(u_gt) * torch.sin(v_gt)
        y_gt = r * torch.sin(u_gt) * torch.sin(v_gt)
        z_gt = r * torch.cos(v_gt)

        ax.plot_surface(
            x_gt, y_gt, z_gt,
            color="gray", alpha=0.05
        )

    return fig



def plot_ellipsoid_plotly(a, b, c, yaw, pitch, roll, points, r=0.62, show_gt=True):
    u, v = torch.linspace(0, 2 * torch.pi, 60), torch.linspace(0, torch.pi, 30)
    u, v = torch.meshgrid(u, v, indexing="ij")
    x = a * torch.cos(u) * torch.sin(v)
    y = b * torch.sin(u) * torch.sin(v)
    z = c * torch.cos(v)

    # Rotation
    angles = torch.tensor([yaw, pitch, roll])
    R = rotation_matrix_3d(torch.deg2rad(angles))
    ellipsoid = torch.stack([x, y, z], dim=-1).reshape(-1, 3).T.double()  # (3, N)
    rotated = (R @ ellipsoid).T.reshape(x.shape + (3,))

    # Residuals for color
    ellipsoid_xyz = rotated.reshape(-1, 3).T
    radii = torch.norm(ellipsoid_xyz, dim=0)
    residuals = (radii - r).reshape(rotated.shape[:2]).cpu().numpy()

    surface = go.Surface(
        x=rotated[..., 0].cpu().numpy(),
        y=rotated[..., 1].cpu().numpy(),
        z=rotated[..., 2].cpu().numpy(),
        surfacecolor=residuals,
        colorscale='RdBu',
        cmin=-0.5,
        cmax=0.5,
        opacity=1,
        showscale=True
    )

    # Noisy points (rotated)
    pts = (R @ points.double()).detach().cpu().numpy()
    scatter = go.Scatter3d(
        x=pts[0], y=pts[1], z=pts[2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.5)
    )

    # Optional: GT sphere wireframe
    if show_gt:
        u_gt, v_gt = torch.meshgrid(torch.linspace(0, 2*torch.pi, 40), torch.linspace(0, torch.pi, 20))
        x_gt = r * torch.cos(u_gt) * torch.sin(v_gt)
        y_gt = r * torch.sin(u_gt) * torch.sin(v_gt)
        z_gt = r * torch.cos(v_gt)
        sphere = go.Surface(
            x=x_gt, y=y_gt, z=z_gt,
            showscale=False,
            opacity=0.1,
            colorscale=[[0, "gray"], [1, "gray"]],
            surfacecolor=torch.zeros_like(x_gt)
        )
        data = [surface, scatter, sphere]
    else:
        data = [surface, scatter]

    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode='cube'
    ))
    fig.update_layout(
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
            )
        )
    return fig




def plot_silhouettes(projected_pts, target_pts, step=None,
                     boundary_points=None, hulls=None, show_alpha=False):
    """
    projected_pts: (B, V, 2, N)
    target_pts:    (V, M, 2)
    boundary_points: Optional List[List[Tensor]] of shape (B x V)
    hulls:          Optional List[List[Polygon]] of shape (B x V)
    show_alpha:     Whether to show alpha shape + boundary points
    """
    n_batches, n_views, _, _ = projected_pts.shape
    fig, axs = plt.subplots(n_batches, n_views, figsize=(4 * n_views, 4 * n_batches), squeeze=False)

    for b in range(n_batches):
        for v in range(n_views):
            ax = axs[b, v]

            proj = projected_pts[b, v].detach().cpu().numpy()     # (2, N)
            targ = target_pts[v].detach().cpu().numpy().T         # (2, M)

            ax.scatter(*proj, color='orange', s=2, label='Projected Points')
            ax.scatter(*targ, color='blue', s=2, label='Target Points')

            if show_alpha and boundary_points is not None and hulls is not None:
                boundary = boundary_points[b][v].detach().cpu().numpy()
                ax.scatter(*boundary, color='red', s=2, label='Boundary Points')

                polygon = Polygon(hulls[b][v].exterior.coords,
                                  facecolor='lightblue', edgecolor='blue', alpha=0.4)
                ax.add_patch(polygon)

            ax.set_title(f'Batch {b+1}, View {v+1}')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.grid(True)
            ax.legend(loc='lower right', fontsize='x-small')

    plt.tight_layout()
    return fig
