import wandb
import torch
import matplotlib.pyplot as plt
from .chamfer import get_boundary
import plotly.graph_objects as go
import numpy as np
from pytorch3d.structures import Meshes
import trimesh
from PIL import ImageChops
from PIL import Image as PILImage
import plotly.io as pio
from io import BytesIO
import math
from alpha_shapes.boundary import Boundary
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def plot_boundary(ax, boundary: Boundary):
    """Plot a boundary using matplotlib's PathPatch."""
    exterior = Path(boundary.exterior)
    holes = [Path(hole) for hole in boundary.holes]
    path = Path.make_compound_path(exterior, *holes)
    patch = PathPatch(path, facecolor="lightblue", lw=0.5, alpha=0.2, ec="b")
    ax.add_patch(patch)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def padded_to_packed(xs, lengths):
    packed = []
    batch_size = xs.size(0)
    for b in range(batch_size):
        n = lengths[b]
        packed.append(xs[b][:n])
    packed = torch.cat(packed, dim=0)
    return packed

def create_padded_tensor(vertices, vert2mesh, max_V, B):
    padded = torch.zeros((B, max_V, 3),device=vertices.device)
    for i in range(B):
        mesh_vertices = vertices[vert2mesh == i]
        num_vertices = mesh_vertices.shape[0]
        padded[i, :num_vertices, :] = mesh_vertices
    return padded

def plot_vertices(verts_list):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    if not isinstance(verts_list, list):
        x, y, z = verts_list.clone().detach().cpu().squeeze().unbind(1)
        ax.scatter3D(x, z, -y)
    else:
        colors = ['b',  'g', 'r' , 'm', 'c', 'y']  # Define some colors for different sets
        marker_size = 5  # Make points smaller
    
        for i, verts in enumerate(verts_list):
            x, y, z = verts.clone().detach().cpu().squeeze().unbind(1)
            ax.scatter3D(x, z, -y, color=colors[i % len(colors)], s=marker_size, label=f"Set {i+1}")    
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_aspect("equal")
    ax.view_init(190, 30)
    ax.legend()
    plt.show()


def plot_projections(projverts, projmats, edgemaps,
                     boundary_points=None, hulls=None, loops=None, alpha=10.0):
    # Disable interactive mode
    P, _, _ = projmats.shape
    edge_coords, edge_lens = edgemaps

    # Choose grid layout (e.g., 3x4 for 12)
    cols = 4
    rows = math.ceil(P / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()  # Flatten for easy indexing

    for i in range(P):
        proj_2d_hom = (projmats[i] @ torch.cat([
            projverts, torch.ones(projverts.shape[0], 1, device=projverts.device)
        ], dim=1).T).T
        proj_2d = proj_2d_hom[:, :2] / proj_2d_hom[:, 2:3]

        # Detach and move to CPU before plotting
        proj_2d = proj_2d.detach().cpu().numpy()
        ax = axes[i]
        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c='orange', s=8, label="Projected Vertices")
        valid_edges = edge_coords[i, :edge_lens[i]].cpu()
        ax.scatter(valid_edges[:, 0], valid_edges[:, 1], c='b', s=1, label="Target Silhouette")
        # --- Optionally plot hulls ---
        if hulls is not None and i < len(hulls):
            for h in hulls[i]:
                plot_boundary(ax, h)

        # --- Optionally plot mesh loops ---
        if loops is not None and i < len(loops):
            for loop in loops[i]:  # loop: Tensor (L, 2)
                pts = loop.detach().cpu().numpy()
                ax.plot(pts[:, 0], pts[:, 1], linestyle='-', color='r', linewidth=1)

        # --- Optionally scatter individual boundary points (not needed if hulls/loops exist) ---
        if boundary_points is not None and i < len(boundary_points):
            pts = boundary_points[i].detach().cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], c='r', s=2, label="Boundary Pts")

        ax.set_title(f"Projection {i+1}", fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.axis("equal")
        ax.invert_yaxis()

    for j in range(P, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return img


def visualise_meshes(srcmesh, tgtmesh):
    vertices = np.asarray(srcmesh.verts_packed())
    faces = np.asarray(srcmesh.faces_packed())

    # Create a Plotly 3D mesh
    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        color="lightblue"
    )])

    vertices = np.asarray(tgtmesh.verts_packed())
    faces = np.asarray(tgtmesh.faces_packed())

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.1,
        color="red"
    ))

    # Update layout for better presentation
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        title="3D Mesh Visualization"
    )

    # Show the figure
    fig.show()


def visualise_heatmap(src: Meshes, tgt: Meshes, cmin=None, cmax=None):
    mesh_X = trimesh.Trimesh(vertices=src[0].verts_packed().detach().cpu().numpy(), 
                                faces=src[0].faces_packed().detach().cpu().numpy())
    mesh_Y = trimesh.Trimesh(vertices=tgt[0].verts_packed().detach().cpu().numpy(), 
                                faces=tgt[0].faces_packed().detach().cpu().numpy())

    # Get vertices
    X_vertices = mesh_X.vertices
    Y_tree = trimesh.proximity.ProximityQuery(mesh_Y)

    # For each vertex in X, find closest point on Y
    closest_points, _, _ = Y_tree.on_surface(X_vertices)

    # Compute signed distance based on norm from origin
    X_norms = np.linalg.norm(X_vertices, axis=1)
    Y_norms = np.linalg.norm(closest_points, axis=1)
    signed_dists = X_norms - Y_norms  # positive = further out than Y

    # Normalize or clip for better colormap contrast if needed
    max_val = np.max(np.abs(signed_dists))
    if cmin is None or cmax is None:
        cmin, cmax = -max_val, max_val  # white will now be centered at 0

    # Plot using Plotly
    i, j, k = mesh_X.faces.T
    fig = go.Figure(data=[
        go.Mesh3d(
            x=X_vertices[:, 0],
            y=X_vertices[:, 1],
            z=X_vertices[:, 2],
            i=i, j=j, k=k,
            intensity=signed_dists,
            colorscale='RdBu',
            reversescale=True,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title='Signed Distance'),
            showscale=True,
            # flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.9),
            lightposition=dict(x=100, y=200, z=0),
            opacity=1.0
        )
    ])
    fig.update_layout(
        title='Mesh X Colored by Signed Distance from Mesh Y',
        scene=dict(aspectmode='data')
    )
    fig.show()
    return cmin,cmax


def crop_img(img, bg_color=(255, 255, 255), threshold=5):
    bg = PILImage.new(img.mode, img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    mask = diff.convert('L').point(lambda p: 255 if p > threshold else 0)
    bbox = mask.getbbox()
    return img.crop(bbox) if bbox else img


def scale_and_crop(img, target_w, target_h):
    original_w, original_h = img.size
    scale_factor = max(target_w / original_w, target_h / original_h) + 0.1
    scaled_img = img.resize((int(original_w * scale_factor), int(original_h * scale_factor)), PILImage.LANCZOS)

    left = (scaled_img.width - target_w) // 2
    upper = (scaled_img.height - target_h) // 2
    return scaled_img.crop((left, upper, left + target_w, upper + target_h))


def compute_signed_distances(src, tgt):
    mesh_X = trimesh.Trimesh(vertices=src[0].verts_packed().detach().cpu().numpy(), faces=src[0].faces_packed().detach().cpu().numpy())
    mesh_Y = trimesh.Trimesh(vertices=tgt[0].verts_packed().detach().cpu().numpy(), faces=tgt[0].faces_packed().detach().cpu().numpy())

    if not mesh_Y.is_watertight or len(mesh_Y.triangles) == 0:
        raise ValueError("Target mesh is invalid or empty")


    X_vertices = mesh_X.vertices
    Y_tree = trimesh.proximity.ProximityQuery(mesh_Y)
    closest_points, _, _ = Y_tree.on_surface(X_vertices)
    
    X_norms = np.linalg.norm(X_vertices, axis=1)
    Y_norms = np.linalg.norm(closest_points, axis=1)
    return X_norms - Y_norms


def generate_3d_visualization(src, tgt, signed_dists, cmin, cmax):
    mesh_X = trimesh.Trimesh(vertices=src[0].verts_packed().detach().cpu().numpy(), faces=src[0].faces_packed().detach().cpu().numpy())
    i, j, k = mesh_X.faces.T
    fig = go.Figure(data=[
        go.Mesh3d(
            x=mesh_X.vertices[:, 0],
            y=mesh_X.vertices[:, 1],
            z=mesh_X.vertices[:, 2],
            i=i, j=j, k=k,
            intensity=signed_dists,
            colorscale='RdBu',
            reversescale=True,
            cmin=cmin, cmax=cmax,
            colorbar=dict(title='Signed Distance'),
            showscale=True,
            flatshading=True,
            lighting=dict(ambient=0.8, diffuse=0.7),
            lightposition=dict(x=100, y=200, z=0),
            opacity=1.0
        )
    ])
    fig.update_layout(
        # title='Mesh X Colored by Signed Distance from Mesh Y',
        scene=dict(
            aspectmode='data',
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z")),         
    )
    return fig


def capture_camera_views(fig, camera_views, target_w, target_h):
    images = []
    for cam in camera_views:
        fig.update_layout(scene_camera=cam)
        png_bytes = pio.to_image(fig, format='png', width=400, height=400)
        img = PILImage.open(BytesIO(png_bytes)).convert('RGB')
        img = crop_img(img)
        images.append(scale_and_crop(img, target_w, target_h))
    return images


def create_colorbar(cmin, cmax):
    dummy_fig = go.Figure(data=[go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=0.0001, color=[0], colorscale='RdBu', cmin=cmin, cmax=cmax, colorbar=dict(title=' '))
    )])
    dummy_fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                            margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    colorbar_img = PILImage.open(BytesIO(pio.to_image(dummy_fig, format='png', width=200, height=400)))
    return crop_img(colorbar_img)


def create_final_image(images, colorbar_img, target_w, target_h):
    grid = PILImage.new('RGB', (2 * target_w, 2 * target_h))
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (target_w, 0))
    grid.paste(images[2], (0, target_h))
    grid.paste(images[3], (target_w, target_h))

    final_img = PILImage.new('RGB', (2 * target_w + colorbar_img.width, 2 * target_h), color=(255, 255, 255))
    final_img.paste(grid, (0, 0))
    final_img.paste(colorbar_img, (2 * target_w, (2 * target_h - colorbar_img.height) // 2))
    return final_img


def create_heatmap(src, tgt, cmin=None, cmax=None):
    signed_dists = compute_signed_distances(src, tgt)
    if cmin is None or cmax is None:
        max_val = np.max(np.abs(signed_dists))
        cmin, cmax = -max_val, max_val

    fig = generate_3d_visualization(src, tgt, signed_dists, cmin, cmax)
    if False:
        camera_views = [
            dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            dict(eye=dict(x=-1.2, y=1.2, z=1.2)),
            dict(eye=dict(x=1.2, y=-1.2, z=1.2)),
            dict(eye=dict(x=0.0, y=0.0, z=2.4)),
        ]
        
        images = capture_camera_views(fig, camera_views, 256, 256)
        colorbar_img = create_colorbar(cmin, cmax)
        final_img = create_final_image(images, colorbar_img, 256, 256)

    # # Display the final image
    # dpi = 100  # You can adjust this if needed
    # figsize = (final_img.width / dpi, final_img.height / dpi)

    # fig = plt.figure(figsize=figsize, dpi=dpi)
    # ax = plt.Axes(fig, [0, 0, 1, 1])  # [left, bottom, width, height] in figure coordinates
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(final_img)
    # plt.show()

    return fig, cmin, cmax


def compute_boundary_distance_heatmap(src: Meshes, boundary_mask, D_all, cmin=None, cmax=None):
     """
     Computes a 3D heatmap coloring each vertex in `src` based on its geodesic distance to the nearest boundary vertex.
     """
     verts3d = src.verts_padded().detach()[0]  # (V, 3)
     V = verts3d.shape[0]
     faces = src.faces_packed().detach()
 
     with torch.no_grad():
         boundary_ids = torch.where(boundary_mask[:V])[0]
         D = D_all[:V, :V]
         boundary_dists = D[:, boundary_ids]  # (V, B)
         min_dist, _ = torch.min(boundary_dists, dim=1)
         heatmap_values = min_dist.float().cpu().numpy()
 
     if cmin is None or cmax is None:
         max_val = np.percentile(heatmap_values, 95)
         cmin, cmax = 0, max_val
 
     mesh = trimesh.Trimesh(vertices=verts3d.cpu().numpy(), faces=faces.cpu().numpy())
     i, j, k = mesh.faces.T
 
     fig = go.Figure(data=[
         go.Mesh3d(
             x=mesh.vertices[:, 0],
             y=mesh.vertices[:, 1],
             z=mesh.vertices[:, 2],
             i=i, j=j, k=k,
             intensity=heatmap_values,
             colorscale='Greens',
             cmin=cmin, cmax=cmax,
             colorbar=dict(title='Hop Distance to Boundary'),
             showscale=True,
             lighting=dict(ambient=0.8, diffuse=0.9),
             lightposition=dict(x=100, y=200, z=0),
             opacity=1.0
         )
     ])
     fig.update_layout(
         # title='Mesh X Colored by Signed Distance from Mesh Y',
         scene=dict(
             aspectmode='data',
             xaxis=dict(title="X"),
             yaxis=dict(title="Y"),
             zaxis=dict(title="Z")),         
     )
     fig.update_layout(title='Distance to Nearest Boundary (3D Geodesic)', scene=dict(aspectmode='data'))
     return fig, cmin, cmax