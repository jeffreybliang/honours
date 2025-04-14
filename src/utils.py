import torch
import matplotlib.pyplot as plt
from .chamfer import get_boundary
import plotly.graph_objects as go
import numpy as np
from pytorch3d.structures import Meshes
import trimesh


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

def plot_projections(projverts, projmats, edgemaps):
    plt.ioff()  # Disable interactive mode

    P, _, _ = projmats.shape
    edge_coords, edge_lens = edgemaps
    fig, axes = plt.subplots(1, P, figsize=(2 * P, 2))  # Increase figure size
    if P == 1:
        axes = [axes]  # Ensure iterable for a single subplot case

    for i in range(P):
        proj_2d_hom = (projmats[i] @ torch.cat([projverts, torch.ones(projverts.shape[0], 1)], dim=1).T).T
        proj_2d = proj_2d_hom[:, :2] / proj_2d_hom[:, 2:3]  # Normalize by depth

        boundary_pts = get_boundary(proj_2d)
        valid_edges = edge_coords[i, :edge_lens[i]]

        ax = axes[i]
        ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c='b', s=8, label="Projected Vertices")
        ax.scatter(valid_edges[:, 0], valid_edges[:, 1], c='r', s=3, label="Edge Coordinates")
        ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='g', s=3, label="Boundary Points")

        ax.set_title(f"Projection {i+1}", fontsize=10)  # Reduce title size
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        
        ax.tick_params(axis='both', which='major', labelsize=6)  # Reduce tick label size
        ax.axis("equal")
        ax.invert_yaxis()

    plt.tight_layout(pad=0.5)  # Reduce whitespace
    plt.subplots_adjust(wspace=0.1)  # Reduce horizontal space
    plt.show()
    plt.close(fig)

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

