import torch
import matplotlib.pyplot as plt
from chamfer import get_boundary

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
