from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn
from shapely.ops      import unary_union
from shapely.geometry import Polygon
import numpy as np


def get_boundary(projverts: torch.Tensor, faces: torch.Tensor, fnorms: torch.Tensor, P: torch.Tensor, eps=1e-3):
    """
    Args:
        projverts: (V, 2) projected 2D points
        faces: (F, 3) long tensor of face indices
        fnorms: (F, 3) face normals (in world coordinates)
        P: (3, 4) projection matrix
    Returns:
        (L_total, 2) tensor of boundary points (torch, differentiable)
    """
    # 1) Compute visibility mask (front-facing)
    R = P[:, :3]  # (3, 3)
    dot_z = (fnorms.double() @ R.T)[:, 2]
    visible = dot_z > 0
    vis_idx = torch.nonzero(visible).squeeze(-1)
    
    # 2) Project each visible triangle as a Polygon
    V2 = projverts.detach().cpu().numpy()
    F_np = faces[vis_idx].cpu().numpy()
    tris = [Polygon(V2[f]) for f in F_np]
    unioned = unary_union(tris).buffer(eps).buffer(-eps)

    # 3) Extract exterior + interior rings
    rings = []
    if unioned.geom_type == 'Polygon':
        rings.append(np.array(unioned.exterior.coords))
        for interior in unioned.interiors:
            rings.append(np.array(interior.coords))
    elif unioned.geom_type == 'MultiPolygon':
        print("unioned.geom_type == 'MultiPolygon'")
        unioned = max(unioned.geoms, key=lambda p: p.area)
        rings.append(np.array(unioned.exterior.coords))
    else:
        raise ValueError(f"Unexpected geometry type: {unioned.geom_type}")

    # 4) Snap rings to closest projected vertices
    loops = []
    for ring in rings:
        coords_t = torch.from_numpy(ring).to(projverts)           # (L, 2)
        diffs = coords_t[:, None, :] - projverts[None, :, :]      # (L, V, 2)
        d2 = (diffs ** 2).sum(dim=2)                              # (L, V)
        idx = torch.argmin(d2, dim=1)
        idx_u = torch.unique_consecutive(idx)
        if idx_u.numel() > 1 and idx_u[0] == idx_u[-1]:
            idx_u = idx_u[:-1]
        loops.append(projverts[idx_u])

    return torch.cat(loops, dim=0)

class PyTorchChamferLoss(nn.Module):
    def __init__(self, src: Meshes, tgt: Meshes, projmatrices, edgemap_info):
        super().__init__()
        self.src = src  # (B meshes)
        self.tgt = tgt  # (B meshes)
        self.projmatrices = projmatrices # (P, 3, 4)
        self.edgemaps = edgemap_info[0] # (P, max_Ni, 2)
        self.edgemaps_len = edgemap_info[1] # (P,)
    
    def project_vertices(self, vertices):
        """
        Projects a set of vertices into multiple views using different projection matrices.
        Args:
            vertices: Tensor of shape (N, 3), representing 3D vertex positions.
        Returns:
            Tensor of shape (P, N, 2), containing projected 2D points in each view.
        """
        V = vertices.shape[0]
        projection_matrices = self.projmatrices

        ones = torch.ones((V, 1), dtype=vertices.dtype, device=vertices.device)
        vertices_homogeneous = torch.cat([vertices, ones], dim=1).double()  # Shape: (V, 4)

        # Perform batched matrix multiplication (P, 3, 4) @ (V, 4, 1) -> (P, V, 3)
        projected = torch.einsum("pij,vj->pvi", projection_matrices, vertices_homogeneous)  # (P, V, 3)
        
        projected_cartesian = projected[:, :, :2] / projected[:, :, 2:3]  # (P, V, 2)

        return projected_cartesian

        
    def forward(self, y):
        B, P = len(self.src), self.projmatrices.size(0)
        vertices = y
        # project vertices
        num_verts_per_mesh = self.src.num_verts_per_mesh()
        projected_vertices = [] # (B, P, V, 2)
        for b in range(B):
            end = num_verts_per_mesh[b]
            projverts = self.project_vertices(vertices[b][:end,:])  # Shape: (P, V, 2)
            projected_vertices.append(projverts)  # Store without padding

        # get boundaries
        faces_padded = self.src.faces_padded()              # (B, F_max, 3)
        deformed_meshes = self.src.update_padded(y)
        fnorms_padded = deformed_meshes.faces_normals_padded()  # (B, F_max, 3)

        boundaries = [] 
        boundary_lengths = torch.zeros(B, P)
        num_faces = self.src.num_faces_per_mesh()  # list of length B
        for b, batch in enumerate(projected_vertices):
            boundaries_b = []
            faces_b = faces_padded[b][:num_faces[b]]
            fnorms_b = fnorms_padded[b][:num_faces[b]]

            for p, projverts in enumerate(batch):
                boundary = get_boundary(projverts, faces_b, fnorms_b, self.projmatrices[p])
                boundaries_b.append(boundary)
                boundary_lengths[b,p] = len(boundary)
            padded_boundaries = torch.nn.utils.rnn.pad_sequence(boundaries_b, batch_first=True, padding_value=0.0)
            boundaries.append(padded_boundaries)

        # perform chamfer
        chamfer_loss = torch.zeros(B)
        for b in range(B):
            boundaries_b = boundaries[b].float()
            edgemaps_b = self.edgemaps[b].float()
            res, _ = chamfer_distance(  x=boundaries_b,
                                        y=edgemaps_b,
                                        x_lengths=boundary_lengths[b].long(),
                                        y_lengths=self.edgemaps_len[b].long(),
                                        batch_reduction="mean",
                                        point_reduction="mean")
            chamfer_loss[b] = res.sum()
        return chamfer_loss.double() * 10

