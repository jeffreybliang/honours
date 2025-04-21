from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn

def get_boundary(projected_pts, alpha=10.0):
    # Create a detached copy for Alpha_Shaper
    projected_pts_detached = projected_pts.detach().clone()
    
    # Use the detached copy with Alpha_Shaper
    shaper = Alpha_Shaper(projected_pts_detached)
    alpha_shape = shaper.get_shape(alpha)
    while isinstance(alpha_shape, shapely.MultiPolygon) or isinstance(alpha_shape, shapely.GeometryCollection):
        alpha -= 1
        alpha_shape = shaper.get_shape(alpha)
    boundary = torch.tensor(alpha_shape.exterior.coords.xy, dtype=torch.double)
    
    # Find indices of boundary points
    boundary_indices = torch.where(
        torch.any(torch.isclose(projected_pts_detached[:, None], boundary.T, atol=1e-6).all(dim=-1), dim=1)
    )[0]
    
    # Index back into the original tensor with gradients
    boundary_pts = projected_pts[boundary_indices]
    return boundary_pts

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
        boundaries = [] 
        boundary_lengths = torch.zeros(B, P)
        for b, batch in enumerate(projected_vertices):
            boundaries_b = []
            for p, projverts in enumerate(batch):
                boundary = get_boundary(projverts)
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

