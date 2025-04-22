from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn
import numpy as np

def get_boundary(projected_pts: torch.Tensor, alpha: float = 12.0):
    """
    projected_pts : (V,2) *on GPU* – 2‑D projected vertices
    alpha         : α‑shape parameter

    Returns
    --------
    boundary_pts  : (B,2)  same tensor type/device as input
    b_mask        : (V,)   bool mask, True for boundary vertices
    """
    # ── shapely runs on CPU, so work on a detached CPU copy ─────────────
    proj_cpu = projected_pts.detach().clone().cpu()

    shaper       = Alpha_Shaper(proj_cpu)
    alpha_shape  = shaper.get_shape(alpha)
    while isinstance(alpha_shape, (shapely.MultiPolygon,
                                   shapely.GeometryCollection)):
        alpha -= 1
        alpha_shape = shaper.get_shape(alpha)

    coords_xy = np.stack(alpha_shape.exterior.coords.xy, axis=-1)  # (B,2)
    boundary_cpu = torch.as_tensor(coords_xy, dtype=torch.double)   # CPU

    # ── match coordinates back to *CPU* copy of proj vertices ───────────
    #     (all‑close on both x & y, then “any” across B points)
    match = torch.isclose(
        proj_cpu.unsqueeze(1),                # (V,1,2)
        boundary_cpu.unsqueeze(0),            # (1,B,2)
        atol=1e-6
    ).all(dim=-1).any(dim=1)                 # (V,) bool

    # indices & mask back to original device
    b_mask = match.to(projected_pts.device)   # (V,) bool
    b_idx  = b_mask.nonzero(as_tuple=False).view(-1)

    boundary_pts = projected_pts[b_idx]       # keeps gradient flow
    return boundary_pts, b_mask

class PyTorchChamferLoss(nn.Module):
    def __init__(self, src: Meshes, tgt: Meshes, projmatrices, edgemap_info, boundary_mask=None):
        super().__init__()
        self.src = src  # (B meshes)
        self.tgt = tgt  # (B meshes)
        self.projmatrices = projmatrices # (P, 3, 4)
        self.edgemaps = edgemap_info[0] # (P, max_Ni, 2)
        self.edgemaps_len = edgemap_info[1] # (P,)
        self.boundary_mask = boundary_mask    # keep the reference

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
        num_verts_per_mesh = self.src.num_verts_per_mesh()

        # 1) project every mesh into every view
        projected = []
        for b in range(B):
            Vb = num_verts_per_mesh[b]
            projected.append(self.project_vertices(vertices[b, :Vb]))  # (P, Vb, 2)

        # 2) reset the shared boundary mask
        with torch.no_grad():
            self.boundary_mask.zero_()

        boundaries_pad   = []                              # list of (P, B_max, 2)
        boundary_lengths = torch.zeros(B, P, device=y.device)

        # 3) per‑mesh → per‑view boundary extraction
        for b in range(B):
            Vb = num_verts_per_mesh[b]
            boundaries_b = []
            for p in range(P):
                boundary_p, mask_p = get_boundary(projected[b][p])      # (B_p,2)
                with torch.no_grad():
                    self.boundary_mask[:Vb].logical_or_(mask_p)      # OR merge
                boundaries_b.append(boundary_p)                         # collect
                boundary_lengths[b, p] = boundary_p.size(0)

            padded = torch.nn.utils.rnn.pad_sequence(boundaries_b,
                                                    batch_first=True,
                                                    padding_value=0.0)
            boundaries_pad.append(padded)                               # (P, B_max, 2)

        # 4) Chamfer distance per mesh
        chamfer_loss = torch.zeros(B, device=y.device)
        for b in range(B):
            res, _ = chamfer_distance(
                        x=boundaries_pad[b].float(),
                        y=self.edgemaps[b].float(),
                        x_lengths=boundary_lengths[b].long(),
                        y_lengths=self.edgemaps_len[b].long(),
                        batch_reduction="mean",
                        point_reduction="mean")
            chamfer_loss[b] = res.sum()

        return chamfer_loss.double()
