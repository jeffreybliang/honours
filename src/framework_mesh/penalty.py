from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn
import numpy as np
from chamfer import get_boundary
from .functions import calculate_volume
from .utils import *
import wandb

class PenaltyMethod(nn.Module):
    def __init__(self, src: Meshes, tgt: Meshes, projmatrices, edgemap_info, lambda_proj=1.0, lambda_vol=1.0, boundary_mask=None):
        super().__init__()
        self.src = src  # (B meshes)
        self.tgt = tgt  # (B meshes)

        # calculate init volumes
        B = len(src)

        self.projmatrices = projmatrices # (P, 3, 4)
        self.edgemaps = edgemap_info[0] # (P, max_Ni, 2)
        self.edgemaps_len = edgemap_info[1] # (P,)
        self.boundary_mask = boundary_mask    # keep the reference
        
        self.lambda_proj = lambda_proj
        self.lambda_vol = lambda_vol
        self.register_buffer("target_volumes", torch.tensor([calculate_volume(src[b].verts_packed(), src[b].faces_packed()) for b in range(B)], dtype=torch.double))

        self.iter = 0

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


    def volume_constraint(self, y):
        """
        Calculates volume of projected points
        Assumes same number of vertices in each projected mesh currently
        """
        B = len(self.src)
        vertices = y
        y_packed = padded_to_packed(vertices, self.src.num_verts_per_mesh())
        verts_packed = y_packed.view(-1,3) # (sum(V_i), 3)

        faces_packed = self.src.faces_packed()  # (sum(F_i), 3)
        face_vertices = verts_packed[faces_packed]  # (sum(F_i), 3, 3)
        
        # Calculate tetrahedron volumes for each face
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        cross_product = torch.cross(v0, v1, dim=-1)  # (F, 3)
        face_volumes = torch.sum(cross_product * v2, dim=-1) / 6.0  # (F,)
        volumes = torch.zeros(B, device=verts_packed.device, dtype=face_volumes.dtype)
        volumes.scatter_add_(0, self.src.faces_packed_to_mesh_idx(), face_volumes)

        volumes = volumes.abs()
        norm_sq = (volumes - self.target_volumes) ** 2
        return norm_sq  # Shape: (B,)    
        
    def lsq_proj(self, xs: torch.Tensor, y: torch.Tensor):
        """
        Calculates sum of squared differences between source and target meshes.

        Args:
            xs (torch.Tensor): a padded (B, max Vi, 3) tensor of the original vertices
            y (torch.Tensor): a padded (B, max Vi, 3) tensor of the projected vertices        
        """
        src_verts = padded_to_packed(xs, self.src.num_verts_per_mesh()).view(-1,3)
        tgt_verts = padded_to_packed(y, self.src.num_verts_per_mesh()).view(-1,3)

        sqr_diffs = torch.square(src_verts - tgt_verts).sum(dim=-1) # (sum(V_i))
        B = len(self.src)
        sse = torch.zeros(B, dtype=sqr_diffs.dtype)
        sse.scatter_add_(0, self.src.verts_packed_to_mesh_idx(), sqr_diffs)
        return sse
        

    def silhouette_chamfer(self, y):
        B, P = len(self.src), self.projmatrices.size(0)
        num_verts_per_mesh = self.src.num_verts_per_mesh()
        vertices = y
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

    def forward(self, xs:torch.Tensor, y:torch.Tensor):
        """
        xs: original vertices
        y: projected vertices
        """
        if self.lambda_proj != 0 and self.lambda_vol != 0:
            lsq_proj_loss = self.lsq_proj(xs, y)
            volume_loss = self.volume_constraint(y)
            silhouette_chamfer_loss = self.silhouette_chamfer(y)
            wandb.log({
                "chamfer": silhouette_chamfer_loss,
                "proj_error": lsq_proj_loss,
                "vol_error": volume_loss
            },
            commit=False)
            return silhouette_chamfer_loss + self.lambda_proj * lsq_proj_loss + self.lambda_vol * volume_loss

        elif self.lambda_vol == 0:
            # the only decision variables are the xs
            silhouette_chamfer_loss = self.silhouette_chamfer(y)
            wandb.log({
                "chamfer": silhouette_chamfer_loss,
            },
            commit=False)
            return silhouette_chamfer_loss