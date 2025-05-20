from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn
import numpy as np
from .chamfer import *
from .functions import calculate_volume
from .utils import *
import wandb

class PenaltyMethod(nn.Module):
    def __init__(self, src: Meshes, tgt: Meshes, projmatrices, edgemap_info, lambda_vol=1.0, boundary_mask=None, device=torch.device("cpu"), doublesided=False, target_volume=None, alpha=12.0):
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.device = device

        # Move everything to the specified device
        self.projmatrices = projmatrices.to(device)
        self.edgemaps = [e.to(device) for e in edgemap_info[0]]
        self.edgemaps_len = edgemap_info[1]
        self.boundary_mask = boundary_mask.to(device) if boundary_mask is not None else None
        self.lambda_vol = lambda_vol
        self.iter = 0
        self.alpha = alpha
        print("alpha is ", alpha)

        # Precompute and store target volumes on the correct device
        B = len(src)
        if target_volume:
            target_volumes = torch.tensor(
                [target_volume for b in range(B)],
                dtype=torch.double,
                device=device
            )
            self.register_buffer("target_volumes", target_volumes)
        self.doublesided = doublesided

    def project_vertices(self, vertices):
        V = vertices.shape[0]
        ones = torch.ones((V, 1), dtype=vertices.dtype, device=vertices.device)
        vertices_homogeneous = torch.cat([vertices, ones], dim=1).double()
        projected = torch.einsum("pij,vj->pvi", self.projmatrices, vertices_homogeneous)
        return projected[:, :, :2] / projected[:, :, 2:3]

    def volume_constraint(self, xs):
        B = len(self.src)
        y_packed = padded_to_packed(xs, self.src.num_verts_per_mesh()).view(-1, 3)
        faces_packed = self.src.faces_packed()
        face_vertices = y_packed[faces_packed]
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        cross_product = torch.cross(v0, v1, dim=-1)
        face_volumes = torch.sum(cross_product * v2, dim=-1) / 6.0
        volumes = torch.zeros(B, device=xs.device, dtype=face_volumes.dtype)
        volumes.scatter_add_(0, self.src.faces_packed_to_mesh_idx(), face_volumes)
        volumes = volumes.abs()
        return (volumes - self.target_volumes) ** 2

    def silhouette_chamfer(self, xs):
        B, P = len(self.src), self.projmatrices.size(0)
        vertices = xs
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
        boundary_lengths = torch.zeros(B, P, device=xs.device)

        # faces_padded = self.src.faces_padded() if self.mode == "mesh" else None
        # fnorms_padded = self.src.update_padded(xs).faces_normals_padded() if self.mode == "mesh" else None
        # num_faces = self.src.num_faces_per_mesh() if self.mode == "mesh" else None

        all_boundary_pts = [[] for _ in range(B)]
        all_hulls = [[] for _ in range(B)]
        all_loops = [[] for _ in range(B)]

        # 3) per‑mesh → per‑view boundary extraction
        for b in range(B):
            Vb = num_verts_per_mesh[b]
            boundaries_b = []
            for p in range(P):
                # if self.mode == "alpha":
                boundary_p, mask_p, hulls_p = get_boundary_alpha(projected[b][p], alpha=self.alpha)
                # else:
                #     faces_b = faces_padded[b][:num_faces[b]]
                #     fnorms_b = fnorms_padded[b][:num_faces[b]]
                #     boundary_p, mask_p, loops_p = get_boundary_mesh(projected[b][p], faces=faces_b, fnorms=fnorms_b, P=self.projmatrices[p])
                #     hulls_p = []  # for consistency
                #     all_loops[b].append(loops_p)

                with torch.no_grad():
                    self.boundary_mask[:Vb].logical_or_(mask_p)

                boundaries_b.append(boundary_p)
                boundary_lengths[b, p] = boundary_p.size(0)
                all_boundary_pts[b].append(boundary_p)
                all_hulls[b].append(hulls_p)

            padded = torch.nn.utils.rnn.pad_sequence(boundaries_b,
                                                    batch_first=True,
                                                    padding_value=0.0)
            boundaries_pad.append(padded)                               # (P, B_max, 2)

        # 4) Chamfer distance per mesh
        chamfer_loss = torch.zeros(B, device=xs.device)
        for b in range(B):
            res, _ = chamfer_distance(
                        x=boundaries_pad[b].float(),
                        y=self.edgemaps[b].float(),
                        x_lengths=boundary_lengths[b].long(),
                        y_lengths=self.edgemaps_len[b].long(),
                        batch_reduction="mean",
                        point_reduction="mean",
                        single_directional=not self.doublesided)
            chamfer_loss[b] = res.sum()

        return chamfer_loss.double(), all_boundary_pts, all_hulls, all_loops


    def forward(self, xs: torch.Tensor):
        silhouette_chamfer_loss, all_boundary_pts, all_hulls, all_loops = self.silhouette_chamfer(xs)

        if self.lambda_vol != 0:
            volume_loss = self.volume_constraint(xs)
            return {
                "chamfer": silhouette_chamfer_loss,
                "vol_error": volume_loss
            }, all_boundary_pts, all_hulls, all_loops

        return {
            "chamfer": silhouette_chamfer_loss,
            "vol_error": 0
        }, all_boundary_pts, all_hulls, all_loops
