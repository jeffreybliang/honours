from alpha_shapes import *
import shapely
import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from torch import nn
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from alpha_shapes.boundary import get_boundaries

def get_boundary(mode, projected_pts: torch.Tensor, alpha: float = 10.0,
                 faces=None, fnorms=None, P=None):
    if mode == "alpha":
        return get_boundary_alpha(projected_pts, alpha)
    if mode == "mesh":
        if faces is None or fnorms is None or P is None:
            raise ValueError("faces, fnorms, and P must be provided in mesh mode")
        return get_boundary_mesh(projected_pts, faces, fnorms, P)
    raise ValueError(f"Unknown boundary mode: {mode}")

def get_boundary_mesh(projverts: torch.Tensor, faces: torch.Tensor, fnorms: torch.Tensor, P: torch.Tensor, eps=1e-2):
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
        # for interior in unioned.interiors:
        #     rings.append(np.array(interior.coords))
    elif unioned.geom_type == 'MultiPolygon':
        # unioned = max(unioned.geoms, key=lambda p: p.area)
        # rings.append(np.array(unioned.exterior.coords))
        for geom in unioned.geoms:
            rings.append(np.array(geom.exterior.coords))
            # for interior in geom.interiors:
            #     rings.append(np.array(interior.coords))
    else:
        raise ValueError(f"Unexpected geometry type: {unioned.geom_type}")

    # 4) Snap rings to closest projected vertices
    loops = []
    boundary_indices = []

    for ring in rings:
        coords_t = torch.from_numpy(ring).to(projverts)           # (L, 2)
        diffs = coords_t[:, None, :] - projverts[None, :, :]      # (L, V, 2)
        d2 = (diffs ** 2).sum(dim=2)                              # (L, V)
        idx = torch.argmin(d2, dim=1)
        idx_u = torch.unique_consecutive(idx)
        if idx_u.numel() > 1 and idx_u[0] == idx_u[-1]:
            idx_u = idx_u[:-1]
        loops.append(projverts[idx_u])
        boundary_indices.append(idx_u)

    boundary_indices = torch.cat(boundary_indices)
    boundary_mask = torch.zeros(projverts.shape[0], dtype=torch.bool, device=projverts.device)
    boundary_mask[boundary_indices] = True

    boundary_pts = torch.cat(loops, dim=0)
    return boundary_pts, boundary_mask, loops


def get_boundary_alpha(projected_pts: torch.Tensor, alpha: float = 15.0):
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

    shaper = Alpha_Shaper(proj_cpu)
    shape = shaper.get_shape(alpha)
    boundaries = get_boundaries(shape)

    # while alpha >= 0:
    #     try:
    #         shaper = Alpha_Shaper(proj_cpu)
    #         shape = shaper.get_shape(alpha)
    #         boundaries = get_boundaries(shape)

    #         if len(boundaries) == 1:
    #             break  # ✅ Got a single polygon
    #     except Exception as e:
    #         print(f"[alpha={alpha:.2f}] Failed to extract shape: {e}")
    #         boundaries = []

    #     alpha -= 1  # Try a smaller α
    # alpha = max(alpha, 0)

    if not boundaries:
        return torch.empty((0, 2), device=projected_pts.device), torch.zeros(projected_pts.shape[0], dtype=torch.bool, device=projected_pts.device)

    coords = np.concatenate(
        [b.exterior[:-1] for b in boundaries] 
        # + [hole for b in boundaries for hole in b.holes[:-1]]
        ,axis=0
    )

    boundary_cpu = torch.as_tensor(coords, dtype=torch.double)  # <- fixed here

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
    return boundary_pts, b_mask, boundaries

class PyTorchChamferLoss(nn.Module):
    def __init__(self, src: Meshes, tgt: Meshes, projmatrices, edgemap_info, boundary_mask=None, doublesided=False, mode="alpha", alpha=12.0):
        super().__init__()
        self.src = src  # (B meshes)
        self.tgt = tgt  # (B meshes)
        self.projmatrices = projmatrices # (P, 3, 4)
        self.edgemaps = edgemap_info[0] # (P, max_Ni, 2)
        self.edgemaps_len = edgemap_info[1] # (P,)
        self.boundary_mask = boundary_mask    # keep the reference
        self.doublesided = doublesided
        self.mode = mode
        self.alpha = alpha

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

        faces_padded = self.src.faces_padded() if self.mode == "mesh" else None
        fnorms_padded = self.src.update_padded(y).faces_normals_padded() if self.mode == "mesh" else None
        num_faces = self.src.num_faces_per_mesh() if self.mode == "mesh" else None

        all_boundary_pts = [[] for _ in range(B)]
        all_hulls = [[] for _ in range(B)]
        all_loops = [[] for _ in range(B)]

        # 3) per‑mesh → per‑view boundary extraction
        for b in range(B):
            Vb = num_verts_per_mesh[b]
            boundaries_b = []
            for p in range(P):
                if self.mode == "alpha":
                    boundary_p, mask_p, hulls_p = get_boundary_alpha(projected[b][p], alpha=self.alpha)
                else:
                    faces_b = faces_padded[b][:num_faces[b]]
                    fnorms_b = fnorms_padded[b][:num_faces[b]]
                    boundary_p, mask_p, loops_p = get_boundary_mesh(projected[b][p], faces=faces_b, fnorms=fnorms_b, P=self.projmatrices[p])
                    hulls_p = []  # for consistency
                    all_loops[b].append(loops_p)

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
        chamfer_loss = torch.zeros(B, device=y.device)
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
