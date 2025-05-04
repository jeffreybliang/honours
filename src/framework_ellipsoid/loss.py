import math
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
import alpha_shapes
import numpy as np
from .plotting import *
from .utils import rotation_matrix_3d_batch

def A_from_u_batch(u):
    Lambda = torch.diag_embed(1 / u[:, :3] ** 2)
    Q = rotation_matrix_3d_batch(u[:, 3:])
    return Q @ Lambda @ Q.transpose(-1, -2)

def schur_complement_batch(M):
    A, B, C, D, E, F = M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]
    return torch.stack([
        torch.stack([A - E ** 2 / C, D - E * F / C], dim=1),
        torch.stack([D - E * F / C, B - F ** 2 / C], dim=1)
    ], dim=1)

def pos_sqrt(A):
    return torch.linalg.cholesky(A, upper=True)

def sample_pts(sqrtA, m=50):
    t = 2.0 * math.pi * torch.linspace(0.0, 1.0, m, dtype=torch.double)
    points = torch.stack([torch.cos(t), torch.sin(t)])
    inverse_sqrtA = torch.linalg.inv(sqrtA)
    sampled_pts = inverse_sqrtA @ points
    return sampled_pts


def make_homogeneous_torch(M):
    """Convert (B, 3, 3) matrix into (B, 4, 4) homogeneous form."""
    B = M.shape[0]
    Q = torch.eye(4, dtype=M.dtype, device=M.device).expand(B, 4, 4).clone()
    Q[:, :3, :3] = M
    return Q

def homogeneous_projection_batch_torch(A, R):
    """
    Args:
        A: (B, 3, 3) ellipsoid matrix (e.g., A_from_u_batch)
        R: (B, 3, 3) rotation matrices

    Returns:
        C: (B, 2, 2) projected 2D ellipse matrices
    """
    B = A.shape[0]
    P = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=A.dtype, device=A.device).expand(B, -1, -1)

    M = R @ A @ R.transpose(-1, -2)
    Q = make_homogeneous_torch(M)
    Qinv = torch.linalg.inv(Q)
    C_star = P @ Qinv @ P.transpose(-1, -2)

    # Dehomogenize
    C = torch.linalg.inv(C_star[:, :2, :2]) / C_star[:, 2, 2].view(-1, 1, 1)
    return C

def build_target_circle(r,m):
    """
    Returns: (1, M, 2) tensor of 2D points
    """
    t = 2 * torch.pi * torch.linspace(0.0, 1.0, m, dtype=torch.double)
    circle = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)  # (M, 2)
    return circle.unsqueeze(0)  # (1, M, 2)


class SampledProjectionChamferLoss(nn.Module):
    def __init__(self, views, m=50):
        super().__init__()
        self.rot_mats, self.target_pts = views
        self.m = m
        self.last_projected_pts = None  # store for silhouette plots

    def forward(self, input, p=1.6075):
        A = A_from_u_batch(input).double()  # (1, 3, 3)
        R = self.rot_mats  # (N, 3, 3)
        A = A.expand(R.shape[0], -1, -1)     # make A match R

        ellipses = homogeneous_projection_batch_torch(A, R)  # (N, 2, 2)
        matrix_sqrts = pos_sqrt(ellipses)
        sampled_pts = sample_pts(matrix_sqrts, m=self.m)  # (N, 2, m)
        self.last_projected_pts = sampled_pts.unsqueeze(0)  # → (1, N, 2, m)
        sampled_pts = sampled_pts.transpose(1, 2)         # → (N, m, 2)
        
        res, _ = chamfer_distance(
            sampled_pts.float(), self.target_pts.float(),
            batch_reduction=None,
            point_reduction="mean"
        )
        return res


    def plot_silhouettes(self, step=None):
        fig = plot_silhouettes(
            projected_pts=self.last_projected_pts,
            target_pts=self.target_pts,
            show_alpha=False,
            step=step
        )
        return fig





def nearest_boundary_points(original_pts, boundary_np, atol=1e-6):
    """
    original_pts: Tensor (2, N)  -- from projected_pts
    boundary_np: np.ndarray (P, 2) -- from alpha shape output
    Returns: Tensor (2, P) of closest matching original points
    """
    boundary = torch.tensor(boundary_np.T, dtype=original_pts.dtype, device=original_pts.device)  # (2, P)
    matched = []

    for i in range(boundary.shape[1]):
        diff = original_pts - boundary[:, i].unsqueeze(1)
        dists = torch.norm(diff, dim=0)
        min_idx = torch.argmin(dists)
        if dists[min_idx] < atol:
            matched.append(original_pts[:, min_idx])
        else:
            # fallback if no close match: just use the closest anyway
            matched.append(original_pts[:, min_idx])

    return torch.stack(matched, dim=1)  # (2, P)



class BoundaryProjectionChamferLoss(nn.Module):
    def __init__(self, views, m=50, sqrt_m=25, alpha=0.0):
        super().__init__()
        self.rot_mats, self.target_pts = views  # target_pts: (N, M, 2)
        self.m = m
        self.sqrt_m = sqrt_m
        self.alpha = alpha
        self.last_projected_pts = None         # (B, V, 2, N)
        self.last_boundary_points = None       # list[list[Tensor]] of shape B × V
        self.last_hulls = None                 # list[list[Polygon]] of shape B × V

    def sample_unit_sphere(self, sqrt_m, device):
        """Sample m points on the unit sphere S²"""
        phi = 2 * math.pi * torch.linspace(0.0, 1.0, sqrt_m, dtype=torch.double, device=device)
        theta = torch.acos(1 - 2 * torch.linspace(0.0, 1.0, sqrt_m, dtype=torch.double, device=device))
        phi, theta = torch.meshgrid(phi, theta, indexing="ij")
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=0)

    def sample_ellipsoid_surface_uniform(self, A: torch.Tensor, n: int, oversample: float = 2.0):
        # eigvals, eigvecs = torch.linalg.eigh(A)
        # A_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
        L = torch.linalg.cholesky(A)  # B in pseudocode: upper-triangular
        A_inv_sqrt = torch.linalg.inv(L)

        accepted = []
        batch = int(n * oversample)

        while sum(p.shape[1] for p in accepted) < n:
            u = torch.randn(3, batch, dtype=torch.float64)
            u /= torch.norm(u, dim=0, keepdim=True)

            x = A_inv_sqrt @ u
            accepted.append(x)
            norm_A = torch.sqrt(torch.sum(x * (A @ x), dim=0))  # shape: (N,)
            x = x / norm_A.unsqueeze(0)

            weights = torch.norm(A_inv_sqrt @ u, dim=0)
            probs = 1.0 / weights**3
            probs /= probs.max()

            accept = torch.rand(batch) < probs
            accepted.append(x[:, accept])

        result = torch.cat(accepted, dim=1)[:, :n]
        return result
        

    def forward(self, input, p=1.6075):
        B = len(self.rot_mats)
        A = A_from_u_batch(input).double()  # (1, 3, 3)
        Au = A[0]
        L = torch.linalg.cholesky(Au)  # B in pseudocode: upper-triangular
        Binv = torch.linalg.inv(L)        # B⁻¹ (3×3)

        # z = self.sample_unit_sphere(self.sqrt_m, device=input.device)  # (3, M)
        # E = Binv @ z  # (3, M), ellipsoid surface points
        n = self.sqrt_m * self.sqrt_m
        E = self.sample_ellipsoid_surface_uniform(Au, n)
        # Step 5: Loop over each view
        boundary_pts = []
        boundary_lengths = []
        hulls = []
        projections = []

        max_len = 0
        for k in range(B):
            Rk = self.rot_mats[k]  # (3, 3)
            y = Rk @ E             # rotate: (3, M)
            proj = y[:2, :]        # (2, M)
            projections.append(proj)  # store for (B, 2, M) → stack later

            proj_np = proj.T.detach().cpu().numpy()  # (M, 2)
            shaper = alpha_shapes.Alpha_Shaper(proj_np)
            shape = shaper.get_shape(alpha=0.0)
            coords = np.stack(shape.exterior.coords.xy, axis=-1)  # (P, 2)

            boundary = nearest_boundary_points(proj, coords)  # (2, P)
            boundary_pts.append(boundary)
            hulls.append(shape)
            boundary_lengths.append(boundary.size(1))
            max_len = max(max_len, boundary.size(1))

        # Store projected_pts (B, 2, M) → (B, 2, M) → (B, 2, M)
        self.last_projected_pts = torch.stack(projections, dim=0).unsqueeze(0) 
        # For consistent interface with silhouette plotter: (B, V=1)
        self.last_boundary_points = [boundary_pts]
        self.last_hulls = [hulls]                 

        # Step 6: Pad and batch
        padded = torch.full((B, max_len, 2), float('nan'), dtype=torch.double, device=input.device)
        for i, b in enumerate(boundary_pts):
            padded[i, :b.shape[1]] = b.T  # (2, N) → (N, 2)

        x_lengths = torch.tensor(boundary_lengths, dtype=torch.int64, device=input.device)
        y_lengths = torch.full((B,), self.target_pts.shape[1], dtype=torch.int64, device=input.device)

        # Step 7: Chamfer distance
        res, _ = chamfer_distance(
            padded.float(), self.target_pts.float(),
            x_lengths=x_lengths,
            y_lengths=y_lengths,
            batch_reduction=None,
            point_reduction="mean"
        )

        return res


    def plot_silhouettes(self, step=None):
        fig = plot_silhouettes(
            projected_pts=self.last_projected_pts,
            target_pts=self.target_pts,
            boundary_points=self.last_boundary_points,
            hulls=self.last_hulls,
            show_alpha=True,
            step=step
        )
        return fig
