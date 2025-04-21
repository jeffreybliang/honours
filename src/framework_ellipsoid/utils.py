import torch
import math
import numpy as np
from torch import cos, sin
import torch.nn.functional as F

def sample_ellipsoid_surface(sqrt_m, a, b, c, yaw, pitch, roll, noise_std=1e-4):
    phi = 2.0 * math.pi * torch.linspace(0.0, 1.0, sqrt_m).double()
    theta = math.pi * torch.linspace(0.0, 1.0, sqrt_m).double()
    phi, theta = torch.meshgrid(phi, theta, indexing='ij')
    x = a * torch.sin(theta) * torch.cos(phi)
    y = b * torch.sin(theta) * torch.sin(phi)
    z = c * torch.cos(theta)
    coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=0)
    angles = torch.tensor([yaw, pitch, roll])
    if torch.any(angles != 0):
        R = rotation_matrix_3d(torch.tensor([yaw, pitch, roll]))
        coords = R @ coords
    noisy = coords + noise_std * torch.randn_like(coords)
    return noisy.unsqueeze(0)  # shape: (1, 3, N)


def ellipsoid_volume(a, b, c):
    return 4/3 * torch.pi * a * b * c

def ellipsoid_surface_area(a, b, c, p=1.6075):
    a_p = a ** p
    b_p = b ** p
    c_p = c ** p
    return 4 * torch.pi * (1/3 * (a_p * b_p + a_p * c_p + b_p * c_p)) ** (1/p)

def rotation_matrix_3d(angles):
    alpha, beta, gamma = angles[0], angles[1], angles[2]
    R = torch.stack([
        torch.stack([cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)]),
        torch.stack([sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)]),
        torch.stack([-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma)])
    ])
    return R

def rotation_matrix_3d_batch(angles):
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
    R = torch.stack([
        torch.stack([
            torch.cos(alpha) * torch.cos(beta),
            torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma) - torch.sin(alpha) * torch.cos(gamma),
            torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(gamma)
        ], dim=1),
        torch.stack([
            torch.sin(alpha) * torch.cos(beta),
            torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma) + torch.cos(alpha) * torch.cos(gamma),
            torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma) - torch.cos(alpha) * torch.sin(gamma)
        ], dim=1),
        torch.stack([
            -torch.sin(beta),
            torch.cos(beta) * torch.sin(gamma),
            torch.cos(beta) * torch.cos(gamma)
        ], dim=1)
    ], dim=1)
    return R

def get_angles(rotation):
    pitch = - np.arcsin(rotation[2,0])
    denom = 1 / np.sqrt(1 - (rotation[2,0] ** 2))
    roll = np.arctan2(rotation[2,1]/denom, rotation[2,2]/denom)
    yaw = np.arctan2(rotation[1,0]/denom, rotation[0,0]/denom)
    return np.rad2deg([yaw, pitch, roll])

def extract_params(u):
    if torch.any(u) < 0:
        print("WARNING: Negative axes lengths.")
    a, b, c = (torch.abs(u[:3])).tolist()
    yaw, pitch, roll = np.rad2deg(u[3:].tolist()) % 360
    return a, b, c, yaw, pitch, roll


def find_orthonormal_basis(v, use_zaxis_projection=False):
    """
    Given a 3D vector v, returns two unit vectors orthonormal to it.
    
    Args:
        v: Tensor of shape (3,), the input vector.
        use_zaxis_projection: If True, u will be the projection of z-axis onto the plane orthogonal to v.
    
    Returns:
        Tuple (u, w): both unit vectors such that
            - u ⊥ v
            - w = v × u (also orthonormal to both)
            - u and w form an orthonormal basis of the plane orthogonal to v
    """
    v = F.normalize(v, dim=0)

    if use_zaxis_projection:
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=v.device, dtype=v.dtype)

        # If v is too close to z-axis, fallback to x-axis to avoid numerical issues
        if torch.allclose(v, z_axis, atol=1e-3) or torch.allclose(v, -z_axis, atol=1e-3):
            z_axis = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype)

        # Project z_axis onto plane orthogonal to v
        proj = z_axis - torch.dot(z_axis, v) * v
        u = F.normalize(proj, dim=0)
    else:
        # Use a standard method to pick an arbitrary non-parallel vector
        if torch.abs(v[0]) < 0.9:
            other = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype)
        else:
            other = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype)
        u = torch.cross(v, other)
        u = F.normalize(u, dim=0)

    # Compute second orthonormal vector
    w = torch.cross(v, u)
    w = F.normalize(w, dim=0)

    return u, w


def build_view_matrices(cfg):
    mode = cfg.get("view_mode", "angles")

    if mode == "angles":
        angles_deg = torch.tensor(cfg["view_angles"], dtype=torch.double)
        angles_rad = torch.deg2rad(angles_deg)
        R = rotation_matrix_3d_batch(angles_rad)  # (N, 3, 3)

    elif mode == "planes":
        normals = torch.tensor(cfg["view_normals"], dtype=torch.double)  # (N, 3)
        use_zaxis = cfg.get("use_zaxis_projection", False)
        R = torch.empty((len(normals), 3, 3), dtype=torch.double)
        for i, n in enumerate(normals):
            z = F.normalize(n, dim=0)
            u, w = find_orthonormal_basis(z, use_zaxis_projection=use_zaxis)
            R[i] = torch.stack([u, w, z], dim=0)
    else:
        raise ValueError(f"Unsupported view_mode: {mode}")

    return R
