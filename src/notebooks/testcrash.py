import math
import numpy as np
from torch import cos, sin
import scipy.optimize as opt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import psutil
import os

process = psutil.Process(os.getpid())

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../../../ddn/")
from ddn.pytorch.node import *

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import alphashape
from descartes import PolygonPatch
import pytorch3d

from pytorch3d.io import IO
import plotly.graph_objects as go
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from plotly.subplots import make_subplots
from pytorch3d.renderer import (
    HeterogeneousRayBundle,
    ray_bundle_to_ray_points,
    RayBundle,
    TexturesAtlas,
    TexturesVertex,
)
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import join_meshes_as_scene, Meshes, Pointclouds

import os
import torch
from pytorch3d.io import load_obj, save_obj,load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import faulthandler
faulthandler.enable()

def get_signed_tet_volume(face_vertices: torch.Tensor) -> torch.Tensor:
    """
    Compute signed tetrahedron volumes for a batch of faces.

    Args:
        face_vertices (torch.Tensor): Tensor of shape (F, 3, 3), where
                                      F is the number of faces, and each face
                                      consists of 3 vertices in 3D.

    Returns:
        torch.Tensor: A tensor of shape (F,) containing signed volumes.
    """
    v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
    
    # Compute determinant of the 3x3 matrix [v0, v1, v2]
    volumes = torch.det(torch.stack([v0, v1, v2], dim=-1)) / 6.0  # Shape: (F,)

    return volumes

def get_volume_batch(meshes: Meshes):
    verts_packed = meshes.verts_packed()  # (sum(V_i), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_i), 3)
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()  # Index of first face per mesh
    n_meshes = len(meshes)
    volumes = torch.zeros(n_meshes, device=verts_packed.device)

    for i in range(n_meshes):
        start = mesh_to_face[i]
        end = start + meshes.num_faces_per_mesh()[i]
        face_vertices = verts_packed[faces_packed[start:end]]  # (F, 3, 3)
        volumes[i] = get_signed_tet_volume(face_vertices).sum()  # Sum over all faces

    return volumes.abs()  # Returns a tensor of shape (num_meshes,)


import sys
print(sys.executable)

init_mesh = load_objs_as_meshes(["./Blender/sphere_4.obj"])
init_vol = get_volume_batch(init_mesh)[0].item()
balloon = load_objs_as_meshes(["./Blender/balloon_4.obj"])

base = "./Blender/"
import os 
# shapes = ["balloon", "rstrawberry", "sphere", "parabola"]
# obj_paths = []
# for s in shapes:
#     obj_paths.append(os.path.join(base, s+"_2.obj"))

# balloon, strawberry, sphere, parabola = load_objs_as_meshes(obj_paths)    

def least_squares(u0, target):
    """
    u0 are vertices
    """
    if not torch.is_tensor(u0):
        u0 = torch.tensor(u0)
    if not torch.is_tensor(target):
        target = torch.tensor(target)

    res = torch.square(u0 - target).sum()
    print("lstsq", res)
    return res.double() * 1e3

def least_squares_grad(u0, target):
    if torch.is_tensor(u0):
        u0 = u0.detach().clone()
    else:
        u0 = torch.tensor(u0)
        
    if torch.is_tensor(target):
        target = target.detach().clone()
    else:
        target = torch.tensor(target)
        
    # Ensure that u0 requires gradients
    u0.requires_grad = True
    
    with torch.enable_grad():
        res = torch.square(u0 - target).sum()

    # Compute the gradient
    obj_grad = torch.autograd.grad(res, u0)[0]
    if torch.isnan(obj_grad).any() or torch.isinf(obj_grad).any():
        print("Warning: NaN or Inf detected in gradient! Replacing with zeros.")
        obj_grad = torch.zeros_like(obj_grad)

    print("obj_grad", obj_grad, obj_grad.size(), obj_grad.norm())
    return obj_grad.double() * 1e3

def get_volume(u, faces, init_vol):
    if not torch.is_tensor(u):
        u = torch.tensor(u)
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces)
    if not torch.is_tensor(init_vol):
        init_vol = torch.tensor(init_vol)
    
    vertices = u.view(-1,3)
    face_vertices = vertices[faces]  # (F, 3, 3)
    volume = get_signed_tet_volume(face_vertices).sum()
    res = volume.abs() - init_vol
    print("vol", res)
    return res.double() * 1e3

def get_volume_grad(u, faces, init_vol):
    if not torch.is_tensor(u):
        u = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    else:
        u = u.clone().detach().requires_grad_(True)
    
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces, dtype=torch.long)
    
    if not torch.is_tensor(init_vol):
        init_vol = torch.tensor(init_vol, dtype=torch.float64)
    
    with torch.enable_grad():
        vertices = u.view(-1, 3)
        face_vertices = vertices[faces]  # (F, 3, 3)
        volume = get_signed_tet_volume(face_vertices).sum()
        res = volume.abs() - init_vol
    
    volume_grad = torch.autograd.grad(res, u, retain_graph=True)[0]
    print("volume grad", volume_grad, volume_grad.size(), volume_grad.norm())
    if torch.isnan(volume_grad).any() or torch.isinf(volume_grad).any():
        print("Warning: NaN or Inf detected in gradient! Replacing with zeros.")
        volume_grad = torch.zeros_like(volume_grad)

    return volume_grad.double() * 1e3


def project(meshes: Meshes, targets: Meshes, with_jac=False):
    n_batches = len(meshes)
    n_vtxs = len(meshes[0].verts_packed().flatten())
    print("n_vtxs", n_vtxs)
    results = torch.zeros(n_batches, n_vtxs, dtype=torch.double)
    losses = torch.zeros(n_batches, 1, dtype=torch.double)
    for batch_number, mesh in enumerate(meshes):
        init_vol = get_volume_batch(mesh).double().detach().cpu().numpy()
        print("batch number", batch_number)
        vertices = mesh.verts_packed().double().flatten().detach().numpy()
        faces = mesh.faces_packed().detach().numpy().astype(np.int64)
        target_vtx = targets[batch_number].verts_packed().flatten().detach().numpy()
        eq_const = {
            'type': 'eq',
            'fun' : lambda u: get_volume(u, faces, init_vol).detach().cpu().numpy(),
            # 'jac' : lambda u: get_volume_grad(u, faces, init_vol).detach().cpu().numpy().astype(np.float64),
            'tol' : 1e-2
        }
        print("starting optimisation")
        res = opt.minimize(
            lambda u: least_squares(u, target_vtx).detach().cpu().numpy().astype(np.float64),
            vertices, 
            # jac=lambda u: least_squares_grad(u, target_vtx).cpu().numpy().astype(np.float64),
            method='SLSQP', 
            constraints=[eq_const],
            options={'ftol': 1e-4, 'disp': True, 'maxiter': 100}
        )   
        print("finished")
        if not res.success:
            print("FIT failed:", res.message)
        results[batch_number] = torch.tensor(res.x.flatten(), dtype=torch.double, requires_grad=True)
        losses[batch_number] = torch.tensor(res.fun, dtype=torch.double, requires_grad=False)
    return results, losses

project(init_mesh, balloon)