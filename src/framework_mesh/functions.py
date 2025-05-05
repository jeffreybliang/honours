import torch
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
import trimesh
import numpy as np

torch.autograd.set_detect_anomaly(True)

def least_squares(u0, tgt_vtxs, device=torch.device("cpu")):
    """
    u0 are vertices
    """
    if not torch.is_tensor(u0):
        u0 = torch.tensor(u0, device=device)
    if not torch.is_tensor(tgt_vtxs):
        tgt_vtxs = torch.tensor(tgt_vtxs, device=device)
    res = torch.square(u0 - tgt_vtxs.flatten()).sum()
    return res.double()

def least_squares_grad(u0, tgt_vtxs, device=torch.device("cpu")):
    if torch.is_tensor(u0):
        u0 = u0.detach().clone()
    else:
        u0 = torch.tensor(u0, device=device)
    if torch.is_tensor(tgt_vtxs):
        tgt_vtxs = tgt_vtxs.detach().clone()
    else:
        tgt_vtxs = torch.tensor(tgt_vtxs, device=device)
        
    # Ensure that u0 requires gradients
    gradient = 2 * (u0 - tgt_vtxs.flatten())
    return gradient.double()


def calculate_volume(vertices, faces):
    face_vertices = vertices[faces]  # (F, 3, 3)
    v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
    
    # Compute determinant of the 3x3 matrix [v0, v1, v2]
    face_volumes = torch.det(torch.stack([v0, v1, v2], dim=-1)) / 6.0  # Shape: (F,)
    volume = face_volumes.sum()
    return volume.abs()


def volume_constraint(x, faces, tgt_vol, device=torch.device("cpu")):
    """
    Calculate the volume of a mesh using PyTorch tensors.
    Args:
        vertices_torch: Nx3 tensor of vertex coordinates
        faces: Mx3 array of face indices
    Returns:
        volume: Total volume of the mesh as a PyTorch scalar
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, device=device)
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces, device=device)
    if not torch.is_tensor(tgt_vol):
        tgt_vol = torch.tensor(tgt_vol, device=device)

    vertices = x.view(-1,3)
    faces = faces.view(-1,3).int()
    volume = calculate_volume(vertices, faces)
    res = volume.abs() - tgt_vol
    return res.double()

def volume_constraint_grad(x, faces, device=torch.device("cpu")):
    if torch.is_tensor(x):
        x = x.detach().clone()
    else:
        x = torch.tensor(x, device=device)
    if torch.is_tensor(faces):
        faces = faces.detach().clone()
    else:
        faces = torch.tensor(faces, device=device)
    faces = faces.to(dtype=torch.int64)

    vertices_torch = x.view(-1, 3)
    p0 = vertices_torch[faces[:, 0]]  # (F, 3)
    p1 = vertices_torch[faces[:, 1]]  # (F, 3)
    p2 = vertices_torch[faces[:, 2]]  # (F, 3)

    grad_p0 = torch.cross(p1, p2, dim=1) / 6.0
    grad_p1 = torch.cross(p2, p0, dim=1) / 6.0
    grad_p2 = torch.cross(p0, p1, dim=1) / 6.0

    grad_verts = torch.zeros_like(vertices_torch)
    grad_verts.scatter_add_(0, faces[:, 0].unsqueeze(1).expand(-1, 3), grad_p0)
    grad_verts.scatter_add_(0, faces[:, 1].unsqueeze(1).expand(-1, 3), grad_p1)
    grad_verts.scatter_add_(0, faces[:, 2].unsqueeze(1).expand(-1, 3), grad_p2)

    analytical_grad = grad_verts.flatten()
    return analytical_grad 

def chamfer_gt(mesh, src:Meshes, tgt:Meshes):
    res,_ = chamfer_distance(x=mesh.detach().float(), 
                             y=tgt.verts_padded().float().detach(),
                             x_lengths=src.num_verts_per_mesh(),
                             y_lengths=tgt.num_verts_per_mesh(),
                             batch_reduction=None,
                             point_reduction="mean")
    # print("Chamfer", res.size())
    return res.tolist() # (B,)


def sse_gt(mesh, src:Meshes, tgt:Meshes):
    sqr_diff = torch.square(mesh - tgt.verts_padded().detach())
    sse = sqr_diff.sum(dim=(1, 2))
    # print("sqrdiff", sqr_diff.size(),"sse", sse.size(), "tgt", tgt.verts_padded().size())
    return sse.tolist() # (B,)


def iou_gt(mesh, src: Meshes, tgt: Meshes, engine='manifold'):
    batch_size = len(mesh)
    ious = []
    gt = tgt.verts_padded()

    for b in range(batch_size):
        num_verts_src = src.num_verts_per_mesh()[b].item()
        num_verts_tgt = tgt.num_verts_per_mesh()[b].item()
        
        mesh_trimesh = trimesh.Trimesh(
            vertices=mesh[b][:num_verts_src].detach().cpu().numpy(), 
            faces=src[b].faces_packed().detach().cpu().numpy()
        )
        gt_trimesh = trimesh.Trimesh(
            vertices=gt[b][:num_verts_tgt].detach().cpu().numpy(), 
            faces=tgt[b].faces_packed().detach().cpu().numpy()
        )

        try:
            intersection = mesh_trimesh.intersection(gt_trimesh, engine=engine, check_volume=True)
        except ValueError as e:
            print(f"Intersection fallback triggered: {e}")
            intersection = mesh_trimesh.intersection(gt_trimesh, engine=engine, check_volume=False)

        try:
            union = mesh_trimesh.union(gt_trimesh, engine=engine, check_volume=True)
        except ValueError as e:
            print(f"Union fallback triggered: {e}")
            union = mesh_trimesh.union(gt_trimesh, engine=engine, check_volume=False)

        if union.volume == 0:
            iou = 0.0
        else:
            iou = intersection.volume / union.volume
            iou = float(np.clip(iou, 0.0, 1.0))  # Ensure the IoU is between 0 and 1

        ious.append(iou)

    return ious  # (B,)
