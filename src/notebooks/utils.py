import torch
import numpy as np
import os
import re
from collections import defaultdict
import cv2

def load_camera_matrices(path, matrix_types=None):
    """
    Loads camera matrices from .npy files in the specified directory.

    Parameters:
        path (str): Path to the directory containing camera matrix files.
        matrix_types (set or list, optional): Specifies which matrix types to load (e.g., {'K', 'RT'}).
        If None, all available matrices ('K', 'RT', 'P') will be loaded.

    Returns:
        dict: A dictionary mapping camera numbers to their respective matrices.
    """
    cameras = defaultdict(dict)
    file_pattern = re.compile(r"^Camera_(\d+)_(K|RT|P)\.npy$")
    
    if matrix_types is not None:
        matrix_types = set(matrix_types)  # Ensure it's a set for quick lookup
    
    for filename in sorted(os.listdir(path)):  # Sort filenames alphabetically
        match = file_pattern.match(filename)
        if match:
            cam_number, matrix_type = match.groups()
            cam_number = int(cam_number)  # Convert camera number to integer
            if matrix_types is None or matrix_type in matrix_types:
                filepath = os.path.join(path, filename)
                cameras[cam_number][matrix_type] = torch.tensor(np.load(filepath))
    
    return cameras

def load_renders(renders_path):
    renders = defaultdict(dict)
    pattern = re.compile(r"([a-zA-Z]+)(\d+)\.png")
    for filename in sorted(os.listdir(renders_path), key=lambda x: (re.match(pattern, x).group(1), int(re.match(pattern, x).group(2))) if re.match(pattern, x) else (x, float('inf'))):
        match = pattern.match(filename)
        if match:
            word, number = match.groups()
            number = int(number)  # Convert number to integer for sorting

            image_path = os.path.join(renders_path, filename)
            image = cv2.imread(image_path)  # Load image using OpenCV
            if image is not None:
                renders[word][number] = image  # Store image in the nested dictionary

    return renders

def get_projmats_and_edgemap_info(view_idx, target_mesh: str, matrices, edgemaps, edgemaps_len):
    """
    Retrieves the projection matrices and target edgemap information for the specified view indices and target mesh.

    Parameters:
        view_idx (list): List of indices for which to retrieve projection matrices and edgemaps.
        target_mesh (str): The target mesh name (e.g., 'balloon') to extract edgemaps and lengths for.
        matrices (dict): Dictionary containing camera matrices.
        edgemaps (dict): Dictionary containing edgemaps for various meshes.
        edgemaps_len (dict): Dictionary containing the lengths of the edgemaps for each mesh.

    Returns:
        tuple: A tuple containing the projection matrices (torch.Tensor) and the target edgemap information (tuple of torch.Tensors).
    """
    # Get the projection matrices for the specified view indices
    projmats = torch.stack([matrices[view_idx[i]]["P"] for i in range(len(view_idx))])

    # Get the target edgemaps for the specified target mesh
    tgt_edgemaps = torch.nn.utils.rnn.pad_sequence([edgemaps[target_mesh][i] for i in view_idx], batch_first=True, padding_value=0.0)
    tgt_edgemaps_len = torch.tensor([edgemaps_len[target_mesh][i] for i in view_idx])

    # Pack the target edgemaps and their lengths
    tgt_edgemap_info = (tgt_edgemaps, tgt_edgemaps_len)

    return projmats, tgt_edgemap_info


def project_batched_vertices(self, vertices):
    """
    Projects batched sets of vertices into multiple views using a shared set of projection matrices.

    Args:
        vertices: Tensor of shape (B, N, 3), representing 3D vertex positions in B batches.

    Returns:
        Tensor of shape (B, P, N, 2), containing projected 2D points for each batch and each view.
    """
    B, N, _ = vertices.shape  # B: batch size, N: number of vertices
    projection_matrices = self.projmatrices  # Shape: (P, 3, 4), P projection matrices

    ones = torch.ones((B, N, 1), dtype=vertices.dtype, device=vertices.device)
    vertices_homogeneous = torch.cat([vertices, ones], dim=2)  # Shape: (B, N, 4)

    # Expand projection matrices to apply them to all batches -> (B, P, 3, 4)
    projection_matrices = projection_matrices.unsqueeze(0).expand(B, -1, -1, -1)

    # Perform batched matrix multiplication using einsum -> (B, P, N, 3)
    projected = torch.einsum("bpij,bnj->bpni", projection_matrices, vertices_homogeneous)

    # Convert to Cartesian coordinates
    projected_cartesian = projected[..., :2] / projected[..., 2:3]  # Shape: (B, P, N, 2)

    return projected_cartesian



def manual_gradient(u0, tgt_vtxs, src_edges, eta=1):
    """
    Compute the gradient manually without using autograd.
    """
    u0 = torch.tensor(u0, dtype=torch.float32, requires_grad=False)
    tgt_vtxs = torch.tensor(tgt_vtxs, dtype=torch.float32)

    # Convert src_edges to tensor
    src_edges = torch.tensor(src_edges, dtype=torch.long)  
    u, v = src_edges[:, 0], src_edges[:, 1]

    # Compute edge length differences
    src_dist = torch.square(u0[u] - u0[v]).sum(dim=1)  # d(X_i, X_j)^2
    tgt_dist = torch.square(tgt_vtxs[u] - tgt_vtxs[v]).sum(dim=1)  # d(Y_i, Y_j)^2
    length_diff = tgt_dist - src_dist   # ell_{ij}

    # Compute per-edge gradient contribution
    edge_grad = -4 * eta * length_diff[:, None] * (u0[u] - u0[v])  # Shape: (num_edges, d)

    # Initialize gradient accumulation tensor
    grad = torch.zeros_like(u0)

    # Scatter gradients to vertices using index_add_
    grad.index_add_(0, u, edge_grad)  # Accumulate gradients for node u
    grad.index_add_(0, v, -edge_grad)  # Accumulate gradients for node v (opposite sign)
    
    # Add gradient from L_sse term
    grad += 2 * (u0 - tgt_vtxs)

    return grad
