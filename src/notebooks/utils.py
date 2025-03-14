import torch
import numpy as np


import os
import re
from collections import defaultdict
import cv2

def read_matrices_from_file(filename):
    matrices = {}
    current_cam = None
    
    with open(filename, "r") as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("Camera:"):
            current_cam = line.split("Camera: ")[1]
            matrices[current_cam] = {}
        elif any(matrix_type in line for matrix_type in ["K (Intrinsic", "RT (Extrinsic", "P (Projection"]):
            matrix_type = line.split(" ")[0]  
            i += 1
            matrices[current_cam][matrix_type] = torch.tensor(np.loadtxt(lines[i:i+3]), dtype=torch.float32)
            i += 2  
        i += 1
        
    return matrices

def load_camera_matrices(path):
    cameras = defaultdict(dict)
    file_pattern = re.compile(r"^(.*)_(K|RT|P)\.npy$")
    for filename in os.listdir(path):
        match = file_pattern.match(filename)
        if match:
            print(filename)
            cam_name, matrix_type = match.groups()
            filepath = os.path.join(path, filename)
            cameras[cam_name][matrix_type] = np.load(filepath)
    return cameras
    

def load_renders(renders_path):
    renders = defaultdict(list)
    pattern = re.compile(r"([a-zA-Z]+)(\d+)\.png")
    for filename in sorted(os.listdir(renders_path), key=lambda x: (re.match(pattern, x).group(1), int(re.match(pattern, x).group(2))) if re.match(pattern, x) else (x, float('inf'))):
        match = pattern.match(filename)
        if match:
            word, number = match.groups()
            number = int(number)  # Convert number to integer for sorting

            # Load image (PIL) or store filename
            image_path = os.path.join(renders_path, filename)
            image = cv2.imread(image_path)  # Load image using OpenCV
            if image is not None:
                renders[word].append(image)
    return renders



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