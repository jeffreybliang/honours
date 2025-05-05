import torch
import numpy as np
import os
import re
from collections import defaultdict
import cv2
from cv2.typing import MatLike
from scipy.interpolate import splprep, splev


def load_camera_matrices(path, matrix_types=None, device=torch.device("cpu")):
    cameras = defaultdict(dict)
    cam_names = set()
    file_pattern = re.compile(r"^Cam_([A-Za-z]+_\d+)_(K|RT|P)\.npy$")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if matrix_types is not None:
        matrix_types = set(matrix_types)

    for filename in sorted(os.listdir(path)):
        match = file_pattern.match(filename)
        if match:
            cam_name, matrix_type = match.groups()
            if matrix_types is None or matrix_type in matrix_types:
                filepath = os.path.join(path, filename)
                cameras[cam_name][matrix_type] = torch.tensor(np.load(filepath), device=device)
                cam_names.add(cam_name)

    cam_names = sorted(cam_names)
    cam_name_to_id = {name: idx for idx, name in enumerate(cam_names)}
    cam_id_to_name = {idx: name for name, idx in cam_name_to_id.items()}
    cameras_id = {cam_name_to_id[name]: cameras[name] for name in cam_names}

    return cameras_id, cam_name_to_id, cam_id_to_name

def load_renders(renders_path, object_name):
    renders = {}
    pattern = re.compile(rf"^{object_name}_Cam_(Overhead|Above|Ground)_(\d+)\.png")

    for filename in sorted(os.listdir(renders_path)):
        match = pattern.match(filename)
        if match:
            view_type, view_number = match.groups()
            cam_name = f"{view_type}_{view_number}"
            image_path = os.path.join(renders_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                renders[cam_name] = image
    return renders  # Dict[str, np.ndarray]

def canny_edge_map(img: MatLike, options):
    equalise, t1, t2 = options
    # convert to grayscale
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if equalise:
        img_greyscale = cv2.equalizeHist(img_greyscale)
    # apply edge detection
    edge_map = cv2.Canny(img_greyscale, threshold1=t1, threshold2=t2)
    # return edge map
    return edge_map

def load_edgemaps(renders: dict, edgemap_options: dict, device=torch.device("cpu")):
    edgemaps = {}
    edgemaps_len = {}

    for cam_name, image in renders.items():
        if cam_name in edgemap_options:
            edge_option = edgemap_options[cam_name]
            edges = canny_edge_map(image, edge_option)
            edge_coords = np.argwhere(edges > 0)[:, [1, 0]]
            edgemaps[cam_name] = torch.tensor(edge_coords)
            edgemaps_len[cam_name] = len(edge_coords)

    return edgemaps, edgemaps_len


# def get_projmats_and_edgemap_info(view_idx, target_mesh: str, matrices, edgemaps, edgemaps_len):
#     """
#     Retrieves the projection matrices and target edgemap information for the specified view indices and target mesh.

#     Parameters:
#         view_idx (list): List of indices for which to retrieve projection matrices and edgemaps.
#         target_mesh (str): The target mesh name (e.g., 'balloon') to extract edgemaps and lengths for.
#         matrices (dict): Dictionary containing camera matrices.
#         edgemaps (dict): Dictionary containing edgemaps for various meshes.
#         edgemaps_len (dict): Dictionary containing the lengths of the edgemaps for each mesh.

#     Returns:
#         tuple: A tuple containing the projection matrices (torch.Tensor) and the target edgemap information (tuple of torch.Tensors).
#     """
#     # Get the projection matrices for the specified view indices
#     projmats = torch.stack([matrices[view_idx[i]]["P"] for i in range(len(view_idx))])

#     # Get the target edgemaps for the specified target mesh
#     tgt_edgemaps = torch.nn.utils.rnn.pad_sequence([edgemaps[target_mesh][i] for i in view_idx], batch_first=True, padding_value=0.0)
#     tgt_edgemaps_len = torch.tensor([edgemaps_len[target_mesh][i] for i in view_idx])

#     # Pack the target edgemaps and their lengths
#     tgt_edgemap_info = (tgt_edgemaps, tgt_edgemaps_len)

#     return projmats, tgt_edgemap_info

# def load_edgemaps(renders, options):
#     edgemaps = {}
#     edgemaps_len = {}

#     # Iterate through the renders dictionary, which is grouped by mesh names
#     for mesh_name, mesh_renders in renders.items():
#         views = {}
#         views_len = {}
        
#         # Find the edgemap options for the current mesh name
#         if mesh_name in options:
#             edgemap_options = options[mesh_name]

#             # Iterate through each render (view) for the current mesh
#             for num, img in mesh_renders.items():
#                 # Apply the corresponding edgemap options for this render number (num)
#                 if num in edgemap_options:
#                     edge_option = edgemap_options[num]
#                     edges = canny_edge_map(img, edge_option)
#                     if True:
#                         edge_coords = np.argwhere(edges > 0)
#                         edge_coords = edge_coords[:, [1, 0]]  # Convert to (x, y) coordinates

#                     views[num] = torch.tensor(edge_coords)
#                     views_len[num] = len(edge_coords)

#         # Store the results for the current mesh
#         edgemaps[mesh_name] = views
#         edgemaps_len[mesh_name] = views_len

#     return edgemaps, edgemaps_len


