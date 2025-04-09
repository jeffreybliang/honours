import torch
import numpy as np
import os
import re
from collections import defaultdict
import cv2
from cv2.typing import MatLike
from scipy.interpolate import splprep, splev


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

def canny_edge_map(img: MatLike, options):
    equalise, t1, t2 = options
    # convert to grayscale
    img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if equalise:
        img_greyscale = cv2.equalizeHist(img_greyscale)
    # apply edge detection
    # edge_map = cv2.Canny(img_greyscale, threshold1=20, threshold2=100)
    # edge_map = cv2.Canny(img_greyscale, threshold1=15, threshold2=250)
    edge_map = cv2.Canny(img_greyscale, threshold1=t1, threshold2=t2)
    # return edge map
    return edge_map

def load_edgemaps(renders, options):
    edgemaps = {}
    edgemaps_len = {}

    # Iterate through the renders dictionary, which is grouped by mesh names
    for mesh_name, mesh_renders in renders.items():
        views = {}
        views_len = {}
        
        # Find the edgemap options for the current mesh name
        if mesh_name in options:
            edgemap_options = options[mesh_name]

            # Iterate through each render (view) for the current mesh
            for num, img in mesh_renders.items():
                # Apply the corresponding edgemap options for this render number (num)
                if num in edgemap_options:
                    edge_option = edgemap_options[num]
                    edges = canny_edge_map(img, edge_option)
                    if True:
                        edge_coords = np.argwhere(edges > 0)
                        edge_coords = edge_coords[:, [1, 0]]  # Convert to (x, y) coordinates
                    else:
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        all_subpixel_edges = []
                        print("number of contours", len(contours))
                        for contour in contours:
                            if len(contour) < 5:
                                continue
                            contour = contour.squeeze()  # Remove single-dim (N,1,2) -> (N,2)
                            x, y = contour[:, 0], contour[:, 1]

                            # Fit a spline to the contour
                            try:
                                tck, u = splprep([x, y], s=1.0)  # s controls smoothing
                                u_fine = np.linspace(0, 1, len(x)*5)  # More points = higher "resolution"
                                x_fine, y_fine = splev(u_fine, tck)

                                subpixel_points = np.vstack((x_fine, y_fine)).T
                                all_subpixel_edges.append(subpixel_points)
                                edge_coords = np.concatenate(all_subpixel_edges)
                            except Exception as e:
                                print(f"Skipping a contour due to error: {e}")
                                continue

                    views[num] = torch.tensor(edge_coords)
                    views_len[num] = len(edge_coords)

        # Store the results for the current mesh
        edgemaps[mesh_name] = views
        edgemaps_len[mesh_name] = views_len

    return edgemaps, edgemaps_len


