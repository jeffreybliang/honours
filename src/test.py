import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_meshes(source_path, target_path):
    """
    Load source and target meshes from OBJ files.
    
    Args:
        source_path: Path to source mesh OBJ file
        target_path: Path to target mesh OBJ file
    
    Returns:
        source_vertices: Vertices of source mesh
        source_faces: Faces of source mesh
        target_vertices: Vertices of target mesh
        target_faces: Faces of target mesh
    """
    source_mesh = trimesh.load_mesh(source_path)
    target_mesh = trimesh.load_mesh(target_path)
    print(source_mesh.vertices.shape)
    # Verify that meshes have the same number of vertices
    if source_mesh.vertices.shape[0] != target_mesh.vertices.shape[0]:
        raise ValueError("Meshes must have the same number of vertices")
    
    # Verify that meshes have the same number of faces
    if source_mesh.faces.shape[0] != target_mesh.faces.shape[0]:
        raise ValueError("Meshes must have the same number of faces")
    return source_mesh.vertices, np.array(source_mesh.faces), target_mesh.vertices, np.array(target_mesh.faces)

def calculate_volume_torch(vertices_torch, faces):
    """
    Calculate the volume of a mesh using PyTorch tensors.
    
    Args:
        vertices_torch: Nx3 tensor of vertex coordinates
        faces: Mx3 array of face indices
    
    Returns:
        volume: Total volume of the mesh as a PyTorch scalar
    """
    volume = torch.tensor(0.0, requires_grad=True)
    
    for face in faces:
        v1, v2, v3 = [vertices_torch[idx] for idx in face]
        # Calculate cross product v2 × v3
        cross_product = torch.cross(v2, v3)
        # Calculate dot product v1 · (v2 × v3)
        triple_product = torch.dot(v1, cross_product)
        # Volume of tetrahedron = (1/6) * |triple scalar product|
        tetrahedron_volume = torch.abs(triple_product) / 6.0
        volume = volume + tetrahedron_volume
    
    return volume

def objective_function(x, target_vertices, num_vertices):
    """
    Calculate the squared sum of differences between vertices.
    
    Args:
        x: Flattened array of vertex coordinates to optimize
        target_vertices: Target mesh vertices
        num_vertices: Number of vertices in the mesh
    
    Returns:
        squared_diff: Sum of squared differences between vertices
    """
    # Reshape x to match the shape of source_vertices
    x_reshaped = x.reshape(num_vertices, 3)
    # Calculate squared difference
    squared_diff = np.sum((x_reshaped - target_vertices) ** 2)
    return squared_diff

def save_histogram(data, title, filename):
    """
    Save a histogram of the provided data.
    
    Args:
        data: Numpy array of gradient values
        title: Title of the histogram
        filename: Path to save the histogram image
    """
    plt.figure()
    plt.hist(data, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.xscale("symlog", linthresh=1e-8)
    plt.title(title)
    plt.xlabel("Gradient Values")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def objective_function_grad(x, target_vertices, num_vertices):
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    target_vertices_torch = torch.tensor(target_vertices, dtype=torch.float32)
    x_reshaped = x_torch.reshape(num_vertices, 3)
    squared_diff = torch.sum((x_reshaped - target_vertices_torch) ** 2)
    squared_diff.backward()
    
    gradient = x_torch.grad.detach().numpy() 
    # save_histogram(gradient.flatten(), "Objective Function Gradient", "objective_gradient_hist.png")
    print("obj mean", gradient.mean())
    return gradient

def volume_constraint_grad(x, faces, target_volume):
    num_vertices = len(x) // 3
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    vertices_torch = x_torch.reshape(num_vertices, 3)

    # ===== Method 1: Autograd-based gradient =====
    volume = calculate_volume_torch(vertices_torch, faces)
    volume.backward()
    autograd_grad = x_torch.grad.detach().numpy()

    # ===== Method 2: Analytical gradient with scatter_add =====
    faces = torch.tensor(faces).detach()
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

    analytical_grad = grad_verts.flatten().detach().numpy()

    # ===== Print and Compare =====
    print("Autograd Gradient:", autograd_grad)
    print("Analytical Gradient:", analytical_grad)

    print("Autograd Gradient Mean:", autograd_grad.mean())
    print("Analytical Gradient Mean:", analytical_grad.mean())

    diff = autograd_grad - analytical_grad
    print("Max difference:", abs(diff).max())
    print("Mean difference:", abs(diff).mean())

    return autograd_grad  # You can return either one, depending on what you need.

def volume_constraint(x, faces, target_volume):
    """
    Constraint function for maintaining the volume.
    
    Args:
        x: Flattened array of vertex coordinates
        faces: Face indices
        target_volume: Target volume to maintain
    
    Returns:
        volume_diff: Difference between calculated volume and target volume
    """
    num_vertices = len(x) // 3
    vertices = x.reshape(num_vertices, 3)
    
    # Convert to PyTorch for volume calculation
    vertices_torch = torch.tensor(vertices, dtype=torch.float32)
    current_volume = calculate_volume_torch(vertices_torch, faces).detach().cpu().numpy().item()
    # print("vol constraint")

    return (current_volume - target_volume)

def optimize_mesh(source_vertices, source_faces, target_vertices, target_faces):
    """
    Optimize the source mesh to minimize distance to target mesh while maintaining volume.
    
    Args:
        source_vertices: Source mesh vertices
        source_faces: Source mesh faces
        target_vertices: Target mesh vertices
        target_faces: Target mesh faces
    
    Returns:
        optimized_vertices: Optimized mesh vertices
    """
    num_vertices = source_vertices.shape[0]
    
    # Calculate the target volume
    vertices_torch = torch.tensor(target_vertices, dtype=torch.float32)
    target_volume = calculate_volume_torch(vertices_torch, source_faces).item()
    
    # Initial guess is the source vertices
    x0 = source_vertices.flatten()
    # x0 += np.random.normal(scale=1e-3, size=x0.shape)  # Add small noise

    # Define the constraint with gradient
    constraint = {
        'type': 'eq',
        'fun': volume_constraint,
        'args': (source_faces, target_volume),
        'jac': volume_constraint_grad
    }
    
    # Run the optimization with provided gradients
    result = minimize(
        objective_function,
        x0,
        args=(target_vertices, num_vertices),
        method='SLSQP',
        jac=objective_function_grad,
        constraints=[constraint],
        options={'ftol': 1e-6, 'iprint': 2, 'maxiter': 100}
    )
    
    # Reshape the result back to 3D vertices
    optimized_vertices = result.x.reshape(num_vertices, 3)
    
    return optimized_vertices, result

import time

def main():
    # File paths
    source_path = "../../Blender/sphere_3.obj"
    target_path =  "../../Blender/balloon_3.obj"
    # Load meshes
    source_vertices, source_faces, target_vertices, target_faces = load_meshes(source_path, target_path)
    

    start_time = time.time()
    optimized_vertices, result = optimize_mesh(source_vertices, source_faces, target_vertices, target_faces)
    end_time = time.time()
    print(f"optimize_mesh time: {end_time - start_time:.4f} seconds")

    # Print optimization results
    print(f"Optimization successful: {result.success}")
    print(f"Final objective value: {result.fun}")
    print(f"Number of iterations: {result.nit}")
    
    # Calculate initial and final differences
    initial_diff = np.sum((source_vertices - target_vertices) ** 2)
    final_diff = np.sum((optimized_vertices - target_vertices) ** 2)
    
    print(f"Initial squared difference: {initial_diff}")
    print(f"Final squared difference: {final_diff}")
    print(f"Improvement: {initial_diff - final_diff} ({(1 - final_diff/initial_diff) * 100:.2f}%)")
    
    # Calculate volumes to verify constraint
    vertices_torch = torch.tensor(source_vertices, dtype=torch.float32)
    source_volume = calculate_volume_torch(vertices_torch, source_faces).item()
    
    vertices_torch = torch.tensor(optimized_vertices, dtype=torch.float32)
    optimized_volume = calculate_volume_torch(vertices_torch, source_faces).item()
    
    print(f"Source volume: {source_volume}")
    print(f"Target volume: {calculate_volume_torch(torch.tensor(target_vertices, dtype=torch.float32), target_faces).item()}")

    print(f"Optimized volume: {optimized_volume}")
    print(f"Volume difference: {abs(source_volume - optimized_volume)}")
    
    # Save optimized mesh

if __name__ == "__main__":
    main()