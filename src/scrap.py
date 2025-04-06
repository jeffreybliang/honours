import torch

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
