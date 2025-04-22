import torch
from collections import deque

# --------------------------------------------------------------------- #
# 0.  Connectivity helpers (run once after loading a mesh)              #
# --------------------------------------------------------------------- #
def build_edge_lists(faces, device=None):
    """
    faces : (F,3) int64   – triangular faces of one mesh
    returns  edge_src, edge_dst  (E,) int64  undirected, both directions
    """
    e = torch.vstack([faces[:, [0, 1]],
                      faces[:, [1, 2]],
                      faces[:, [2, 0]]])
    e = torch.cat([e, e.flip(1)]).unique(dim=0)        # undirected unique
    if device: e = e.to(device)
    return e[:, 0], e[:, 1]                            # src , dst


def bfs_hop_distance(V, edge_src, edge_dst, boundary_idx, k_max=10):
    """
    Multi‑source BFS.  Returns  (V,B) int16 hop‑distance matrix
    clipped at k_max+1 (treated as "∞").
    """
    B = boundary_idx.numel()
    D = torch.full((V, B), k_max + 1, dtype=torch.int16)   # init with "∞"
    # frontier per boundary vertex
    frontier = [deque([b.item()]) for b in boundary_idx]
    visited  = [set([b.item()])   for b in boundary_idx]

    for dist in range(k_max + 1):
        for b, q in enumerate(frontier):
            for _ in range(len(q)):
                v = q.popleft()
                D[v, b] = dist
                # push neighbours
                neigh_mask = (edge_src == v)              # (E,)
                neigh = edge_dst[neigh_mask].tolist()
                for u in neigh:
                    if u not in visited[b]:
                        visited[b].add(u)
                        q.append(u)
    return D


# ================================================================
# 0.  Common utilities
# ================================================================
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor            # comes with PyG




def register_and_test(verts, loss_fn, make_hook_ref, make_hook_pyg,
                      name="hook", atol=1e-6):
    """
    verts      : original (V,3) tensor with requires_grad_
    loss_fn    : lambda(tensor) → scalar  (dummy loss to trigger backward)
    make_hook_ref / make_hook_pyg : callables that return backward hooks
    """
    v_ref = verts.clone().detach().requires_grad_(True)
    v_pyg = verts.clone().detach().requires_grad_(True)

    # register hooks
    v_ref.register_hook(make_hook_ref())
    v_pyg.register_hook(make_hook_pyg())

    loss_fn(v_ref).backward()
    loss_fn(v_pyg).backward()

    diff = (v_ref.grad - v_pyg.grad).abs().max().item()
    print(f"{name}: max |ref‑pyg| = {diff:.3e}")
    if diff > atol:
        print("WARNING: gradients diverge!")

# ================================================================
# 1.  Jacobi (k Laplacian sweeps)
# ================================================================
def make_jacobi_ref(edge_src, edge_dst, boundary_mask, k):
    V = boundary_mask.size(0)
    onesE = torch.ones(edge_src.size(0), 1, device=edge_src.device)

    def hook(grad_in):
        g = grad_in.view(-1, 3).clone()      # ← flattened
        # print(g)
        for _ in range(k):
            g_nb = torch.zeros_like(g)
            g_nb.scatter_add_(0,
                              edge_dst.unsqueeze(1).expand(-1, 3),
                              g[edge_src])
            deg  = torch.zeros(V, 1, device=g.device)
            deg.scatter_add_(0, edge_dst.unsqueeze(1), onesE)

            g_nb = g_nb / deg.clamp_min_(1.0)
            g    = torch.where(boundary_mask.unsqueeze(1), g, g_nb)
            # g = g_nb
        # print(g)
        # has_neighbors = deg.squeeze() > 0
        # touched = (~boundary_mask) & has_neighbors
        # print("Interior verts w/ boundary neighbors:", touched.sum().item())
        return g.view_as(grad_in)
    return hook


def make_jacobi_pyg(edge_index, boundary_mask, k):
    class LapSmooth(MessagePassing):
        def __init__(self, k):
            super().__init__(aggr='mean')
            self.k = k
        def forward(self, x, edge_index, keep):
            for _ in range(self.k):
                x = self.propagate(edge_index, x=x, keep=keep)
            return x
        def message(self, x_j):
            return x_j
        def update(self, aggr_out, x, keep):
            return torch.where(keep.unsqueeze(1), x, aggr_out)

    lap = LapSmooth(k)
    def hook(g):
        return lap(g, edge_index, boundary_mask)
    return hook

# ================================================================
# 2.  Inverse hop‑distance with self‑weight   (dense mm vs SparseTensor)
# ================================================================
def make_invhop_ref(boundary_idx, D_hop, eps=1e-3):
    inv    = 1.0 / (D_hop.float() + eps)
    inv[D_hop == D_hop.max()] = 0.0
    w_self = torch.ones(inv.size(0), 1, device=inv.device)
    row_sum= w_self + inv.sum(1, keepdim=True)
    W_norm = inv / row_sum
    w_self = w_self / row_sum

    bmask = torch.zeros(inv.size(0), dtype=torch.bool, device=inv.device)
    bmask[boundary_idx] = True

    def hook(g):
        g_b   = g[boundary_idx]
        g_new = W_norm @ g_b + w_self * g
        g_new[bmask] = g[bmask]
        return g_new
    return hook


def make_invhop_pyg(boundary_idx, D_hop, eps=1e-3):
    inv = 1.0 / (D_hop.float() + eps)
    inv[D_hop == D_hop.max()] = 0.0
    w_self = torch.ones(inv.size(0), 1, device=inv.device)
    row_sum= w_self + inv.sum(1, keepdim=True)
    W_norm = inv / row_sum
    w_self = w_self / row_sum

    # sparse COO → torch_sparse.SparseTensor  (GPU friendly)
    row, col = W_norm.nonzero(as_tuple=True)
    W_sp = SparseTensor(row=row, col=col,
                        value=W_norm[row, col],
                        sparse_sizes=W_norm.shape).coalesce()

    bmask = torch.zeros(inv.size(0), dtype=torch.bool, device=inv.device)
    bmask[boundary_idx] = True

    def hook(g):
        g_b   = g[boundary_idx]                     # (B,3)
        g_new = W_sp @ g_b + w_self * g             # spmm
        g_new[bmask] = g[bmask]
        return g_new
    return hook

# ================================================================
# 3.  k‑hop bounded, equal weights (k‑ring)
# ================================================================
def make_khop_ref(boundary_idx, D_hop, k, eps=1e-3):
    mask    = D_hop <= k
    W       = mask.float()
    row_sum = W.sum(1, keepdim=True).clamp_min_(eps)
    W       = W / row_sum

    def hook(g):
        g_b = g[boundary_idx]
        return W @ g_b
    return hook


def make_khop_pyg(edge_index, boundary_mask, k):
    # Same LapSmooth class but aggr='mean' already gives equal weights
    class KRing(MessagePassing):
        def __init__(self, k):
            super().__init__(aggr='mean')
            self.k = k
        def forward(self, x, edge_index, keep):
            for _ in range(self.k):
                x = self.propagate(edge_index, x=x, keep=keep)
            return x
        def message(self, x_j):
            return x_j
        def update(self, aggr_out, x, keep):
            return torch.where(keep.unsqueeze(1), x, aggr_out)
    kring = KRing(k)
    def hook(g):
        return kring(g, edge_index, boundary_mask)
    return hook


def make_jacobi_hook_debug(edge_src, edge_dst,
                           boundary_mask, k_iter=5,
                           every=1,                      # print every N calls
                           tag="JACOBI"):
    """
    Wraps make_jacobi_hook with a one‑line debug statement.
    """
    base_hook = make_jacobi_ref(edge_src, edge_dst,
                                 boundary_mask, k_iter)

    call_counter = {"n": 0}          # mutable container closes over hook

    def debug_hook(grad_in):
        call_counter["n"] += 1
        g_raw    = grad_in.view(-1, 3)
        g_smooth = base_hook(grad_in)
        g_diff   = (g_smooth.view(-1, 3) - g_raw).abs()

        if (call_counter["n"] - 1) % every == 0:
            num_changed = (g_diff.sum(-1) > 1e-6).sum().item()
            num_nonzero = (g_raw.norm(dim=-1) > 1e-6).sum().item()
            print(f"[{tag}] iter {call_counter['n']}:")
            print(f"   boundary_mask.sum()     = {boundary_mask.sum().item()}")
            print(f"   nonzero grad_in         = {num_nonzero}")
            print(f"   verts with Δg > 1e-6    = {num_changed}")
            print(f"   max |Δg|                = {g_diff.max().item():.2e}")

        return g_smooth

    return debug_hook
