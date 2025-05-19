import torch
from collections import deque

def build_edge_lists(faces, device=None):
    e = torch.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = torch.cat([e, e.flip(1)]).unique(dim=0)
    if device: e = e.to(device)
    return e[:, 0], e[:, 1]

def bfs_hop_distance(V, edge_src, edge_dst, boundary_idx, k_max=10, device=torch.device("cpu")):
    B = boundary_idx.numel()
    D = torch.full((V, B), k_max + 1, dtype=torch.int16, device=device)

    # Make sure indices are on CPU for Python-native ops like deque/set
    edge_src = edge_src.cpu()
    edge_dst = edge_dst.cpu()
    boundary_idx = boundary_idx.cpu()

    frontier = [deque([b.item()]) for b in boundary_idx]
    visited  = [set([b.item()])   for b in boundary_idx]

    for dist in range(k_max + 1):
        for b, q in enumerate(frontier):
            for _ in range(len(q)):
                v = q.popleft()
                D[v, b] = dist
                for u in edge_dst[edge_src == v].tolist():
                    if u not in visited[b]:
                        visited[b].add(u)
                        q.append(u)
    return D


# -------------------------------
# 1. Jacobi smoother
# -------------------------------
def make_jacobi_hook(edge_src, edge_dst, boundary_mask, k=5, constrained=False, debug=False):
    V = boundary_mask.size(0)
    onesE = torch.ones(edge_src.size(0), 1, device=edge_src.device)

    def hook(grad_in):
        g = grad_in.view(-1, 3).clone()
        og_g = grad_in.view(-1, 3).clone()
        for _ in range(k):
            g_nb = torch.zeros_like(g)
            g_nb.scatter_add_(0, edge_dst[:, None].expand(-1, 3), g[edge_src])
            deg = torch.zeros(V, 1, device=g.device)
            deg.scatter_add_(0, edge_dst[:, None], onesE)
            # if constrained:
            #     g_nb += og_g
            # else:
            g_nb += g
            deg += 1.0
            g_nb = g_nb / (deg.clamp_min_(1.0))
            if constrained:
                g = torch.where(boundary_mask[:, None], og_g, g_nb)
            else:
                g = g_nb

        if debug:
            g_diff = (g - grad_in.view(-1, 3)).abs()
            print("[Jacobi] Δg > 1e-6:", (g_diff.norm(dim=-1) > 1e-6).sum().item(),
                  "| mean Δg:", g_diff.mean().item())

        return g.view_as(grad_in)

    return hook


# -------------------------------
# 2. Inverse-hop smoother
# -------------------------------
def make_invhop_hook(D_all, boundary_mask, k=-1, eps=1e-4, constrained=True, debug=False):
    V = boundary_mask.size(0)

    def hook(grad_in):
        g = grad_in.view(-1, 3)

        with torch.no_grad():
            boundary_idx = boundary_mask.nonzero(as_tuple=False).view(-1)
            D_hop = D_all[:, boundary_idx].to(g.device)  # (V, B)

            if k >= 0:
                D_mask = (D_hop <= k)
                D_hop = D_hop.masked_fill(~D_mask, float('inf'))

            inv = 1.0 / (D_hop.float() + eps)
            inv[D_hop == float('inf')] = 0.0  # zero out masked entries

            w_self = torch.ones(V, 1, device=g.device)
            row_sum = w_self + inv.sum(1, keepdim=True)
            W = inv / row_sum
            w_self = w_self / row_sum

            g_b = g[boundary_idx]
            g_new = W @ g_b + w_self * g

            if constrained:
                bmask = torch.zeros(V, dtype=torch.bool, device=g.device)
                bmask[boundary_idx] = True
                g_new[bmask] = g[bmask]

            if debug:
                g_diff = (g_new - g).abs()
                print("[InvHop-k] Δg > 1e-6:", (g_diff.norm(dim=-1) > 1e-6).sum().item(),
                      "| max Δg:", g_diff.max().item())

        return g_new.view_as(grad_in)

    return hook


# -------------------------------
# 3. K-hop bounded smoother
# -------------------------------
def make_khop_hook(D_all, boundary_mask, k=5, eps=1e-3, constrained=True, debug=False):
    V = boundary_mask.size(0)

    def hook(grad_in):
        g = grad_in.view(-1, 3)

        with torch.no_grad():
            boundary_idx = boundary_mask.nonzero(as_tuple=False).view(-1)
            D_hop = D_all[:, boundary_idx].to(g.device)  # (V, B)

        mask = D_hop <= k
        W_boundary = mask.float()  # (V, B)

        row_sum = W_boundary.sum(1, keepdim=True) + 1  # +1 for self
        W_boundary = W_boundary / row_sum.clamp_min(eps)  # (V, B)
        w_self = 1.0 / row_sum.clamp_min(eps)            # (V, 1)

        g_b = g[boundary_idx]  # (B, 3)
        g_new = W_boundary @ g_b + w_self * g  # (V, 3)

        if constrained:
            bmask = torch.zeros(V, dtype=torch.bool, device=g.device)
            bmask[boundary_idx] = True
            g_new[bmask] = g[bmask]

        if debug:
            g_diff = (g_new - g).abs()
            print("[Khop+Self] Δg > 1e-6:", (g_diff.norm(dim=-1) > 1e-6).sum().item(),
                  "| max Δg:", g_diff.max().item())

        return g_new.view_as(grad_in)

    return hook

# -------------------------------
# Unified selector
# -------------------------------
def select_hook(
    method: str,
    edge_src=None,
    edge_dst=None,
    boundary_mask=None,
    D_all=None,             # used in cached hooks
    k=5,
    constrained=False,
    debug=False,
):
    if method == "jacobi":
        return make_jacobi_hook(edge_src, edge_dst, boundary_mask, k=k, constrained=constrained, debug=debug)
    elif method == "invhop":
        return make_invhop_hook(
            D_all=D_all,
            boundary_mask=boundary_mask,
            constrained=constrained,
            debug=debug
        )
    elif method == "khop":
        return make_khop_hook(
            D_all=D_all,
            boundary_mask=boundary_mask,
            k=k,
            constrained=constrained,
            debug=debug
        )
    else:
        raise ValueError(f"Unknown smoothing method: {method}")