from .dataloader import DataLoader
import json
from .node import *
from .chamfer import *
from .utils import *
from .io import *
from .gradient import *
from typing import List, Tuple, Union
import random
import wandb
import os
from tqdm import trange


class ExperimentRunner:
    def __init__(self, experiment_config: Union[str, dict], data_loader: DataLoader) -> None:
        # If experiment_config is a path, load from file
        if isinstance(experiment_config, str) and os.path.exists(experiment_config):
            with open(experiment_config, "r") as f:
                self.cfg = json.load(f)
        elif isinstance(experiment_config, dict):
            self.cfg = experiment_config
        else:
            raise ValueError("Invalid experiment_config. Must be a path or dict.")
        # Store the experiment details
        self.project = self.cfg["project"]
        self.experiment_name = self.cfg["name"]
        self.experiment_description = self.cfg["description"]
        self.src_mesh = self.cfg["src_mesh"]
        self.target_meshes = self.cfg["target_meshes"]
        
        # Parse views configuration for each target mesh
        self.views_config = {}
        for mesh_name, view_config in self.cfg["views"].items():
            self.views_config[mesh_name] = {
                "mode": view_config["mode"],
                "view_names": view_config["view_names"],
                "num_views": view_config["num_views"]
            }
            self.num_views = view_config["num_views"]

        
        # Training settings
        self.n_iters = self.cfg["training"]["n_iters"]
        self.lr = self.cfg["training"]["lr"]
        self.momentum = self.cfg["training"]["momentum"]
        self.verbose = self.cfg["verbose"]
        self.vis_enabled = self.cfg["vis"]["enabled"]
        self.vis_freq = self.cfg["vis"]["frequency"]
        # Use the DataLoader to load data
        self.data_loader = data_loader
        # edgemap_options = {k: v for k,v in data_loader.edgemap_options.items() if k in self.target_meshes}
        # self.edgemaps, self.edgemaps_len = load_edgemaps(data_loader.renders, edgemap_options)
        
        self.wandb = self.cfg["wandb"]

        self.smoothing = self.cfg["gradient"]["smoothing"]
        self.smoothing_method = self.cfg["gradient"]["method"]
        self.smoothing_k = self.cfg["gradient"]["k"]
        self.smoothing_constrained = self.cfg["gradient"]["constrained"]
        self.smoothing_debug = self.cfg["gradient"]["debug"]
        if self.smoothing_debug:
            print("Smoothing debug is on")

        velocity_cfg = self.cfg.get("velocity", {})
        self.velocity_enabled = velocity_cfg.get("enabled", False)
        self.velocity_k = velocity_cfg.get("k", 0)
        self.beta = velocity_cfg.get("beta", 1)

# ============================================================================================================

    def run(self):
        device = self.data_loader.device
        print(f"Running on device: {device}")
        return self.pipeline(self.src_mesh, self.target_meshes, device)

# ============================================================================================================


    def pipeline(self, src_name: str, tgt_names: List[str], device: torch.device):
        """Pipeline function to run the experiment on a source mesh and a list of target meshes."""
        if self.wandb:
            run = wandb.init(
                project=self.project,
                name=self.experiment_name,
                notes=self.experiment_description,
                group="clean",
                config={
                    "data":    self.data_loader.cfg,
                    "experiment":       self.cfg
                }
            )
            wandb.define_metric("outer/chamfer", summary="min")
            wandb.define_metric("outer/gt/chamfer", summary="min")
            wandb.define_metric("outer/gt/iou", summary="max")

        view_names_used = {}
        step_offset = 0

        prev_verts = None
        prev_displacement = None
        src_mesh = self.get_mesh(src_name).to(device)
        initial_volume = calculate_volume(src_mesh[0].verts_packed(), src_mesh[0].faces_packed()).item()

        for tgt_name in tgt_names:
            print(f"Target: {tgt_name}")
            tgt_mesh = self.get_gt_mesh(tgt_name).to(device)

            # Use encapsulated loading logic
            projmats, tgt_edgemap_info, view_ids = self.get_projmats_and_edgemap_info(tgt_name, device)
            gt_projmats, gt_edgemap_info, _ = self.get_gt_projmats_and_edgemap_info(tgt_name, device)

            # Reverse map view_ids to view_names
            _, _, cam_id_to_name = self.data_loader.load_camera_matrices()
            view_names = [cam_id_to_name[i] for i in view_ids]
            view_names_used[tgt_name] = view_names

            edgemap_info = ([tgt_edgemap_info[0]], [tgt_edgemap_info[1]])

            if self.velocity_enabled and prev_displacement is not None:
                print("Accounting for velocity")
                faces = src_mesh[0].faces_packed().to(prev_verts.device)
                edge_src, edge_dst = build_edge_lists(faces, device=prev_verts.device)
                smoothed_disp = smooth_displacement_jacobi(prev_displacement, edge_src, edge_dst, k=self.velocity_k)
                extrapolated_verts = prev_verts[0] + self.beta * smoothed_disp
                src_mesh = Meshes(verts=extrapolated_verts[None, ...], faces=src_mesh.faces_padded())

                mean_disp = (self.beta * smoothed_disp).norm(dim=1).mean().item()
                if self.wandb:
                    wandb.log({"velocity/mean_displacement": mean_disp}, step=step_offset)

            # Train the mesh transformation
            final_verts = self.train_loop(
                src_mesh, tgt_mesh,
                projmats, edgemap_info,
                gt_projmats, gt_edgemap_info,
                n_iters=self.n_iters,
                step_offset=step_offset,
                lr=self.lr,
                moment=self.momentum,
                device=device,
                target_volume=initial_volume
            )
            
            if prev_verts is not None:
                prev_displacement = final_verts[0] - prev_verts[0]
            prev_verts = final_verts.detach()
            src_mesh = Meshes(verts=final_verts.float(), faces=src_mesh.faces_padded())

            # update step offset
            step_offset += self.n_iters

        if self.wandb:
            run.config["view_names"] = view_names_used
            run.finish()

# ============================================================================================================

    def train_loop(self, src: Meshes, tgt: Meshes, projmats, edgemap_info, gt_projmats, gt_edgemap_info, 
                n_iters, step_offset, lr, moment, device, target_volume):
        node = ConstrainedProjectionNode(src, target_volume, self.wandb)
        verts_init = src.verts_padded()
        verts_init.requires_grad = True
        verts = verts_init.clone().detach().requires_grad_(True).to(device)
        V_max = src.num_verts_per_mesh().max().item()
        boundary_mask = torch.zeros(V_max, dtype=torch.bool, device=device)

        faces = src[0].faces_packed().to(device)
        edge_src, edge_dst = build_edge_lists(faces, device)
        all_idx = torch.arange(V_max, device=device)
        D_all = bfs_hop_distance(V_max, edge_src, edge_dst, all_idx, k_max=10)

        if self.smoothing:
            hook = select_hook(
                method=self.smoothing_method,         # "jacobi", "invhop", or "khop"
                edge_src=edge_src,
                edge_dst=edge_dst,
                boundary_mask=boundary_mask,
                D_all=D_all,
                k=self.smoothing_k,
                constrained=self.smoothing_constrained,
                debug=self.smoothing_debug,
            )
            verts.register_hook(hook)

        chamfer_loss = PyTorchChamferLoss(src, tgt, projmats, edgemap_info, boundary_mask=boundary_mask)
        optimiser = torch.optim.SGD([verts], lr=lr, momentum=moment)
        a,b = edgemap_info
        a,b = a[0], b[0]
        projectionplot = plot_projections(verts.detach().squeeze().double(), gt_projmats, gt_edgemap_info)
        cmin,cmax = None,None
        bmin,bmax = None,None

        initheatmap,cmin,cmax = create_heatmap(Meshes(verts=verts.detach(), faces=src[0].faces_packed().unsqueeze(0)), tgt[0], cmin, cmax)
        if self.wandb:
            wandb.log({"plt/projections": wandb.Image(projectionplot),
                        "plt/heatmap": wandb.Plotly(initheatmap)}, step= step_offset)

        min_loss = float("inf")
        best_verts = None
    
        pbar = trange(n_iters, desc="Training", leave=True, mininterval=5)  # Always visible
        for i in pbar:
            optimiser.zero_grad(set_to_none=True)
            node.iter = i + step_offset
            projverts = ConstrainedProjectionFunction.apply(node, verts)
            loss = chamfer_loss(projverts)

            colour = bcolors.FAIL
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_verts = projverts.detach().clone()
                colour = bcolors.OKGREEN
                
            loss.backward()
            optimiser.step()

            pbar.set_description(f"Loss: {loss.item()*10:.4f}")

            # log
            if self.wandb:
                wandb.log(
                    data = {
                    "outer/chamfer": loss * 10,
                    "outer/vol": calculate_volume(projverts[0], src[0].faces_packed()),
                    "outer/gt/chamfer": chamfer_gt(projverts, src, tgt)[0] * 10,
                    "outer/gt/iou": iou_gt(projverts, src, tgt)[0]
                    },
                    step = i + step_offset
                )

            if self.verbose:
                print(f"{i:4d} Loss: {colour}{loss.item():.3f}{bcolors.ENDC} Volume: {calculate_volume(projverts[0], src[0].faces_packed()):.3f}")
                print(f"GT Chamfer: [{', '.join(f'{x:.3f}' for x in chamfer_gt(projverts, src, tgt))}] "
                    f"GT IoU: [{', '.join(f'{x:.3f}' for x in iou_gt(projverts, src, tgt))}]")
            def should_log(i):
                return i < 50 or (i % self.vis_freq == self.vis_freq-1)  # True at i = 1, 2, 4, 8, 16, ...
            if self.vis_enabled and should_log(i):
                projectionplot = plot_projections(projverts.detach().squeeze().double(), gt_projmats, gt_edgemap_info)
                tmp_mesh = Meshes(verts=projverts.detach(), faces=src[0].faces_packed().unsqueeze(0))
                heatmap,cmin,cmax = create_heatmap(tmp_mesh, tgt[0], cmin, cmax)
                boundary_fig, bmin, bmax = compute_boundary_distance_heatmap(
                    tmp_mesh, boundary_mask, D_all, bmin, bmax
                )
                if self.wandb:
                    wandb.log({
                         "plt/projections": wandb.Image(projectionplot),
                         "plt/heatmap": wandb.Plotly(heatmap),
                         "plt/boundary_heatmap": wandb.Plotly(boundary_fig)
                         }, 
                         step=i + step_offset)        
        return best_verts

    def get_gt_mesh(self,name):
        return self.data_loader.get_gt_mesh(name)

    def get_mesh(self, name):
        return self.data_loader.get_mesh(name)
    
    # def get_edgemaps(self, name):
    #     return self.edgemaps[name], self.edgemaps_len[name]

    def get_camera_matrices(self):
        return self.data_loader.load_camera_matrices()

    def get_projmats_and_edgemap_info(self, mesh_name, device=torch.device("cpu")):
        view_conf = self.views_config[mesh_name]
        edgemap_opts = self.data_loader.edgemap_options[mesh_name]
        renders = self.data_loader.load_renders(mesh_name)

        view_names = (
            view_conf["view_names"]
            if view_conf["mode"] == "manual"
            else random.sample(view_conf["view_names"], view_conf["num_views"])
        )
        print(f"{view_names} from {view_conf['view_names']}")

        edgemaps, edgemaps_len = load_edgemaps(renders, edgemap_opts)
        camera_matrices, cam_name_to_id, _ = self.data_loader.load_camera_matrices()

        view_idx = [cam_name_to_id[name] for name in view_names]
        projmats = torch.stack([camera_matrices[cam_name_to_id[name]]["P"] for name in view_names])
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence(
            [edgemaps[name] for name in view_names],
            batch_first=True, padding_value=0.0
        )
        tgt_edgemaps_len = torch.tensor([edgemaps_len[name] for name in view_names])

        return projmats, (tgt_edgemaps, tgt_edgemaps_len), view_idx



    def get_gt_projmats_and_edgemap_info(self, mesh_name, device=torch.device("cpu")):
        edgemap_opts = self.data_loader.edgemap_options[mesh_name]
        renders = self.data_loader.load_renders(mesh_name)
        edgemaps, edgemaps_len = load_edgemaps(renders, edgemap_opts)

        camera_matrices, cam_name_to_id, _ = self.data_loader.load_camera_matrices()
        view_names = sorted(cam_name_to_id.keys())  # alphabetical order
        view_idx = [cam_name_to_id[name] for name in view_names]

        projmats = torch.stack([camera_matrices[cam_name_to_id[name]]["P"] for name in view_names]).to(device)
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence(
            [edgemaps[name] for name in view_names],
            batch_first=True, padding_value=0.0
        ).to(device)
        tgt_edgemaps_len = torch.tensor([edgemaps_len[name] for name in view_names], device=device)

        return projmats, (tgt_edgemaps, tgt_edgemaps_len), view_idx
    
def smooth_displacement_jacobi(displacement, edge_src, edge_dst, k=1):
    V = displacement.size(0)
    d = displacement.clone()
    for _ in range(k):
        d_nb = torch.zeros_like(d)
        d_nb.scatter_add_(0, edge_dst[:, None].expand(-1, 3), d[edge_src])
        deg = torch.zeros(V, 1, device=d.device)
        deg.scatter_add_(0, edge_dst[:, None], torch.ones_like(edge_dst[:, None], dtype=deg.dtype))
        d_nb += d
        deg += 1.0
        d = d_nb / deg
    return d
