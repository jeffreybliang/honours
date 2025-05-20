from .dataloader import DataLoader
import json
from .node import *
from .chamfer import *
from .utils import *
from .io import *
from .gradient import *
from .penalty import PenaltyMethod
from typing import List, Tuple, Union
import random
import wandb
import os
from tqdm import trange
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing


class ExperimentRunner:
    def __init__(self, experiment_config: Union[str, dict], data_loader: DataLoader) -> None:
        if isinstance(experiment_config, str) and os.path.exists(experiment_config):
            with open(experiment_config, "r") as f:
                self.cfg = json.load(f)
        elif isinstance(experiment_config, dict):
            self.cfg = experiment_config
        else:
            raise ValueError("Invalid experiment_config. Must be a path or dict.")

        self.project = self.cfg["project"]
        self.experiment_name = self.cfg["name"]
        self.experiment_description = self.cfg["description"]
        self.src_mesh = self.cfg["src_mesh"]
        self.target_meshes = self.cfg["target_meshes"]

        self.views_config = {
            mesh_name: {
                "mode": view_config["mode"],
                "view_names": [str(i) for i in view_config["view_idx"]],
                "num_views": view_config["num_views"]
            }
            for mesh_name, view_config in self.cfg["views"].items()
        }
        self.num_views = next(iter(self.views_config.values()))["num_views"]

        self.n_iters = self.cfg["training"]["n_iters"]
        self.lr = self.cfg["training"]["lr"]
        self.momentum = self.cfg["training"]["momentum"]
        self.opt = self.cfg["training"].get("optimiser", "sgd")

        self.verbose = self.cfg["verbose"]
        self.vis_enabled = self.cfg["vis"]["enabled"]
        self.vis_freq = self.cfg["vis"]["frequency"]

        self.data_loader = data_loader
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

        self.method = self.cfg.get("method", "projection")
        self.lambda_vol = self.cfg.get("penalty", {}).get("lambda_init", 0.0)
        self.projection_mode = self.cfg["projection"].get("mode", "alpha")
        self.alpha = self.cfg["projection"].get("alpha", 10.0)
        
# ============================================================================================================

    def run(self):
        device = self.data_loader.device
        print(f"Running on device: {device}")
        return self.pipeline(self.src_mesh, self.target_meshes, device)

    def pipeline(self, src_name: str, tgt_names: List[str], device: torch.device):
        if self.wandb:
            run = wandb.init(
                project=self.project,
                name=self.experiment_name,
                notes=self.experiment_description,
                group="clean",
                config={
                    "data": self.data_loader.cfg,
                    "experiment": self.cfg
                }
            )
            wandb.define_metric("outer/chamfer", summary="min")
            wandb.define_metric("outer/gt/chamfer", summary="min")
            wandb.define_metric("outer/gt/iou", summary="max")

        view_ids_used = {}
        step_offset = 0

        prev_verts = None
        prev_displacement = None
        src_mesh = self.get_mesh(src_name).to(device)
        initial_volume = 4.39763096 # 1.4 * pi but for mesh, so a little smaller
        # initial_volume = calculate_volume(src_mesh[0].verts_packed(), src_mesh[0].faces_packed()).item()

        for tgt_name in tgt_names:
            print(f"Target: {tgt_name}")
            tgt_mesh = self.get_gt_mesh(tgt_name).to(device)

            projmats, tgt_edgemap_info, view_ids = self.get_projmats_and_edgemap_info(tgt_name, device)
            gt_projmats, gt_edgemap_info, _ = self.get_gt_projmats_and_edgemap_info(tgt_name, device)

            _, _, cam_id_to_name = self.data_loader.load_camera_matrices()
            view_names = [cam_id_to_name[i] for i in view_ids]
            # view_names_used[tgt_name] = view_names
            # Reverse map view_ids to view_names
            view_ids_used[tgt_name] = view_ids

            edgemap_info = ([tgt_edgemap_info[0]], [tgt_edgemap_info[1]])

            if self.velocity_enabled and prev_displacement is not None:
                print("Accounting for velocity")
                faces = src_mesh[0].faces_packed().to(prev_verts.device)
                edge_src, edge_dst = build_edge_lists(faces, device=prev_verts.device)
                smoothed_disp = smooth_displacement_jacobi(prev_displacement, edge_src, edge_dst, k=self.velocity_k)
                extrapolated_verts = prev_verts[0] + self.beta * smoothed_disp
                src_mesh = Meshes(verts=extrapolated_verts[None, ...], faces=src_mesh.faces_padded())

                mean_disp = (self.beta * smoothed_disp).mean(dim=0)
                if self.wandb:
                    wandb.log({
                        "velocity/mean_disp_x": mean_disp[0].item(),
                        "velocity/mean_disp_y": mean_disp[1].item(),
                        "velocity/mean_disp_z": mean_disp[2].item(),
                    }, step=step_offset)

            final_verts = self.train_loop(
                src_mesh, tgt_mesh, projmats, edgemap_info,
                gt_projmats, gt_edgemap_info, self.n_iters,
                step_offset, self.lr, self.momentum, device, initial_volume
            )

            if prev_verts is not None:
                prev_displacement = final_verts[0] - prev_verts[0]
            prev_verts = final_verts.detach()
            src_mesh = Meshes(verts=final_verts.float(), faces=src_mesh.faces_padded())
            step_offset += self.n_iters

        if self.wandb:
            run.config["view_ids"] = view_ids_used
            run.finish()

    def train_loop(self, src, tgt, projmats, edgemap_info, gt_projmats, gt_edgemap_info,
                n_iters, step_offset, lr, moment, device: torch.device, target_volume=None):
        verts_init = src.verts_padded()
        verts_init.requires_grad = True

        if self.method == "penalty":
            xs = verts_init.clone().detach().requires_grad_(True).to(device)
        else:
            verts = verts_init.clone().detach().requires_grad_(True).to(device)

        faces = src[0].faces_packed().to(device)
        V = verts_init.size(1)
        boundary_mask = torch.zeros(V, dtype=torch.bool, device=device)
        edge_src, edge_dst = build_edge_lists(faces, device)
        all_idx = torch.arange(V, device=device)
        D_all = bfs_hop_distance(V, edge_src, edge_dst, all_idx, k_max=10, device=device)

        if self.smoothing:
            hook = select_hook(
                method=self.smoothing_method,
                edge_src=edge_src,
                edge_dst=edge_dst,
                boundary_mask=boundary_mask,
                D_all=D_all,
                k=self.smoothing_k,
                constrained=self.smoothing_constrained,
                debug=self.smoothing_debug,
            )
        if self.method == "penalty":
            xs.register_hook(hook)
        else:
            verts.register_hook(hook)

        if self.method == "penalty":
            loss_fn = PenaltyMethod(src, tgt, projmats, edgemap_info,
                                    lambda_vol=self.lambda_vol, boundary_mask=boundary_mask, device=device, doublesided=self.cfg["chamfer"]["doublesided"], target_volume=target_volume)
            optimiser = torch.optim.SGD([xs], lr=lr, momentum=moment)
        else:
            node = ConstrainedProjectionNode(src, target_volume, self.wandb)
            loss_fn = PyTorchChamferLoss(src, tgt, projmats, edgemap_info, boundary_mask=boundary_mask)
            optimiser = torch.optim.SGD([verts], lr=lr, momentum=moment)

        a, b = edgemap_info
        a, b = a[0], b[0]
        mesh_input = xs if self.method == "penalty" else verts
        projectionplot = plot_projections(mesh_input.detach().squeeze().double(), gt_projmats, gt_edgemap_info)
        cmin, cmax = None, None
        bmin,bmax = None,None

        initheatmap, cmin, cmax = create_heatmap(Meshes(verts=mesh_input.detach(), faces=faces.unsqueeze(0)), tgt[0], cmin, cmax)
        if self.wandb and self.vis_enabled:
            wandb.log({"plt/projections": wandb.Image(projectionplot),
                    "plt/heatmap": wandb.Plotly(initheatmap)}, step=step_offset)

        min_loss = float("inf")
        best_verts = None
        pbar = trange(n_iters, desc="Training", leave=True)

        if self.method == "penalty":
            penalty = self.cfg.get("penalty", {})
            lambda_vol = penalty.get("lambda_init", 100.0)
            lambda_max = penalty.get("lambda_max", 1e6)
            rate = penalty.get("rate", -1)
            linear = penalty.get("linear", -1)
        for i in pbar:
            optimiser.zero_grad(set_to_none=True)
    

            if self.method == "penalty":
                loss_fn.iter = i + step_offset
                loss_dict, boundary_pts, hulls, loops = loss_fn(xs)
                penalty_loss = lambda_vol * loss_dict["vol_error"]
                loss = loss_dict["chamfer"] + penalty_loss
                old_lambda_vol = lambda_vol
                if rate > 0:
                    lambda_vol = min(lambda_vol * rate, lambda_max)
                    loss *= 10
                elif linear > 0:
                    lambda_vol = min(lambda_vol + linear, lambda_max)
                projverts = xs
            else:
                node.iter = i + step_offset
                projverts = ConstrainedProjectionFunction.apply(node, verts)
                loss, boundary_pts, hulls, loops = loss_fn(verts)

            tmp_mesh = Meshes(verts=projverts.float().clone().detach(), faces=src[0].faces_packed().unsqueeze(0))

            colour = bcolors.FAIL
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_verts = projverts.detach().clone()
                colour = bcolors.OKGREEN

            loss.backward()
            optimiser.step()
            pbar.set_description(f"Loss: {loss.item()*10:.4f}")

            # log
            tmp_mesh = Meshes(verts=projverts.detach(), faces=src[0].faces_packed().unsqueeze(0))
            if self.wandb:
                if self.method == "penalty":
                    wandb.log(
                        data = {
                        "outer/vol_penalty_lambda": old_lambda_vol,
                        "outer/vol_error": loss_dict["vol_error"],
                        "outer/penalty": penalty_loss,
                        },
                        step = i + step_offset
                    )

                wandb.log(
                    data = {
                    "outer/chamfer": loss,
                    "outer/vol": calculate_volume(projverts[0], src[0].faces_packed()),
                    "outer/gt/chamfer": chamfer_gt(tmp_mesh, tgt)[0],
                    "outer/gt/iou": iou_gt(projverts, src, tgt)[0],
                    "gradient/norm": verts.grad.norm(dim=1).mean()
                    },
                    step = i + step_offset
                )

            if self.verbose:
                print(f"{i:4d} Loss: {colour}{loss.item():.3f}{bcolors.ENDC} Volume: {calculate_volume(projverts[0], src[0].faces_packed()).item():.3f} Chamfer: {loss_dict['chamfer'].item():.3f} Penalty: {penalty_loss:.3f} Tgt: {target_volume} VolErr: {loss_dict['vol_error']:.3f}")
                print(f"GT Chamfer: [{', '.join(f'{x:.3f}' for x in chamfer_gt(tmp_mesh, tgt))}] "
                    f"GT IoU: [{', '.join(f'{x:.3f}' for x in iou_gt(projverts, src, tgt))}]")
            def should_log(i):
                return i < 50 or (i % self.vis_freq == self.vis_freq-1)  # True at i = 1, 2, 4, 8, 16, ...
            if self.vis_enabled and should_log(i):
                projectionplot = plot_projections(
                    projverts.detach().squeeze().double(),
                    gt_projmats,
                    gt_edgemap_info,
                    boundary_points=boundary_pts[0],  # B=1 case
                    hulls=hulls[0],
                    loops=loops[0]
                )
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

    def get_gt_mesh(self, name):
        return self.data_loader.get_gt_mesh(name)

    def get_mesh(self, name):
        return self.data_loader.get_mesh(name)

    def get_camera_matrices(self):
        return self.data_loader.load_camera_matrices()

    def get_projmats_and_edgemap_info(self, mesh_name, device=torch.device("cpu")):
        view_conf = self.views_config[mesh_name]
        edgemap_opts = self.data_loader.edgemap_options[mesh_name]
        renders = self.data_loader.load_renders(mesh_name)
        print("Loaded renders:", list(renders.keys()))

        full_idx = view_conf["view_names"]
        view_names = (
            full_idx if view_conf["mode"] == "manual"
            else random.sample(full_idx, view_conf["num_views"])
        )
        print(f"{view_names} from view_idx: {view_conf['view_names']}")

        edgemaps, edgemaps_len = load_edgemaps(renders, edgemap_opts)
        camera_matrices, cam_name_to_id, _ = self.data_loader.load_camera_matrices()

        view_idx = [cam_name_to_id[name] for name in view_names]
        projmats = torch.stack([camera_matrices[cam_name_to_id[name]]["P"] for name in view_names]).to(device)
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence(
            [edgemaps[name] for name in view_names], batch_first=True, padding_value=0.0
        ).to(device)
        tgt_edgemaps_len = torch.tensor([edgemaps_len[name] for name in view_names], device=device)

        return projmats, (tgt_edgemaps, tgt_edgemaps_len), view_idx

    def get_gt_projmats_and_edgemap_info(self, mesh_name, device=torch.device("cpu")):
        view_conf = self.views_config[mesh_name]
        edgemap_opts = self.data_loader.edgemap_options[mesh_name]
        renders = self.data_loader.load_renders(mesh_name)
        edgemaps, edgemaps_len = load_edgemaps(renders, edgemap_opts)

        camera_matrices, cam_name_to_id, _ = self.data_loader.load_camera_matrices()
        # view_names = sorted(cam_name_to_id.keys())

        # Decide on view names based on mode
        full_idx = view_conf["view_names"]  # Already stringified
        view_names = (
            full_idx if view_conf["mode"] == "manual"
            else random.sample(full_idx, view_conf["num_views"])
        )
        view_idx = [cam_name_to_id[name] for name in view_names]

        projmats = torch.stack([
            camera_matrices[cam_name_to_id[name]]["P"].to(device)
            for name in view_names
        ])
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence(
            [edgemaps[name].to(device) for name in view_names],
            batch_first=True, padding_value=0.0
        )
        tgt_edgemaps_len = torch.tensor(
            [edgemaps_len[name] for name in view_names],
            device=device
        )

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
