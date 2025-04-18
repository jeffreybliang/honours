from .dataloader import DataLoader
import json
from .node import *
from .chamfer import *
from .utils import *
from .io import *
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
                "view_idx": view_config["view_idx"],
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
        edgemap_options = {k: v for k,v in data_loader.edgemap_options.items() if k in self.target_meshes}
        self.edgemaps, self.edgemaps_len = load_edgemaps(data_loader.renders, edgemap_options)


    def run(self):
        return self.pipeline(self.src_mesh, self.target_meshes)

    def pipeline(self, src_name: str, tgt_names: List[str]):
        """Pipeline function to run the experiment on a source mesh and a list of target meshes."""
        
        run = wandb.init(
            project=self.project,
            name=self.experiment_name,
            notes=self.experiment_description,
            group="clean",
            config={
                "iters":    self.n_iters,
                "lr":       self.lr,
                "momentum": self.momentum,
                "source": src_name,
                "targets": tgt_names[0] if len(tgt_names) == 1 else tgt_names,
                "num_views": self.num_views,
            }
        )
        wandb.define_metric("outer/chamfer", summary="min")
        wandb.define_metric("outer/gt/chamfer", summary="min")
        wandb.define_metric("outer/gt/sse", summary="min")
        wandb.define_metric("outer/gt/iou", summary="max")

        view_idxs = {}
        step_offset = 0

        src_mesh = self.get_mesh(src_name)
        for tgt_name in tgt_names:
            tgt_mesh = self.get_gt_mesh(tgt_name)

            # Select the view points and get corresponding projmats
            projmats, tgt_edgemap_info, view_idx = self.get_projmats_and_edgemap_info(tgt_name)
            view_idxs[tgt_name] = view_idx
            edgemap_info = ([tgt_edgemap_info[0]], [tgt_edgemap_info[1]])

            # Train the mesh transformation
            final_verts = self.train_loop(src_mesh, tgt_mesh, projmats, edgemap_info, n_iters=self.n_iters, step_offset = step_offset, lr=self.lr, moment=self.momentum)
            
            # update step offset
            step_offset += self.n_iters

            # Update source mesh for warmstarting
            src_mesh = Meshes(verts=final_verts, faces=tgt_mesh.faces_padded())
            src_name = tgt_name
            # Visualization step (optional)
            if self.vis_enabled:
                visualise_meshes(src_mesh, tgt_mesh)
        run.config["view_idxs"] = view_idxs
        run.finish()

    def train_loop(self, src: Meshes, tgt: Meshes, projmats, edgemap_info, n_iters, step_offset, lr, moment, verbose=False):
        node = ConstrainedProjectionNode(src)
        verts_init = src.verts_padded() # (B, max Vi, 3)
        verts_init.requires_grad = True
        projverts_init = ConstrainedProjectionFunction.apply(node, verts_init) # (B, max Vi, 3)
        chamfer_loss = PyTorchChamferLoss(src, tgt, projmats, edgemap_info)
        history = [projverts_init]
        verts = verts_init.clone().detach().requires_grad_(True)
        optimiser = torch.optim.SGD([verts], lr=lr, momentum=moment)
        a,b = edgemap_info
        a,b = a[0], b[0]
        # plot_projections(verts.detach().squeeze().double(), projmats, (a,b))
        min_loss = float("inf")
        best_verts = None
    
        pbar = trange(n_iters, desc="Training", leave=True)  # Always visible
        for i in pbar:
            optimiser.zero_grad(set_to_none=True)
            node.iter = i + step_offset
            projverts = ConstrainedProjectionFunction.apply(node, verts)
            history.append(projverts.detach().clone())
            loss = chamfer_loss(projverts)
            colour = bcolors.FAIL
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_verts = projverts.detach().clone()
                colour = bcolors.OKGREEN
            loss.backward()
            optimiser.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")

            # log
            wandb.log(
                data = {
                "outer/chamfer": loss,
                "outer/vol": calculate_volume(projverts[0], src[0].faces_packed()),
                "outer/gt/chamfer": chamfer_gt(projverts, src, tgt)[0],
                "outer/gt/sse": sse_gt(projverts, src, tgt)[0],
                "outer/gt/iou": iou_gt(projverts, src, tgt)[0]
                },
                step = i + step_offset
            )

            if self.verbose:
                print(f"{i:4d} Loss: {colour}{loss.item():.3f}{bcolors.ENDC} Volume: {calculate_volume(projverts[0], src[0].faces_packed()):.3f}")
                print(f"GT Chamfer: [{', '.join(f'{x:.3f}' for x in chamfer_gt(projverts, src, tgt))}] "
                    f"GT SSE: [{', '.join(f'{x:.3f}' for x in sse_gt(projverts, src, tgt))}] "
                    f"GT IoU: [{', '.join(f'{x:.3f}' for x in iou_gt(projverts, src, tgt))}]")
                if self.vis_enabled and i % self.vis_freq == self.vis_freq-1:
                    plot_projections(projverts.detach().squeeze().double(), projmats, (a,b))
        return best_verts

    def get_gt_mesh(self,name):
        return self.data_loader.gt_meshes[name]

    def get_mesh(self, name):
        return self.data_loader.meshes[name]
    
    def get_edgemaps(self, name):
        return self.edgemaps[name], self.edgemaps_len[name]

    def get_camera_matrices(self):
        return self.data_loader.camera_matrices

    def get_projmats_and_edgemap_info(self, mesh_name):
        camera_matrices = self.get_camera_matrices()
        edgemaps, edgemaps_len = self.get_edgemaps(mesh_name)

        if self.views_config[mesh_name]["mode"] == "manual":
            view_idx = self.views_config[mesh_name]["view_idx"]
        else: # random
            num_views = self.views_config[mesh_name]["num_views"]
            view_idx = random.sample(self.views_config[mesh_name]["view_idx"], num_views)
        print(f"{view_idx} from {self.views_config[mesh_name]['view_idx']}")
        projmats = torch.stack([camera_matrices[idx]["P"] for idx in view_idx])
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence([edgemaps[i] for i in view_idx], batch_first=True, padding_value=0.0)
        tgt_edgemaps_len = torch.tensor([edgemaps_len[i] for i in view_idx])

        return projmats, (tgt_edgemaps, tgt_edgemaps_len), view_idx
