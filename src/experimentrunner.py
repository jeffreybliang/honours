from .dataloader import DataLoader
import json
from .node import *
from .chamfer import *
from .utils import *
from typing import List, Tuple
import random

class ExperimentRunner:
    def __init__(self, config_path: str, data_loader: DataLoader) -> None:
    # Load the experiment config from a JSON file
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
        
        # Store the experiment details
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
        
        # Training settings
        self.n_iters = self.cfg["training"]["n_iters"]
        self.lr = self.cfg["training"]["lr"]
        self.momentum = self.cfg["training"]["momentum"]
        
        # Use the DataLoader to load data
        self.data_loader = data_loader

    def run(self):
        return self.pipeline(self.src_mesh, self.target_meshes)

    def pipeline(self, src_name: str, tgt_names: List[str]):
        """Pipeline function to run the experiment on a source mesh and a list of target meshes."""
        
        src_mesh = self.get_mesh(src_name)
        for tgt_name in tgt_names:
            tgt_mesh = self.get_mesh(tgt_name)

            # Select the view points and get corresponding projmats
            projmats, tgt_edgemap_info, view_idx = self.get_projmats_and_edgemap_info(tgt_name)
            batch_info = ([tgt_edgemap_info[0]], [tgt_edgemap_info[1]])

            # Train the mesh transformation
            final_verts = self.train_loop(src_mesh, tgt_mesh, projmats, batch_info, n_iters=self.n_iters, lr=self.lr, moment=self.momentum)

            # Update source mesh for warmstarting
            src_mesh = Meshes(verts=final_verts, faces=tgt_mesh.faces_padded())

            # Visualization step (optional)
            visualise_meshes(src_mesh, tgt_mesh)

    def train_loop(self, src: Meshes, tgt: Meshes, projmats, edgemap_info, n_iters, lr, moment, verbose=True):
        node = ConstrainedProjectionNode(src, tgt)
        verts_init = src.verts_padded() # (B, max Vi, 3)
        verts_init.requires_grad = True
        projverts_init = node.solve(verts_init)
        # apply solve
        projverts_init = ConstrainedProjectionFunction.apply(node, verts_init) # (B, max Vi, 3)
        chamfer_loss = PyTorchChamferLoss(src, tgt, projmats, edgemap_info)
        history = [projverts_init]
        verts = verts_init.clone().detach().requiresgrad(True)
        optimiser = torch.optim.SGD([verts], lr=lr, momentum=moment)
        a,b = edgemap_info
        a,b = a[0], b[0]
        plot_projections(verts.detach().squeeze().double(), projmats, (a,b))
        min_loss = float("inf")
        best_verts = None
        for i in range(n_iters):
            optimiser.zero_grad(set_to_none=True)
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
            if verbose:
                print(f"{i:4d} Loss: {colour}{loss.item():.3f}{bcolors.ENDC} Volume: {calculate_volume(projverts[0], src[0].faces_packed()):.3f}")
                print(f"GT Chamfer: [{', '.join(f'{x:.3f}' for x in chamfer_gt(projverts, src, tgt))}] "
                    f"GT SSE: [{', '.join(f'{x:.3f}' for x in sse_gt(projverts, src, tgt))}] "
                    f"GT IoU: [{', '.join(f'{x:.3f}' for x in iou_gt(projverts, src, tgt))}]")
                if i % 50 == 49:
                    plot_projections(projverts.detach().squeeze().double(), projmats, (a,b))
        return best_verts

    def get_mesh(self, name):
        return self.data_loader.meshes[name]
    
    def get_edgemaps(self, name):
        dataloader = self.data_loader
        return dataloader.edgemaps[name], dataloader.edgemaps_len[name]

    def get_camera_matrices(self, name):
        return self.data_loader.camera_matrices[name]

    def get_projmats_and_edgemap_info(self, mesh_name):
        camera_matrices = self.get_camera_matrices(mesh_name)
        edgemaps, edgemaps_len = self.get_edgemaps(mesh_name)

        if self.views_config[mesh_name]["mode"] == "manual":
            view_idx = self.views_config[mesh_name]["view_idx"]
        else: # random
            num_views = self.views_config[mesh_name]["num_views"]
            view_idx = random.sample(list(camera_matrices.keys()), num_views)

        projmats = torch.stack([camera_matrices[idx]["P"] for idx in view_idx])
        tgt_edgemaps = torch.nn.utils.rnn.pad_sequence([edgemaps[i] for i in view_idx], batch_first=True, padding_value=0.0)
        tgt_edgemaps_len = torch.tensor([edgemaps_len[i] for i in view_idx])

        return projmats, (tgt_edgemaps, tgt_edgemaps_len), view_idx
