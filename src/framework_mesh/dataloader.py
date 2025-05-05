import json
import os
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from .io import *

class DataLoader:
    def __init__(self, config) -> None:

        if isinstance(config, str) and os.path.exists(config):
            with open(config, "r") as f:
                self.cfg = json.load(f)
        elif isinstance(config, dict):
            self.cfg = config
        else:
            raise ValueError("Invalid experiment_config. Must be a path or dict.")

        # Set up paths and other config settings
        self.mesh_dir = self.cfg["paths"]["mesh_dir"]
        self.mesh_res = self.cfg["paths"]["mesh_res"]
        self.renders_path = self.cfg["paths"]["renders_path"]
        self.material = self.cfg["paths"]["material"]
        self.materials_path = os.path.join(self.renders_path, self.material)
        self.matrices_path = self.cfg["paths"]["matrices_path"]
        
        self.edgemap_options = {
            mesh["name"]: mesh.get("edgemap_options", {})
            for mesh in self.cfg["meshes"]
        }
        # self.camera_matrices = load_camera_matrices(path=self.matrices_path, matrix_types="P")
        # self.renders = load_renders(self.renders_path)

        # Load meshes using PyTorch3D
        # self.meshes, self.gt_meshes = self.load_meshes()

    def load_renders(self, objname):
        return load_renders(self.materials_path, objname)
    
    def load_camera_matrices(self):
        return load_camera_matrices(self.matrices_path, matrix_types="P")
        # return cam_id_map  # {int_id: {P: ...}}



    # def load_meshes(self):
    #     """
    #     Load meshes using PyTorch3D's load_obj function.
    #     Returns a dictionary of PyTorch3D Meshes objects.
    #     """
    #     meshes = {}
    #     gt_meshes = {}
    #     for mesh_info in self.cfg["meshes"]:
    #         mesh_name = mesh_info["name"]
    #         if mesh_name == "sphere":
    #             mesh_path = os.path.join(self.mesh_dir, f"{mesh_name}_{self.mesh_res}.obj")
    #             verts, faces, aux = load_obj(mesh_path)
    #             # Create Meshes object in PyTorch3D
    #             meshes[mesh_name] = Meshes(verts=[verts], faces=[faces.verts_idx])

    #         gt_mesh_path = os.path.join(self.mesh_dir, f"{mesh_name}.obj")
    #         gt_verts, gt_faces, aux = load_obj(gt_mesh_path)
    #         gt_meshes[mesh_name] = Meshes(verts=[gt_verts], faces=[gt_faces.verts_idx])

    #     return meshes, gt_meshes
    

    def get_mesh(self, mesh_name):
        """
        Load a predicted mesh by name with resolution suffix.
        Returns a PyTorch3D Meshes object or None if file not found.
        """
        mesh_path = os.path.join(self.mesh_dir, f"{mesh_name}_{self.mesh_res}.obj")
        if not os.path.exists(mesh_path):
            return None

        verts, faces, _ = load_obj(mesh_path)
        return Meshes(verts=[verts], faces=[faces.verts_idx])

    def get_gt_mesh(self, mesh_name):
        """
        Load a ground-truth mesh by name (no resolution suffix).
        Returns a PyTorch3D Meshes object or None if file not found.
        """
        gt_path = os.path.join(self.mesh_dir, f"{mesh_name}.obj")
        if not os.path.exists(gt_path):
            return None

        verts, faces, _ = load_obj(gt_path)
        return Meshes(verts=[verts], faces=[faces.verts_idx])
