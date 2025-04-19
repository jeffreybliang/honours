import torch
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.utils import sample_ellipsoid_surface,build_view_matrices
from framework_ellipsoid.loss import SampledProjectionChamferLoss,build_target_circle 
from .innernodes import UnitVolConstrainedProjectionNode, EllipseConstrainedProjectionFunction

class ChamferSampledProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rot_mats = build_view_matrices(cfg)
        target_cfg = cfg["target"]
        target_pts = build_target_circle(target_cfg["radius"], target_cfg["m"]).expand(self.rot_mats.size(0), -1, -1)
        self.views = (self.rot_mats, target_pts)
        initial_angles_deg = cfg.get("initial_angles", [0, 0, 0])
        self.initial_angles = torch.deg2rad(initial_angles_deg)

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return UnitVolConstrainedProjectionNode(self.m)

    def get_loss(self):
        return SampledProjectionChamferLoss(self.views, m=self.cfg["chamfer_pts"])

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
