import torch
import torch.nn as nn
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from .innernodes import UnitVolConstrainedProjectionNode, EllipseConstrainedProjectionFunction

class SurfaceAreaLoss(nn.Module):
    def __init__(self, p=1.6075):
        super().__init__()
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a, b, c = input[:, 0], input[:, 1], input[:, 2]
        a_p = a ** self.p
        b_p = b ** self.p
        c_p = c ** self.p
        return 4 * torch.pi * ((1/3 * (a_p * b_p + a_p * c_p + b_p * c_p)) ** (1/self.p))


class VolumeConstrainedProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)
        initial_angles_deg = cfg.get("initial_angles", [0, 0, 0])
        self.initial_angles = torch.deg2rad(initial_angles_deg)

    def generate_data(self):
        from framework_ellipsoid.utils import sample_ellipsoid_surface
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return UnitVolConstrainedProjectionNode(self.m)

    def get_loss(self):
        return SurfaceAreaLoss(p=self.p)

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
