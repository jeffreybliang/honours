import torch
import torch.nn as nn
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.utils import sample_ellipsoid_surface
from .innernodes import NAAUnitSAConstrainedProjectionNode, EllipseConstrainedProjectionFunction

class InverseProductLoss(nn.Module):
    def forward(self, input):
        return 1 / (input[:, 0] * input[:, 1] * input[:, 2] + 1e-8)


class NonAxisAlignedProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)
        initial_angles_deg = cfg.get("initial_angles", [0, 0, 0])
        self.initial_angles = torch.deg2rad(initial_angles_deg)
        self.m = self.sqrt_m * self.sqrt_m

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return NAAUnitSAConstrainedProjectionNode(self.m, self.p)

    def get_loss(self):
        return InverseProductLoss()

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
