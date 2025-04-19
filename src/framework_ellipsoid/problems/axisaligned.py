import torch
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.utils import sample_ellipsoid_surface
import torch.nn as nn
from .innernodes import UnitSAConstrainedProjectionNode, EllipseConstrainedProjectionFunction


class SqrtProductLoss(nn.Module):
    def forward(self, input):
        return torch.sqrt(input[:, 0] * input[:, 1] * input[:, 2])


class AxisAlignedProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw = pitch = roll = 0.0
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return UnitSAConstrainedProjectionNode(self.m, self.p)

    def get_loss(self):
        return SqrtProductLoss()

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
