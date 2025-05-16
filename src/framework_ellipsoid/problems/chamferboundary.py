import torch
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.utils import sample_ellipsoid_surface,build_view_matrices
from .innernodes import UnitVolConstrainedProjectionNode, EllipseConstrainedProjectionFunction
from framework_ellipsoid.loss import BoundaryProjectionChamferLoss,build_target_circle 
from framework_ellipsoid.utils import ellipsoid_surface_area, ellipsoid_volume

class ChamferBoundaryProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rot_mats = build_view_matrices(cfg)

        target_cfg = cfg.get("target", {})
        radius = target_cfg.get("radius", 1.0)
        m_pts = target_cfg.get("m", 50)
        self.m_sampled = cfg.get("m_sample", 900)

        target_pts = build_target_circle(radius, m_pts).expand(self.rot_mats.size(0), -1, -1)
        self.views = (self.rot_mats, target_pts)

        initial_angles_deg = torch.tensor(cfg.get("initial_angles", [0, 0, 0]), dtype=torch.double)
        self.initial_angles = torch.deg2rad(initial_angles_deg)
        self.m_pts = m_pts

        self.alpha = cfg.get("alpha", 5.0)

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        print(f"Initial volume is {ellipsoid_volume(a,b,c)} and initial surface area {ellipsoid_surface_area(a,b,c)}")
        return sample_ellipsoid_surface(self.m_sampled, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return UnitVolConstrainedProjectionNode(self.m, self.wandb)

    def get_loss(self):
        return BoundaryProjectionChamferLoss(self.views, m_sampled=self.m_sampled, alpha=self.alpha)

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
