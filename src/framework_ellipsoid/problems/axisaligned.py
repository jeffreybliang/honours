import wandb
import torch
import math
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.utils import sample_ellipsoid_surface
from ddn.pytorch.node import EqConstDeclarativeNode, DeclarativeFunction
import torch.nn as nn
import scipy.optimize as opt
import numpy as np

class EllipsoidConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m, p=1.6075):
        super().__init__(eps=1e-4)
        self.m = m
        self.n = (3 * m,)
        self.p = p

    def objective(self, xs, y):
        data = xs.view(-1, 3, self.m).transpose(1, 2).pow(2)
        A = data  # shape: (B, m, 3)
        b = torch.ones((A.shape[0], A.shape[1], 1), dtype=torch.double, device=xs.device)
        y = y.view(-1, 3, 1)
        return torch.sum((A @ y - b).pow(2), dim=(1, 2))

    def equality_constraints(self, xs, y):
        p = self.p
        sqrt_y = torch.sqrt(y)
        u1, u2, u3 = sqrt_y[:, 0], sqrt_y[:, 1], sqrt_y[:, 2]
        avg = (u1**p + u2**p + u3**p) / 3
        return 4 * math.pi * avg**(1/p) - u1*u2*u3

    def solve(self, xs):
        n_batches = xs.size(0)
        data = xs.view(n_batches, 3, self.m)
        A_np = data.detach().cpu().numpy().transpose(0, 2, 1)**2
        b_np = np.ones((self.m,))
        results = torch.zeros(n_batches, 3, dtype=torch.double)

        for b in range(n_batches):
            u0 = np.linalg.lstsq(A_np[b], b_np, rcond=None)[0]
            def constraint(u):
                return 4 * math.pi * (1/3 * (math.sqrt(u[0])**self.p + math.sqrt(u[1])**self.p + math.sqrt(u[2])**self.p))**(1/self.p) - math.sqrt(u[0]*u[1]*u[2])
            cons = {'type': 'eq', 'fun': constraint}
            res = opt.minimize(lambda u: np.sum((A_np[b] @ u - b_np)**2), u0, constraints=[cons])
            results[b] = torch.tensor(res.x, dtype=torch.double, requires_grad=True)
            wandb.log({"inner/loss": res.fun}, commit=False)

        return results, None


class EllipseConstrainedProjectionFunction(DeclarativeFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SqrtProductLoss(nn.Module):
    def forward(self, input):
        return torch.sqrt(input[:, 0] * input[:, 1] * input[:, 2])


class AxisAlignedProblem(BaseEllipsoidProblem):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sqrt_m = cfg["sqrt_m"]
        self.nu = cfg.get("noise", 1e-4)
        self.p = cfg.get("p", 1.6075)
        self.initial_axes = cfg["initial_axes"]  # [a, b, c]
        self.m = self.sqrt_m * self.sqrt_m

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw = pitch = roll = 0.0
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return EllipsoidConstrainedProjectionNode(self.m, self.p)

    def get_loss(self):
        return SqrtProductLoss()

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
