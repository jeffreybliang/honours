import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as opt
from ddn.pytorch.node import EqConstDeclarativeNode, DeclarativeFunction
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from framework_ellipsoid.functions import (
    objective_function,
    objective_function_grad,
    vol_constraint_function,
    vol_constraint_function_grad,
    initialise_u
)
import wandb

class EllipsoidConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m):
        super().__init__(eps=1e-4)
        self.m = m
        self.n = (3 * m,)

    def objective(self, xs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_batches = xs.size(0)
        data = xs.view(n_batches, 3, self.m)
        y = y.view(n_batches, 6)
        semiaxes = y[:, :3]
        angles = y[:, 3:]
        L_diag = 1 / semiaxes ** 2
        L = torch.diag_embed(L_diag)

        angles_rad = torch.deg2rad(angles)
        cos = torch.cos(angles_rad)
        sin = torch.sin(angles_rad)

        R = torch.zeros((n_batches, 3, 3), dtype=torch.double, device=xs.device)
        cy, cp, cr = cos[:, 0], cos[:, 1], cos[:, 2]
        sy, sp, sr = sin[:, 0], sin[:, 1], sin[:, 2]

        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr

        A = torch.bmm(R, torch.bmm(L, R.transpose(1, 2)))
        XT_AX = torch.einsum('bji,bjk,bki->bi', data, A, data)
        return ((XT_AX - 1) ** 2).sum(dim=1)

    def equality_constraints(self, xs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a, b, c = y[:, 0], y[:, 1], y[:, 2]
        return 4/3 * torch.pi * a * b * c - 1

    def solve(self, xs: torch.Tensor, method="pca", with_jac=True):
        n_batches = xs.size(0)
        results = torch.zeros(n_batches, 6, dtype=torch.double)

        for b in range(n_batches):
            X = xs[b].view(3, -1).detach().cpu().numpy()
            u0 = initialise_u(X, method)

            eq_const = {
                'type': 'eq',
                'fun': lambda u: vol_constraint_function(u).item()
            }
            if with_jac:
                eq_const['jac'] = lambda u: vol_constraint_function_grad(u).numpy()

            ineq_const = {
                'type': 'ineq',
                'fun': lambda u: np.array([np.pi - u[3], np.pi - u[4], np.pi - u[5], u[3], u[4], u[5]])
            }

            res = opt.minimize(
                lambda u: objective_function(u, X).item(),
                u0,
                jac=(lambda u: objective_function_grad(u, X).numpy()) if with_jac else None,
                constraints=[eq_const, ineq_const],
                method='SLSQP',
                options={'ftol': 1e-9, 'disp': False, 'maxiter': 200}
            )

            if not res.success:
                print("SOLVE failed:", res.message)

            results[b] = torch.tensor(res.x, dtype=torch.double, requires_grad=True)
            wandb.log({"inner/loss": res.fun}, commit=False)

        return results, None


class EllipseConstrainedProjectionFunction(DeclarativeFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        self.sqrt_m = cfg["sqrt_m"]
        self.nu = cfg.get("noise", 1e-4)
        self.p = cfg.get("p", 1.6075)
        self.initial_axes = cfg["initial_axes"]
        self.initial_angles = cfg.get("initial_angles", [0, 0, 0])
        self.m = self.sqrt_m * self.sqrt_m

    def generate_data(self):
        from framework_ellipsoid.utils import sample_ellipsoid_surface
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return EllipsoidConstrainedProjectionNode(self.m)

    def get_loss(self):
        return SurfaceAreaLoss(p=self.p)

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
