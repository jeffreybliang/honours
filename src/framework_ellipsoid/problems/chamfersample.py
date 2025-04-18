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
from framework_ellipsoid.utils import sample_ellipsoid_surface,build_view_matrices
from framework_ellipsoid.loss import SampledProjectionChamferLoss,build_target_circle 

class EllipsoidConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m):
        super().__init__(eps=1e-6)
        self.m = m
        self.n = (3 * m,)
        self.u_prev = None

    def objective(self, xs, y):
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

        # Compute b tensor for all batches
        b_ones = torch.ones_like(XT_AX)        
        if self.u_prev is None or torch.equal(y[:,:], self.u_prev):
            obj_val = torch.sum((XT_AX - b_ones).pow(2), dim=1)
        else:
            obj_val = torch.sum((XT_AX - b_ones).pow(2), dim=1) + torch.norm(y[:,:] - self.u_prev, p=2)**2
        self.u_prev = y
        return obj_val
    
    def equality_constraints(self, xs, y):
        # y is of shape [m x number of parameters] (same as u)  
        a = y[:,0]
        b = y[:,1]
        c = y[:,2]
        constraint_val = 4/3 * torch.pi * a * b * c - 1
        return constraint_val

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

class ChamferSampledProblem(BaseEllipsoidProblem):
    def __init__(self, cfg, views):
        super().__init__(cfg)
        self.rot_mats = build_view_matrices(cfg)
        target_cfg = cfg["target"]
        target_pts = build_target_circle(target_cfg["radius"], target_cfg["m"]).expand(self.rot_mats.size(0), -1, -1)
        self.views = (self.rot_mats, target_pts)
        self.sqrt_m = cfg["sqrt_m"]
        self.nu = cfg.get("noise", 1e-4)
        self.p = cfg.get("p", 1.6075)
        self.initial_axes = cfg["initial_axes"]
        self.initial_angles = cfg.get("initial_angles", [0, 0, 0])
        self.m = self.sqrt_m * self.sqrt_m

    def generate_data(self):
        a, b, c = self.initial_axes
        yaw, pitch, roll = self.initial_angles
        return sample_ellipsoid_surface(self.sqrt_m, a, b, c, yaw, pitch, roll, self.nu)

    def get_node(self):
        return EllipsoidConstrainedProjectionNode(self.m)

    def get_loss(self):
        return SampledProjectionChamferLoss(self.views, m=self.cfg["chamfer_pts"])

    def wrap_node_function(self, node, x):
        return EllipseConstrainedProjectionFunction.apply(node, x)
