import torch
import torch.nn as nn
import numpy as np
import scipy.optimize as opt
from ddn.pytorch.node import EqConstDeclarativeNode, DeclarativeFunction
import math
import wandb
from framework_ellipsoid.functions import *


class EllipseConstrainedProjectionFunction(DeclarativeFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class UnitSAConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m, wandbBool, p=1.6075):
        super().__init__(eps=1e-4)
        self.m = m
        self.n = (3 * m,)
        self.p = p
        self.wandb = wandbBool

    def objective(self, xs, y):
        n_batches = xs.size(0)
        data = xs.view(n_batches, 3, -1)  # reshape to (n_batches, 3, m)
        A = torch.transpose(data, 1, 2).pow(2)  # A has shape [n_batches, m, 3]        
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
        data = xs.view(n_batches, 3, -1)
        A_np = data.detach().cpu().numpy().transpose(0, 2, 1)**2
        b_np = np.ones((A_np.shape[1]))
        results = torch.zeros(n_batches, 3, dtype=torch.double)

        for b in range(n_batches):
            u0 = np.linalg.lstsq(A_np[b], b_np,rcond=None)[0]
            def constraint(u):
                if (np.any(u < 0.0)):
                    return np.array([1]) 
                else:
                    return 4 * math.pi * (1/3 * (math.sqrt(u[0])**self.p + math.sqrt(u[1])**self.p + math.sqrt(u[2])**self.p))**(1/self.p) - math.sqrt(u[0]*u[1]*u[2])
            cons = {'type': 'eq', 'fun': lambda u: constraint(u)}
            res = opt.minimize(lambda u: np.sum((A_np[b] @ u - b_np)**2), u0, method='SLSQP', constraints=[cons], options={'ftol': 1e-9, 'disp': False})
            results[b] = torch.tensor(res.x, dtype=torch.double, requires_grad=True)
            if self.wandb:
                wandb.log({"inner/loss": res.fun}, commit=False)

        return results, None

class NAAUnitSAConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m, wandbBool, p=1.6075):
        super().__init__(eps=1e-4)
        self.m = m
        self.n = (3 * m,)
        self.p = p
        self.wandb = wandbBool

    def objective(self, xs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_batches = xs.size(0)
        data = xs.view(n_batches, 3, -1)
        y = y.view(n_batches, 6)
        semiaxes = y[:, :3]
        L_diag = 1 / semiaxes ** 2
        L = torch.diag_embed(L_diag)

        angles = y[:, 3:]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

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

    def equality_constraints(self, xs, y):
        p = self.p
        # # inverse squared method
        # u = 1/y[:,:3]**2
        # u1 = u[:,0]
        # u2 = u[:,1]
        # u3 = u[:,2]
        # # Compute the components of the constraint
        # term1 = (1 / torch.sqrt(u1 * u2)) ** p
        # term2 = (1 / torch.sqrt(u1 * u3)) ** p
        # term3 = (1 / torch.sqrt(u2 * u3)) ** p
        # # Calculate the average and constraint value
        # average = (1/3) * (term1 + term2 + term3)
        # constraint_val = 4 * math.pi * (average ** (1 / p)) - 1      
  
        # normal method
        a_p = y[:,0]**p
        b_p = y[:,1]**p
        c_p = y[:,2]**p
        constraint_val = 4 * torch.pi * (1/3 * (a_p*b_p + a_p*c_p + b_p*c_p))**(1/p) - 1
        return constraint_val
    

    def solve(self, xs: torch.Tensor, method="bb", with_jac=True):
        n_batches = xs.size(0)
        results = torch.zeros(n_batches, 6, dtype=torch.double)

        for b in range(n_batches):
            X = xs[b].view(3, -1).detach().cpu().numpy()
            u0 = initialise_u(X, method)

            eq_const = {
                'type': 'eq',
                'fun': lambda u: sa_constraint_function(u).cpu().numpy()
            }
            if with_jac:
                eq_const['jac'] = lambda u: sa_constraint_function_grad(u).numpy()

            ineq_const = {
                'type': 'ineq',
                'fun': lambda u: np.array([np.pi - u[3], np.pi - u[4], np.pi - u[5], u[3], u[4], u[5]])
            }

            res = opt.minimize(
                lambda u: objective_function(u, X).cpu().numpy(),
                u0,
                jac=(lambda u: objective_function_grad(u, X).numpy()) if with_jac else None,
                constraints=[eq_const, ineq_const],
                method='SLSQP',
                options={'ftol': 1e-9, 'disp': False, 'maxiter': 200}
            )

            if not res.success:
                print("SOLVE failed:", res.message)

            results[b] = torch.tensor(res.x, dtype=torch.double, requires_grad=True)
            if self.wandb:
                wandb.log({"inner/loss": res.fun}, commit=False)

        return results, None


class UnitVolConstrainedProjectionNode(EqConstDeclarativeNode):
    def __init__(self, m, wandbBool, p=1.6075):
        super().__init__(eps=1e-6)
        self.m = m
        self.n = (3 * m,)
        self.u_prev = None
        self.wandb = wandbBool
        self.p=p

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
            if self.wandb:
                wandb.log({"inner/loss": res.fun}, commit=False)

        return results, None
