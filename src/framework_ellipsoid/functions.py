import torch
import numpy as np
from sklearn.decomposition import PCA
from .utils import rotation_matrix_3d, get_angles

def vol_constraint_function(u, p=1.6075):
    if not torch.is_tensor(u):
        u = torch.tensor(u)
    a = u[0]
    b = u[1]
    c = u[2]
    res = 4/3 * torch.pi * (a*b*c) - 1
    return res

def vol_constraint_function_grad(u):
    if torch.is_tensor(u):
        u = u.detach().clone()
    else:
        u = torch.tensor(u)
    u.requires_grad = True
    with torch.enable_grad():
        res = vol_constraint_function(u)
    constr_grad = torch.autograd.grad(res, u)[0]


def sa_constraint_function(u, p=1.6075):
    if not torch.is_tensor(u):
        u = torch.tensor(u)
    a, b, c = u[0], u[1], u[2]
    res = 4 * torch.pi * (1/3 * (a**p * b**p + a**p * c**p + b**p * c**p))**(1/p) - 1
    return res

def sa_constraint_function_grad(u):
    if torch.is_tensor(u):
        u = u.detach().clone()
    else:
        u = torch.tensor(u)
    u.requires_grad = True
    with torch.enable_grad():
        res = sa_constraint_function(u)
    return torch.autograd.grad(res, u)[0]

def objective_function(u, X, u_prev=None):
    if not torch.is_tensor(u):
        u = torch.tensor(u).double()
    if not torch.is_tensor(X):
        X = torch.tensor(X).double()
    L = torch.diag(1/u[:3]**2).double()
    R = rotation_matrix_3d(u[3:6]).double() # assumes radians
    A = R @ L @ R.T
    XT_AX = torch.einsum('ji,jk,ki->i', X, A, X)
    b = torch.ones(X.shape[1])
    if u_prev is None:
        u_prev = u 
    elif not torch.is_tensor(u_prev):   
        u_prev = torch.tensor(u_prev).double()
    res = torch.sum((XT_AX - b) ** 2) + torch.norm(u_prev - u)**2
    return res

def objective_function_grad(u, X, u_prev=None):
    if torch.is_tensor(u):
        u = u.detach().clone()
    else:
        u = torch.tensor(u)
    if torch.is_tensor(X):
        X = X.detach().clone()
    else:
        X = torch.tensor(X).double()
    if u_prev is not None:
        if torch.is_tensor(u_prev):
            u_prev = u_prev.detach().clone()
        else:
            u_prev = torch.tensor(u_prev)
    else:
        u_prev = u
    u.requires_grad = True
    with torch.enable_grad():
        res = objective_function(u, X, u_prev).double()
    obj_grad = torch.autograd.grad(res, u)[0].double()
    return obj_grad

def initialise_u(data, method="default"):
    if method == "default":
        return np.ones(6)
    elif method == "bb":
        h, w, l = get_bounding_box_dims(data) / 2
        u0 = np.zeros(6)
        u0[:3] = np.array([h, w, l])
        u0[3:] = np.random.uniform(low=0, high=90, size=3)
        return u0
    elif method == "pca":
        return pca(data)

def pca(data):
    data = data.T
    pca = PCA(n_components=3)
    pca.fit(data)
    semiaxes = np.sqrt(pca.explained_variance_ * np.array([2, 4, 4]))
    rotation = np.fliplr(pca.components_)
    angles = np.array(get_angles(rotation))
    return np.concatenate([semiaxes, angles])

def get_bounding_box_dims(points):
    min_x, min_y, min_z = np.min(points, axis=1)
    max_x, max_y, max_z = np.max(points, axis=1)
    height = max_z - min_z
    width = max_x - min_x
    length = max_y - min_y
    return np.sort([height, width, length])
