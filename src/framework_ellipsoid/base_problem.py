import torch
import abc

class BaseEllipsoidProblem(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sqrt_m = cfg["sqrt_m"]
        self.seed = cfg["seed"]
        torch.seed = self.seed
        self.nu = cfg.get("noise", 1e-4)
        self.p = cfg.get("p", 1.6075)
        self.initial_axes = cfg["initial_axes"]  # [a, b, c]
        self.m = self.sqrt_m * self.sqrt_m

    @abc.abstractmethod
    def generate_data(self):
        ...

    @abc.abstractmethod
    def get_node(self):
        ...

    @abc.abstractmethod
    def get_loss(self):
        ...

    def wrap_node_function(self, node, x):
        """Override this to use DeclarativeFunction.apply(node, x) if needed"""
        return node.solve(x)[0]
