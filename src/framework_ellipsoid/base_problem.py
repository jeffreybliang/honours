import abc

class BaseEllipsoidProblem(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg

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
