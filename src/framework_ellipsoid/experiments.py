from .problems import PROBLEM_REGISTRY
import torch
import wandb
from .utils import ellipsoid_volume, ellipsoid_surface_area
from framework_ellipsoid.base_problem import BaseEllipsoidProblem

class EllipsoidExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        problem_name = cfg["problem"]
        if problem_name not in PROBLEM_REGISTRY:
            raise ValueError(f"Unknown problem type: {problem_name}")
        self.problem: BaseEllipsoidProblem = PROBLEM_REGISTRY[problem_name](cfg)

    def run(self):
        x = self.problem.generate_data()
        node = self.problem.get_node()
        loss_fn = self.problem.get_loss()

        x.requires_grad = True
        y_init = node.solve(x)[0]

        run = wandb.init(
            project=self.cfg.get("project", "ellipsoid-exp"),
            name=self.cfg.get("name", f"{self.cfg['problem']}-{self.cfg.get('target', {}).get('m', 'unknown')}"),
            config=self.cfg,    
            reinit=True,
            mode="offline" if self.cfg.get("offline", False) else "online"
        )
        wandb.define_metric("outer/loss", summary="min")
        wandb.define_metric("inner/loss", summary="min")


        optimiser = torch.optim.SGD([x], lr=self.cfg["lr"], momentum=self.cfg.get("momentum", 0.0))
        for i in range(self.cfg["n_iters"]):
            optimiser.zero_grad()
            y = self.problem.wrap_node_function(node, x)
            outer_loss = loss_fn(y)
            outer_loss.backward()
            optimiser.step()

            wandb.log({"outer/loss": outer_loss.item()}, step=i)

        if self.cfg.get("verbose", False):
            y_detached = y.detach().squeeze()
            a_hat = y_detached[0].item()
            b_hat = y_detached[1].item()
            c_hat = y_detached[2].item()

            if y_detached.numel() >= 6:  # check if angles exist
                yaw_hat, pitch_hat, roll_hat = torch.rad2deg(y_detached[3:6]).tolist()
                print(f"{i:5d} ellipsoid estimate ({a_hat:0.3}, {b_hat:0.3}, {c_hat:0.3}, "
                    f"{yaw_hat:0.4}°, {pitch_hat:0.4}°, {roll_hat:0.4}°) "
                    f"has volume {ellipsoid_volume(a_hat, b_hat, c_hat):0.3} and "
                    f"surface area {ellipsoid_surface_area(a_hat, b_hat, c_hat, self.problem.p):0.5}. "
                    f"LR {optimiser.param_groups[0]['lr']}")
            else:
                print(f"{i:5d} ellipsoid estimate ({a_hat:0.3}, {b_hat:0.3}, {c_hat:0.3}) "
                    f"has volume {ellipsoid_volume(a_hat, b_hat, c_hat):0.3} and "
                    f"surface area {ellipsoid_surface_area(a_hat, b_hat, c_hat, self.problem.p):0.5}. "
                    f"LR {optimiser.param_groups[0]['lr']}")
        run.finish()