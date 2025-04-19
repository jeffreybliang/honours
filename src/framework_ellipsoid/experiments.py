from .problems import PROBLEM_REGISTRY
import torch
import wandb
from .utils import ellipsoid_volume, ellipsoid_surface_area
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from .plotting import *
from .problems import ChamferBoundaryProblem, ChamferSampledProblem

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

        vis_cfg = self.cfg.get("vis", {})
        vis_enabled = vis_cfg.get("enabled", False)
        vis_freq = vis_cfg.get("frequency", 10)
        vis_backend = vis_cfg.get("backend", "mpl")


        optimiser = torch.optim.SGD([x], lr=self.cfg["lr"], momentum=self.cfg.get("momentum", 0.0))
        for i in range(self.cfg["n_iters"]):
            optimiser.zero_grad()
            y = self.problem.wrap_node_function(node, x)
            res = loss_fn(y)
            outer_loss = res.sum()
            outer_loss.backward()
            optimiser.step()

            wandb.log({
                "outer/loss": outer_loss.item(),
                "chamfer/mean": res.mean().item(),
                **{f"chamfer/e{j+1}": res[j].item() for j in range(res.numel())}
            }, step=i)

            y_detached = y.detach().squeeze()
            a_hat = y_detached[0].item()
            b_hat = y_detached[1].item()
            c_hat = y_detached[2].item()
            vol = ellipsoid_volume(a_hat, b_hat, c_hat)
            sa = ellipsoid_surface_area(a_hat, b_hat, c_hat, self.problem.p)
            wandb.log({
                "geometry/a": a_hat,
                "geometry/b": b_hat,
                "geometry/c": c_hat,
                "geometry/volume": vol,
                "geometry/surface_area": sa
            }, step=i)

            r = self.cfg["target"]["radius"]
            V_gt = ellipsoid_volume(r,r,r)
            S_gt = ellipsoid_surface_area(r,r,r)
            wandb.log({
                "gt/axes_L2": ((a_hat - r)**2 + (b_hat - r)**2 + (c_hat - r)**2),
                "gt/axes_rel_L2": (((a_hat - r)/r)**2 + ((b_hat - r)/r)**2 + ((c_hat - r)/r)**2),
                "gt/a" : (a_hat - r)**2,
                "gt/b" : (b_hat - r)**2,
                "gt/c" : (c_hat - r)**2,
                "gt/volume_abs": abs(vol - V_gt),
                "gt/volume_rel": abs(vol - V_gt) / V_gt,
                "gt/surface_area_abs": abs(sa - S_gt),
                "gt/surface_area_rel": abs(sa - S_gt) / S_gt
            })

            if y_detached.numel() >= 6:  # check if angles exist
                yaw_hat, pitch_hat, roll_hat = torch.rad2deg(y_detached[3:6]).tolist()
                wandb.log({
                    "geometry/yaw": yaw_hat,
                    "geometry/pitch": pitch_hat,
                    "geometry/roll": roll_hat
                }, step=i)

            if vis_enabled and i % vis_freq == 0:
                if vis_backend == "matplotlib":
                    fig = plot_ellipsoid_mpl(
                        a_hat, b_hat, c_hat,
                        yaw_hat, pitch_hat, roll_hat,
                        x[0].detach().cpu(),
                        r=self.cfg["target"]["radius"]
                    )
                    wandb.log({f"vis/ellipsoid": wandb.Image(fig)}, step=i)
                    plt.close(fig)

                elif vis_backend == "plotly":
                    fig = plot_ellipsoid_plotly(
                        a_hat, b_hat, c_hat,
                        yaw_hat, pitch_hat, roll_hat,
                        x[0].detach().cpu(),
                        r=self.cfg["target"]["radius"]
                    )
                    wandb.log({f"vis/ellipsoid": fig}, step=i)

                if isinstance(self.problem, (ChamferSampledProblem, ChamferBoundaryProblem)):
                    silhouette = getattr(self.problem.get_loss(), "plot_silhouettes", None)
                    if callable(silhouette):
                        fig = silhouette(step=i)
                        wandb.log({
                            "vis/silhouettes": wandb.Image(fig)
                        },
                        step=i)
                        plt.close(fig)



            if self.cfg.get("verbose", False):
                if y_detached.numel() >= 6:  # check if angles exist
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