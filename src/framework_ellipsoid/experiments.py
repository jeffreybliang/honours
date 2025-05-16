from .problems import PROBLEM_REGISTRY
import torch
import wandb
from .utils import ellipsoid_volume, ellipsoid_surface_area
from framework_ellipsoid.base_problem import BaseEllipsoidProblem
from .plotting import *
from .problems import ChamferBoundaryProblem, ChamferSampledProblem
from tqdm import trange
import math
import io


class EllipsoidExperiment:
    def __init__(self, cfg):
        self.cfg = cfg
        problem_name = cfg["problem"]
        self.problem_name = problem_name
        if problem_name not in PROBLEM_REGISTRY:
            raise ValueError(f"Unknown problem type: {problem_name}")
        self.problem: BaseEllipsoidProblem = PROBLEM_REGISTRY[problem_name](cfg)
        self.wandb = cfg.get("wandb", False)
        self.training = cfg["training"]
        self.vmin,self.vmax = None,None

    def run(self):
        data = self.problem.generate_data()
        n_batches = data.size(0)
        x = data.view(n_batches, -1)
        x.requires_grad = True
        node = self.problem.get_node()
        loss_fn = self.problem.get_loss()

        # y_init = node.solve(x)[0]
        
        if self.wandb:
            run = wandb.init(
                project=self.cfg.get("project", "ellipsoid-exp"),
                name=self.cfg.get("name", f"{self.cfg['problem']}-{self.cfg.get('target', {}).get('m', 'unknown')}"),
                config={
                    "problem": self.cfg["problem"],
                    # "axes_id": self.cfg["axes_id"],         # letter like "A"
                    # "rotation_id": self.cfg["rotation_id"], # letter like "R2"
                    # "trial": self.cfg["trial"],
                    **self.cfg
                },
                reinit=True,
            )
            wandb.define_metric("outer/loss", summary="min")
            wandb.define_metric("inner/loss", summary="min")

        vis_cfg = self.cfg.get("vis", {})
        vis_enabled = vis_cfg.get("enabled", False)
        vis_freq = vis_cfg.get("frequency", 10)

        optimiser = torch.optim.SGD([x], lr=self.training["lr"], momentum=self.training.get("momentum", 0.0))
        pbar = trange(self.training["n_iters"], desc="Training", leave=True)  # Always visible

        for i in pbar:
            optimiser.zero_grad()
            y = self.problem.wrap_node_function(node, x)
            res = loss_fn(y)
            outer_loss = res.mean()
            outer_loss.backward()
            wandb.log({
                "gradient_norm": torch.norm(x.grad)},
                commit=False
            )
            pbar.set_description(f"Loss: {outer_loss.item():.4f}")
            if self.wandb:
                wandb.log({
                    "outer/loss": outer_loss.item(),
                    "chamfer/mean": res.mean().item(),
                    **{f"chamfer/e{j+1}": res[j].item() for j in range(res.numel())}
                }, step=i)

            y_detached = y.detach().squeeze()
            if self.problem_name == "axisaligned":
                y_detached[:3] = 1/torch.sqrt(y_detached[:3])
            a_hat = y_detached[0].item()
            b_hat = y_detached[1].item()
            c_hat = y_detached[2].item()
            if y_detached.numel() >= 6:  # check if angles exist
                    yaw_hat, pitch_hat, roll_hat = torch.rad2deg(y_detached[3:6]).tolist()


            if self.wandb:
                self.log_geometry_and_errors(y_detached, self.cfg, i)
                if vis_enabled and i % vis_freq == 0:
                    self.log_visualisations_if_enabled(self.problem, loss_fn, self.cfg, y_detached, x, i)

            if self.cfg.get("verbose", False):
                if y_detached.numel() >= 6:  # check if angles exist
                    print(f"{i:5d} ellipsoid estimate ({a_hat:.3f}, {b_hat:.3f}, {c_hat:.3f}, "
                        f"{yaw_hat:.4f}°, {pitch_hat:.4f}°, {roll_hat:.4f}°) "
                        f"has volume {ellipsoid_volume(a_hat, b_hat, c_hat):.3f} and "
                        f"surface area {ellipsoid_surface_area(a_hat, b_hat, c_hat, self.problem.p):.5f}. "
                        f"LR {optimiser.param_groups[0]['lr']}")
                else:
                    print(f"{i:5d} ellipsoid estimate ({a_hat:.3f}, {b_hat:.3f}, {c_hat:.3f}) "
                        f"has volume {ellipsoid_volume(a_hat, b_hat, c_hat):.3f} and "
                        f"surface area {ellipsoid_surface_area(a_hat, b_hat, c_hat, self.problem.p):.5f}. "
                        f"LR {optimiser.param_groups[0]['lr']}")
                    
            optimiser.step()

        if self.wandb:
            run.finish()

    def log_geometry_and_errors(self, y_detached, cfg, step):
        a, b, c = y_detached[0:3].tolist()
        vol = ellipsoid_volume(a, b, c)
        sa = ellipsoid_surface_area(a, b, c, cfg.get("p", 1.6075))

        wandb.log({
            "geometry/a": a,
            "geometry/b": b,
            "geometry/c": c,
            "geometry/volume": vol,
            "geometry/surface_area": sa
        }, step=step)

        r = cfg["target"]["radius"]
        V_gt = ellipsoid_volume(r, r, r)
        S_gt = ellipsoid_surface_area(r, r, r)

        wandb.log({
            "gt/axes_L2": (a - r) ** 2 + (b - r) ** 2 + (c - r) ** 2,
            "gt/axes_rel_L2": ((a - r) / r) ** 2 + ((b - r) / r) ** 2 + ((c - r) / r) ** 2,
            "gt/a": (a - r) ** 2,
            "gt/b": (b - r) ** 2,
            "gt/c": (c - r) ** 2,
            "gt/volume_abs": abs(vol - V_gt),
            "gt/volume_rel": abs(vol - V_gt) / V_gt,
            "gt/surface_area_abs": abs(sa - S_gt),
            "gt/surface_area_rel": abs(sa - S_gt) / S_gt
        }, step=step)

        if y_detached.numel() >= 6:
            yaw, pitch, roll = torch.rad2deg(y_detached[3:6]).tolist()
            wandb.log({
                "geometry/yaw": yaw,
                "geometry/pitch": pitch,
                "geometry/roll": roll
            }, step=step)


    def log_visualisations_if_enabled(self, problem, loss_fn, cfg, y_detached, x, step):
        vis_cfg = cfg.get("vis", {})
        a, b, c = y_detached[0:3].tolist()
        if y_detached.numel() >= 6:
            yaw, pitch, roll = torch.rad2deg(y_detached[3:6]).tolist()
        else:
            yaw = pitch = roll = 0.0

        r = cfg["target"]["radius"]
        backend = vis_cfg.get("backend", "plotly")

        points = x[0].detach().view(3,-1)

        if backend == "mpl":
            fig,vmin,vmax = plot_ellipsoid_mpl(a, b, c, yaw, pitch, roll, points.cpu(), r=r, vmin=self.vmin, vmax=self.vmax)
            wandb.log({f"vis/ellipsoid": wandb.Image(fig)}, step=step)
            plt.close()
        elif backend == "plotly":
            fig,vmin,vmax = plot_ellipsoid_plotly(a, b, c, yaw, pitch, roll, points.cpu(), r=r, vmin=self.vmin, vmax=self.vmax)
            wandb.log({f"vis/ellipsoid": wandb.Plotly(fig)}, step=step)
        
        if self.vmin is None or self.vmax is None:
            self.vmin = vmin
            self.vmax = vmax

        if isinstance(problem, (ChamferSampledProblem, ChamferBoundaryProblem)):
            silhouette = getattr(loss_fn, "plot_silhouettes", None)
            if callable(silhouette):
                fig = silhouette(step=step)
                wandb.log({f"vis/silhouettes": wandb.Image(fig)}, step=step)
                plt.close()
