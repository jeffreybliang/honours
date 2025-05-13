# worker.py
import warnings
import argparse
import json
from .experiments import EllipsoidExperiment  # adjust as needed
warnings.filterwarnings("ignore")

def get_base_config():
    return {
        "project": "grid_init_uniform_sampling",
        "name": "auto-run",
        "m_init": 1000,
        "seed": 42,
        "noise": 1e-2,
        "p": 1.6075,
        "initial_axes": [0.2, 0.2, 0.2],
        "initial_angles": [0, 0, 0],
        "target": {"radius": 0.2820947918, "m": 300, "sqrt_m": 30},
        "view_mode": "angles",
        "view_angles": [
            [-90.0, -90.0, 0.0], [180.0, 0.0, 90.0], [90.0, 0.0, 0.0],
            [-150.0, -35.2643897, 45.0], [-30.0, -35.2643897, -45.0],
            [150.0, 35.2643897, 45.0], [30.0, 35.2643897, -45.0],
            [150.0, 35.2643897, 45.0]
        ],
        "view_normals": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1], [-1, 1, 1]],
        "view_indices": [0, 2, 4, 6],
        "training": {"n_iters": 200, "lr": 5e-1, "momentum": 0.8},
        "verbose": False,
        "vis": {"enabled": True, "frequency": 1, "backend": "plotly"},
        "wandb": True,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--axes", nargs=3, type=float, required=True)
    parser.add_argument("--angles", nargs=3, type=float, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=5)
    parser.add_argument("--target_m", type=int)
    parser.add_argument("--target_radius", type=float)
    parser.add_argument("--m_sample", type=int)
    parser.add_argument("--vis_enabled", type=int, choices=[0, 1], default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = get_base_config()

    cfg.update({
        "problem": args.problem,
        "project": args.project,
        "name": args.name,
        "initial_axes": args.axes,
        "initial_angles": args.angles,
        "trial": args.trial,
        "alpha": args.alpha,
        "vis_enabled": bool(args.vis_enabled),
    })
    cfg["vis"] = {
        **cfg["vis"],
        "enabled": bool(int(args.vis_enabled))
    }

    if args.target_m:
        cfg["target"]["m"] = args.target_m
    if args.target_radius:
        cfg["target"]["radius"] = args.target_radius
    if args.m_sample is not None:
        cfg["m_sample"] = args.m_sample

    experiment = EllipsoidExperiment(cfg)
    experiment.run()

if __name__ == "__main__":
    main()
