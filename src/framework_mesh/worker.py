import argparse
import json
import torch
import warnings
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner

warnings.filterwarnings("ignore")

def main(args):
    with open(args.data_path) as f:
        data_cfg = json.load(f)
    with open(args.exp_base_path) as f:
        exp_cfg = json.load(f)

    # Apply mesh resolution
    data_cfg['paths']['mesh_res'] = args.mesh_res
    dataloader = DataLoader(data_cfg, args.device)

    # Patch config
    exp_cfg.update({
        "velocity": {
            "enabled": bool(args.velocity_enabled),
            "k": args.velocity_k,
            "beta": args.velocity_beta
        },
        "gradient": {
            "smoothing": bool(args.smoothing_enabled),
            "method": "jacobi",
            "k": 5,
            "constrained": args.constrained,
            "debug": False
        },
        "training": {
            "n_iters": 100,
            "lr": args.lr,
            "momentum": args.momentum,
            "optimiser": args.optimiser,
        },
        "chamfer": {
            "doublesided": args.doublesided
        },
        "name": args.name,
        "vis": {
            "enabled": bool(args.vis_enabled),
            "frequency": 2
        }    
    })
    if args.projection_mode is not None:
        exp_cfg["projection"]["mode"] = args.projection_mode
    if args.alpha is not None:
        exp_cfg["projection"]["alpha"] = args.alpha

    target = args.target_object
    exp_cfg["target"] = args.target_object
    exp_cfg["target_meshes"] = [target]
    exp_cfg["views"] = {
        target: {
            "mode": "manual",
            "view_idx": list(range(12)),
            "num_views": 12
        }
    }
    exp_cfg["target"] = args.target_object

    # Run experiment
    runner = ExperimentRunner(exp_cfg, dataloader)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--exp_base_path", required=True)
    parser.add_argument("--mesh_res", type=int, required=True)
    parser.add_argument("--constrained", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--optimiser", default="SGD")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--velocity_k", type=int, default=1)
    parser.add_argument("--velocity_beta", type=float, default=1.0)
    parser.add_argument("--velocity_enabled", type=int, default=0)
    parser.add_argument("--smoothing_enabled", type=int, default=0)
    parser.add_argument("--vis_enabled", type=int, default=1)
    parser.add_argument("--doublesided", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--ground_label", default="ground")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--projection_mode", choices=["alpha", "mesh"], required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--target_object", required=True)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
