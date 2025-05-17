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

    data_cfg['paths']['mesh_res'] = args.mesh_res
    dataloader = DataLoader(data_cfg, args.device)

    # Patch config only if args are provided
    exp_cfg.setdefault("velocity", {})
    if args.velocity_enabled is not None:
        exp_cfg["velocity"]["enabled"] = bool(args.velocity_enabled)
    if args.velocity_k is not None:
        exp_cfg["velocity"]["k"] = args.velocity_k
    if args.velocity_beta is not None:
        exp_cfg["velocity"]["beta"] = args.velocity_beta

    exp_cfg.setdefault("gradient", {})
    if args.smoothing_enabled is not None:
        exp_cfg["gradient"]["smoothing"] = bool(args.smoothing_enabled)
    exp_cfg["gradient"].update({
        "method": "jacobi",
        "k": 5,
        "constrained": args.constrained,
        "debug": False
    })

    exp_cfg.setdefault("training", {})
    if args.lr is not None:
        exp_cfg["training"]["lr"] = args.lr
    if args.momentum is not None:
        exp_cfg["training"]["momentum"] = args.momentum
    if args.optimiser is not None:
        exp_cfg["training"]["optimiser"] = args.optimiser
    exp_cfg["training"].setdefault("n_iters", 100)

    exp_cfg.setdefault("chamfer", {})
    if args.doublesided is not None:
        exp_cfg["chamfer"]["doublesided"] = args.doublesided

    if args.vis_enabled is not None:
        exp_cfg["vis"] = {"enabled": bool(args.vis_enabled), "frequency": 2}

    if args.projection_mode is not None:
        exp_cfg.setdefault("projection", {})
        exp_cfg["projection"]["mode"] = args.projection_mode
    if args.alpha is not None:
        exp_cfg.setdefault("projection", {})
        exp_cfg["projection"]["alpha"] = args.alpha

    exp_cfg["name"] = args.name
    exp_cfg["target"] = args.target_object
    exp_cfg["target_meshes"] = [args.target_object]
    exp_cfg["views"] = {
        args.target_object: {
            "mode": "manual",
            "view_idx": list(range(12)),
            "num_views": 12
        }
    }

    runner = ExperimentRunner(exp_cfg, dataloader)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--exp_base_path", required=True)
    parser.add_argument("--mesh_res", type=int, required=True)
    parser.add_argument("--constrained", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--optimiser", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--velocity_k", type=int, default=None)
    parser.add_argument("--velocity_beta", type=float, default=None)
    parser.add_argument("--velocity_enabled", type=int, default=None)
    parser.add_argument("--smoothing_enabled", type=int, default=None)
    parser.add_argument("--vis_enabled", type=int, default=None)
    parser.add_argument("--doublesided", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--ground_label", default="ground")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", required=True)
    parser.add_argument("--projection_mode", choices=["alpha", "mesh"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--target_object", required=True)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
