# worker.py
import argparse
import json
import torch
import warnings
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner

warnings.filterwarnings("ignore")

def main(args):
    # Load config files
    with open(args.data_path) as f:
        data_cfg = json.load(f)
    with open(args.exp_base_path) as f:
        exp_cfg = json.load(f)

    # Apply mesh resolution
    data_cfg['paths']['mesh_res'] = args.mesh_res
    dataloader = DataLoader(data_cfg, args.device)

    # Update experiment config
    exp_cfg.update({
        "velocity": {
            "enabled": True,
            "k": args.velocity_k,
            "beta": args.velocity_beta
        },
        "gradient": {
            "smoothing": True,
            "method": "jacobi",
            "k": 5,
            "constrained": args.constrained,
            "debug": False
        },
        "training": {
            "n_iters": 150,
            "lr": args.lr,
            "momentum": args.momentum,
            "optimiser": args.optimiser,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "weight_decay": args.weight_decay,
        },
        "chamfer": {
            "doublesided": args.doublesided
        },
        "name": args.name
    })

    # Run experiment
    runner = ExperimentRunner(exp_cfg, dataloader)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--exp_base_path", required=True)
    parser.add_argument("--mesh_res", type=int, required=True)
    parser.add_argument("--constrained", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--optimiser", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--velocity_k", type=int, default=1)
    parser.add_argument("--velocity_beta", type=float, default=1.0)
    parser.add_argument("--doublesided", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--ground_label", default="ground")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", required=True)

    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
