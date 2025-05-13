import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    data_path = "./framework_mesh/data_noground.json"
    exp_path = "./framework_mesh/exp_spherediff.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_path, 'r') as f:
        exp_config_base = json.load(f)

    mesh_res = 2
    velocity_k = 1
    velocity_beta = 1.0

    data_config['paths']['mesh_res'] = mesh_res
    dataloader = DataLoader(data_config, device)

    for constrained in [False]:
        exp_config = json.loads(json.dumps(exp_config_base))  # deep copy

        # Velocity settings (unchanged)
        exp_config["velocity"] = {
            "enabled": True,
            "k": velocity_k,
            "beta": velocity_beta
        }

        # Gradient smoothing
        exp_config["gradient"] = {
            "smoothing": True,
            "method": "jacobi",
            "k": 5,
            "constrained": constrained,
            "debug": False
        }

        # Training hyperparameters
        # exp_config["training"] = {
        #     "n_iters": 150,
        #     "lr": 5e-5,
        #     "momentum": 0.8
        # }

        # Chamfer loss: double-sided
        exp_config["chamfer"] = {
            "doublesided": True
        }

        # Naming
        name = f"clean_vbeta_{velocity_beta}_vk_{velocity_k}_gradsmooth_True_constrained_{constrained}_res_{mesh_res}"
        exp_config["name"] = name

        print(f"[RUN] {name}")
        runner = ExperimentRunner(exp_config, dataloader)
        runner.run()


if __name__ == "__main__":
    main()
