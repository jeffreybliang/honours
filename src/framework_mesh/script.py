import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    data_path = "./framework_mesh/data_noground.json"
    exp_path = "./framework_mesh/exp_penalty.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_path, 'r') as f:
        exp_config_base = json.load(f)

    velocity_k = 1
    velocity_beta = 1.0
    iters_to_sweep = [100]

    for mesh_res in [3]:
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)

        constrained_options = [False]

        for constrained in constrained_options:
            for n_iters in iters_to_sweep:
                exp_config = json.loads(json.dumps(exp_config_base))  # deep copy

                exp_config["method"] = "penalty"

                exp_config["velocity"] = {
                    "enabled": True,
                    "k": velocity_k,
                    "beta": velocity_beta
                }

                exp_config["gradient"] = {
                    "smoothing": True,
                    "method": "jacobi",
                    "k": 5,
                    "constrained": constrained,
                    "debug": False
                }

                exp_config["training"] = {
                    "n_iters": n_iters,
                    "lr": 5e-5,
                    "momentum": 0.9
                }

                exp_config["chamfer"] = {
                    "doublesided": True
                }

                name = (
                    f"PENALTY_clean_vbeta_{velocity_beta}_vk_{velocity_k}_"
                    f"gradsmooth_True_constrained_{constrained}_res_{mesh_res}_n{n_iters}"
                )
                exp_config["name"] = name

                print(f"[RUN] {name}")
                runner = ExperimentRunner(exp_config, dataloader)
                runner.run()


if __name__ == "__main__":
    main()
