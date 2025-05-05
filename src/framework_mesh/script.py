import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    data_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/obliqueconfig_local.json"
    exp_config_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/exp_oblique.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        exp_config_base = json.load(f)

    mesh_resolutions = [2, 3]
    grad_smoothing_options = [False, True]
    velocity_ks = [1, 3]
    velocity_betas = [0.5, 1.0]

    for mesh_res in mesh_resolutions:
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)

        for grad_smoothing in grad_smoothing_options:
            for constrained in ([False, True] if grad_smoothing else [False]):
                for k in velocity_ks:
                    for beta in velocity_betas:
                        # Deep copy experiment config
                        exp_config = json.loads(json.dumps(exp_config_base))

                        # Velocity (displacement) smoothing
                        exp_config["velocity"] = {
                            "enabled": True,
                            "k": k,
                            "beta": beta
                        }

                        # Gradient smoothing
                        exp_config["gradient"] = {
                            "smoothing": grad_smoothing,
                            "method": "jacobi",
                            "k": 5,
                            "constrained": constrained,
                            "debug": False
                        }

                        # Naming
                        name = f"vbeta_{beta}_vk_{k}_"
                        name += f"gradsmooth_{grad_smoothing}"
                        if grad_smoothing:
                            name += f"_constrained_{constrained}"
                        name += f"_res_{mesh_res}"
                        exp_config["name"] = name

                        print(f"[RUN] {name}")
                        runner = ExperimentRunner(exp_config, dataloader)
                        runner.run()

if __name__ == "__main__":
    main()
