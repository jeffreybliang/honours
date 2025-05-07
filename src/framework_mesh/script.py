import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def run_with_configs(data_path, exp_path, mesh_res_list, enable_doublesided=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_path, 'r') as f:
        exp_config_base = json.load(f)

    grad_smoothing = True
    velocity_k = 1
    velocity_beta = 1.0

    for mesh_res in mesh_res_list:
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)

        for constrained in [False, True] if mesh_res != 3 else [False]:  # for res=3 only run constrained=False unless explicitly stated
            exp_config = json.loads(json.dumps(exp_config_base))
            exp_config["velocity"] = {
                "enabled": True,
                "k": velocity_k,
                "beta": velocity_beta
            }

            exp_config["gradient"] = {
                "smoothing": grad_smoothing,
                "method": "jacobi",
                "k": 5,
                "constrained": constrained,
                "debug": False
            }

            # Enable double-sided Chamfer if requested
            if enable_doublesided:
                exp_config["chamfer"] = {
                    "doublesided": True
                }

            name = f"vbeta_{velocity_beta}_vk_{velocity_k}_gradsmooth_{grad_smoothing}_constrained_{constrained}_res_{mesh_res}"
            exp_config["name"] = name

            print(f"[RUN] {name}")
            runner = ExperimentRunner(exp_config, dataloader)
            runner.run()

def main():
    # First set: obliqueconfig, res=2
    run_with_configs(
        "./framework_mesh/obliqueconfig.json",
        "./framework_mesh/exp_oblique.json",
        mesh_res_list=[2],
        enable_doublesided=False
    )

    # Second set: data_noground, res=2 and 3, with double-sided Chamfer
    run_with_configs(
        "./framework_mesh/data_noground.json",
        "./framework_mesh/exp_oblique.json",
        mesh_res_list=[2, 3],
        enable_doublesided=True
    )

    # Final run: obliqueconfig, res=3, constrained=True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open("./framework_mesh/obliqueconfig.json", 'r') as f:
        data_config = json.load(f)
    with open("./framework_mesh/exp_oblique.json", 'r') as f:
        exp_config_base = json.load(f)

    data_config['paths']['mesh_res'] = 3
    dataloader = DataLoader(data_config, device)

    exp_config = json.loads(json.dumps(exp_config_base))
    exp_config["velocity"] = {"enabled": True, "k": 1, "beta": 1.0}
    exp_config["gradient"] = {
        "smoothing": True,
        "method": "jacobi",
        "k": 5,
        "constrained": True,
        "debug": False
    }
    exp_config["name"] = "vbeta_1.0_vk_1_gradsmooth_True_constrained_True_res_3"

    print(f"[RUN] {exp_config['name']}")
    runner = ExperimentRunner(exp_config, dataloader)
    runner.run()

if __name__ == "__main__":
    main()
