import torch
import json
from framework_mesh.dataloader import DataLoader
from framework_mesh.experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def run_experiment(data_path, exp_config_path, device):
    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        base_config = json.load(f)

    for smoothing in [False, True]:
        # Deep copy the config for each setting
        exp_config = json.loads(json.dumps(base_config))
        exp_config["gradient"]["smoothing"] = smoothing

        # Update name to reflect setting
        suffix = "_gradsmooth_True" if smoothing else "_gradsmooth_False"
        exp_config["name"] = exp_config["name"] + suffix

        dataloader = DataLoader(data_config, device)
        runner = ExperimentRunner(exp_config, dataloader)
        print(f"[RUNNING] {exp_config['name']}")
        runner.run()

def run_experiment(data_path, exp_config_path, device):
    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        base_config = json.load(f)

    for smoothing in [False, True]:
        # Deep copy the config for each setting
        exp_config = json.loads(json.dumps(base_config))
        exp_config["gradient"]["smoothing"] = smoothing

        # Update name to reflect setting
        suffix = "_gradsmooth_True" if smoothing else "_gradsmooth_False"
        exp_config["name"] = exp_config["name"] + suffix

        dataloader = DataLoader(data_config, device)
        runner = ExperimentRunner(exp_config, dataloader)
        print(f"[RUNNING] {exp_config['name']}")
        runner.run()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Balloon
    # run_experiment(
    #     data_path="/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/skyconfig_balloon.json",
    #     exp_config_path="/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/exp_balloon.json",
    #     device=device
    # )

    # Parabola
    run_experiment(
        data_path="/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/skyconfig_parabola.json",
        exp_config_path="/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/exp_parabola.json",
        device=device
    )

if __name__ == "__main__":
    main()
