# worker.py
import argparse
import torch
import json
from framework_mesh.dataloader import DataLoader
from framework_mesh.experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def run_experiment(exp_path, mesh_res, constrained, ground):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    data_path = f"./framework_mesh/data_{ground}.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)
    with open(exp_path, 'r') as f:
        exp_config_base = json.load(f)

    data_config['paths']['mesh_res'] = mesh_res
    dataloader = DataLoader(data_config, device)

    exp_config = json.loads(json.dumps(exp_config_base))  # deep copy

    exp_config["velocity"] = {
        "enabled": True,
        "k": 1,
        "beta": 1.0
    }

    exp_config["gradient"] = {
        "smoothing": True,
        "method": "jacobi",
        "k": 5,
        "constrained": constrained,
        "debug": False
    }

    exp_config["chamfer"] = {
        "doublesided": ground == "noground"
    }

    exp_name = exp_path.split("/")[-1].split(".")[0]
    name = f"vel_{True}_{exp_name}_res{mesh_res}_constr{constrained}_{ground}"
    exp_config["name"] = name

    print(f"[WORKER RUN] {name}")
    runner = ExperimentRunner(exp_config, dataloader)
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--mesh_res", type=int, required=True)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--ground", type=str, choices=["ground", "noground"], required=True)
    args = parser.parse_args()

    run_experiment(args.exp_path, args.mesh_res, args.constrained, args.ground)
