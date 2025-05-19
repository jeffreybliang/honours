import argparse
import json
import torch
from framework_mesh.dataloader import DataLoader  # Adjust import as needed
from framework_mesh.experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--exp_path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.data_path, "r") as f:
        data_config = json.load(f)
    with open(args.exp_path, "r") as f:
        exp_config = json.load(f)

    print(f"[INFO] Running {exp_config['name']} on {device}")

    # Instantiate dataloader with device
    dataloader = DataLoader(data_config, device=device)

    # Run experiment
    runner = ExperimentRunner(exp_config, dataloader)
    runner.run()

if __name__ == "__main__":
    main()
