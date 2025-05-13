# run_worker.py
import torch
import json
import argparse
from pathlib import Path
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--exp_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.data_path, 'r') as f:
        data_config = json.load(f)
    with open(args.exp_path, 'r') as f:
        exp_config = json.load(f)

    mesh_res = data_config['paths']['mesh_res']
    dataloader = DataLoader(data_config, device)

    print(f"[RUN] {exp_config['name']}")
    runner = ExperimentRunner(exp_config, dataloader)
    runner.run()


if __name__ == "__main__":
    main()
