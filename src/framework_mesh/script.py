import torch
import json
from itertools import product
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


import torch
import json
from framework_mesh.dataloader import DataLoader
from framework_mesh.experimentrunner import ExperimentRunner

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # --- First experiment ---
    # data_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/skyconfig_balloon.json"
    # exp_config_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/exp_gradients.json"

    # with open(data_path, 'r') as f:
    #     data_config = json.load(f)

    # with open(exp_config_path, 'r') as f:
    #     exp_config = json.load(f)

    # dataloader = DataLoader(data_config, device)
    # runner = ExperimentRunner(exp_config, dataloader)
    # runner.run()

    # --- Second experiment ---
    data_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/skyconfig_strawberry.json"
    exp_config_path = "/Users/jeffreyliang/Documents/Honours/honours/src/framework_mesh/exp_strawberry.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        exp_config = json.load(f)

    dataloader = DataLoader(data_config, device)
    runner = ExperimentRunner(exp_config, dataloader)
    runner.run()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
