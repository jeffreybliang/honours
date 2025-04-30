import torch
import json
from itertools import product
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    data_path = "/home/jeffrey/honours/src/framework_mesh/skyconfig_gpu.json"
    exp_config_path = "/home/jeffrey/honours/src/framework_mesh/exp_gradients.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        exp_config = json.load(f)

    methods = ["invhop", "khop"]
    constrained_options = [True, False]
    mesh_resolutions = [2, 3]

    for mesh_res in mesh_resolutions:
        # Update mesh resolution
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)

        for method in methods:
            for constrained in constrained_options:
                # Update gradient method and constraint setting
                exp_config['gradient']['method'] = method
                exp_config['gradient']['constrained'] = constrained

                # Update experiment name to be descriptive
                exp_config['name'] = f"pipeline_{method}_constrained_{constrained}_res_{mesh_res}"

                # Create runner and run
                runner = ExperimentRunner(exp_config, dataloader)
                runner.run()

if __name__ == "__main__":
    main()
