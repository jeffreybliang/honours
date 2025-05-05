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
        exp_config = json.load(f)

    smoothing = exp_config.get("gradient", {}).get("smoothing", False)
    mesh_resolutions = [2, 3]

    for mesh_res in mesh_resolutions:
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)

        if not smoothing:
            exp_config['name'] = f"nosmooth_res_{mesh_res}"
            print(f"[RUN] smoothing=False | mesh_res={mesh_res}")
            runner = ExperimentRunner(exp_config, dataloader)
            runner.run()

        else:
            method = "jacobi"
            constrained = False
            exp_config['gradient']['method'] = method
            exp_config['gradient']['constrained'] = constrained
            exp_config['name'] = f"smooth_{method}_constrained_{constrained}_res_{mesh_res}"
            print(f"[RUN] smoothing=True | method={method} | constrained={constrained} | mesh_res={mesh_res}")
            runner = ExperimentRunner(exp_config, dataloader)
            runner.run()

if __name__ == "__main__":
    main()
