import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    data_path = "/home/jeffrey/honours/src/framework_mesh/obliqueconfig_gpu.json"
    exp_config_path = "/home/jeffrey/honours/src/framework_mesh/exp_oblique.json"

    with open(data_path, 'r') as f:
        data_config = json.load(f)

    with open(exp_config_path, 'r') as f:
        exp_config_base = json.load(f)

    mesh_resolutions = [2, 3]
    smoothing_options = [False, True]

    for mesh_res in mesh_resolutions:
        for smoothing in smoothing_options:
            data_config['paths']['mesh_res'] = mesh_res
            dataloader = DataLoader(data_config, device)

            exp_config = json.loads(json.dumps(exp_config_base))  # deep copy
            exp_config.setdefault("gradient", {})["smoothing"] = smoothing

            if not smoothing:
                exp_config['name'] = f"nosmooth_res_{mesh_res}"
                print(f"[RUN] smoothing=False | mesh_res={mesh_res}")
                runner = ExperimentRunner(exp_config, dataloader)
                runner.run()
            else:
                method = "jacobi"
                for constrained in [False, True]:
                    exp_config['gradient']['method'] = method
                    exp_config['gradient']['constrained'] = constrained
                    exp_config['name'] = f"smooth_{method}_constrained_{constrained}_res_{mesh_res}"
                    print(f"[RUN] smoothing=True | method={method} | constrained={constrained} | mesh_res={mesh_res}")
                    runner = ExperimentRunner(exp_config, dataloader)
                    runner.run()

if __name__ == "__main__":
    main()
