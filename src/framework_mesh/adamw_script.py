import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def run_experiments(data_path, exp_base_path, mesh_res_values, velocity_k, velocity_beta,
                    doublesided, constrained_map, ground_label, device):
    # Load configurations
    with open(data_path, 'r') as f:
        data_config = json.load(f)
    with open(exp_base_path, 'r') as f:
        exp_config_base = json.load(f)

    for mesh_res in mesh_res_values:
        data_config['paths']['mesh_res'] = mesh_res
        dataloader = DataLoader(data_config, device)
        for constrained in constrained_map.get(mesh_res, []):
            # Deep copy config
            exp_config = json.loads(json.dumps(exp_config_base))

            exp_config['velocity'] = {
                'enabled': True,
                'k': velocity_k,
                'beta': velocity_beta
            }

            exp_config['gradient'] = {
                'smoothing': True,
                'method': 'jacobi',
                'k': 5,
                'constrained': constrained,
                'debug': False
            }

            exp_config['training'] = {
                'n_iters': 150,
                'lr': 5e-3,
                'momentum': 0.9,
                'optimiser': "AdamW",
                'beta1': 0.9,
                'beta2': 0.9,
                'weight_decay': 0.0
            }

            exp_config['chamfer'] = {
                'doublesided': doublesided
            }

            name = (
                f"{ground_label}_AdamW_vbeta_{velocity_beta}"
                f"_vk_{velocity_k}_gradsmooth_True"
                f"_constrained_{constrained}_res_{mesh_res}"
            )
            exp_config['name'] = name

            print(f"[RUN] {name}")
            runner = ExperimentRunner(exp_config, dataloader)
            runner.run()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')  # force CPU
    exp_base_path = './framework_mesh/exp_oblique_AdamW.json'
    velocity_k = 1
    velocity_beta = 1.0

    constrained_map = {
        2: [False, True],
        3: [False]
    }

    run_experiments(
        data_path='./framework_mesh/data_noground.json',
        exp_base_path=exp_base_path,
        mesh_res_values=[3],
        velocity_k=velocity_k,
        velocity_beta=velocity_beta,
        doublesided=True,
        constrained_map=constrained_map,
        ground_label='noground',
        device=device
    )

    run_experiments(
        data_path='./framework_mesh/data_ground.json',
        exp_base_path=exp_base_path,
        mesh_res_values=[2, 3],
        velocity_k=velocity_k,
        velocity_beta=velocity_beta,
        doublesided=False,
        constrained_map=constrained_map,
        ground_label='ground',
        device=device
    )
