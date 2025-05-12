import torch
import json
from .dataloader import DataLoader
from .experimentrunner import ExperimentRunner
import warnings
warnings.filterwarnings("ignore")


def run_experiments(data_path, exp_base_path, mesh_res, velocity_k, velocity_beta,
                    doublesided, constrained_values, ground_label, device):
    # Load configurations
    with open(data_path, 'r') as f:
        data_config = json.load(f)
    with open(exp_base_path, 'r') as f:
        exp_config_base = json.load(f)

    # Update mesh resolution in data loader
    data_config['paths']['mesh_res'] = mesh_res
    dataloader = DataLoader(data_config, device)

    # Loop over constrained settings
    for constrained in constrained_values:
        # Deep copy of base experiment config
        exp_config = json.loads(json.dumps(exp_config_base))

        # Velocity settings
        exp_config['velocity'] = {
            'enabled': True,
            'k': velocity_k,
            'beta': velocity_beta
        }

        # Gradient smoothing settings
        exp_config['gradient'] = {
            'smoothing': True,
            'method': 'jacobi',
            'k': 5,
            'constrained': constrained,
            'debug': False
        }

        # Training hyperparameters
        exp_config['training'] = {
            'n_iters': 100,
            'lr': 5e-5,
            'momentum': 0.9
        }

        # Chamfer loss setting
        exp_config['chamfer'] = {
            'doublesided': doublesided
        }

        # Name the experiment
        name = (
            f"{ground_label}_vbeta_{velocity_beta}" \
            f"_vk_{velocity_k}_gradsmooth_True" \
            f"_constrained_{constrained}_res_{mesh_res}"
        )
        exp_config['name'] = name

        print(f"[RUN] {name}")
        runner = ExperimentRunner(exp_config, dataloader)
        runner.run()


if __name__ == '__main__':
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')  # override to CPU

    # Common parameters
    mesh_res = 2
    velocity_k = 1
    velocity_beta = 1.0
    exp_base_path = './framework_mesh/exp_oblique.json'

    # No-ground experiments (doublesided chamfer)
    run_experiments(
        data_path='./framework_mesh/data_noground.json',
        exp_base_path=exp_base_path,
        mesh_res=mesh_res,
        velocity_k=velocity_k,
        velocity_beta=velocity_beta,
        doublesided=True,
        constrained_values=[False, True],
        ground_label='noground',
        device=device
    )

    # With-ground experiments (singlesided chamfer)
    run_experiments(
        data_path='./framework_mesh/data_ground.json',
        exp_base_path=exp_base_path,
        mesh_res=mesh_res,
        velocity_k=velocity_k,
        velocity_beta=velocity_beta,
        doublesided=False,
        constrained_values=[False, True],
        ground_label='ground',
        device=device
    )
