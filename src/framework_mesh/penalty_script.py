import json
import subprocess
import argparse
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool

velocity_k = 1
velocity_beta = 1.0
iters_to_sweep = [150]
constrained = False
ground_options = ["ground", "noground"]
mesh_resolutions = [2, 3]

worker_script = "python3 -m framework_mesh.worker"  # adjust if not using as module
outdir = Path("./framework_mesh/sweep_configs")
outdir.mkdir(exist_ok=True)

def launch_job(args):
    data_config, exp_config, name = args

    data_out = outdir / f"data_{name}.json"
    exp_out = outdir / f"exp_{name}.json"

    with open(data_out, 'w') as f:
        json.dump(data_config, f, indent=2)
    with open(exp_out, 'w') as f:
        json.dump(exp_config, f, indent=2)

    cmd = f"{worker_script} --data_path {data_out} --exp_path {exp_out}"
    print(f"[LAUNCHING] {cmd}")
    subprocess.run(cmd, shell=True)


def collect_jobs(filter_penalty_type=None):
    jobs = []

    for ground_label in ground_options:
        data_path_template = f"./framework_mesh/data_{ground_label}.json"
        with open(data_path_template, 'r') as f:
            data_config_base = json.load(f)

        for mesh_res in mesh_resolutions:
            for n_iters in iters_to_sweep:
                for use_penalty in [False, True]:
                    penalty_label = "volpen" if use_penalty else "novolpen"
                    if filter_penalty_type and filter_penalty_type != penalty_label:
                        continue  # skip if doesn't match the filter

                    data_config = deepcopy(data_config_base)
                    data_config['paths']['mesh_res'] = mesh_res

                    exp_path = "./framework_mesh/exp_penalty.json"
                    with open(exp_path, 'r') as f:
                        exp_config = json.load(f)

                    exp_config["method"] = "penalty"
                    if use_penalty:
                        exp_config["penalty"] = {
                            "lambda_init": 0.01,
                            "lambda_max": 1e7,
                            "rate": 500,
                        }
                    else:
                        exp_config["penalty"] = {
                            "lambda_init": 0.0,
                            "lambda_max": 0.0,
                            "rate": 1.0
                        }

                    exp_config["velocity"] = {
                        "enabled": True,
                        "k": velocity_k,
                        "beta": velocity_beta
                    }

                    exp_config["gradient"] = {
                        "smoothing": True,
                        "method": "jacobi",
                        "k": 5,
                        "constrained": constrained,
                        "debug": False
                    }

                    exp_config["training"] = {
                        "n_iters": n_iters,
                        "lr": 1e-4,
                        "momentum": 0.9
                    }

                    exp_config["chamfer"] = {
                        "doublesided": ground_label == "noground"
                    }

                    name = (
                        f"PENALTY_{penalty_label}_{ground_label}_vbeta_{velocity_beta}_"
                        f"vk_{velocity_k}_gradsmooth_True_constrained_{constrained}_"
                        f"res_{mesh_res}_n{n_iters}"
                    )
                    exp_config["name"] = name

                    jobs.append((data_config, exp_config, name))

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty_type", type=str, choices=["volpen", "novolpen"], default=None,
                        help="Optional: filter to only run 'volpen' or 'novolpen' jobs")
    args = parser.parse_args()

    jobs = collect_jobs(filter_penalty_type=args.penalty_type)
    print(f"[INFO] Launching {len(jobs)} jobs with max concurrency = 3")

    with Pool(processes=3) as pool:
        pool.map(launch_job, jobs)
