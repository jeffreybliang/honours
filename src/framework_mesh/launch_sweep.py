import json
import subprocess
from multiprocessing import Pool
from pathlib import Path
from copy import deepcopy

# Constants
mesh_res = 3
velocity_k = 1
velocity_beta = 1.0
n_iters = 150
exp_base_path = "./framework_mesh/exp_oblique.json"
outdir = Path("./framework_mesh/sweep_configs")
outdir.mkdir(exist_ok=True)

# Paths for datasets
data_paths = {
    "ground": "./framework_mesh/data_real_ground.json",
    "noground": "./framework_mesh/data_real_noground.json",
}

worker_script = "python3 -m framework_mesh.worker"

def launch_job(args):
    data_path, exp_path, device = args

    cmd = f"{worker_script} --data_path {data_path} --exp_path {exp_path} --device {device}"
    print(f"[LAUNCHING] {cmd}")
    subprocess.run(cmd, shell=True)

def build_jobs():
    jobs = []

    # Load base experiment config
    with open(exp_base_path, 'r') as f:
        base_exp = json.load(f)

    for method in ["normal", "penalty"]:
        for penalty_case in ["none", "lambda0", "lambda01"]:
            if method == "normal" and penalty_case != "none":
                continue
            if method == "penalty" and penalty_case == "none":
                continue

            for ground in ["ground", "noground"]:
                data_path = data_paths[ground]
                doublesided = ground == "noground"
                device = "cuda" if method == "penalty" else "cpu"

                exp = deepcopy(base_exp)

                exp["method"] = "penalty" if method == "penalty" else "projection"

                exp["velocity"] = {
                    "enabled": True,
                    "k": velocity_k,
                    "beta": velocity_beta
                }

                exp["gradient"] = {
                    "smoothing": True,
                    "method": "jacobi",
                    "k": 5,
                    "constrained": False,
                    "debug": False
                }

                exp["training"] = {
                    "n_iters": n_iters,
                    "lr": 1e-4,
                    "momentum": 0.9
                }

                exp["chamfer"] = {
                    "doublesided": doublesided
                }

                if method == "penalty":
                    if penalty_case == "lambda0":
                        exp["penalty"] = {
                            "lambda_init": 0.0,
                            "lambda_max": 0.0,
                            "rate": 1.0
                        }
                    elif penalty_case == "lambda01":
                        exp["penalty"] = {
                            "lambda_init": 0.01,
                            "lambda_max": 1e7,
                            "linear": 100
                        }

                name = f"{method}_{penalty_case}_{ground}_res{mesh_res}"
                exp["name"] = name
                exp_out_path = outdir / f"exp_{name}.json"

                with open(exp_out_path, "w") as f:
                    json.dump(exp, f, indent=2)

                jobs.append((data_path, str(exp_out_path), device))

    return jobs

if __name__ == "__main__":
    jobs = build_jobs()
    print(f"[INFO] Launching {len(jobs)} jobs with concurrency = 3")
    with Pool(processes=3) as pool:
        pool.map(launch_job, jobs)
