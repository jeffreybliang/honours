import argparse
import json
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cuda"
mesh_res = 2
trials = 10

objects = [
    "Balloon", "Spiky", "Uneven", "Parabola", "Cube", "Strawberry",
    "Cylinder", "Diamond", "Biconvex", "Ellipsoid"
]

alpha_override = {
    "Balloon": 2, "Spiky": 10, "Uneven": 10, "Parabola": 10,
    "Cube": 5, "Strawberry": 2, "Cylinder": 2, "Diamond": 2,
    "Biconvex": 5, "Ellipsoid": 2,
}

view_counts = [1, 2, 3, 4, 6, 8, 9, 10, 12]
projection_modes = ["alpha"]

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def sweep_penalty_views():
    jobs = []

    for penalty_mode in ["lambda0"]:
        for mode in projection_modes:
            for obj in objects:
                alpha = alpha_override.get(obj, 5)
                for num_views in view_counts:
                    trial_count = trials - num_views//2
                    for trial in range(trial_count):
                        vis = int(trial == 0)
                        penalty_args = {}

                        if penalty_mode == "lambda0":
                            penalty_args = {
                                "penalty_lambda_init": 0.0,
                                "penalty_lambda_max": 0.0
                            }
                        elif penalty_mode == "lambda01":
                            penalty_args = {
                                "penalty_lambda_init": 0.01,
                                "penalty_lambda_max": 1e7,
                                "penalty_linear": 100
                            }

                        run_name = f"{obj}_views-{num_views}_penalty-{penalty_mode}_t{trial:02d}"

                        cmd = make_args(
                            data_path=data_path,
                            exp_base_path=exp_path,
                            target_object=obj,
                            projection_mode=mode,
                            mesh_res=mesh_res,
                            alpha=alpha,
                            name=run_name,
                            device=device,
                            vis_enabled=vis,
                            # velocity_enabled=1,
                            # velocity_k=1,
                            # velocity_beta=1.0,
                            smoothing_enabled=0,
                            constrained="false",
                            view_mode="random",
                            num_views=num_views,
                            project=f"Shape Num Views Res{mesh_res}",
                            optimiser="SGD",
                            n_iters=100,
                            lr=5e-6,
                            momentum=0.87,
                            **penalty_args
                        )
                        jobs.append(cmd)

    return jobs

def main(n_workers):
    jobs = sweep_penalty_views()
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    main(args.n_workers)
