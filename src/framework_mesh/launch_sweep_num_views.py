import argparse
import json
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"
mesh_res = 2

objects = [
    "Balloon", "Spiky", "Uneven", "Parabola", "Cube", "Strawberry"
]

alpha_override = {
    "Balloon": 2,
    "Spiky": 10,
    "Uneven": 10,
    "Parabola": 10,
    "Cube": 5,
    "Strawberry": 2,
}

view_counts = [1, 2, 3, 4, 6, 9, 12]
projection_modes = ["alpha"]  # Singleton for future extensibility
trials = 4

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

def sweep_view_counts():
    jobs = []
    for mode in projection_modes:
        for obj in objects:
            alpha = alpha_override[obj]
            for num_views in view_counts:
                for trial in range(trials):
                    vis = int(trial == 0)
                    run_name = f"{obj}_views-{num_views}_proj-{mode}_t{trial:02d}"
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
                        velocity_enabled=0,
                        smoothing_enabled=0,
                        view_mode="random",
                        num_views=num_views,
                        project=f"Shape Num Views Res{mesh_res}",
                        n_iters=100,
                        lr=5e-6,
                        momentum=0.85
                    )
                    jobs.append(cmd)
    return jobs

def main(n_workers):
    jobs = sweep_view_counts()
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=2)
    args = parser.parse_args()
    main(args.n_workers)
