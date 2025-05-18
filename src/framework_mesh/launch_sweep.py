import argparse
import json
import subprocess
from multiprocessing import Pool

# --- CONFIG ---
data_path = "./framework_mesh/data_diffuse.json"
exp_path = "./framework_mesh/exp_all_diffuse.json"
device = "cpu"

objects = [
    "Balloon", "Biconvex", "Bottle", "Cube", "Cylinder",
    "Diamond", "Ellipsoid", "Parabola", "Spiky", "Sponge",
    "Strawberry", "Tear", "Turnip", "Uneven"
]

# bad_views = {
#     "Balloon": [],
#     "Biconvex": [7, 8],
#     "Bottle": [],
#     "Cube": [],
#     "Cylinder": [3, 4, 5, 11],
#     "Diamond": [],
#     "Ellipsoid": [0, 2],
#     "Parabola": [2,6,8],
#     "Spiky": [],
#     "Sponge": [7],
#     "Strawberry": [2, 4, 10],
#     "Tear": [0,3],
#     "Turnip": [],
#     "Uneven":[]
# }

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

alpha_override = {
    "Diamond": 2,
    "Balloon": 2,
    "Strawberry": 2,
    "Spiky": 8,
    "Parabola": 10,
    "Cube": 5,
    "Cylinder": 2,
    "Bottle": 5,
    "Biconvex": 5,
    "Uneven": 8,
    "Tear": 7,
    "Turnip": 5,
    "Ellipsoid": 2,
    "Sponge": 5
}


def sweep_projection_modes(projection_modes, trials=1):
    mesh_res = 3
    jobs = []
    for mode in projection_modes:
        for obj in objects:
            alpha = alpha_override.get(obj, 5)
            for doublesided in [0, 1]:
                for trial in range(trials):
                    run_name = f"{obj}_proj-{mode}_ds{doublesided}_t{trial:02d}"
                    cmd = make_args(
                        data_path=data_path,
                        exp_base_path=exp_path,
                        target_object=obj,
                        projection_mode=mode,
                        mesh_res=mesh_res,
                        alpha=alpha,
                        name=run_name,
                        device=device,
                        vis_enabled=1,
                        velocity_enabled=0,
                        smoothing_enabled=0,
                        lr=5e-6 if doublesided==1 else 5e-6,
                        momentum=0.9,
                        doublesided=doublesided,
                        project="Mesh vs Alpha v2 Res3",
                        n_iters=120,
                    )
                    jobs.append(cmd)
    return jobs

def main(n_workers):
    projection_modes = ["alpha", "mesh"]
    jobs = sweep_projection_modes(projection_modes, trials=1)
    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    args = parser.parse_args()
    main(args.n_workers)
