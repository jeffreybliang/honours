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

def make_args(**kwargs):
    args = ["python", "-m", "framework_mesh.worker"]
    for k, v in kwargs.items():
        args += [f"--{k}", str(v)]
    return args

def run_process(cmd):
    subprocess.run(cmd)

alpha_override = {
    "Parabola": 9,
    "Spiky": 8,
    "Uneven": 7,
    "Sponge": 6,
    "Cube": 1,
    "Ellipsoid": 3,
    "Cylinder": 2,
    "Diamond": 2,
    "Bottle": 4,
}

def sweep_projection_modes(projection_modes, trials=4):
    mesh_res = 2
    jobs = []
    for mode in projection_modes:
        for obj in objects:
            alpha = alpha_override.get(obj, 5)  # default to 5 if not in dict 
            for trial in range(trials):
                run_name = f"{obj}_proj-{mode}_t{trial:02d}"
                cmd = make_args(
                    data_path=data_path,
                    exp_base_path=exp_path,
                    target_object=obj,
                    projection_mode=mode,
                    mesh_res=mesh_res,
                    alpha=alpha,
                    name=run_name,
                    device=device,
                    vis_enabled=int(trial < 1),
                    velocity_enabled=0,
                    smoothing_enabled=0,
                )
                jobs.append(cmd)
    return jobs


def sweep_alpha_values(alpha_range, mesh_res_values=(2,), trials=1):
    jobs = []
    mode = "alpha"
    for mesh_res in mesh_res_values:
        ntrials = trials if mesh_res == 2 else 1
        for alpha in alpha_range:
            for obj in objects:
                for trial in range(ntrials):
                    run_name = f"{obj}_alpha-{alpha}_res{mesh_res}_t{trial:02d}"
                    cmd = make_args(
                        data_path=data_path,
                        exp_base_path=exp_path,
                        target_object=obj,
                        projection_mode=mode,
                        mesh_res=mesh_res,
                        alpha=alpha,
                        name=run_name,
                        device=device,
                        vis_enabled=int(trial < 1),
                        velocity_enabled=0,
                        smoothing_enabled=0,
                        project=f"REDO Shape Alpha Sweep Res{mesh_res}",
                        lr=5e-6,
                        momentum=0.9,
                        n_iters=80
                    )
                    jobs.append(cmd)
    return jobs


def main(n_workers, mode="alpha"):
    print("mode is", mode)
    # if mode == "projection":
    #     jobs = sweep_projection_modes(["alpha", "mesh"], trials=4)

    alpha_range = [0, 2, 5, 10, 15, 20]
    mesh_res_vals = [2]
    jobs = sweep_alpha_values(alpha_range=alpha_range, 
                                mesh_res_values=mesh_res_vals, 
                                trials=1)

    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--mode", type=str, default="alpha", choices=["projection", "alpha"])
    args = parser.parse_args()
    main(args.n_workers, args.mode)
