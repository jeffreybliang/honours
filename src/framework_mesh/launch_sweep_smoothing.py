import argparse
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

def sweep_smoothing_k_and_constraints(
    smoothing_ks=(1, 3, 5),
    constrained_flags=(True, False),
    mesh_res_vals=(2, 3),
    trials=2
):
    jobs = []
    mode = "alpha"

    for mesh_res in mesh_res_vals:
        ntrials = trials if mesh_res == 2 else 1
        for obj in objects:
            alpha = alpha_override.get(obj, 5)
            for k in smoothing_ks:
                for constrained in constrained_flags:
                    for trial in range(ntrials):
                        run_name = f"{obj}_res{mesh_res}_k{k}_constr{int(constrained)}_t{trial:02d}"
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
                            smoothing_enabled=1,
                            smoothing_k=k,
                            constrained=constrained,
                            project=f"Shape Smoothing Sweep Res{mesh_res}"
                        )
                        jobs.append(cmd)
    return jobs

def main(n_workers, mode="smoothing"):
    print("mode is", mode)
    
    if mode == "smoothing":
        jobs = sweep_smoothing_k_and_constraints(
            smoothing_ks=[0, 1, 2, 3, 5, 7],
            constrained_flags=[True, False],
            trials=2
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    print(f"Launching {len(jobs)} jobs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        pool.map(run_process, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--mode", type=str, default="smoothing", choices=["smoothing"])
    args = parser.parse_args()
    main(args.n_workers, args.mode)
